from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

class HybridAPIRetriever(BaseRetriever):
    """
    Custom retriever that combines BM25 keyword search with Chroma vector similarity 
    to retrieve the most relevant API descriptions.
    """

    def __init__(self, chroma_db: Chroma, k: int = 5):
        """
        Initialize the retriever.

        :param chroma_db: The Chroma vector database.
        :param k: Number of top results to return.
        """
        self.k = k
        self.chroma_db = chroma_db  # Use Chroma for vector search

        # Load API descriptions from Chroma metadata (used for BM25)
        self.api_documents = chroma_db.get()["documents"]
        self.api_descriptions = [doc["page_content"] for doc in self.api_documents]
        self.api_ids = [doc["metadata"]["name"] for doc in self.api_documents]

        # BM25 for keyword-based search
        self.bm25 = BM25Okapi([desc.split() for desc in self.api_descriptions])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Retrieve relevant API descriptions based on a hybrid search method.

        :param query: The input query string.
        :return: A list of relevant API descriptions wrapped in Document objects.
        """

        # --- Step 1: Perform BM25 Keyword Search ---
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.k]
        bm25_results = [(self.api_ids[i], bm25_scores[i]) for i in bm25_top_indices]

        # --- Step 2: Perform Chroma Vector Search ---
        vector_results = self.chroma_db.similarity_search(query, k=self.k)
        vector_names = {doc.metadata["name"]: doc for doc in vector_results}

        # --- Step 3: Combine BM25 + Vector Results ---
        final_scores = {}
        for api_name, bm25_score in bm25_results:
            vector_score = 1.0 if api_name in vector_names else 0.0  # Check if present in Chroma results
            final_scores[api_name] = bm25_score + vector_score

        # Sort APIs based on combined scores
        ranked_apis = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # --- Step 4: Convert to LangChain Documents ---
        retrieved_docs = []
        for api_name, score in ranked_apis[:self.k]:
            api_info = vector_names.get(api_name)  # Get full API description from Chroma
            retrieved_docs.append(Document(page_content=api_info.page_content, metadata={"name": api_name, "score": score}))

        return retrieved_docs
