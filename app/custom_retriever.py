from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from pydantic import PrivateAttr

class HybridAPIRetriever(BaseRetriever):
    """
    Custom retriever that combines BM25 keyword search with Chroma vector similarity.
    """

    _chroma_db: Chroma = PrivateAttr()  # Prevents Pydantic from pickling ChromaDB
    k: int = 5  # Number of results to retrieve
    _api_documents: List[Document] = PrivateAttr()
    _api_descriptions: List[str] = PrivateAttr()
    _api_ids: List[str] = PrivateAttr()
    _bm25: BM25Okapi = PrivateAttr()

    def __init__(self, chroma_db: Chroma, k: int = 5):
        """
        Initialize the retriever.

        :param chroma_db: The Chroma vector database.
        :param k: Number of top results to return.
        """
        super().__init__()
        self._chroma_db = chroma_db  # Store Chroma as a private attribute
        self.k = k

        # Load API descriptions from Chroma metadata (used for BM25)
        api_data = self._chroma_db.get()

        if "documents" not in api_data or not api_data["documents"]:
            raise ValueError("Chroma vector store is empty or improperly configured.")

        # Convert raw documents into LangChain Document objects
        self._api_documents = [
            Document(
                page_content=doc,
                metadata=api_data["metadatas"][i] if api_data["metadatas"] else {}
            ) 
            for i, doc in enumerate(api_data["documents"])
        ]

        # Extract API IDs from "seq_num" (fallback to "API_X" if missing)
        self._api_ids = [
            str(doc.metadata.get("seq_num", f"API_{i}"))  
            for i, doc in enumerate(self._api_documents)
        ]

        # Initialize BM25 model
        self._api_descriptions = [doc.page_content for doc in self._api_documents]
        self._bm25 = BM25Okapi([desc.split() for desc in self._api_descriptions])


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Retrieve relevant API descriptions based on hybrid search (BM25 + Chroma).
        """
        # --- Step 1: Perform BM25 Keyword Search ---
        bm25_scores = self._bm25.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.k]
        bm25_results = [(self._api_ids[i], bm25_scores[i]) for i in bm25_top_indices]

        # --- Step 2: Perform Chroma Vector Search ---
        vector_results = self._chroma_db.similarity_search(query, k=self.k)

        # Fix: Ensure vector search results use correct metadata key
        vector_names = {}
        for doc in vector_results:
            seq_num = str(doc.metadata.get("seq_num", f"API_{vector_results.index(doc)}"))  # Use seq_num as ID
            vector_names[seq_num] = doc  # Store properly indexed results
        
        # Add BM25 results to vector_names if they do not exist
        for api_name, _ in bm25_results:
            if api_name not in vector_names:
            # Find the corresponding document in _api_documents
                doc_index = self._api_ids.index(api_name)
                vector_names[api_name] = self._api_documents[doc_index]
              
        # --- Step 3: Combine BM25 + Vector Results ---
        final_scores = {}
        all_api_names = set([api_name for api_name, _ in bm25_results] + list(vector_names.keys()))

        for api_name in all_api_names:
            bm25_score = next((score for name, score in bm25_results if name == api_name), 0.0)
            vector_score = 5.0 if api_name in vector_names else 0.0  # Check if present in Chroma results
            final_scores[api_name] = bm25_score + vector_score

        # Sort APIs based on combined scores
        ranked_apis = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # --- Step 4: Convert to LangChain Documents ---
        retrieved_docs = []
        for api_name, score in ranked_apis[:self.k]:
            api_info = vector_names.get(api_name)  # Get document from vector results

            # Ensure api_info is always a valid Document
            if not isinstance(api_info, Document):  
                api_info = Document(
                    page_content="No description available." if api_info is None else str(api_info),
                    metadata={"seq_num": api_name, "source": "Retrieved from ChromaDB"}
                )

            retrieved_docs.append(Document(
                page_content=api_info.page_content,
                metadata={"seq_num": api_name, "score": score, "source": api_info.metadata.get("source", "Unknown")}
            ))

        return retrieved_docs