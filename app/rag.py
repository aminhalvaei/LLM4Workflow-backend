import json
from collections import defaultdict
from pathlib import Path
from typing import List, Union

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from app.custom_retriever import HybridAPIRetriever
from app.json_loader import JSONLoader
from app.utils import MODEL


REWRITE_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate {k} alternative queries to retrieve relevant 
APIs (to execute tasks in scientific workflows) from a vector database. 
By generating better queries based on the user's task list, your goal is to help the user overcome some of the limitations of distance-based similarity search. 
Provide these alternative queries separated by newlines. 
Original user task list: {text}
"""


def unique_union_documents(docs: List[Document]) -> List[Document]:
    """Deduplicate documents."""
    unique_documents_dict = {
        (doc.page_content, json.dumps(doc.metadata, sort_keys=True)): doc
        for doc in docs
    }

    unique_documents = list(unique_documents_dict.values())
    print(f"**Retrieved {len(unique_documents)} relevant documents**")
    return unique_documents


def create_rewrite_chain():
    def parse(ai_message: AIMessage) -> List[str]:
        """Split the AI message into a list of queries"""
        content_list = ai_message.content.strip().split("\n\n")
        return content_list

    rewrite_chain = ChatPromptTemplate.from_template(REWRITE_QUERY_PROMPT) | MODEL | parse
    return rewrite_chain


rewrite_query_chain = create_rewrite_chain()


class RAG:
    def __init__(self, db_directory, collection_name='default', doc_path=None):
        self.persist_directory = db_directory
        self.model = MODEL
        self.collection_name = collection_name
        self.documents = None
        if doc_path:
            self.doc_path = Path(doc_path)
            self.documents = self.doc_loader()
        else:
            self.retriever = self.create_retriever()


    def doc_loader(self):
        loader = JSONLoader(
            file_path=self.doc_path,
            text_content=False, json_lines=False)
        data = loader.load()
        return data

    # This method is used to make the retriever inside of the class
    def create_retriever(self) -> HybridAPIRetriever:
        """Create a custom retriever using HybridAPIRetriever with ChromaDB."""
        embeddings = OpenAIEmbeddings(disallowed_special=())

        # Load Chroma vector store
        chroma_vectorstore = Chroma(
            persist_directory=self.persist_directory, 
            collection_name=self.collection_name, 
            embedding_function=embeddings
        )

        # Use our custom retriever with ChromaDB
        return HybridAPIRetriever(chroma_db=chroma_vectorstore, k=3)
    

    # This is the main method to use from outside of the RAG class to access the API Retrieval 
    def mq_retrieve_documents(self, queries):
        relevant_docs = []
        
        # Retrieve documents for each query
        for query in queries:
            docs_with_score = self.retriever.invoke(query)
            relevant_docs.extend(docs_with_score)

        # Group by document identifier and sum scores
        grouped_docs = defaultdict(lambda: {'doc': None, 'total_score': 0})
        
        for doc in relevant_docs:
            # Use a unique identifier from metadata
            doc_key = doc.metadata.get("seq_num", doc.page_content)
            doc_score = doc.metadata.get('score', 0)

            if grouped_docs[doc_key]['doc'] is None:
                grouped_docs[doc_key]['doc'] = doc
            
            # Sum the score for all occurrences
            grouped_docs[doc_key]['total_score'] += doc_score

        # Update metadata with total_score
        for doc_info in grouped_docs.values():
            doc_info['doc'].metadata['total_score'] = doc_info['total_score']

        # Prepare final list of unique documents
        relevant_docs = [doc_info['doc'] for doc_info in grouped_docs.values()]
        
        # Sort the final result based on the total_score
        sorted_docs = sorted(relevant_docs, key=lambda doc: doc.metadata.get('total_score', 0), reverse=True)
        
        unique_union_docs = unique_union_documents(sorted_docs)
        return [{'doc': doc, 'status': 1} for doc in unique_union_docs]
        

    def db_retrieve_documents(self, queries: List[str]):
        relevant_docs_with_query = defaultdict(list)
        for idx, query in enumerate(queries):
            docs = self.retriever.invoke(query)
            relevant_docs_with_query[idx + 1].extend(docs)
        return dict(relevant_docs_with_query)

