"""
Vector store logic using ChromaDB and HuggingFace embeddings.
"""
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )

def add_documents_to_vectorstore(vectorstore: Chroma, chunks: List[Document]) -> Chroma:
    vectorstore.add_documents(chunks)
    return vectorstore
