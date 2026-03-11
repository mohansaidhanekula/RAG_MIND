"""
Document loaders for PDF, DOCX, TXT/MD, and Web pages.
"""

import os
import requests
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_web(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Attach source URL as metadata
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def load_file(file_path: str) -> List[Document]:
    """Route to correct loader based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in {".txt", ".md"}:
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
