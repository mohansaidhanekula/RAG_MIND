"""
RAG chain: retrieval + Groq generation with source citations.
"""
from typing import List, Tuple, Generator
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

GROQ_MODELS = {
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "gemma2-9b-it": "Gemma 2 9B"
}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context documents.

Rules:
- Answer ONLY using the provided context. Do not use prior knowledge.
- If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
- Always cite which document/source your answer comes from.
- Be concise and precise.

Context:
{context}
"""

USER_PROMPT = "{question}"

def format_context(docs: List[Document]) -> str:
    """Format retrieved docs into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"Document {i}")
        page = doc.metadata.get("page", "")
        page_str = f" (page {page + 1})" if page != "" else ""
        parts.append(f"[Source {i}: {source}{page_str}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def build_rag_chain(api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0):
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(
        model_name=model,
        groq_api_key=api_key,
        temperature=temperature,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])
    return prompt | llm

def query_rag(
    vectorstore,
    question: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    k: int = 5,
    temperature: float = 0.0,
):
    """
    Returns a (stream_generator, source_docs) tuple.
    Caller is responsible for consuming the stream.
    """
    docs = vectorstore.similarity_search(question, k=k)
    context = format_context(docs)

    chain = build_rag_chain(api_key, model, temperature)
    stream = chain.stream({"context": context, "question": question})

    return stream, docs
