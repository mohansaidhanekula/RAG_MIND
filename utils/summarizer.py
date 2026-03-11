"""
Auto Document Summarization using Groq.
Supports: per-document summary, multi-doc overview, key topics extraction.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document analyst. Given document chunks, produce:
1. A concise 3-5 sentence summary of the main content
2. 5 key topics/themes as bullet points
3. Document type (report, article, manual, legal, academic, etc.)

Format your response exactly like this:
SUMMARY:
<your summary here>

KEY TOPICS:
• <topic 1>
• <topic 2>
• <topic 3>
• <topic 4>
• <topic 5>

DOCUMENT TYPE: <type>
"""),
    ("human", "Document content:\n\n{content}"),
])

OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Given summaries from multiple documents, 
create a unified overview that:
1. Identifies common themes across documents
2. Highlights key differences or contradictions
3. Provides a 5-7 sentence synthesis

Be concise and insightful."""),
    ("human", "Document summaries:\n\n{summaries}"),
])


def get_llm(api_key: str, model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    return ChatGroq(model=model, groq_api_key=api_key, temperature=0.2, streaming=False)


def summarize_document(
    docs: List[Document],
    api_key: str,
    source_name: str,
    model: str = "llama-3.3-70b-versatile",
    max_chars: int = 8000,
) -> Dict[str, str]:
    """
    Summarize a list of chunks from a single document.
    Returns dict with summary, key_topics, doc_type.
    """
    # Concatenate chunks up to max_chars to avoid context overflow
    combined = "\n\n".join(d.page_content for d in docs)[:max_chars]

    llm    = get_llm(api_key, model)
    chain  = SUMMARY_PROMPT | llm
    result = chain.invoke({"content": combined})
    raw    = result.content if hasattr(result, "content") else str(result)

    # Parse structured response
    summary    = _extract_section(raw, "SUMMARY:", "KEY TOPICS:").strip()
    key_topics = _extract_section(raw, "KEY TOPICS:", "DOCUMENT TYPE:").strip()
    doc_type   = _extract_section(raw, "DOCUMENT TYPE:", None).strip()

    return {
        "source":     source_name,
        "summary":    summary or raw,
        "key_topics": key_topics,
        "doc_type":   doc_type,
        "raw":        raw,
    }


def multi_doc_overview(
    summaries: List[Dict[str, str]],
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """Generate a cross-document synthesis from individual summaries."""
    if len(summaries) < 2:
        return summaries[0]["summary"] if summaries else "No documents to overview."

    combined = "\n\n---\n\n".join(
        f"[{s['source']}]\n{s['summary']}" for s in summaries
    )
    llm    = get_llm(api_key, model)
    chain  = OVERVIEW_PROMPT | llm
    result = chain.invoke({"summaries": combined})
    return result.content if hasattr(result, "content") else str(result)


def _extract_section(text: str, start_marker: str, end_marker) -> str:
    """Extract text between two markers."""
    try:
        start = text.index(start_marker) + len(start_marker)
        if end_marker:
            end = text.index(end_marker, start)
            return text[start:end]
        return text[start:]
    except ValueError:
        return ""
