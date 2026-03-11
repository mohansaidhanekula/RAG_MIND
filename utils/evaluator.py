"""
RAG Evaluation Module.
Scores answers on 4 dimensions (all free, no external API):

1. Faithfulness      — Does the answer stay within the retrieved context?
2. Context Relevance — Are the retrieved chunks relevant to the question?
3. Answer Relevance  — Does the answer address the question?
4. ROUGE-L           — Lexical overlap between answer and source context.
"""

import re
from typing import List, Dict
from langchain_core.documents import Document


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute LCS length for ROUGE-L."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-efficient DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between hypothesis and reference."""
    h = _tokenize(hypothesis)
    r = _tokenize(reference)
    if not h or not r:
        return 0.0
    lcs = _lcs_length(h, r)
    precision = lcs / len(h)
    recall    = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


# ── Scoring Functions ──────────────────────────────────────────────────────────

def score_context_relevance(question: str, docs: List[Document]) -> float:
    """
    How relevant are the retrieved chunks to the question?
    Uses average Jaccard similarity between question tokens and each chunk.
    Score: 0.0 – 1.0
    """
    if not docs:
        return 0.0
    q_tokens = _tokenize(question)
    scores = [_jaccard(q_tokens, _tokenize(doc.page_content)) for doc in docs]
    return round(sum(scores) / len(scores), 3)


def score_faithfulness(answer: str, docs: List[Document]) -> float:
    """
    Does the answer stay grounded in the retrieved context?
    Measures what fraction of answer tokens appear in the combined context.
    Score: 0.0 – 1.0
    """
    if not docs or not answer.strip():
        return 0.0
    context = " ".join(doc.page_content for doc in docs)
    ctx_tokens = set(_tokenize(context))
    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
    return round(overlap / len(ans_tokens), 3)


def score_answer_relevance(question: str, answer: str) -> float:
    """
    Does the answer address the question?
    Uses Jaccard similarity between question and answer tokens.
    Score: 0.0 – 1.0
    """
    return round(_jaccard(_tokenize(question), _tokenize(answer)), 3)


def score_rouge_l(answer: str, docs: List[Document]) -> float:
    """
    Lexical overlap (ROUGE-L F1) between answer and best matching chunk.
    Score: 0.0 – 1.0
    """
    if not docs or not answer.strip():
        return 0.0
    scores = [rouge_l(answer, doc.page_content) for doc in docs]
    return max(scores)


# ── Main Eval Entry Point ──────────────────────────────────────────────────────

def evaluate_response(
    question: str,
    answer: str,
    source_docs: List[Document],
) -> Dict[str, float]:
    """
    Run all 4 metrics and return a scored dict + overall score.
    """
    ctx_rel   = score_context_relevance(question, source_docs)
    faithful  = score_faithfulness(answer, source_docs)
    ans_rel   = score_answer_relevance(question, answer)
    rouge     = score_rouge_l(answer, source_docs)

    # Weighted overall score
    overall = round(
        0.35 * faithful +
        0.25 * ctx_rel  +
        0.25 * ans_rel  +
        0.15 * rouge,
        3,
    )

    return {
        "faithfulness":       faithful,
        "context_relevance":  ctx_rel,
        "answer_relevance":   ans_rel,
        "rouge_l":            rouge,
        "overall":            overall,
    }


def score_label(score: float) -> str:
    """Human-readable quality label."""
    if score >= 0.75:
        return "🟢 Excellent"
    elif score >= 0.5:
        return "🟡 Good"
    elif score >= 0.25:
        return "🟠 Fair"
    else:
        return "🔴 Poor"
