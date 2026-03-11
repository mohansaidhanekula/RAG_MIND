"""
Hybrid Search: BM25 (keyword) + Semantic (vector) search combined.
This dramatically improves retrieval quality — catching both exact keyword matches
and conceptually similar content that semantic-only search might miss.
"""

from typing import List, Dict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


class HybridRetriever:
    """
    Combines BM25 keyword search with ChromaDB semantic search.
    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, documents: List[Document], vectorstore, k: int = 5):
        self.documents = documents
        self.vectorstore = vectorstore
        self.k = k

        # Build BM25 index over all document chunks
        corpus = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def bm25_search(self, query: str, top_n: int) -> List[Document]:
        """Return top-n docs by BM25 keyword score."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [self.documents[i] for i in top_indices]

    def semantic_search(self, query: str, top_n: int) -> List[Document]:
        """Return top-n docs by vector similarity."""
        return self.vectorstore.similarity_search(query, k=top_n)

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Document],
        semantic_results: List[Document],
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Merge two ranked lists using RRF.
        RRF score = sum(1 / (rank + rrf_k)) across both lists.
        Higher = more relevant.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:100]  # use content as unique key
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + rrf_k)
            doc_map[key] = doc

        for rank, doc in enumerate(semantic_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + rrf_k)
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[: self.k]]

    def search(self, query: str) -> List[Document]:
        """Run hybrid search and return top-k fused results."""
        fetch_n = self.k * 2  # fetch more from each, then fuse
        bm25_docs    = self.bm25_search(query, fetch_n)
        semantic_docs = self.semantic_search(query, fetch_n)
        return self.reciprocal_rank_fusion(bm25_docs, semantic_docs)
