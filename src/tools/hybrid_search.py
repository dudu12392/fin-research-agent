"""Hybrid search: BM25 + Vector + Cross-Encoder reranking.

Combines sparse lexical retrieval (BM25) with dense semantic retrieval
(ChromaDB vectors), then re-ranks candidates with a Cross-Encoder.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi


class HybridSearcher:
    """Two-stage hybrid retriever with BM25 + Vector fusion + Cross-Encoder rerank.

    Stage 1 — Candidate Recall:
      - Vector: ChromaDB dense retrieval (top_k * 2)
      - BM25: sparse keyword retrieval (top_k * 2)
      - Fusion: weighted score combination

    Stage 2 — Rerank:
      - Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each
        candidate against the original query, producing final ranking.
    """

    def __init__(
        self,
        chroma_collection: Any,
        chunks: list[dict[str, Any]],
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialise with ChromaDB collection and chunk list.

        Args:
            chroma_collection: ChromaDB collection object (must have .query()).
            chunks: List of dicts with 'content' and 'metadata' keys.
            cross_encoder_model: HuggingFace cross-encoder name.
        """
        self.collection = chroma_collection
        self.chunks = chunks

        # ── Build BM25 index ──────────────────────────────────
        self.corpus_texts: list[str] = [c["content"] for c in chunks]
        tokenized = [text.split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        # ── Cross-Encoder (lazy-loaded on first use) ──────────
        self._cross_encoder = None
        self._ce_model_name = cross_encoder_model

        print(f"✅ HybridSearcher ready: {len(chunks)} docs, BM25 + Cross-Encoder")

    @property
    def cross_encoder(self):
        """Lazy-load the Cross-Encoder to avoid overhead on init."""
        if self._cross_encoder is None:
            import os
            from sentence_transformers import CrossEncoder

            # Use HF mirror for China mainland
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            print("⏳ Loading Cross-Encoder model...")
            try:
                self._cross_encoder = CrossEncoder(
                    self._ce_model_name,
                    max_length=512,
                )
            except Exception:
                # Fallback: use sentence-transformers/all-MiniLM-L6-v2
                # as a lightweight cross-encoder surrogate
                fallback = "sentence-transformers/all-MiniLM-L6-v2"
                print(f"   Primary model failed, trying fallback: {fallback}")
                self._cross_encoder = CrossEncoder(fallback, max_length=512)
            print("   Cross-Encoder loaded.")
        return self._cross_encoder

    # ── Public API ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Hybrid search with reranking.

        Args:
            query: Natural-language query string.
            top_k: Number of final results to return.
            vector_weight: Weight for vector score (0=BM25-only, 1=vector-only).

        Returns:
            List of result dicts with keys: chunk, score, source, rank.
        """
        # Stage 1a: Vector retrieval
        vec_results = self._vector_search(query, top_k * 2)

        # Stage 1b: BM25 retrieval
        bm25_results = self._bm25_search(query, top_k * 2)

        # Stage 1c: Fusion
        merged = self._merge_results(vec_results, bm25_results, vector_weight)

        # Take top candidates before rerank
        candidates = merged[:top_k]

        # Stage 2: Cross-Encoder rerank
        reranked = self._rerank(query, candidates)

        # Assign final ranks
        for rank, result in enumerate(reranked, 1):
            result["rank"] = rank

        return reranked[:top_k]

    # ── Stage 1: Vector retrieval ─────────────────────────────────

    def _vector_search(self, query: str, n: int) -> list[dict[str, Any]]:
        """Dense vector search via ChromaDB.

        Returns list of {chunk_idx, score, source: 'vector'}.
        """
        results = self.collection.query(query_texts=[query], n_results=n)
        ids = results["ids"][0]
        distances = results.get("distances", [[0] * n])[0]

        # ChromaDB returns cosine distance; convert to similarity 0-1
        output: list[dict[str, Any]] = []
        for doc_id, dist in zip(ids, distances):
            # Parse chunk index from id: "AAPL_2025_123"
            try:
                chunk_idx = int(doc_id.rsplit("_", 1)[-1])
            except (ValueError, IndexError):
                chunk_idx = 0
            output.append({
                "chunk_idx": chunk_idx,
                "score": 1.0 / (1.0 + dist),  # distance→similarity
                "source": "vector",
            })
        return output

    # ── Stage 1: BM25 retrieval ───────────────────────────────────

    def _bm25_search(self, query: str, n: int) -> list[dict[str, Any]]:
        """Sparse keyword search via BM25.

        Returns list of {chunk_idx, score, source: 'bm25'}.
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-n indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        # Normalise scores to 0-1
        max_score = scores[top_indices[0]] if top_indices else 1.0

        output: list[dict[str, Any]] = []
        for idx in top_indices:
            norm_score = scores[idx] / max_score if max_score > 0 else 0.0
            output.append({
                "chunk_idx": idx,
                "score": norm_score,
                "source": "bm25",
            })
        return output

    # ── Stage 1: Fusion ───────────────────────────────────────────

    def _merge_results(
        self,
        vec_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        vector_weight: float,
    ) -> list[dict[str, Any]]:
        """Weighted reciprocal rank fusion of vector and BM25 results.

        Computes final_score = vector_weight * vec_score + (1-vector_weight) * bm25_score
        for each unique chunk, then sorts descending.
        """
        scores: dict[int, dict[str, float]] = {}

        for r in vec_results:
            idx = r["chunk_idx"]
            scores.setdefault(idx, {"bm25": 0.0, "vector": 0.0})
            scores[idx]["vector"] = max(scores[idx]["vector"], r["score"])

        for r in bm25_results:
            idx = r["chunk_idx"]
            scores.setdefault(idx, {"bm25": 0.0, "vector": 0.0})
            scores[idx]["bm25"] = max(scores[idx]["bm25"], r["score"])

        merged = []
        for idx, s in scores.items():
            final_score = vector_weight * s["vector"] + (1 - vector_weight) * s["bm25"]
            merged.append({
                "chunk_idx": idx,
                "score": final_score,
                "source": "hybrid",
            })

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged

    # ── Stage 2: Cross-Encoder rerank ─────────────────────────────

    def _rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score each candidate against the query with Cross-Encoder."""
        if not candidates:
            return []

        ce = self.cross_encoder
        pairs = [(query, self.corpus_texts[c["chunk_idx"]]) for c in candidates]
        ce_scores = ce.predict(pairs, show_progress_bar=False)

        # Attach CE scores and chunk metadata, then re-sort
        for candidate, ce_score in zip(candidates, ce_scores):
            idx = candidate["chunk_idx"]
            candidate["ce_score"] = float(ce_score)
            candidate["score"] = float(ce_score)
            # Attach full metadata from original chunks
            candidate["metadata"] = self.chunks[idx].get("metadata", {})
            candidate["content"] = self.corpus_texts[idx][:200]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
