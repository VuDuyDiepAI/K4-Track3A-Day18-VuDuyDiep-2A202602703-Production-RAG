from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            cache = getattr(CrossEncoderReranker, "_model_cache", {})
            if self.model_name not in cache:
                cache[self.model_name] = CrossEncoder(
                    self.model_name, local_files_only=os.getenv("LAB_OFFLINE") == "1")
            CrossEncoderReranker._model_cache = cache
            self._model = cache[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        import numpy as np

        if not documents or top_k <= 0:
            return []
        pairs = [(query, document["text"]) for document in documents]
        scores = np.asarray(self._load_model().predict(
            pairs, batch_size=8, show_progress_bar=False)).reshape(-1)
        ranked = sorted(zip(scores, documents), key=lambda pair: float(pair[0]), reverse=True)
        return [RerankResult(text=document["text"],
                             original_score=float(document.get("score", 0.0)),
                             rerank_score=float(score), metadata=document.get("metadata", {}),
                             rank=i)
                for i, (score, document) in enumerate(ranked[:top_k])]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        from flashrank import Ranker, RerankRequest

        if not documents or top_k <= 0:
            return []
        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": document["text"]}
                    for i, document in enumerate(documents)]
        ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
        return [RerankResult(text=documents[result["id"]]["text"],
                             original_score=float(documents[result["id"]].get("score", 0)),
                             rerank_score=float(result["score"]),
                             metadata=documents[result["id"]].get("metadata", {}), rank=i)
                for i, result in enumerate(ranked[:top_k])]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
