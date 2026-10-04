from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    metadata = dict(metadata or {})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    if not sentences:
        return []
    if not hasattr(chunk_semantic, "_model"):
        chunk_semantic._model = SentenceTransformer(
            "all-MiniLM-L6-v2", local_files_only=os.getenv("LAB_OFFLINE") == "1")
    embeddings = chunk_semantic._model.encode(sentences, normalize_embeddings=True)
    groups, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
        if similarity < threshold:
            groups.append(" ".join(current))
            current = []
        current.append(sentences[i])
    groups.append(" ".join(current))
    return [Chunk(text=group, metadata={
        **metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    import hashlib

    if not 0 < child_size < parent_size:
        raise ValueError("require 0 < child_size < parent_size")
    metadata = dict(metadata or {})
    prefix = hashlib.sha256(
        (str(metadata.get("source", "")) + "\0" + text).encode("utf-8")
    ).hexdigest()[:16]

    def split_bounded(value, size):
        parts, value = [], value.strip()
        while len(value) > size:
            cut = max(value.rfind("\n", 0, size + 1), value.rfind(" ", 0, size + 1))
            if cut < size // 2:
                cut = size
            parts.append(value[:cut].strip())
            value = value[cut:].strip()
        if value:
            parts.append(value)
        return parts

    parent_texts, current = [], ""
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        for part in split_bounded(paragraph, parent_size):
            combined = current + "\n\n" + part if current else part
            if len(combined) > parent_size:
                parent_texts.append(current)
                current = part
            else:
                current = combined
    if current:
        parent_texts.append(current)
    parents, children = [], []
    for i, parent_text in enumerate(parent_texts):
        pid = f"{prefix}_parent_{i}"
        parents.append(Chunk(text=parent_text, metadata={
            **metadata, "strategy": "hierarchical", "chunk_type": "parent", "parent_id": pid}))
        for j, child_text in enumerate(split_bounded(parent_text, child_size)):
            children.append(Chunk(text=child_text, metadata={
                **metadata, "strategy": "hierarchical", "chunk_type": "child",
                "chunk_index": j}, parent_id=pid))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = dict(metadata or {})
    chunks, lines, section, fence = [], [], "", None

    def flush():
        content = "\n".join(lines).strip()
        if content:
            chunks.append(Chunk(text=content, metadata={
                **metadata, "section": section, "strategy": "structure",
                "chunk_index": len(chunks)}))

    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(\x60{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = (token[0], len(token))
            elif token[0] == fence[0] and len(token) >= fence[1]:
                fence = None
            lines.append(line)
            continue
        if fence is None and re.match(r"^#{1,6}\s+", line):
            flush()
            lines = []
            section = line.strip()
        lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
