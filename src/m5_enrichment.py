from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    from config import GEMINI_API_KEY, GEMINI_ENRICH_MODEL, GEMINI_BASE_URL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY
    if api_key and api_key != "sk-..." and os.getenv("LAB_NO_API") != "1":
        try:
            if GEMINI_API_KEY:
                import time
                from config import GEMINI_REQUEST_INTERVAL
                time.sleep(GEMINI_REQUEST_INTERVAL)
            from openai import OpenAI
            response = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL if GEMINI_API_KEY else None,
                              timeout=120, max_retries=12 if GEMINI_API_KEY else 1).chat.completions.create(
                model=GEMINI_ENRICH_MODEL if GEMINI_API_KEY else "gpt-4o-mini", messages=[
                    {"role": "system", "content": "Tóm tắt trong 2-3 câu bằng tiếng Việt. Giữ nguyên con số, điều kiện và phủ định."},
                    {"role": "user", "content": text}],
                max_tokens=4096 if GEMINI_API_KEY else 150, temperature=0)
            return response.choices[0].message.content.strip()[:max(1, len(text) * 2)]
        except Exception as exc:
            print(f"  Summarize fallback: {type(exc).__name__}")
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    from config import GEMINI_API_KEY, GEMINI_ENRICH_MODEL, GEMINI_BASE_URL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY
    if api_key and api_key != "sk-..." and os.getenv("LAB_NO_API") != "1":
        try:
            if GEMINI_API_KEY:
                import time
                from config import GEMINI_REQUEST_INTERVAL
                time.sleep(GEMINI_REQUEST_INTERVAL)
            from openai import OpenAI
            response = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL if GEMINI_API_KEY else None,
                              timeout=120, max_retries=12 if GEMINI_API_KEY else 1).chat.completions.create(
                model=GEMINI_ENRICH_MODEL if GEMINI_API_KEY else "gpt-4o-mini", messages=[
                    {"role": "system", "content": f"Tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu trên một dòng. Không thêm dữ kiện."},
                    {"role": "user", "content": text}],
                max_tokens=4096 if GEMINI_API_KEY else 200, temperature=0)
            questions = response.choices[0].message.content.strip().splitlines()
            return [question.strip().lstrip("0123456789.-) ")
                    for question in questions if question.strip()][:n_questions]
        except Exception as exc:
            print(f"  HyQA fallback: {type(exc).__name__}")
    import re
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{sentence.rstrip('.')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    from config import GEMINI_API_KEY, GEMINI_ENRICH_MODEL, GEMINI_BASE_URL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY
    if api_key and api_key != "sk-..." and os.getenv("LAB_NO_API") != "1":
        try:
            if GEMINI_API_KEY:
                import time
                from config import GEMINI_REQUEST_INTERVAL
                time.sleep(GEMINI_REQUEST_INTERVAL)
            from openai import OpenAI
            response = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL if GEMINI_API_KEY else None,
                              timeout=120, max_retries=12 if GEMINI_API_KEY else 1).chat.completions.create(
                model=GEMINI_ENRICH_MODEL if GEMINI_API_KEY else "gpt-4o-mini", messages=[
                    {"role": "system", "content": "Viết một câu ngắn mô tả nguồn và chủ đề đoạn văn. Chỉ dùng thông tin được cung cấp."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"}],
                max_tokens=4096 if GEMINI_API_KEY else 80, temperature=0)
            context = response.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  Contextual fallback: {type(exc).__name__}")
    prefix = f"Trích từ {document_title}.\n\n" if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    from config import GEMINI_API_KEY, GEMINI_ENRICH_MODEL, GEMINI_BASE_URL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY
    if api_key and api_key != "sk-..." and os.getenv("LAB_NO_API") != "1":
        try:
            if GEMINI_API_KEY:
                import time
                from config import GEMINI_REQUEST_INTERVAL
                time.sleep(GEMINI_REQUEST_INTERVAL)
            import json
            from openai import OpenAI
            response = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL if GEMINI_API_KEY else None,
                              timeout=120, max_retries=12 if GEMINI_API_KEY else 1).chat.completions.create(
                model=GEMINI_ENRICH_MODEL if GEMINI_API_KEY else "gpt-4o-mini",
                response_format={"type": "json_object"}, messages=[
                    {"role": "system", "content": 'Trích metadata và trả JSON với topic, entities (list), category (policy|hr|it|finance), language (vi|en). Chỉ dùng thông tin trong đoạn văn.'},
                    {"role": "user", "content": text}],
                max_tokens=4096 if GEMINI_API_KEY else 150, temperature=0)
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise TypeError("Metadata must be a JSON object")
            return {key: result[key] for key in ("topic", "entities", "category", "language")
                    if key in result}
        except Exception as exc:
            print(f"  Metadata fallback: {type(exc).__name__}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import json
    import re

    from config import GEMINI_API_KEY, GEMINI_ENRICH_MODEL, GEMINI_BASE_URL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY
    if api_key and api_key != "sk-..." and os.getenv("LAB_NO_API") != "1":
        try:
            if GEMINI_API_KEY:
                import time
                from config import GEMINI_REQUEST_INTERVAL
                time.sleep(GEMINI_REQUEST_INTERVAL)
            from openai import OpenAI
            response = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL if GEMINI_API_KEY else None,
                              timeout=120, max_retries=12 if GEMINI_API_KEY else 1).chat.completions.create(
                model=GEMINI_ENRICH_MODEL if GEMINI_API_KEY else "gpt-4o-mini",
                response_format={"type": "json_object"}, temperature=0,
                messages=[
                    {"role": "system", "content": (
                        "Phân tích đoạn văn và trả JSON với summary (2-3 câu), questions (3 câu hỏi), "
                        "context (1 câu về nguồn/chủ đề), metadata (topic, entities, category, language). "
                        "Chỉ dùng thông tin được cung cấp; giữ con số, điều kiện, phủ định và phiên bản.")},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"}],
                max_tokens=4096 if GEMINI_API_KEY else 500)
            result = json.loads(response.choices[0].message.content)
            if not isinstance(result, dict):
                raise TypeError("Enrichment must be a JSON object")
            metadata = result.get("metadata", {})
            questions = result.get("questions", [])
            return {
                "summary": str(result.get("summary", "")),
                "questions": [q for q in questions if isinstance(q, str)][:3]
                             if isinstance(questions, list) else [],
                "context": str(result.get("context", "")),
                "metadata": {key: metadata[key] for key in ("topic", "entities", "category", "language")
                             if key in metadata} if isinstance(metadata, dict) else {},
            }
        except Exception as exc:
            print(f"  Enrichment fallback: {type(exc).__name__}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return {
        "summary": " ".join(sentences[:2]),
        "questions": [f"{sentence.rstrip('.')}?" for sentence in sentences[:3]],
        "context": f"Trích từ tài liệu {source}." if source else "",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
