from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    import math
    from config import OPENAI_API_KEY
    from config import GEMINI_API_KEY, GEMINI_EVAL_MODEL, GEMINI_EMBEDDING_MODEL, GEMINI_BASE_URL
    from config import GEMINI_REQUEST_INTERVAL
    api_key = GEMINI_API_KEY or OPENAI_API_KEY

    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    fallback = {metric: 0.0 for metric in metrics}
    fallback.update({"per_question": [], "status": "unavailable"})
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation columns must have equal lengths")
    if not questions or not api_key or api_key == "sk-..." or os.getenv("LAB_NO_API") == "1":
        fallback["error"] = "No evaluation samples, no valid API key, or LAB_NO_API=1; RAGAS not run"
        return fallback
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths})
        if GEMINI_API_KEY:
            import asyncio
            from openai import OpenAI, AsyncOpenAI
            from ragas.llms.base import BaseRagasLLM
            from langchain_core.outputs import Generation, LLMResult

            class GeminiRagasLLM(BaseRagasLLM):
                def __init__(self):
                    super().__init__()
                    self.client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL,
                                         timeout=120, max_retries=12)

                @staticmethod
                def request_kwargs(prompt, temperature, stop):
                    kwargs = {"model": GEMINI_EVAL_MODEL, "temperature": temperature,
                              "max_tokens": 4096,
                              "response_format": {"type": "json_object"},
                              "messages": [{"role": "user", "content": prompt.to_string()}]}
                    if stop:
                        kwargs["stop"] = stop
                    return kwargs

                @staticmethod
                def content(response):
                    text = response.choices[0].message.content
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Gemini returned an empty response")
                    return text.strip()

                def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
                    import time
                    kwargs = self.request_kwargs(prompt, temperature, stop)
                    generations = []
                    for _ in range(n):
                        time.sleep(GEMINI_REQUEST_INTERVAL)
                        generations.append(Generation(text=self.content(
                            self.client.chat.completions.create(**kwargs))))
                    return LLMResult(generations=[generations])

                async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
                    kwargs = self.request_kwargs(prompt, temperature, stop)
                    # RAGAS answer relevancy needs n=3; Gemini supplies one completion per request.
                    async with AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL,
                                           timeout=120, max_retries=12) as client:
                        responses = []
                        for _ in range(n):
                            await asyncio.sleep(GEMINI_REQUEST_INTERVAL)
                            responses.append(await client.chat.completions.create(**kwargs))
                    return LLMResult(generations=[[
                        Generation(text=self.content(response)) for response in responses]])

            # Gemini returns one completion per request, so strictness=1 keeps
            # answer_relevancy at one judge call per question instead of three.
            answer_relevancy.strictness = 1
            llm = GeminiRagasLLM()
            embeddings = OpenAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL, api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL,
                check_embedding_ctx_length=False, chunk_size=1, request_timeout=120, max_retries=12)
        else:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        result = evaluate(
            dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, embeddings=embeddings,
            run_config=RunConfig(timeout=600 if GEMINI_API_KEY else 120, max_retries=2,
                                 max_workers=1 if GEMINI_API_KEY else 2),
            raise_exceptions=False)
        rows = result.to_pandas().to_dict(orient="records")
        if len(rows) != len(questions):
            raise ValueError("RAGAS returned a different number of rows")
        per_question = []
        unscored_questions = []
        for i, row in enumerate(rows):
            values = {metric: float(row.get(metric, float("nan"))) for metric in metrics}
            missing = [metric for metric, value in values.items() if not math.isfinite(value)]
            if missing:
                unscored_questions.append({"question": questions[i], "metrics": missing})
            per_question.append(EvalResult(
                question=questions[i], answer=answers[i], contexts=contexts[i],
                ground_truth=ground_truths[i], **values))
        valid_scores = {metric: [getattr(row, metric) for row in per_question
                                 if math.isfinite(getattr(row, metric))] for metric in metrics}
        if not any(valid_scores.values()):
            raise ValueError("RAGAS returned no valid scores")
        return {
            **{metric: sum(values) / len(values) if values else 0.0
               for metric, values in valid_scores.items()},
            "per_question": per_question,
            "status": "partial" if unscored_questions else "ok",
            "metric_coverage": {metric: len(values) for metric, values in valid_scores.items()},
            "unscored_questions": unscored_questions}
    except Exception as exc:
        fallback["status"] = "error"
        fallback["error"] = f"{type(exc).__name__}: {exc}".replace(api_key, "<redacted>")
        print(f"  RAGAS evaluation failed: {fallback['error']}")
        return fallback

    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    import math

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten context-only prompt and lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {metric: float(getattr(result, metric)) for metric in diagnostic_tree}
        if not all(math.isfinite(value) for value in scores.values()):
            continue
        worst = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({
            "question": result.question, "answer": result.answer,
            "ground_truth": result.ground_truth, "contexts": result.contexts,
            "worst_metric": worst, "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis, "suggested_fix": fix,
            "error_tree": "Output wrong -> Context complete? -> Sources relevant? -> Inspect retrieval/generation",
        })
    return sorted(failures, key=lambda failure: failure["score"])[:max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
