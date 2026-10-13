import argparse
import json
from pathlib import Path

import config
import rag


def load_dataset(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("items", [])
    return [item for item in raw if item.get("question")]


def lexical_metrics(question: str, answer: str, contexts: list[str]) -> dict:
    question_terms = {t for t in question.lower().split() if len(t) > 3}
    answer_lower = answer.lower()
    hit_rate = (
        sum(1 for term in question_terms if term in answer_lower) / len(question_terms)
        if question_terms
        else 0.0
    )
    context_lower = " ".join(contexts).lower()
    context_recall = (
        sum(1 for term in question_terms if term in context_lower) / len(question_terms)
        if question_terms
        else 0.0
    )
    refusal_hit = "нет информации" in answer_lower and len(answer) < 400
    return {
        "answer_keyword_hit_rate": round(hit_rate, 4),
        "context_recall": round(context_recall, 4),
        "refused": refusal_hit,
    }


def evaluate(dataset: list[dict]) -> dict:
    rows = []
    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        result = rag.ask(question, item.get("top_k"))
        contexts = [c.text for c in result.chunks]
        metrics = lexical_metrics(question, result.text, contexts)
        reference = item.get("answer", "")
        metrics["exact_match"] = bool(reference) and result.text.strip() == reference.strip()
        metrics["grounded"] = result.grounded
        metrics["question"] = question
        metrics["expected_answer"] = reference
        metrics["generated_answer"] = result.text
        metrics["sources"] = [c.citation for c in result.chunks]
        rows.append(metrics)
        print(f"[{index}/{len(dataset)}] {question[:60]} -> {metrics}")

    def mean(key: str) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0

    return {
        "samples": len(rows),
        "grounded_rate": mean("grounded"),
        "context_recall": mean("context_recall"),
        "answer_keyword_hit_rate": mean("answer_keyword_hit_rate"),
        "exact_match": mean("exact_match"),
        "rows": rows,
    }


def evaluate_with_ragas(dataset: list[dict]) -> dict:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    items = []
    for item in dataset:
        result = rag.ask(item["question"], item.get("top_k"))
        items.append(
            {
                "question": item["question"],
                "answer": result.text,
                "contexts": [c.text for c in result.chunks],
                "ground_truth": item.get("answer", ""),
            }
        )

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
    )

    result = ragas_evaluate(
        dataset=Dataset.from_list(items),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas().to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества RAG")
    parser.add_argument("--dataset", default=str(Path(config.EVAL_DIR) / "dataset.json"))
    parser.add_argument("--out", default=str(Path(config.EVAL_DIR) / "report.json"))
    parser.add_argument("--ragas", action="store_true", help="дополнительно прогнать RAGAS")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    if not dataset:
        raise SystemExit(f"датасет пуст: {args.dataset}")

    report = evaluate(dataset)
    if args.ragas:
        report["ragas"] = evaluate_with_ragas(dataset)

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"отчёт: {args.out}")


if __name__ == "__main__":
    main()