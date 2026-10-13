from dataclasses import dataclass

from openai import OpenAI

import config
import vectorstore as vs
from capabilities import is_capabilities_question, render as render_capabilities


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk: int
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.source}:{self.chunk}]"


@dataclass
class Answer:
    text: str
    chunks: list[RetrievedChunk]
    grounded: bool
    corrected: bool = False
    request_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "grounded": self.grounded,
            "corrected": self.corrected,
            "request_id": self.request_id,
            "sources": [
                {
                    "source": c.source,
                    "chunk": c.chunk,
                    "score": round(c.score, 4),
                    "text": c.text,
                }
                for c in self.chunks
            ],
        }


NO_ANSWER_MARKER = "нет информации"


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or config.RETRIEVE_TOP_K
    client = vs.get_qdrant()
    target = vs.target_name()
    if not client.collection_exists(target):
        return []
    results = client.query_points(
        collection_name=target,
        query=vs.embed_query(query),
        limit=top_k,
        with_payload=True,
    ).points
    return [
        RetrievedChunk(
            text=payload.get("text", ""),
            source=payload.get("source", "unknown"),
            chunk=payload.get("chunk", 0),
            score=float(score),
        )
        for payload, score in ((p.payload, p.score) for p in results)
    ]


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    examples: list[dict] | None = None,
) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(f"<fragment id=\"{chunk.citation}\">\n{chunk.text}\n</fragment>")
    head = ""
    if examples:
        samples = "\n\n".join(
            f"<example>\nВопрос: {item['question']}\nПравильный ответ: {item['answer']}\n</example>"
            for item in examples
        )
        head = f"Примеры исправленных ответов, учти их стиль и точность:\n\n{samples}\n\n"
    return (
        f"Документация:\n\n" + "\n\n".join(blocks) + f"\n\n{head}Вопрос: {question}"
    )


def get_client() -> OpenAI:
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=180)


EMPTY_INDEX_MESSAGE = "Индекс документации пуст."


def empty_index_message() -> str:
    try:
        import ingest

        snapshot = ingest.job.snapshot()
    except Exception:
        return f"{EMPTY_INDEX_MESSAGE} Запустите индексацию командой `make ingest`."
    if snapshot.get("status") in ("running", "paused"):
        done, total = snapshot.get("done", 0), snapshot.get("total", 0)
        progress = f" ({done}/{total})" if total else ""
        state = "собирается" if snapshot["status"] == "running" else "приостановлена"
        return (
            f"Индекс документации пуст: переиндексация {state}{progress}. "
            "Поиск заработает автоматически после её завершения."
        )
    return f"{EMPTY_INDEX_MESSAGE} Запустите индексацию командой `make ingest`."


def capabilities_answer() -> Answer:
    return Answer(text=render_capabilities(), chunks=[], grounded=True)


def find_correction(question: str) -> dict | None:
    try:
        import corrections

        exact, _ = corrections.match(question)
        return exact
    except Exception:
        return None


def correction_examples(question: str) -> list[dict]:
    try:
        import corrections

        _, examples = corrections.match(question)
        return examples
    except Exception:
        return []


def ask(question: str, top_k: int | None = None) -> Answer:
    if is_capabilities_question(question):
        return capabilities_answer()

    exact = find_correction(question)
    if exact:
        return Answer(
            text=exact["answer"],
            chunks=[],
            grounded=True,
            corrected=True,
        )

    chunks = retrieve(question, top_k)
    if not chunks:
        return Answer(
            text=empty_index_message(),
            chunks=[],
            grounded=False,
        )

    client = get_client()
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(question, chunks, correction_examples(question)),
            },
        ],
        temperature=0.1,
    )
    text = (completion.choices[0].message.content or "").strip()
    grounded = NO_ANSWER_MARKER not in text.lower()
    return Answer(text=text, chunks=chunks, grounded=grounded)