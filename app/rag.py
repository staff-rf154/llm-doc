from dataclasses import dataclass

from openai import OpenAI

import config
import domains
import logtools
import settings
import vectorstore as vs
from capabilities import is_capabilities_question, render as render_capabilities


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk: int
    score: float
    url: str = ""
    title: str = ""
    page: int = 0
    kind: str = "file"

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
    reference: dict | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "grounded": self.grounded,
            "corrected": self.corrected,
            "request_id": self.request_id,
            "reference": self.reference,
            "sources": [
                {
                    "source": c.source,
                    "chunk": c.chunk,
                    "score": round(c.score, 4),
                    "text": c.text,
                    "url": c.url,
                    "title": c.title,
                    "page": c.page,
                    "kind": c.kind,
                }
                for c in self.chunks
            ],
        }


NO_ANSWER_MARKER = "нет информации"


def document_name(chunk: RetrievedChunk) -> str:
    """Полное название документа, из которого взят фрагмент."""
    import docpages

    return docpages.human_title(chunk.source, chunk.title, chunk.kind)


def sources_block(chunks: list[RetrievedChunk]) -> str:
    """Блок с полными названиями документов и рекомендацией изучить их детально."""
    documents: dict[str, dict] = {}
    for chunk in chunks:
        entry = documents.setdefault(
            chunk.source,
            {
                "name": document_name(chunk),
                "pages": set(),
                "url": chunk.url,
            },
        )
        if chunk.page:
            entry["pages"].add(int(chunk.page))
        if not entry["url"] and chunk.url:
            entry["url"] = chunk.url
    if not documents:
        return ""
    lines = []
    for source, entry in documents.items():
        pages = ", ".join(str(page) for page in sorted(entry["pages"]))
        line = entry["name"]
        if source and source != entry["name"]:
            line += f" ({source})"
        if pages:
            line += f", с. {pages}"
        if entry["url"]:
            line += f" — {entry['url']}"
        lines.append(f"- {line}")
    plural = "этими документами" if len(lines) > 1 else "этим документом"
    return (
        "\n\nДокументация, из которой взят ответ:\n"
        + "\n".join(lines)
        + f"\n\nРекомендую детально ознакомиться с {plural} — там описаны детали, "
        "шаги и настройки, которых нет в кратком ответе."
    )


def reference_block(doc_name: str, doc_pages: str) -> str:
    """Блок с названием документа для ответа, взятого из исправления администратора."""
    doc_name = (doc_name or "").strip()
    if not doc_name:
        return ""
    import docpages

    title = docpages.human_title(doc_name)
    line = title if title == doc_name else f"{title} ({doc_name})"
    pages = (doc_pages or "").strip()
    if pages:
        line += f", с. {pages}"
    return (
        "\n\nДокументация, из которой взят ответ:\n"
        f"- {line}"
        + "\n\nРекомендую детально ознакомиться с этим документом — "
        "там описаны детали, шаги и настройки, которых нет в кратком ответе."
    )


def query_variants(query: str) -> list[str]:
    """Варианты поискового запроса с подстановкой синонимов из словаря."""
    try:
        import synonymdb

        return synonymdb.expand(query, max_variants=settings.int_value("SYNONYMS_MAX_VARIANTS"))
    except Exception:
        return [query]


def _to_chunk(point) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        text=payload.get("text", ""),
        source=payload.get("source", "unknown"),
        chunk=int(payload.get("chunk", 0)),
        score=float(point.score),
        url=payload.get("url", ""),
        title=payload.get("title", ""),
        page=int(payload.get("page", 0) or 0),
        kind=payload.get("kind", "file") or "file",
    )


def _dense_points(client, target: str, variant: str, limit: int) -> list:
    try:
        return client.query_points(
            collection_name=target,
            query=vs.embed_query(variant),
            using="dense",
            limit=limit,
            with_payload=True,
        ).points
    except Exception:
        return client.query_points(
            collection_name=target,
            query=vs.embed_query(variant),
            limit=limit,
            with_payload=True,
        ).points


def _hybrid_points(client, target: str, variant: str, limit: int):
    """Dense + BM25 с слиянием RRF; при отсутствии sparse-индекса — только dense."""
    if not vs.sparse_enabled() or not vs.supports_sparse(target):
        return _dense_points(client, target, variant, limit)
    sparse = vs.sparse_query(variant)
    if sparse is None:
        return _dense_points(client, target, variant, limit)
    from qdrant_client.http import models

    prefetch = [
        models.Prefetch(query=vs.embed_query(variant), using="dense", limit=limit),
        models.Prefetch(query=sparse, using="sparse", limit=limit),
    ]
    try:
        return client.query_points(
            collection_name=target,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
    except Exception:
        return _dense_points(client, target, variant, limit)


def keep_count() -> int:
    return settings.int_value("RERANKER_TOP_K") or config.RERANKER_TOP_K


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Переранжирование кандидатов cross-encoder'ом."""
    if not settings.bool_value("RERANKER_ENABLED", config.RERANKER_ENABLED) or not chunks:
        return chunks[:top_k]
    candidates = chunks[: (settings.int_value("RERANKER_CANDIDATES") or config.RERANKER_CANDIDATES) or len(chunks)]
    try:
        model = vs.get_reranker()
        pairs = [(query, chunk.text) for chunk in candidates]
        batch = settings.int_value("RERANKER_BATCH") or config.RERANKER_BATCH
        scores = model.predict(pairs, batch_size=batch, show_progress_bar=False)
    except Exception:
        return chunks[:top_k]
    ranked = sorted(
        zip(candidates, [float(score) for score in scores]),
        key=lambda pair: pair[1],
        reverse=True,
    )
    keep = settings.int_value("RERANKER_TOP_K") or config.RERANKER_TOP_K or top_k
    result = []
    for chunk, score in ranked[: max(top_k, keep)]:
        chunk.score = round(score, 4)
        result.append(chunk)
    return result


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.int_value("RETRIEVE_TOP_K")
    client = vs.get_qdrant()
    target = vs.target_name()
    if not client.collection_exists(target):
        return []

    pool = max(settings.int_value("HYBRID_CANDIDATES") or config.HYBRID_CANDIDATES, top_k)
    merged: dict[tuple[str, int], RetrievedChunk] = {}
    for variant in query_variants(query):
        for point in _hybrid_points(client, target, variant, pool):
            chunk = _to_chunk(point)
            key = (chunk.source, chunk.chunk)
            previous = merged.get(key)
            if previous is None or chunk.score > previous.score:
                merged[key] = chunk

    ordered = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    ordered = rerank(query, ordered, top_k)
    return ordered[:top_k]


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    examples: list[dict] | None = None,
) -> str:
    blocks = []
    for chunk in chunks:
        name = document_name(chunk)
        blocks.append(
            f"<fragment id=\"{chunk.citation}\">\n"
            f"Документ: {name} ({chunk.source})\n"
            f"{chunk.text}\n</fragment>"
        )
    lessons = lesson_examples(question)
    blocks_lessons = ""
    if lessons:
        import lessons as lessons_mod

        blocks_lessons = lessons_mod.as_prompt_block(lessons) + "\n\n"
    head = ""
    if examples:
        samples = "\n\n".join(
            f"<example>\nВопрос: {item['question']}\nПравильный ответ: {item['answer']}\n</example>"
            for item in examples
        )
        head = f"Примеры исправленных ответов, учти их стиль и точность:\n\n{samples}\n\n"
    return (
        f"Документация:\n\n"
        + "\n\n".join(blocks)
        + f"\n\n{blocks_lessons}{head}Вопрос: {question}"
    )


def get_client() -> OpenAI:
    return OpenAI(
        base_url=settings.value("LLM_BASE_URL", config.LLM_BASE_URL),
        api_key=settings.value("LLM_API_KEY", config.LLM_API_KEY),
        timeout=180,
    )


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

def lesson_examples(question: str) -> list[dict]:
    try:
        import lessons

        return lessons.relevant(question)
    except Exception:
        return []


def correction_examples(question: str) -> list[dict]:
    try:
        import corrections

        _, examples = corrections.match(question)
        return examples
    except Exception:
        return []


def answer_from_logs(question: str, report: dict) -> Answer:
    """Ответ по выдержкам из логов, без поиска по документации."""
    client = get_client()
    temperature = domains.temperature()
    if temperature is None:
        temperature = settings.float_value("LLM_TEMPERATURE")
    completion = client.chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=[
            {"role": "system", "content": domains.system_prompt()},
            {"role": "user", "content": logtools.build_prompt(question, report)},
        ],
        temperature=temperature,
    )
    text = (completion.choices[0].message.content or "").strip()
    text += logtools.sources_block(report)
    chunks = [
        RetrievedChunk(
            text=item["text"][:4000],
            source=item["name"],
            chunk=index,
            score=1.0,
            title=item["name"],
            kind="log",
        )
        for index, item in enumerate(report.get("files", []), start=1)
    ]
    return Answer(text=text, chunks=chunks, grounded=True)


def ask(question: str, top_k: int | None = None) -> Answer:
    if is_capabilities_question(question):
        return capabilities_answer()

    if domains.log_analysis() and logtools.find_path(question):
        try:
            report = logtools.analyze(question)
        except logtools.LogError as exc:
            return Answer(text=str(exc), chunks=[], grounded=False)
        return answer_from_logs(question, report)

    exact = find_correction(question)
    if exact:
        answer = exact["answer"] + reference_block(
            exact.get("doc_name", ""), exact.get("doc_pages", "")
        )
        return Answer(
            text=answer,
            chunks=[],
            grounded=True,
            corrected=True,
            reference={
                "doc_name": exact.get("doc_name", ""),
                "doc_pages": exact.get("doc_pages", ""),
                "doc_ref": exact.get("doc_ref", ""),
                "author": exact.get("author", ""),
                "updated_at": exact.get("updated_at", ""),
                "correction_id": exact.get("correction_id", 0),
            },
        )

    chunks = retrieve(question, top_k)
    if not chunks:
        return Answer(
            text=empty_index_message(),
            chunks=[],
            grounded=False,
        )

    client = get_client()
    temperature = domains.temperature()
    if temperature is None:
        temperature = settings.float_value("LLM_TEMPERATURE")
    completion = client.chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=[
            {"role": "system", "content": domains.system_prompt()},
            {
                "role": "user",
                "content": build_prompt(question, chunks, correction_examples(question)),
            },
        ],
        temperature=temperature,
    )
    text = (completion.choices[0].message.content or "").strip()
    grounded = NO_ANSWER_MARKER not in text.lower()
    if grounded:
        text += sources_block(chunks)
    return Answer(text=text, chunks=chunks, grounded=grounded)