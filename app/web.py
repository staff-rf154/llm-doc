import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import appdb
import auth
import config
import rag
import settings
from capabilities import is_capabilities_question, render as render_capabilities, suggestions

router = APIRouter()


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Turn] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    show_images: bool = True


def _history(messages: list[Turn]) -> list[dict]:
    turns = [{"role": "system", "content": settings.value("SYSTEM_PROMPT", config.SYSTEM_PROMPT)}]
    for turn in messages[:-1]:
        turns.append({"role": turn.role, "content": turn.content})
    turns.append({"role": "user", "content": messages[-1].content})
    return turns


def _related_images(question: str, allowed: bool = True) -> list[dict]:
    if not allowed or not settings.bool_value("IMAGE_SEARCH_ENABLED", True):
        return []
    try:
        import images

        return images.search(question)
    except Exception:
        return []


def _stream(req: ChatRequest, user: dict) -> StreamingResponse:
    question = req.messages[-1].content
    started = time.time()

    if is_capabilities_question(question):
        text = render_capabilities()
        request_id = appdb.log_request(user, question, text, True, False, 0, 0)
        payload = [
            {"type": "token", "data": text},
            {"type": "prompts", "data": suggestions()},
            {"type": "sources", "data": []},
            {"type": "request", "data": {"request_id": request_id}},
            {"type": "meta", "data": {"grounded": True, "corrected": False}},
        ]
        found = _related_images(question, req.show_images)
        if found:
            payload.append({"type": "images", "data": found})
        payload.append({"type": "done", "data": ""})
        return StreamingResponse(
            _sse(payload),
            media_type="text/event-stream",
        )

    exact = rag.find_correction(question)
    if exact:
        reference = {
            "doc_name": exact.get("doc_name", ""),
            "doc_pages": exact.get("doc_pages", ""),
            "doc_ref": exact.get("doc_ref", ""),
            "author": exact.get("author", ""),
            "updated_at": exact.get("updated_at", ""),
            "correction_id": exact.get("correction_id", 0),
        }
        request_id = appdb.log_request(
            user, question, exact["answer"], True, True, 0,
            int((time.time() - started) * 1000), reference,
        )
        payload = [
            {"type": "token", "data": exact["answer"]},
            {"type": "sources", "data": []},
            {"type": "reference", "data": reference},
            {"type": "request", "data": {"request_id": request_id}},
            {"type": "meta", "data": {"grounded": True, "corrected": True}},
            {"type": "done", "data": ""},
        ]
        return StreamingResponse(
            _sse(payload),
            media_type="text/event-stream",
        )

    chunks = rag.retrieve(question, req.top_k)

    if not chunks:
        text = rag.empty_index_message()
        request_id = appdb.log_request(user, question, text, False, False, 0, 0)
        payload = [
            {"type": "token", "data": text},
            {"type": "sources", "data": []},
            {"type": "request", "data": {"request_id": request_id}},
            {"type": "meta", "data": {"grounded": False, "corrected": False}},
            {"type": "done", "data": ""},
        ]
        return StreamingResponse(
            _sse(payload),
            media_type="text/event-stream",
        )

    prompt = rag.build_prompt(question, chunks, rag.correction_examples(question))
    client = rag.get_client()

    def generate():
        answer_parts: list[str] = []
        try:
            stream = client.chat.completions.create(
                model=settings.value("LLM_MODEL", config.LLM_MODEL),
                messages=_history(req.messages)[:-1] + [{"role": "user", "content": prompt}],
                temperature=settings.float_value("LLM_TEMPERATURE"),
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    answer_parts.append(delta.content)
                    yield _sse_event({"type": "token", "data": delta.content})
        except Exception as exc:
            yield _sse_event({"type": "error", "data": f"Ошибка генерации: {exc}"})
            return

        answer = "".join(answer_parts).strip()
        grounded = rag.NO_ANSWER_MARKER not in answer.lower()
        request_id = appdb.log_request(
            user,
            question,
            answer,
            grounded,
            False,
            len(chunks),
            int((time.time() - started) * 1000),
        )
        yield _sse_event(
            {
                "type": "sources",
                "data": [
                    {
                        "source": c.source,
                        "chunk": c.chunk,
                        "score": round(c.score, 4),
                        "citation": c.citation,
                        "text": c.text,
                        "url": c.url,
                        "title": c.title,
                        "page": c.page,
                        "kind": c.kind,
                    }
                    for c in chunks
                ],
            }
        )
        yield _sse_event({"type": "request", "data": {"request_id": request_id}})
        yield _sse_event(
            {"type": "meta", "data": {"grounded": grounded, "corrected": False}}
        )
        found = _related_images(question, req.show_images)
        if found:
            yield _sse_event({"type": "images", "data": found})
        yield _sse_event({"type": "done", "data": ""})

    return StreamingResponse(generate(), media_type="text/event-stream")


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse(payloads: list[dict]):
    for payload in payloads:
        yield _sse_event(payload)


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(path)


@router.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    path = Path(__file__).parent / "static" / "admin.html"
    return FileResponse(path)


@router.post("/chat", include_in_schema=False)
def chat(
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
) -> StreamingResponse:
    if not req.messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    return _stream(req, user)
