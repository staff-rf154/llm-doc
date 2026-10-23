import json
import queue
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import appdb
import auth
import config
import domains
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


def _history(messages: list[Turn], tenant: str | None = None) -> list[dict]:
    turns = [{"role": "system", "content": domains.system_prompt(tenant or appdb.DEFAULT_TENANT)}]
    for turn in messages[:-1]:
        turns.append({"role": turn.role, "content": turn.content})
    turns.append({"role": "user", "content": messages[-1].content})
    return turns


def _autolearn(request_id: int, tenant: str | None = None) -> None:
    """Автоматическая оценка запроса после ответа (без LLM, дёшево)."""
    if tenant not in (None, appdb.DEFAULT_TENANT):
        return
    try:
        import autolearn

        autolearn.evaluate_request(request_id)
    except Exception:
        pass


def _correction_response(
    user: dict, question: str, exact: dict, started: float
) -> StreamingResponse:
    answer_text = exact["answer"] + rag.reference_block(
        exact.get("doc_name", ""), exact.get("doc_pages", "")
    )
    reference = rag.correction_reference(exact) or {}
    request_id = appdb.log_request(
        user,
        question,
        answer_text,
        True,
        True,
        0,
        int((time.time() - started) * 1000),
        reference,
        origin="web",
    )
    payload = [
        {"type": "token", "data": answer_text},
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


def _stream(req: ChatRequest, user: dict) -> StreamingResponse:
    question = req.messages[-1].content
    tenant = auth.tenant_scope(user) or appdb.DEFAULT_TENANT
    started = time.time()

    if is_capabilities_question(question):
        text = render_capabilities()
        request_id = appdb.log_request(user, question, text, True, False, 0, 0, origin="web")
        payload = [
            {"type": "token", "data": text},
            {"type": "prompts", "data": suggestions()},
            {"type": "sources", "data": []},
            {"type": "request", "data": {"request_id": request_id}},
            {"type": "meta", "data": {"grounded": True, "corrected": False}},
        ]
        payload.append({"type": "done", "data": ""})
        return StreamingResponse(
            _sse(payload),
            media_type="text/event-stream",
        )

    exact, examples = rag.correction_context(question, tenant)
    verify = settings.bool_value(
        "CORRECTION_MODEL_VERIFY", config.CORRECTION_MODEL_VERIFY
    )
    if exact and not verify:
        return _correction_response(user, question, exact, started)

    chunks = rag.retrieve(question, req.top_k, tenant=tenant)

    if not chunks:
        if exact:
            return _correction_response(user, question, exact, started)
        text = rag.empty_index_message()
        request_id = appdb.log_request(user, question, text, False, False, 0, 0, origin="web")
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

    prompt = rag.build_prompt(question, chunks, examples, tenant, exact)
    client = rag.get_client()

    def generate():
        events: queue.Queue = queue.Queue()
        done = object()
        answer_parts: list[str] = []
        heartbeat = settings.int_value("SSE_HEARTBEAT") or config.SSE_HEARTBEAT

        def pump() -> None:
            try:
                stream = client.chat.completions.create(
                    model=settings.value("LLM_MODEL", config.LLM_MODEL),
                    messages=_history(req.messages, tenant)[:-1]
                    + [{"role": "user", "content": prompt}],
                    temperature=domains.temperature(tenant)
                    if domains.temperature(tenant) is not None
                    else settings.float_value("LLM_TEMPERATURE"),
                    stream=True,
                )
                for event in stream:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    if delta.content:
                        answer_parts.append(delta.content)
                        events.put({"type": "token", "data": delta.content})
            except Exception as exc:
                events.put({"type": "error", "data": f"Ошибка генерации: {exc}"})
            finally:
                events.put(done)

        threading.Thread(target=pump, daemon=True).start()
        failed = False
        while True:
            try:
                item = events.get(timeout=heartbeat)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if item is done:
                break
            if item["type"] == "error":
                failed = True
            yield _sse_event(item)
        if failed:
            return

        answer = "".join(answer_parts).strip()
        grounded = rag.NO_ANSWER_MARKER not in answer.lower()
        block = rag.sources_block(chunks) if grounded else ""
        if block:
            yield _sse_event({"type": "token", "data": block})
            answer += block
        corrected = bool(exact) and grounded
        reference = rag.correction_reference(exact) if corrected else None
        request_id = appdb.log_request(
            user,
            question,
            answer,
            grounded,
            corrected,
            len(chunks),
            int((time.time() - started) * 1000),
            reference,
            origin="web",
        )
        _autolearn(request_id, tenant)
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
        if reference:
            yield _sse_event({"type": "reference", "data": reference})
        yield _sse_event({"type": "request", "data": {"request_id": request_id}})
        yield _sse_event(
            {"type": "meta", "data": {"grounded": grounded, "corrected": corrected}}
        )
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
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    path = Path(__file__).parent / "static" / "admin.html"
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.post("/chat", include_in_schema=False)
def chat(
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
) -> StreamingResponse:
    if not req.messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    return _stream(req, user)
