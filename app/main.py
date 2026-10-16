from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

import time
from pathlib import Path

import admin
import appdb
import auth
import config
import capabilities as capabilities_mod
import ingest
import promptdb
import prompts
import rag
import settings
import synonymdb
import vectorstore
import web
from fastapi import Depends

app = FastAPI(
    title="LLM Documentation RAG",
    description="RAG по внутренней документации: retrieval + генерация с цитированием",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    appdb.ensure_schema()
    created = appdb.ensure_admin()
    if created:
        print(created)
    try:
        import corrections

        total = corrections.reindex()
        print(f"исправлений в индексе: {total}")
    except Exception as exc:
        print(f"индекс исправлений недоступен: {exc}")
    try:
        import synonyms

        print(f"терминов в словаре синонимов: {synonyms.seed_if_empty()}")
    except Exception as exc:
        print(f"словарь синонимов недоступен: {exc}")
    try:
        import lessons

        print(f"уроков в индексе обучения: {lessons.reindex()}")
    except Exception as exc:
        print(f"индекс уроков недоступен: {exc}")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


def app_title() -> str:
    """Название базы знаний: значение из настроек, иначе из APP_TITLE."""
    try:
        return appdb.get_setting("title", config.APP_TITLE) or config.APP_TITLE
    except Exception:
        return config.APP_TITLE


@app.get("/api/images/{filename}")
def image_file(filename: str):
    """Изображение из документации, извлечённое при индексации."""
    from fastapi.responses import FileResponse

    safe = Path(filename).name
    path = Path(config.IMAGES_DIR) / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(path)


@app.get("/api/title")
def title() -> dict:
    return {"title": app_title()}


@app.post("/api/login")
def login(payload: LoginRequest, response: Response) -> dict:
    token = appdb.login(payload.username, payload.password)
    if not token:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    auth.session_cookie(response, token)
    return appdb.user_by_token(token) or {}


@app.post("/api/logout")
def logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(auth.COOKIE)
    if token:
        appdb.logout(token)
    auth.clear_cookie(response)
    return {"ok": True}


@app.get("/api/me")
def me(user: dict = Depends(auth.current_user)) -> dict:
    return user


class AskRequest(BaseModel):
    question: str = Field(min_length=2, examples=["Как восстановить доступ к VPN?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    show_images: bool = True


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int | None = Field(default=None, ge=1, le=50)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "collection": config.QDRANT_COLLECTION,
        "title": app_title(),
        "llm_model": settings.value("LLM_MODEL", config.LLM_MODEL),
        "embedding_model": config.EMBEDDING_MODEL,
    }


@app.get("/collection")
def collection() -> dict:
    info = vectorstore.collection_info(vectorstore.target_name())
    info["target"] = config.QDRANT_COLLECTION
    return info


@app.post("/reindex", status_code=202)
def reindex() -> dict:
    """Запустить полную переиндексацию в фоне; поиск продолжает работать."""
    result = ingest.job.start()
    if not result.pop("started", False):
        result["detail"] = "переиндексация уже выполняется"
    return result


@app.get("/prompts")
def starter_prompts() -> dict:
    """Стартовые промты из отдельной БД, отобранные по составу индекса."""
    items = prompts.get_prompts()
    return {
        "prompts": [item["text"] for item in items],
        "items": items,
        "total_in_db": promptdb.count_prompts(),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    """Перечень возможностей ассистента и подсказки из БД промтов."""
    return {
        "groups": capabilities_mod.groups(),
        "prompts": capabilities_mod.suggestions(),
        "text": capabilities_mod.render(),
        "custom": capabilities_mod.is_custom(),
        "source": "кабинет администратора" if capabilities_mod.is_custom() else "код (app/capabilities.py)",
    }


@app.get("/synonyms")
def synonyms() -> dict:
    """Словарь синонимов, расширяющий поисковый запрос."""
    return {"items": synonymdb.list_terms(), "total": synonymdb.count()}


@app.get("/reindex")
def reindex_status() -> dict:
    status = ingest.job.snapshot()
    status["target"] = vectorstore.target_name()
    status["collection_info"] = vectorstore.collection_info(status["target"])
    return status


@app.post("/reindex/pause", status_code=202)
def reindex_pause() -> dict:
    """Поставить фоновую индексацию на паузу со срезом прогресса."""
    result = ingest.job.pause()
    if not result.pop("paused", False):
        result["detail"] = "индексация не запущена"
    return result


@app.post("/reindex/resume", status_code=202)
def reindex_resume() -> dict:
    """Продолжить индексацию с последнего сохранённого среза."""
    return ingest.job.start()


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    chunks = rag.retrieve(req.query, req.top_k)
    return {
        "query": req.query,
        "results": [
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
            for c in chunks
        ],
    }


@app.post("/ask")
def ask(
    req: AskRequest,
    user: dict = Depends(auth.require_user),
) -> dict:
    started = time.time()
    try:
        answer = rag.ask(req.question, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
    request_id = appdb.log_request(
        user,
        req.question,
        answer.text,
        answer.grounded,
        answer.corrected,
        len(answer.chunks),
        int((time.time() - started) * 1000),
    )
    payload = answer.to_dict()
    payload["request_id"] = request_id
    return payload


class RatingRequest(BaseModel):
    vote: str = Field(pattern="^(up|down)$")
    comment: str = ""


@app.post("/api/rate/{request_id}")
def rate(
    request_id: int,
    payload: RatingRequest,
    user: dict = Depends(auth.require_user),
) -> dict:
    appdb.set_rating(request_id, payload.vote, payload.comment)
    return {
        "ok": True,
        "request_id": request_id,
        "vote": payload.vote,
    }


app.include_router(admin.router)
app.include_router(web.router)