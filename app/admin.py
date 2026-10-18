"""API кабинета администратора: пользователи, оценки, исправления, статистика."""

import httpx
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import appdb
import auth
import autolearn
import bootstrap
import capabilities as capabilities_mod
import config
import corrections
import domains
import docpages
import documents as docs_mod
import evaluation
import images
import ingest
import lessons
import models
import questiongen
import settings
import synonymdb
import vectorstore as vs
import verify
import webcrawl

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=4, max_length=256)
    role: str = Field(default="user", pattern="^(user|admin)$")


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=4, max_length=256)
    role: str | None = Field(default=None, pattern="^(user|admin)$")
    active: bool | None = None


class PasswordReset(BaseModel):
    user_id: int
    password: str = Field(min_length=4, max_length=256)


class CorrectionCreate(BaseModel):
    question: str = Field(min_length=2)
    answer: str = Field(min_length=2)
    doc_name: str = Field(min_length=1, description="документ из data/docs")
    doc_pages: str = Field(min_length=1, description="номера страниц, например 12-15")


class CorrectionUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    status: str | None = Field(default=None, pattern="^(active|disabled)$")
    doc_name: str | None = None
    doc_pages: str | None = None


class PageExtractRequest(BaseModel):
    doc_name: str = Field(min_length=1)
    doc_pages: str = Field(min_length=1)


class SettingsUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class TrainTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class TrainRequest(BaseModel):
    history: list[TrainTurn] = Field(default_factory=list)
    question: str = Field(min_length=1)


class LessonCreate(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    question: str = ""
    answer: str = Field(min_length=2)
    focus: str = ""


class LessonUpdate(BaseModel):
    topic: str | None = None
    question: str | None = None
    answer: str | None = None
    focus: str | None = None
    status: str | None = Field(default=None, pattern="^(active|disabled|candidate)$")


@router.post("/train/chat")
def train_chat(
    payload: TrainRequest,
    user: dict = Depends(auth.require_admin),
) -> dict:
    history = [turn.model_dump() for turn in payload.history]
    try:
        answer = lessons.chat(history, payload.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
    return {"answer": answer}


@router.post("/train/draft")
def train_draft(
    payload: TrainRequest,
    user: dict = Depends(auth.require_admin),
) -> dict:
    history = [turn.model_dump() for turn in payload.history]
    history.append({"role": "user", "content": payload.question})
    try:
        draft = lessons.draft(history)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
    return draft


@router.get("/lessons")
def list_lessons(user: dict = Depends(auth.require_admin)) -> dict:
    items = appdb.list_lessons()
    return {
        "items": items,
        "total": len(items),
        "active": len([item for item in items if item["status"] == "active"]),
    }


@router.post("/lessons", status_code=201)
def create_lesson(
    payload: LessonCreate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    return lessons.save_lesson(
        payload.topic,
        payload.question,
        payload.answer,
        payload.focus,
        user["username"],
    )


@router.patch("/lessons/{lesson_id}")
def update_lesson(
    lesson_id: int,
    payload: LessonUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_lesson(lesson_id):
        raise HTTPException(status_code=404, detail="Урок не найден")
    appdb.update_lesson(lesson_id, **payload.model_dump(exclude_none=True))
    lessons.reindex()
    return appdb.get_lesson(lesson_id) or {}


@router.delete("/lessons/{lesson_id}")
def delete_lesson(lesson_id: int, user: dict = Depends(auth.require_admin)) -> dict:
    if not appdb.delete_lesson(lesson_id):
        raise HTTPException(status_code=404, detail="Урок не найден")
    lessons.reindex()
    return {"ok": True}


@router.get("/training/log")
def training_log(user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "items": appdb.list_training_log(),
        "job": lessons.learn_job.snapshot(),
    }


@router.post("/training/run")
def training_run(user: dict = Depends(auth.require_admin)) -> dict:
    """Отправить все исправления на обучение (повторно)."""
    queued = 0
    for correction in appdb.list_corrections(status="active"):
        lessons.learn_job.queue(correction, user["username"])
        queued += 1
    return {"queued": queued}


@router.get("/lessons/check")
def check_lessons(q: str, user: dict = Depends(auth.require_admin)) -> dict:
    """Показать, какие уроки сработают на этот вопрос."""
    return {"query": q, "lessons": lessons.relevant(q)}


class AutolearnRun(BaseModel):
    llm: bool | None = None


def _enrich_evals(items: list[dict]) -> None:
    """Добавляет к оценкам ошибки запроса и признак, чем вопрос уже закрыт."""
    lessons_map: dict = {}
    for item in appdb.list_lessons():
        key = (item.get("question") or "").strip().lower()
        if key and key not in lessons_map:
            lessons_map[key] = {"id": item["id"], "topic": item.get("topic"), "status": item.get("status")}
    corrections_map = appdb.correction_ids_by_question()
    for item in items:
        item["issues"] = appdb.issues_for_request(item["request_id"])
        covered = autolearn.coverage(item.get("question"), lessons_map, corrections_map)
        item["lesson"] = covered["lesson"]
        item["correction"] = covered["correction"]


@router.get("/autolearn")
def autolearn_overview(
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Обзор автообучения: сводка, состояние прогона, темы, кандидаты, параметры."""
    topics = [
        {**topic, "members": appdb.topic_requests(topic["id"], limit=8)}
        for topic in appdb.list_topics()
    ]
    return {
        "summary": autolearn.summary(),
        "job": autolearn.job.snapshot(),
        "topics": topics,
        "candidates": appdb.list_lessons(status="candidate"),
        "params": [item for item in settings.describe() if item["group"] == settings.AUTOLEARN],
        "totals": {
            "evals": appdb.count_request_evals(),
            "problems": appdb.count_request_evals(only_problems=True),
            "issues": appdb.count_issues("open"),
        },
    }


@router.get("/autolearn/evals")
def autolearn_evals(
    offset: int = 0,
    limit: int = 20,
    verdict: str | None = None,
    only_problems: bool = False,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Постраничный список оценок запросов с деталями по каждому проблемному."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    items = appdb.list_request_evals(
        limit=limit, offset=offset, verdict=verdict or None, only_problems=only_problems
    )
    _enrich_evals(items)
    total = appdb.count_request_evals(verdict or None, only_problems)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.get("/autolearn/issues")
def autolearn_issues(
    offset: int = 0,
    limit: int = 20,
    status: str | None = "open",
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Постраничный список обнаруженных ошибок."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    items = appdb.list_issues(status=status, limit=limit, offset=offset)
    total = appdb.count_issues(status)
    return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.post("/autolearn/run", status_code=202)
def autolearn_run(
    payload: AutolearnRun | None = None,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Запустить автообучение: эвристики, ассоциации, LLM-оценка, уроки-кандидаты."""
    return autolearn.job.start(payload.llm if payload else None)


@router.post("/autolearn/lessons/{lesson_id}/approve")
def autolearn_approve(
    lesson_id: int,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Одобрить урок-кандидат: он начнёт подмешиваться в ответы."""
    if not appdb.get_lesson(lesson_id):
        raise HTTPException(status_code=404, detail="Урок не найден")
    appdb.update_lesson(lesson_id, status="active")
    lessons.reindex()
    return appdb.get_lesson(lesson_id) or {}


@router.post("/autolearn/lessons/{lesson_id}/reject")
def autolearn_reject(
    lesson_id: int,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.delete_lesson(lesson_id):
        raise HTTPException(status_code=404, detail="Урок не найден")
    return {"ok": True}


@router.post("/autolearn/issues/{issue_id}/dismiss")
def autolearn_dismiss(
    issue_id: int,
    user: dict = Depends(auth.require_admin),
) -> dict:
    appdb.update_issue(issue_id, status="dismissed")
    return {"ok": True}


class QuestionGenRun(BaseModel):
    count: int | None = Field(default=None, ge=1, le=100)
    sources: list[str] | None = None
    judge: bool | None = None


@router.get("/questiongen")
def questiongen_overview(user: dict = Depends(auth.require_admin)) -> dict:
    """Проверка на сгенерированных вопросах: состояние и сводка."""
    return {
        "job": questiongen.job.snapshot(),
        "summary": questiongen.summary(),
        "sources": list(questiongen.SOURCES),
        "defaults": {
            "count": questiongen._count(),
            "sources": questiongen._sources(),
            "judge": questiongen._judge_enabled(),
        },
        "params": [
            item for item in settings.describe()
            if item["group"] == settings.AUTOLEARN and item["key"].startswith("QUESTIONGEN_")
        ],
    }


@router.post("/questiongen/run", status_code=202)
def questiongen_run(
    payload: QuestionGenRun | None = None,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Сгенерировать вопросы и прогнать их через ассистента (фон)."""
    payload = payload or QuestionGenRun()
    return questiongen.job.start(count=payload.count, sources=payload.sources, judge=payload.judge)


@router.get("/questiongen/report")
def questiongen_report(
    offset: int = 0,
    limit: int = 20,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Постраничный отчёт по сгенерированным вопросам с деталями."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    data = questiongen.report(limit=limit, offset=offset)
    _enrich_evals(data["items"])
    return data


class BootstrapRun(BaseModel):
    cycles: int | None = Field(default=None, ge=1, le=10)
    count: int | None = Field(default=None, ge=1, le=200)


@router.get("/bootstrap")
def bootstrap_overview(user: dict = Depends(auth.require_admin)) -> dict:
    """Первичное обучение: состояние, готовность и параметры."""
    return {
        "job": bootstrap.job.snapshot(),
        "readiness": bootstrap.readiness(),
        "params": [
            item for item in settings.describe()
            if item["group"] == settings.AUTOLEARN and item["key"].startswith("BOOTSTRAP_")
        ],
    }


@router.post("/bootstrap/run", status_code=202)
def bootstrap_run(
    payload: BootstrapRun | None = None,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Запустить первичное обучение по активной сфере (принудительно)."""
    payload = payload or BootstrapRun()
    return bootstrap.maybe_start("manual", force=True, cycles=payload.cycles, count=payload.count)


@router.get("/settings")
def get_settings(user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "title": appdb.get_setting("title", config.APP_TITLE) or config.APP_TITLE,
        "default_title": config.APP_TITLE,
        "groups": settings.groups(),
        "items": settings.describe(),
        "llm_models": _available_llm_models(),
    }


@router.patch("/settings")
def update_settings(
    payload: SettingsUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    appdb.set_setting("title", payload.title.strip())
    return {"title": appdb.get_setting("title", config.APP_TITLE)}


class ParamsUpdate(BaseModel):
    values: dict[str, str]


@router.patch("/settings/params")
def update_params(
    payload: ParamsUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        result = settings.save(payload.values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reindex = result["reindex"]
    warnings = result["warnings"]
    detail = ""
    if reindex:
        detail = "Изменения применены, но требуют переиндексации: " + ", ".join(reindex)
    elif warnings:
        detail = "Проверьте настройки: " + "; ".join(warnings)
    return {
        "ok": True,
        "items": settings.describe(),
        "requires_reindex": reindex,
        "warnings": warnings,
        "detail": detail,
    }


@router.post("/settings/params/{key}/reset")
def reset_param(key: str, user: dict = Depends(auth.require_admin)) -> dict:
    param = next((item for item in settings.PARAMS if item.key == key), None)
    if param is None:
        raise HTTPException(status_code=404, detail="Неизвестный параметр")
    if not param.editable:
        raise HTTPException(status_code=400, detail="Параметр доступен только для чтения")
    settings.reset(key)
    return {"ok": True, "items": settings.describe()}


class ModelName(BaseModel):
    name: str = Field(min_length=1)


class ModelKind(BaseModel):
    kind: str = Field(pattern="^(embedding|reranker|sparse)$")
    name: str = Field(min_length=1)


@router.get("/models")
def models_state(user: dict = Depends(auth.require_admin)) -> dict:
    return models.summary()


@router.post("/models/ollama/pull", status_code=202)
def models_pull(payload: ModelName, user: dict = Depends(auth.require_admin)) -> dict:
    return models.job.start(payload.name)


@router.get("/models/ollama/pull")
def models_pull_state(user: dict = Depends(auth.require_admin)) -> dict:
    return models.job.snapshot()


@router.delete("/models/ollama")
def models_ollama_delete(name: str, user: dict = Depends(auth.require_admin)) -> dict:
    try:
        return models.ollama_delete(name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama недоступна: {exc}") from exc


@router.post("/models/ollama/unload")
def models_ollama_unload(payload: ModelName, user: dict = Depends(auth.require_admin)) -> dict:
    try:
        return models.ollama_unload(payload.name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama недоступна: {exc}") from exc


@router.post("/models/ollama/select")
def models_ollama_select(payload: ModelName, user: dict = Depends(auth.require_admin)) -> dict:
    return models.ollama_select(payload.name)


@router.post("/models/hf/download", status_code=202)
def models_hf_download(payload: ModelKind, user: dict = Depends(auth.require_admin)) -> dict:
    try:
        return models.hf_download(payload.kind, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось скачать модель: {exc}") from exc


@router.post("/models/hf/select")
def models_hf_select(payload: ModelKind, user: dict = Depends(auth.require_admin)) -> dict:
    try:
        return models.hf_select(payload.kind, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/hf")
def models_hf_delete(
    name: str, kind: str = "", user: dict = Depends(auth.require_admin)
) -> dict:
    try:
        return models.hf_delete(kind, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _available_llm_models() -> list[str]:
    """Модели, доступные у текущего LLM-сервера."""
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=settings.value("LLM_BASE_URL", config.LLM_BASE_URL),
            api_key=settings.value("LLM_API_KEY", config.LLM_API_KEY) or "none",
            timeout=5,
        )
        models = client.models.list()
        names = sorted({getattr(item, "id", "") for item in models.data if getattr(item, "id", "")})
        if names:
            return names
    except Exception:
        pass
    current = settings.value("LLM_MODEL", config.LLM_MODEL)
    return [current] if current else []


class RatingFilter(BaseModel):
    vote: str | None = None


class SynonymUpsert(BaseModel):
    term: str = Field(min_length=1, max_length=128)
    synonyms: list[str] = Field(default_factory=list)


class SynonymImport(BaseModel):
    items: dict[str, list[str]]


@router.get("/summary")
def summary(user: dict = Depends(auth.require_admin)) -> dict:
    return appdb.summary()


@router.get("/users")
def users(user: dict = Depends(auth.require_admin)) -> dict:
    return {"users": appdb.list_users()}


@router.post("/users", status_code=201)
def create_user(payload: UserCreate, user: dict = Depends(auth.require_admin)) -> dict:
    if appdb.get_user_by_name(payload.username):
        raise HTTPException(status_code=409, detail="Пользователь уже существует")
    return appdb.create_user(payload.username, payload.password, payload.role)


@router.patch("/users/{user_id}")
def update_user(
    user_id: int,
    payload: UserUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_user(user_id):
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    if payload.password:
        appdb.set_password(user_id, payload.password)
    if payload.role:
        if user_id == user["id"] and payload.role != "admin":
            raise HTTPException(status_code=400, detail="Нельзя снять роль у себя")
        appdb.set_role(user_id, payload.role)
    if payload.active is not None:
        if user_id == user["id"] and not payload.active:
            raise HTTPException(status_code=400, detail="Нельзя отключить себя")
        appdb.set_active(user_id, payload.active)
    return appdb.get_user(user_id) or {}


@router.post("/users/reset-password")
def reset_password(
    payload: PasswordReset,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_user(payload.user_id):
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    appdb.set_password(payload.user_id, payload.password)
    return {"ok": True}


@router.delete("/users/{user_id}")
def delete_user(user_id: int, user: dict = Depends(auth.require_admin)) -> dict:
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Нельзя удалить себя")
    if not appdb.get_user(user_id):
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    appdb.delete_user(user_id)
    return {"ok": True}


@router.get("/requests")
def requests(
    limit: int = 50,
    offset: int = 0,
    vote: str | None = None,
    search: str | None = None,
    username: str | None = None,
    origin: str | None = None,
    sort: str = "date",
    order: str = "desc",
    user: dict = Depends(auth.require_admin),
) -> dict:
    items = appdb.list_requests(
        limit=limit,
        offset=offset,
        vote=vote,
        search=search,
        username=username,
        origin=origin,
        sort=sort,
        order=order,
    )
    by_question = appdb.correction_ids_by_question()
    for item in items:
        item["correction_id"] = by_question.get(item["question"].strip().lower())
        item["is_system"] = item.get("origin") not in ("", "web")
    return {
        "items": items,
        "total": appdb.count_requests(
            vote=vote, username=username, search=search, origin=origin
        ),
    }


@router.get("/stats")
def stats(days: int = 30, user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "summary": appdb.summary(),
        "by_user": appdb.stats_by_user(),
        "by_day": appdb.stats_by_day(days),
        "top_questions": appdb.top_questions(),
    }


@router.get("/documents")
def documents(user: dict = Depends(auth.require_admin)) -> dict:
    """Документы, доступные для ссылки на страницу."""
    return {"items": docpages.list_documents()}


class EvalStart(BaseModel):
    ragas: bool = False


@router.get("/eval")
def eval_status(user: dict = Depends(auth.require_admin)) -> dict:
    """Отчёт по качеству поиска и ответов + состояние фонового прогона."""
    return {"job": evaluation.job.snapshot(), **evaluation.summary()}


@router.post("/eval/start", status_code=202)
def eval_start(
    payload: EvalStart,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Запустить оценку качества по датасету."""
    result = evaluation.job.start(payload.ragas)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/docs")
def docs_list(
    offset: int = 0,
    limit: int = 20,
    q: str = "",
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Файлы документации и их состояние в индексе (с постраничным выводом)."""
    items = docs_mod.list_documents()
    offset = max(0, offset)
    limit = max(1, min(limit, 200))
    needle = (q or "").strip().lower()
    if needle:
        items = [item for item in items if needle in item["name"].lower()]
    total = len(items)
    indexed_all = len([item for item in items if item["indexed"]])
    uploaded_all = len([item for item in items if item["editable"]])
    return {
        "items": items[offset:offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
        "query": q,
        "indexed": indexed_all,
        "uploaded": uploaded_all,
        "max_mb": settings.int_value("UPLOAD_MAX_MB") or config.UPLOAD_MAX_MB,
        "formats": list(docs_mod.FILE_EXTENSIONS),
    }


@router.post("/docs/upload", status_code=201)
async def docs_upload(
    file: UploadFile = File(...),
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Загрузить документ с компьютера администратора."""
    try:
        saved = docs_mod.save_bytes(file.filename or "", await file.read())
        indexed = docs_mod.index_document(saved["name"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить: {exc}") from exc
    return {**saved, **indexed}


class DocUrl(BaseModel):
    url: str = Field(min_length=8, max_length=1000)


@router.post("/docs/url", status_code=201)
def docs_url(payload: DocUrl, user: dict = Depends(auth.require_admin)) -> dict:
    """Скачать документ по http-ссылке и сразу проиндексировать."""
    try:
        saved = docs_mod.fetch_url(payload.url)
        indexed = docs_mod.index_document(saved["name"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Сервер вернул {exc.response.status_code}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось скачать: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось скачать: {exc}") from exc
    return {**saved, **indexed}


@router.post("/docs/{name:path}/index", status_code=202)
def docs_index(name: str, user: dict = Depends(auth.require_admin)) -> dict:
    """Переиндексировать один документ."""
    try:
        return docs_mod.index_document(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/docs/{name:path}")
def docs_delete(
    name: str,
    root: str = "",
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Удалить документ: из загрузок — с диска, из документации — из индекса."""
    if Path(docs_mod.safe_name(name)).suffix.lower() not in docs_mod.FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Формат файла не поддерживается")
    try:
        result = docs_mod.delete_document(name, root or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result["removed_file"] and not result["purged_points"] and not result["excluded"]:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return result


class DocRestore(BaseModel):
    name: str = Field(min_length=1)
    root: str = Field(min_length=1)


@router.post("/docs/restore", status_code=202)
def docs_restore(payload: DocRestore, user: dict = Depends(auth.require_admin)) -> dict:
    """Вернуть исключённый документ в индекс."""
    try:
        return docs_mod.restore_document(payload.name, payload.root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/pages")
def document_pages(
    payload: PageExtractRequest,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Текст указанных страниц документа — основа для исправленного ответа."""
    if not docpages.find_document(payload.doc_name):
        raise HTTPException(status_code=404, detail="Документ не найден в data/docs")
    text = docpages.extract_pages(payload.doc_name, payload.doc_pages)
    return {
        "doc_name": payload.doc_name,
        "doc_pages": docpages.format_pages(docpages.parse_pages(payload.doc_pages)),
        "doc_ref": docpages.build_reference(payload.doc_name, payload.doc_pages),
        "text": text,
    }


@router.get("/corrections")
def list_corrections(user: dict = Depends(auth.require_admin)) -> dict:
    return {"items": appdb.list_corrections()}


@router.post("/corrections", status_code=201)
def create_correction(
    payload: CorrectionCreate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    pages = docpages.format_pages(docpages.parse_pages(payload.doc_pages))
    item = appdb.create_correction(
        payload.question,
        payload.answer,
        user["username"],
        payload.doc_name.strip(),
        pages,
        docpages.build_reference(payload.doc_name, pages),
    )
    corrections.reindex()
    if config.TRAINING_FROM_CORRECTIONS:
        lessons.learn_job.queue(item, user["username"])
    return item


@router.patch("/corrections/{correction_id}")
def update_correction(
    correction_id: int,
    payload: CorrectionUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    current = appdb.get_correction(correction_id)
    if not current:
        raise HTTPException(status_code=404, detail="Исправление не найдено")

    doc_name = payload.doc_name if payload.doc_name is not None else current["doc_name"]
    doc_pages = (
        docpages.format_pages(docpages.parse_pages(payload.doc_pages))
        if payload.doc_pages is not None
        else current["doc_pages"]
    )
    if payload.answer is not None or payload.doc_name is not None or payload.doc_pages is not None:
        if not doc_name.strip() or not doc_pages.strip():
            raise HTTPException(
                status_code=400,
                detail="При редактировании ответа обязательны документ и номера страниц",
            )
    appdb.update_correction(
        correction_id,
        question=payload.question,
        answer=payload.answer,
        status=payload.status,
        doc_name=doc_name,
        doc_pages=doc_pages,
        doc_ref=docpages.build_reference(doc_name, doc_pages),
    )
    corrections.reindex()
    updated = appdb.get_correction(correction_id) or {}
    if config.TRAINING_FROM_CORRECTIONS and updated:
        lessons.learn_job.queue(updated, user["username"])
    return updated


@router.delete("/corrections/{correction_id}")
def delete_correction(
    correction_id: int,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_correction(correction_id):
        raise HTTPException(status_code=404, detail="Исправление не найдено")
    lesson_ids = appdb.training_lessons_for_correction(correction_id)
    appdb.delete_correction(correction_id)
    corrections.deactivate(correction_id)
    for lesson_id in lesson_ids:
        appdb.delete_lesson(lesson_id)
    appdb.delete_training_log_for_correction(correction_id)
    lessons.reindex()
    return {"ok": True}


@router.post("/corrections/{correction_id}/from-request/{request_id}")
def correction_from_request(
    correction_id: int,
    request_id: int,
    payload: CorrectionUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    items = appdb.list_requests(limit=10_000)
    for item in items:
        if item["id"] == request_id:
            break
    else:
        raise HTTPException(status_code=404, detail="Запрос не найден")
    doc_name = payload.doc_name or item["doc_name"]
    doc_pages = payload.doc_pages or item["doc_pages"]
    if not doc_name or not doc_pages:
        raise HTTPException(
            status_code=400,
            detail="Укажите документ и номера страниц: без них ответ редактировать нельзя",
        )
    answer = payload.answer if payload.answer is not None else item["answer"]
    question = payload.question if payload.question is not None else item["question"]
    return update_correction(
        correction_id,
        CorrectionUpdate(
            question=question,
            answer=answer,
            status=payload.status,
            doc_name=doc_name,
            doc_pages=doc_pages,
        ),
        user,
    )


@router.get("/synonyms")
def list_synonyms(user: dict = Depends(auth.require_admin)) -> dict:
    return {"items": synonymdb.list_terms(), "total": synonymdb.count()}


@router.post("/synonyms", status_code=201)
def upsert_synonym(payload: SynonymUpsert, user: dict = Depends(auth.require_admin)) -> dict:
    item = synonymdb.set_term(payload.term, payload.synonyms, user["username"])
    synonymdb.invalidate()
    return item


@router.post("/synonyms/import")
def import_synonyms(
    payload: SynonymImport,
    user: dict = Depends(auth.require_admin),
) -> dict:
    total = synonymdb.import_many(payload.items, author=user["username"])
    synonymdb.invalidate()
    return {"total": total}


@router.delete("/synonyms/{term}")
def delete_synonym(term: str, user: dict = Depends(auth.require_admin)) -> dict:
    if not synonymdb.delete_term(term):
        raise HTTPException(status_code=404, detail="Термин не найден")
    synonymdb.invalidate()
    return {"ok": True}


@router.get("/synonyms/expand")
def expand_synonyms(q: str, user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "query": q,
        "variants": synonymdb.expand(q, max_variants=settings.int_value("SYNONYMS_MAX_VARIANTS")),
    }


@router.post("/corrections/by-question")
def upsert_correction_by_question(
    payload: CorrectionCreate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Обновить исправление с таким же вопросом или создать новое."""
    existing = None
    for item in appdb.list_corrections():
        if item["question"].strip().lower() == payload.question.strip().lower():
            existing = item
            break
    if existing:
        return update_correction(
            existing["id"],
            CorrectionUpdate(
                question=payload.question,
                answer=payload.answer,
                doc_name=payload.doc_name,
                doc_pages=payload.doc_pages,
            ),
            user,
        )
    return create_correction(payload, user)


@router.get("/verify")
def verify_status(user: dict = Depends(auth.require_admin)) -> dict:
    status = verify.job.snapshot()
    status["summary"] = appdb.check_summary()
    return status


@router.post("/verify", status_code=202)
def verify_start(user: dict = Depends(auth.require_admin)) -> dict:
    return verify.job.start()


@router.post("/verify/reset")
def verify_reset(user: dict = Depends(auth.require_admin)) -> dict:
    appdb.reset_checks()
    return {"ok": True, "summary": appdb.check_summary()}


@router.get("/images")
def images_status(user: dict = Depends(auth.require_admin)) -> dict:
    status = images.job.snapshot()
    client = vs.get_qdrant()
    status["exists"] = client.collection_exists(images.collection_name())
    if status["exists"]:
        status["indexed"] = len(images.indexed_points())
    return status


@router.post("/images/start", status_code=202)
def images_start(user: dict = Depends(auth.require_admin)) -> dict:
    return images.job.start()


class CapabilitiesUpdate(BaseModel):
    groups: list[dict]


@router.get("/capabilities")
def get_capabilities(user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "groups": capabilities_mod.groups(),
        "custom": capabilities_mod.is_custom(),
        "from_domain": bool(domains.capabilities()),
        "source": capabilities_mod.source(),
        "text": capabilities_mod.render(),
    }


@router.patch("/capabilities")
def update_capabilities(
    payload: CapabilitiesUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        saved = capabilities_mod.save_groups(payload.groups)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "groups": saved,
        "custom": True,
        "from_domain": bool(domains.capabilities()),
        "source": capabilities_mod.source(),
    }


@router.post("/capabilities/reset")
def reset_capabilities(user: dict = Depends(auth.require_admin)) -> dict:
    saved = capabilities_mod.reset_groups()
    return {
        "groups": saved,
        "custom": False,
        "from_domain": bool(domains.capabilities()),
        "source": capabilities_mod.source(),
    }


class DomainSelect(BaseModel):
    profile: str


class DomainCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    slug: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    capabilities: list[dict] | None = None
    categories: list[str] | None = None
    terminology: list[str] | None = None
    sample_questions: list[str] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    allow_inference: bool | None = None
    log_analysis: bool | None = None


class DomainUpdate(BaseModel):
    title: str | None = None
    slug: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    capabilities: list[dict] | None = None
    categories: list[str] | None = None
    terminology: list[str] | None = None
    sample_questions: list[str] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    allow_inference: bool | None = None
    log_analysis: bool | None = None


def _domain_catalog() -> dict:
    current = domains.current()
    return {
        "current": current.get("slug", ""),
        "profile": current,
        "profiles": domains.list_profiles(),
        "default": domains.DEFAULT_SLUG,
        "env": config.DOMAIN_PROFILE,
    }


@router.get("/domain")
def get_domain(user: dict = Depends(auth.require_admin)) -> dict:
    """Сфера деятельности: активное направление и каталог."""
    return _domain_catalog()


@router.patch("/domain")
def select_domain(
    payload: DomainSelect,
    user: dict = Depends(auth.require_admin),
) -> dict:
    """Выбрать сферу деятельности: подставляются её инструкция и настройки."""
    try:
        domains.set_profile(payload.profile)
        capabilities_mod.reset_groups()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    bootstrap.maybe_start("domain")
    return _domain_catalog()


@router.post("/domain/reset")
def reset_domain(user: dict = Depends(auth.require_admin)) -> dict:
    domains.reset()
    capabilities_mod.reset_groups()
    return _domain_catalog()


@router.post("/domains", status_code=201)
def create_domain(
    payload: DomainCreate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        return domains.create(payload.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/domains/{slug}")
def update_domain(
    slug: str,
    payload: DomainUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        return domains.update(slug, payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/domains/{slug}")
def delete_domain(
    slug: str,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        removed = domains.delete(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Направление не найдено")
    return {"ok": True}


@router.post("/domains/{slug}/duplicate", status_code=201)
def duplicate_domain(
    slug: str,
    user: dict = Depends(auth.require_admin),
) -> dict:
    try:
        return domains.duplicate(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class SiteCreate(BaseModel):
    url: str = Field(min_length=3, max_length=300)
    enabled: bool = True
    max_pages: int | None = None


class SitesUpdate(BaseModel):
    sites: list[SiteCreate]


@router.get("/sites")
def get_sites(user: dict = Depends(auth.require_admin)) -> dict:
    return {
        "sites": webcrawl.sites(),
        "crawl": webcrawl.job.snapshot(),
        "respect_robots": config.WEB_CRAWL_RESPECT_ROBOTS,
        "max_pages": settings.int_value("WEB_CRAWL_MAX_PAGES"),
    }


@router.put("/sites")
def put_sites(
    payload: SitesUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    return {"sites": webcrawl.save_sites([item.model_dump() for item in payload.sites])}


@router.post("/sites/start", status_code=202)
def start_crawl(user: dict = Depends(auth.require_admin)) -> dict:
    if not webcrawl.sites():
        raise HTTPException(status_code=400, detail="Список сайтов пуст")
    return webcrawl.job.start()


@router.get("/index")
def index_status(user: dict = Depends(auth.require_admin)) -> dict:
    status = ingest.job.snapshot()
    status["target"] = vs.target_name()
    status["collection_info"] = vs.collection_info(status["target"])
    return status


@router.post("/index/start")
def index_start(user: dict = Depends(auth.require_admin)) -> dict:
    return ingest.job.start()


@router.post("/index/pause")
def index_pause(user: dict = Depends(auth.require_admin)) -> dict:
    return ingest.job.pause()


@router.post("/index/resume")
def index_resume(user: dict = Depends(auth.require_admin)) -> dict:
    return ingest.job.start()
