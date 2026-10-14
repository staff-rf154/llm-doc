"""API кабинета администратора: пользователи, оценки, исправления, статистика."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import appdb
import auth
import config
import corrections
import docpages
import ingest
import synonymdb
import vectorstore as vs
import verify

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
    user: dict = Depends(auth.require_admin),
) -> dict:
    return {
        "items": appdb.list_requests(limit=limit, offset=offset, vote=vote, search=search),
        "total": appdb.count_requests(vote=vote),
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
    return appdb.get_correction(correction_id) or {}


@router.delete("/corrections/{correction_id}")
def delete_correction(
    correction_id: int,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_correction(correction_id):
        raise HTTPException(status_code=404, detail="Исправление не найдено")
    appdb.delete_correction(correction_id)
    corrections.deactivate(correction_id)
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
        "variants": synonymdb.expand(q, max_variants=config.SYNONYMS_MAX_VARIANTS),
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
