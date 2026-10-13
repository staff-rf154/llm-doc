"""API кабинета администратора: пользователи, оценки, исправления, статистика."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import appdb
import auth
import corrections
import ingest
import vectorstore as vs

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


class CorrectionUpdate(BaseModel):
    question: str | None = None
    answer: str | None = None
    status: str | None = Field(default=None, pattern="^(active|disabled)$")


class RatingFilter(BaseModel):
    vote: str | None = None


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


@router.get("/corrections")
def list_corrections(user: dict = Depends(auth.require_admin)) -> dict:
    return {"items": appdb.list_corrections()}


@router.post("/corrections", status_code=201)
def create_correction(
    payload: CorrectionCreate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    item = appdb.create_correction(payload.question, payload.answer, user["username"])
    corrections.reindex()
    return item


@router.patch("/corrections/{correction_id}")
def update_correction(
    correction_id: int,
    payload: CorrectionUpdate,
    user: dict = Depends(auth.require_admin),
) -> dict:
    if not appdb.get_correction(correction_id):
        raise HTTPException(status_code=404, detail="Исправление не найдено")
    appdb.update_correction(
        correction_id,
        question=payload.question,
        answer=payload.answer,
        status=payload.status,
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
    answer = payload.answer if payload.answer is not None else item["answer"]
    question = payload.question if payload.question is not None else item["question"]
    return update_correction(
        correction_id,
        CorrectionUpdate(question=question, answer=answer, status=payload.status),
        user,
    )


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
