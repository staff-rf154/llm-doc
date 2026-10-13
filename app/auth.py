"""Аутентификация по cookie-сессии: login/logout, доступ текущего пользователя."""

from fastapi import Depends, HTTPException, Request, Response

import appdb

COOKIE = "sid"


class AnonymousUser:
    id = None
    username = ""
    role = "guest"
    active = 0


def session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=appdb.SESSION_DAYS * 24 * 3600,
        path="/",
    )


def clear_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE, path="/")


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE)
    user = appdb.user_by_token(token)
    return user or {"id": None, "username": "", "role": "guest", "active": 0}


def require_user(user: dict = Depends(current_user)) -> dict:
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="Требуется вход")
    return user


def require_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return user


def authenticate(username: str, password: str) -> dict | None:
    token = appdb.login(username, password)
    if not token:
        return None
    return appdb.user_by_token(token)
