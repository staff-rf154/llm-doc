"""Аутентификация по cookie-сессии: login/logout, доступ текущего пользователя."""

from fastapi import Depends, HTTPException, Request, Response

import appdb
import settings

COOKIE = "sid"


class AnonymousUser:
    id = None
    username = ""
    role = "guest"
    tenant_id = "default"
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
    user = appdb.user_by_token(token) or {
        "id": None,
        "username": "",
        "role": "guest",
        "tenant_id": appdb.DEFAULT_TENANT,
        "active": 0,
    }
    settings.CURRENT_TENANT.set(user.get("tenant_id") or appdb.DEFAULT_TENANT)
    return user


def require_user(user: dict = Depends(current_user)) -> dict:
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="Требуется вход")
    return user


def require_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("role") not in ("admin", "system", "tenant_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return user


def require_tenant_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("role") not in ("admin", "system", "tenant_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return user


def require_global_admin(user: dict = Depends(require_user)) -> dict:
    """Глобальные разделы: администратор всей установки или system."""
    if user.get("role") not in ("admin", "system"):
        raise HTTPException(status_code=403, detail="Недоступно для роли тенанта")
    return user


TENANT_ADMIN_PATHS = (
    "/users",
    "/requests",
    "/stats",
    "/summary",
    "/docs",
    "/documents",
    "/corrections",
    "/domain",
    "/domains",
    "/verify",
)


def tenant_guard(request: Request, user: dict = Depends(require_user)) -> dict:
    if user.get("role") != "tenant_admin":
        return user
    path = request.url.path.removeprefix("/api/admin")
    if not any(path == item or path.startswith(item + "/") for item in TENANT_ADMIN_PATHS):
        raise HTTPException(status_code=403, detail="Недоступно для роли тенанта")
    return user


def tenant_scope(user: dict) -> str | None:
    """Тенант для фильтрации данных; None — без ограничения (system/admin)."""
    if user.get("role") in ("admin", "system"):
        return None
    return user.get("tenant_id") or appdb.DEFAULT_TENANT


def require_system(user: dict = Depends(require_user)) -> dict:
    """Доступ только к системному пользователю: сферы деятельности и модели."""
    if user.get("role") != "system":
        raise HTTPException(status_code=403, detail="Недоступно для вашей роли")
    return user


def authenticate(username: str, password: str) -> dict | None:
    token = appdb.login(username, password)
    if not token:
        return None
    return appdb.user_by_token(token)
