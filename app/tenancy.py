"""Тенанты: общие правила путей, payload и фильтров Qdrant."""

from pathlib import PurePosixPath

DEFAULT = "default"
PREFIX = "tenants"


def normalize(tenant: str | None) -> str:
    return (tenant or DEFAULT).strip() or DEFAULT


def rel_prefix(tenant: str | None) -> str:
    tenant = normalize(tenant)
    return "" if tenant == DEFAULT else f"{PREFIX}/{tenant}"


def tenant_of_rel(rel: str) -> str:
    parts = PurePosixPath((rel or "").replace("\\", "/")).parts
    if len(parts) >= 2 and parts[0] == PREFIX:
        return parts[1]
    return DEFAULT


def in_tenant(rel: str, tenant: str | None) -> bool:
    tenant = normalize(tenant)
    if tenant == DEFAULT:
        return tenant_of_rel(rel) == DEFAULT
    return tenant_of_rel(rel) == tenant


def scoped_rel(tenant: str | None, rel: str) -> str:
    prefix = rel_prefix(tenant)
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not prefix:
        return rel
    return f"{prefix}/{rel}" if rel else prefix


def qdrant_filter(tenant: str | None):
    """Фильтр по тенанту; None — без ограничений (system/admin)."""
    if not tenant:
        return None
    from qdrant_client.http import models

    tenant = normalize(tenant)
    match = models.FieldCondition(
        key="tenant_id", match=models.MatchValue(value=tenant)
    )
    if tenant == DEFAULT:
        return models.Filter(
            should=[
                match,
                models.IsEmptyCondition(is_empty=models.PayloadField(key="tenant_id")),
            ]
        )
    return models.Filter(must=[match])
