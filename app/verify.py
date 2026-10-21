"""Сверка исправлений администратора с документацией при индексации.

Для каждого активного исправления заново извлекается текст указанных страниц,
LLM оценивает, подтверждают ли страницы правленый ответ. Расхождение не
отключает исправление — оно помечается как «требует проверки».
"""

import threading
from datetime import datetime

import appdb
import config
import docpages
import rag
import settings
import tenancy
import vectorstore as vs

VERIFY_PROMPT = (
    "Ты — проверяющий редактор технической документации.\n"
    "Ниже приведены страницы документа и ответ, который вручную исправил "
    "администратор базы знаний.\n\n"
    "Определи, подтверждают ли страницы документа исправленный ответ:\n"
    "1) ответ не противоречит тексту страниц;\n"
    "2) ключевые шаги и факты присутствуют на указанных страницах.\n\n"
    "Ответь строго одной строкой в формате:\n"
    "ВЕРДИКТ: VERIFIED или REVIEW\n"
    "КОММЕНТАРИЙ: кратко, до 200 символов, почему такой вердикт."
)

JUDGE_SYSTEM = (
    "Ты проверяешь соответствие ответа администратора тексту документации. "
    "Если страницы документа подтверждают ответ — вердикт VERIFIED, иначе REVIEW. "
    "Если страниц нет или текст нечитаем — REVIEW. Ничего не выдумывай."
)


def verify_one(correction: dict, tenant: str | None = None) -> dict:
    """Проверить одно исправление. Возвращает статус, комментарий и причину."""
    doc_name = correction.get("doc_name") or ""
    doc_pages = correction.get("doc_pages") or ""
    question = correction.get("question", "")
    answer = correction.get("answer", "")
    scope = tenant or correction.get("tenant_id") or tenancy.DEFAULT

    if not doc_name or not doc_pages:
        return {
            "status": "review",
            "comment": "Не указаны документ и номера страниц",
            "collection": "",
        }

    document = docpages.find_document(doc_name, scope)
    if not document:
        return {
            "status": "review",
            "comment": f"Документ {doc_name} не найден в data/docs",
            "collection": "",
        }

    pages_text = docpages.extract_pages(doc_name, doc_pages, scope)
    if not pages_text.strip():
        return {
            "status": "review",
            "comment": "Не удалось извлечь текст указанных страниц",
            "collection": "",
        }
    if len(pages_text) < 40:
        return {
            "status": "review",
            "comment": "На указанных страницах почти нет текста — проверьте номера страниц",
            "collection": "",
        }

    target = vs.target_name()
    indexed_note = _index_note(question, target, scope)

    client = rag.get_client()
    prompt = (
        f"Страницы документа ({correction.get('doc_ref') or doc_name}):\n\n"
        f"{pages_text[:6000]}\n\n"
        f"{indexed_note}\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Ответ администратора: {answer}\n\n"
        "Верни вердикт."
    )
    completion = client.chat.completions.create(
        model=settings.value("LLM_MODEL", config.LLM_MODEL),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    verdict = (completion.choices[0].message.content or "").strip()
    upper = verdict.upper()
    status = "verified" if ("VERIFIED" in upper and "REVIEW" not in upper) else "review"
    comment = _extract_comment(verdict)
    if not comment:
        rest = [
            line.strip()
            for line in verdict.splitlines()
            if line.strip()
            and "ВЕРДИКТ" not in line.upper()
            and "VERIFIED" not in line.upper()
            and "REVIEW" not in line.upper()
        ]
        comment = " ".join(rest)[:200]
    if not comment:
        comment = (
            "Ответ подтверждён указанными страницами"
            if status == "verified"
            else "Страницы не подтверждают ответ"
        )
    return {"status": status, "comment": comment, "collection": target}


def _extract_comment(text: str) -> str:
    for line in text.splitlines():
        if line.upper().startswith("КОММЕНТАРИЙ"):
            return line.split(":", 1)[1].strip()
    return ""


def _index_note(
    question: str, collection: str, tenant: str | None = None, top_k: int = 3
) -> str:
    client = vs.get_qdrant()
    if not client.collection_exists(collection):
        return "В индексе документации пока нет данных."
    try:
        points = client.query_points(
            collection_name=collection,
            query=vs.embed_query(question),
            query_filter=tenancy.qdrant_filter(tenant),
            limit=top_k,
            with_payload=True,
        ).points
    except Exception:
        return "В индексе документации пока нет данных."
    if not points:
        return "В индексе документации пока нет данных."
    blocks = []
    for point in points:
        payload = point.payload or {}
        blocks.append(
            f"[{payload.get('source', 'unknown')}:{payload.get('chunk', 0)}] "
            f"{(payload.get('text', ''))[:800]}"
        )
    return "Близкие фрагменты из индекса документации:\n" + "\n\n".join(blocks)


class VerifyJob:
    """Фоновая сверка исправлений с прогрессом."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "done": 0,
            "total": 0,
            "verified": 0,
            "review": 0,
            "failed": [],
            "error": None,
            "tenant": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _update(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)

    def start(self, tenant: str | None = None) -> dict:
        with self._lock:
            if self.state["status"] == "running" and self._thread and self._thread.is_alive():
                return {**self.state, "started": False}
            self.state = {
                **self._idle(),
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "tenant": tenant,
            }
        self._thread = threading.Thread(target=self._run, args=(tenant,), daemon=True)
        self._thread.start()
        return {**self.snapshot(), "started": True}

    def _run(self, tenant: str | None = None) -> None:
        try:
            result = verify_all(
                progress=lambda **fields: self._update(**fields), tenant=tenant
            )
            self._update(
                status="done",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                **result,
            )
        except Exception as exc:
            self._update(
                status="error",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=f"{type(exc).__name__}: {exc}",
            )


def verify_all(progress=None, tenant: str | None = None) -> dict:
    corrections = [
        item
        for item in appdb.list_corrections(tenant=tenant)
        if item["status"] == "active"
    ]
    if progress:
        progress(done=0, total=len(corrections))

    verified = 0
    review = 0
    failed: list[dict] = []
    for index, correction in enumerate(corrections, start=1):
        try:
            result = verify_one(correction, tenant)
        except Exception as exc:
            failed.append({"source": f"#{correction['id']}", "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                progress(done=index, total=len(corrections), verified=verified, review=review)
            continue
        appdb.set_check(
            correction["id"],
            result["status"],
            result["comment"],
            result["collection"],
        )
        if result["status"] == "verified":
            verified += 1
        else:
            review += 1
        if progress:
            progress(done=index, total=len(corrections), verified=verified, review=review)

    return {"verified": verified, "review": review, "failed": failed}


job = VerifyJob()