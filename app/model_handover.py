"""Перенос обучения при смене моделей генерации и поиска.

Уроки, исправления, синонимы и промты хранятся отдельно от весов модели.
При смене LLM они начинают использоваться сразу. При смене модели эмбеддингов
документы, уроки и исправления пересобираются под новую размерность, а
переключение происходит только после успешной сборки.
"""

import threading
import time
from datetime import datetime

import httpx

import appdb
import config
import corrections
import index_state
import ingest
import lessons
import promptdb
import settings
import synonymdb
import vectorstore as vs

_STATE_LOCK = threading.Lock()
_SWITCH_LOCK = threading.Lock()

_state = {
    "status": "idle",
    "kind": "",
    "model": "",
    "stage": "",
    "done": 0,
    "total": 0,
    "percent": 0,
    "error": "",
    "started_at": "",
    "finished_at": "",
    "result": {},
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update(**fields) -> None:
    with _STATE_LOCK:
        _state.update(fields)


def snapshot() -> dict:
    with _STATE_LOCK:
        return dict(_state)


def _reset(kind: str, model: str) -> None:
    _update(
        status="running",
        kind=kind,
        model=model,
        stage="Подготовка",
        done=0,
        total=0,
        percent=0,
        error="",
        started_at=_now(),
        finished_at="",
        result={},
    )


def _ollama_root() -> str:
    base = (settings.value("LLM_BASE_URL", config.LLM_BASE_URL) or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _warmup_llm(name: str) -> None:
    with httpx.Client(timeout=300) as client:
        response = client.post(
            _ollama_root() + "/api/generate",
            json={
                "model": name,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1},
            },
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise ValueError(str(data["error"]))


def inherited() -> dict:
    lessons_active = [item for item in appdb.list_lessons() if item["status"] == "active"]
    return {
        "lessons": len(lessons_active),
        "corrections": len(appdb.list_corrections(status="active")),
        "synonyms": synonymdb.count(),
        "prompts": promptdb.count_prompts(),
        "generated_questions": appdb.count_generated_questions(),
    }


def _llm_available(name: str) -> bool:
    with httpx.Client(timeout=10) as client:
        response = client.get(_ollama_root() + "/api/tags")
        response.raise_for_status()
        data = response.json()
    return any(item.get("name") == name for item in data.get("models", []))


def select_llm(name: str) -> dict:
    """Проверить и прогреть новую LLM, затем переключить и вернуть наследование."""
    if not _SWITCH_LOCK.acquire(blocking=False):
        raise ValueError("Уже выполняется перенос модели")
    try:
        if not _llm_available(name):
            raise ValueError(f"Модель не найдена на сервере LLM: {name}")
        _warmup_llm(name)
        lessons.reindex()
        corrections.reindex()
        appdb.set_setting("LLM_MODEL", name)
        settings.invalidate()
        return {"selected": name, "inherited": inherited()}
    finally:
        _SWITCH_LOCK.release()


def _sparse_available(name: str) -> bool:
    try:
        vs.get_sparse_model(name)
        return True
    except Exception:
        return False


def select_reranker(name: str) -> dict:
    """Прогреть cross-encoder и назначить его активным."""
    if not _SWITCH_LOCK.acquire(blocking=False):
        raise ValueError("Уже выполняется перенос модели")
    try:
        model = vs.get_reranker(name)
        model.predict([("проверка", "проверка")], show_progress_bar=False)
        appdb.set_setting("RERANKER_MODEL", name)
        settings.invalidate()
        settings._reload_models()
        return {"selected": name, "kind": "reranker", "inherited": inherited()}
    finally:
        _SWITCH_LOCK.release()


def _cleanup_stale_run() -> None:
    run = index_state.get_open_run()
    if not run:
        return
    name = run.get("collection") or ""
    if not name or name == vs.target_name():
        return
    try:
        client = vs.get_qdrant()
        if client.collection_exists(name):
            client.delete_collection(name)
    except Exception:
        pass
    try:
        index_state.forget_run(name)
    except Exception:
        pass


def _drop(name: str | None) -> None:
    if not name:
        return
    try:
        client = vs.get_qdrant()
        if client.collection_exists(name):
            client.delete_collection(name)
    except Exception:
        pass


def _embedding_spec(name: str) -> vs.EmbedSpec:
    device = settings.value("EMBEDDER_DEVICE", config.EMBEDDER_DEVICE)
    model = vs.get_embedder_for(name, device)
    dim = model.get_sentence_embedding_dimension()
    if not dim:
        raise ValueError(f"Не удалось определить размерность модели {name}")
    return vs.EmbedSpec(name, int(dim), device)


def _progress(done: int, total: int, skipped: int, **fields) -> None:
    percent = int(100 * done / total) if total else 0
    _update(done=done, total=total, percent=percent, stage="Документы")


def _run_embedding(name: str) -> None:
    new_docs = None
    old_spec = None
    switched = False
    try:
        _reset("embedding", name)
        if ingest.job.snapshot().get("status") in ("running", "paused"):
            raise ValueError("Сначала завершите текущую индексацию документов")
        _cleanup_stale_run()
        spec = _embedding_spec(name)
        _update(dim=spec.dim, stage="Документы")
        old_spec = vs.EmbedSpec(
            settings.value("EMBEDDING_MODEL", config.EMBEDDING_MODEL),
            vs.embedding_dim(),
            settings.value("EMBEDDER_DEVICE", config.EMBEDDER_DEVICE),
        )

        result = ingest.reindex(progress=_progress, spec=spec, swap=False)
        new_docs = result.get("collection")
        failed = result.get("failed_files") or []
        if failed:
            raise ValueError(
                f"Не удалось проиндексировать файлов: {len(failed)}. "
                f"Первый: {failed[0].get('source', '')}"
            )

        _update(stage="Уроки", percent=0)
        lessons.rebuild(spec=spec)
        _update(stage="Исправления")
        corrections.rebuild(spec=spec)

        _update(stage="Переключение")
        appdb.set_setting("EMBEDDING_MODEL", name)
        appdb.set_setting("EMBED_DIM", str(spec.dim))
        settings.invalidate()
        settings._reload_models()
        vs.swap_target(new_docs)
        switched = True
        clear_error = ""
        try:
            appdb.clear_generated_vectors()
        except Exception as exc:
            clear_error = str(exc)

        result = {"collection": new_docs, "dim": spec.dim, **inherited()}
        if clear_error:
            result["warning"] = clear_error
        _update(
            status="done",
            stage="Готово",
            percent=100,
            finished_at=_now(),
            result=result,
        )
        try:
            import verify

            verify.job.start()
        except Exception:
            pass
        try:
            import bootstrap

            bootstrap.maybe_start("model")
        except Exception:
            pass
    except Exception as exc:
        if not switched and old_spec is not None:
            if new_docs:
                _drop(new_docs)
            appdb.set_setting("EMBEDDING_MODEL", old_spec.model)
            appdb.set_setting("EMBED_DIM", str(old_spec.dim))
            settings.invalidate()
            settings._reload_models()
            try:
                lessons.rebuild(spec=old_spec)
                corrections.rebuild(spec=old_spec)
            except Exception:
                pass
        _update(status="error", stage="Ошибка", error=str(exc), finished_at=_now())
    finally:
        _SWITCH_LOCK.release()


def _run_sparse(name: str) -> None:
    new_docs = None
    switched = False
    old_name = settings.value("SPARSE_MODEL", config.SPARSE_MODEL)
    try:
        _reset("sparse", name)
        if ingest.job.snapshot().get("status") in ("running", "paused"):
            raise ValueError("Сначала завершите текущую индексацию документов")
        if not vs.sparse_enabled():
            appdb.set_setting("SPARSE_MODEL", name)
            settings.invalidate()
            settings._reload_models()
            _update(
                status="done",
                stage="Готово",
                percent=100,
                finished_at=_now(),
                result={"selected": name, **inherited()},
            )
            return
        if not _sparse_available(name):
            raise ValueError(f"Не удалось загрузить sparse-модель: {name}")
        _cleanup_stale_run()
        result = ingest.reindex(progress=_progress, sparse_name=name, swap=False)
        new_docs = result.get("collection")
        failed = result.get("failed_files") or []
        if failed:
            raise ValueError(
                f"Не удалось проиндексировать файлов: {len(failed)}. "
                f"Первый: {failed[0].get('source', '')}"
            )
        _update(stage="Переключение")
        appdb.set_setting("SPARSE_MODEL", name)
        settings.invalidate()
        settings._reload_models()
        vs.swap_target(new_docs)
        switched = True
        _update(
            status="done",
            stage="Готово",
            percent=100,
            finished_at=_now(),
            result={"collection": new_docs, **inherited()},
        )
        try:
            import verify

            verify.job.start()
        except Exception:
            pass
    except Exception as exc:
        if not switched:
            if new_docs:
                _drop(new_docs)
            appdb.set_setting("SPARSE_MODEL", old_name)
            settings.invalidate()
            settings._reload_models()
        _update(status="error", stage="Ошибка", error=str(exc), finished_at=_now())
    finally:
        _SWITCH_LOCK.release()


def start_embedding(name: str) -> dict:
    if not _SWITCH_LOCK.acquire(blocking=False):
        raise ValueError("Уже выполняется перенос модели")
    _reset("embedding", name)
    thread = threading.Thread(target=_run_embedding, args=(name,), daemon=True)
    try:
        thread.start()
    except Exception:
        _SWITCH_LOCK.release()
        raise
    return snapshot()


def start_sparse(name: str) -> dict:
    if not _SWITCH_LOCK.acquire(blocking=False):
        raise ValueError("Уже выполняется перенос модели")
    _reset("sparse", name)
    thread = threading.Thread(target=_run_sparse, args=(name,), daemon=True)
    try:
        thread.start()
    except Exception:
        _SWITCH_LOCK.release()
        raise
    return snapshot()
