"""Управление моделями: LLM (Ollama) и модели поиска (Hugging Face / fastembed).

Позволяет из кабинета вывести список установленных моделей, скачать новую,
удалить, выгрузить из памяти и назначить активную.
"""

import os
import threading
import time
from pathlib import Path

import httpx

import appdb
import config
import settings

HF_KINDS = {
    "embedding": "EMBEDDING_MODEL",
    "reranker": "RERANKER_MODEL",
    "sparse": "SPARSE_MODEL",
}


def ollama_root() -> str:
    """Корень Ollama API: из LLM_BASE_URL убираем завершающий /v1."""
    base = (settings.value("LLM_BASE_URL", config.LLM_BASE_URL) or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def ollama_list() -> dict:
    root = ollama_root()
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(root + "/api/tags")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        return {"available": False, "root": root, "models": [], "error": str(exc)}
    models = []
    for item in data.get("models", []):
        details = item.get("details") or {}
        models.append(
            {
                "name": item.get("name", ""),
                "size": item.get("size", 0),
                "modified_at": item.get("modified_at", ""),
                "family": details.get("family", ""),
                "params": details.get("parameter_size", ""),
                "quant": details.get("quantization_level", ""),
            }
        )
    models.sort(key=lambda item: item["name"])
    return {"available": True, "root": root, "models": models}


def ollama_running() -> list[dict]:
    root = ollama_root()
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(root + "/api/ps")
            response.raise_for_status()
            data = response.json()
    except Exception:
        return []
    result = []
    for item in data.get("models", []):
        result.append(
            {
                "name": item.get("name", ""),
                "size_vram": item.get("size_vram", 0),
                "expires_at": item.get("expires_at", ""),
            }
        )
    return result


def ollama_delete(name: str) -> dict:
    root = ollama_root()
    with httpx.Client(timeout=60) as client:
        response = client.request(
            "DELETE", root + "/api/delete", json={"name": name}
        )
        response.raise_for_status()
    return {"deleted": name}


def ollama_unload(name: str) -> dict:
    root = ollama_root()
    with httpx.Client(timeout=60) as client:
        response = client.post(
            root + "/api/generate",
            json={"model": name, "prompt": "", "keep_alive": 0},
        )
        response.raise_for_status()
    return {"unloaded": name}


def ollama_select(name: str) -> dict:
    appdb.set_setting("LLM_MODEL", name)
    settings.invalidate()
    return {"selected": name}


class PullJob:
    """Фоновая загрузка модели Ollama с прогрессом."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.state: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {
            "status": "idle",
            "model": "",
            "completed": 0,
            "total": 0,
            "percent": 0,
            "error": "",
            "started_at": "",
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def start(self, name: str) -> dict:
        with self._lock:
            if self.state.get("status") == "running":
                return {"started": False, "status": "running", "model": self.state["model"]}
            self.state = {
                "status": "running",
                "model": name,
                "completed": 0,
                "total": 0,
                "percent": 0,
                "error": "",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        self._thread = threading.Thread(target=self._run, args=(name,), daemon=True)
        self._thread.start()
        return {"started": True, "model": name}

    def _run(self, name: str) -> None:
        root = ollama_root()
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    root + "/api/pull",
                    json={"name": name, "stream": True},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            event = httpx.Response(200, content=line).json()
                        except Exception:
                            continue
                        self._apply(event)
            with self._lock:
                self.state["status"] = "done"
                self.state["percent"] = 100
        except Exception as exc:
            with self._lock:
                self.state["status"] = "error"
                self.state["error"] = str(exc)

    def _apply(self, event: dict) -> None:
        with self._lock:
            if event.get("error"):
                self.state["error"] = event["error"]
            total = int(event.get("total") or 0)
            completed = int(event.get("completed") or 0)
            if total:
                self.state["total"] = total
                self.state["completed"] = completed
                self.state["percent"] = int(100 * completed / total)
            if event.get("status"):
                self.state["status_text"] = event["status"]


job = PullJob()


def hf_cache_dir() -> Path:
    home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def hf_downloaded() -> list[str]:
    """Уже скачанные HF-модели из кэша."""
    root = hf_cache_dir()
    if not root.exists():
        return []
    names = []
    for path in root.glob("models--*"):
        name = path.name.replace("models--", "").replace("--", "/")
        names.append(name)
    return sorted(names)


def hf_active() -> dict:
    return {kind: settings.value(key, getattr(config, key, "")) for kind, key in HF_KINDS.items()}


def hf_download(kind: str, name: str) -> dict:
    """Скачать модель в кэш, создав её объект (это и тянет файлы)."""
    if kind not in HF_KINDS:
        raise ValueError("Неизвестный тип модели")
    import vectorstore as vs

    if kind == "embedding":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, device="cpu")
        del model
        vs.get_embedder.cache_clear()
    elif kind == "reranker":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(name, device="cpu")
        del model
        vs.get_reranker.cache_clear()
    else:
        from fastembed.sparse import SparseTextEmbedding

        model = SparseTextEmbedding(model_name=name)
        del model
        vs.get_sparse_model.cache_clear()
    return {"downloaded": name, "kind": kind}


def hf_select(kind: str, name: str) -> dict:
    if kind not in HF_KINDS:
        raise ValueError("Неизвестный тип модели")
    key = HF_KINDS[kind]
    appdb.set_setting(key, name)
    settings.invalidate()
    settings._reload_models()
    return {"selected": name, "kind": kind, "requires_reindex": key == "EMBEDDING_MODEL"}


def hf_delete(kind: str, name: str) -> dict:
    if kind and kind not in HF_KINDS:
        raise ValueError("Неизвестный тип модели")
    actives = {settings.value(key, getattr(config, key, "")) for key in HF_KINDS.values()}
    if name in actives:
        raise ValueError("Нельзя удалить активную модель")
    root = hf_cache_dir()
    folder = "models--" + name.replace("/", "--")
    target = root / folder
    if not target.exists():
        raise ValueError("Модель не найдена в кэше")
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    return {"deleted": name, "kind": kind or "any"}


def summary() -> dict:
    from vectorstore import device_info

    return {
        "llm_model": settings.value("LLM_MODEL", config.LLM_MODEL),
        "ollama": ollama_list(),
        "running": ollama_running(),
        "pull": job.snapshot(),
        "hf": {
            "active": hf_active(),
            "downloaded": hf_downloaded(),
            "cache_dir": str(hf_cache_dir()),
        },
        "device": device_info(),
    }
