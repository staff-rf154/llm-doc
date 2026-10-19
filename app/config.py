import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://ollama:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
DOCS_DIR = os.getenv("DOCS_DIR", "/data/docs")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join(DOCS_DIR, "upload"))
EVAL_DIR = os.getenv("EVAL_DIR", "/data/eval")
LOGS_ROOT = os.getenv("LOGS_ROOT", "/data/logs")
LOGS_MAX_CHARS = int(os.getenv("LOGS_MAX_CHARS", "16000"))
LOGS_MAX_FILES = int(os.getenv("LOGS_MAX_FILES", "200"))
LOGS_MAX_FILE_BYTES = int(os.getenv("LOGS_MAX_FILE_BYTES", str(80 * 1024 * 1024)))
LOGS_MAX_UNPACKED_BYTES = int(os.getenv("LOGS_MAX_UNPACKED_BYTES", str(400 * 1024 * 1024)))
LOGS_CONTEXT_LINES = int(os.getenv("LOGS_CONTEXT_LINES", "2"))
PROMPTS_DB = os.getenv("PROMPTS_DB", "/data/prompts/prompts.db")
PROMPTS_MAX = int(os.getenv("PROMPTS_MAX", "8"))

SOURCE_ROOTS = [DOCS_DIR]

UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "100"))
UPLOAD_URL_TIMEOUT = int(os.getenv("UPLOAD_URL_TIMEOUT", "60"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
EMBED_DIM = 768
PROMPTS_CACHE_TTL = int(os.getenv("PROMPTS_CACHE_TTL", "300"))
INDEX_STATE_DB = os.getenv("INDEX_STATE_DB", "/data/index_state/index.db")
APP_DB = os.getenv("APP_DB", "/data/app/app.db")
SETTINGS_DB = os.getenv("SETTINGS_DB", "/data/settings/settings.db")
APP_TITLE = os.getenv("APP_TITLE", "База знаний")
DOMAIN_PROFILE = os.getenv("DOMAIN_PROFILE", "general")
DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SYSTEM_USERNAME = os.getenv("SYSTEM_USERNAME", "system")
SYSTEM_PASSWORD = os.getenv("SYSTEM_PASSWORD", "")
SESSION_COOKIE = os.getenv("SESSION_COOKIE", "sid")
CORRECTIONS_COLLECTION = os.getenv("CORRECTIONS_COLLECTION", "corrections")
CORRECTION_MATCH_THRESHOLD = float(os.getenv("CORRECTION_MATCH_THRESHOLD", "0.86"))
CORRECTION_EXAMPLES = int(os.getenv("CORRECTION_EXAMPLES", "2"))
CORRECTION_OVERLAP_WORDS = int(os.getenv("CORRECTION_OVERLAP_WORDS", "2"))
SYNONYMS_DB = os.getenv("SYNONYMS_DB", "/data/synonyms/synonyms.db")
SYNONYMS_CACHE_TTL = int(os.getenv("SYNONYMS_CACHE_TTL", "300"))
SYNONYMS_SEED_ON_START = os.getenv("SYNONYMS_SEED_ON_START", "1") not in ("0", "false", "False")
SYNONYMS_MAX_VARIANTS = int(os.getenv("SYNONYMS_MAX_VARIANTS", "6"))
AUTOLEARN_WINDOW = int(os.getenv("AUTOLEARN_WINDOW", "500"))
AUTOLEARN_DAYS = int(os.getenv("AUTOLEARN_DAYS", "30"))
AUTOLEARN_TOPIC_THRESHOLD = float(os.getenv("AUTOLEARN_TOPIC_THRESHOLD", "0.84"))
AUTOLEARN_MIN_TOPIC_HITS = int(os.getenv("AUTOLEARN_MIN_TOPIC_HITS", "3"))
AUTOLEARN_LLM_LIMIT = int(os.getenv("AUTOLEARN_LLM_LIMIT", "20"))
AUTOLEARN_LLM_ENABLED = os.getenv("AUTOLEARN_LLM_ENABLED", "1") not in ("0", "false", "False")
AUTOLEARN_MAX_TOPICS = int(os.getenv("AUTOLEARN_MAX_TOPICS", "300"))
QUESTIONGEN_COUNT = int(os.getenv("QUESTIONGEN_COUNT", "10"))
QUESTIONGEN_SOURCES = os.getenv("QUESTIONGEN_SOURCES", "docs,topics,domain")
QUESTIONGEN_JUDGE = os.getenv("QUESTIONGEN_JUDGE", "1") not in ("0", "false", "False")
BOOTSTRAP_CYCLES = int(os.getenv("BOOTSTRAP_CYCLES", "3"))
BOOTSTRAP_COUNT = int(os.getenv("BOOTSTRAP_COUNT", "30"))
BOOTSTRAP_JUDGE = os.getenv("BOOTSTRAP_JUDGE", "1") not in ("0", "false", "False")
LESSONS_COLLECTION = os.getenv("LESSONS_COLLECTION", "lessons")
TRAINING_FROM_CORRECTIONS = os.getenv("TRAINING_FROM_CORRECTIONS", "1") not in ("0", "false", "False")
LESSONS_MAX = int(os.getenv("LESSONS_MAX", "2"))
LESSONS_MIN_SCORE = float(os.getenv("LESSONS_MIN_SCORE", "0.75"))
WEB_CRAWL_MAX_PAGES = int(os.getenv("WEB_CRAWL_MAX_PAGES", "2000"))
WEB_CRAWL_TIMEOUT = int(os.getenv("WEB_CRAWL_TIMEOUT", "20"))
WEB_CRAWL_DELAY_MS = int(os.getenv("WEB_CRAWL_DELAY_MS", "500"))
WEB_CRAWL_RESPECT_ROBOTS = os.getenv("WEB_CRAWL_RESPECT_ROBOTS", "1") not in ("0", "false", "False")
WEB_CRAWL_USER_AGENT = os.getenv(
    "WEB_CRAWL_USER_AGENT", "Mozilla/5.0 (compatible; DocsBot/1.0)"
)
INGEST_CHECKPOINT_EVERY = int(os.getenv("INGEST_CHECKPOINT_EVERY", "64"))
SSE_HEARTBEAT = int(os.getenv("SSE_HEARTBEAT", "10"))
HYBRID_SPARSE = os.getenv("HYBRID_SPARSE", "1") not in ("0", "false", "False")
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))
HYBRID_FUSION = os.getenv("HYBRID_FUSION", "rrf")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") not in ("0", "false", "False")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "20"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RERANKER_BATCH = int(os.getenv("RERANKER_BATCH", "8"))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
EMBEDDER_DEVICE = os.getenv("EMBEDDER_DEVICE", "auto")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "8"))

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

SYSTEM_PROMPT = (
    "Ты — ассистент по внутренней документации компании. "
    "Отвечай строго на основе переданных фрагментов документации. "
    "Если в фрагментах нет ответа, прямо скажи: "
    "«В документации нет информации по этому вопросу». "
    "Никогда не достраивай ответ по памяти: не выдумывай шаги, команды, "
    "названия параметров, номера страниц и факты — опирайся только на текст фрагментов. "
    "Если фрагментов недостаточно для полного ответа, скажи, чего в них не хватает, "
    "и предложи уточнить вопрос. "
    "Отвечай на языке вопроса. "
    "Указывай источники в формате [doc:chunk], называй полное название документа, "
    "откуда взята информация, и рекомендуй ознакомиться с ним подробнее. "
    "В документации часто повторяется одна и та же информация — объединяй повторы "
    "и давай один цельный ответ без дублирования. "
    "Если пользователь спрашивает, что ты умеешь, перечисли свои возможности "
    "по текущей сфере деятельности: ответы на вопросы по документации простыми "
    "словами, работа с текстом (написание, редактирование, сокращение, перевод), "
    "анализ документов, статей и договоров, помощь с письмами, резюме, отчётами "
    "и презентациями. "
    "Не заявляй, что выходишь в интернет, выполняешь действия в системах или "
    "работаешь с файлами за пределами базы знаний."
)