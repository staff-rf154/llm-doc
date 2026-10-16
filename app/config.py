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
EXTRACTED_DIR = os.getenv("EXTRACTED_DIR", "/data/extracted")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/data/uploads")
EVAL_DIR = os.getenv("EVAL_DIR", "/data/eval")
PROMPTS_DB = os.getenv("PROMPTS_DB", "/data/prompts/prompts.db")
PROMPTS_MAX = int(os.getenv("PROMPTS_MAX", "8"))

SOURCE_ROOTS = [DOCS_DIR, EXTRACTED_DIR, UPLOADS_DIR]

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
APP_TITLE = os.getenv("APP_TITLE", "База знаний")
DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SESSION_COOKIE = os.getenv("SESSION_COOKIE", "sid")
CORRECTIONS_COLLECTION = os.getenv("CORRECTIONS_COLLECTION", "corrections")
CORRECTION_MATCH_THRESHOLD = float(os.getenv("CORRECTION_MATCH_THRESHOLD", "0.86"))
CORRECTION_EXAMPLES = int(os.getenv("CORRECTION_EXAMPLES", "2"))
CORRECTION_OVERLAP_WORDS = int(os.getenv("CORRECTION_OVERLAP_WORDS", "2"))
SYNONYMS_DB = os.getenv("SYNONYMS_DB", "/data/synonyms/synonyms.db")
SYNONYMS_CACHE_TTL = int(os.getenv("SYNONYMS_CACHE_TTL", "300"))
SYNONYMS_SEED_ON_START = os.getenv("SYNONYMS_SEED_ON_START", "1") not in ("0", "false", "False")
SYNONYMS_MAX_VARIANTS = int(os.getenv("SYNONYMS_MAX_VARIANTS", "6"))
IMAGES_DIR = os.getenv("IMAGES_DIR", "/data/images")
IMAGES_COLLECTION = os.getenv("IMAGES_COLLECTION", "images")
IMAGE_MATCH_THRESHOLD = float(os.getenv("IMAGE_MATCH_THRESHOLD", "0.75"))
IMAGE_MAX = int(os.getenv("IMAGE_MAX", "2"))
IMAGE_MIN_SIDE = int(os.getenv("IMAGE_MIN_SIDE", "200"))
IMAGE_PER_PAGE = int(os.getenv("IMAGE_PER_PAGE", "2"))
IMAGE_PER_DOC = int(os.getenv("IMAGE_PER_DOC", "20"))
IMAGE_SKIP_FULLPAGE = float(os.getenv("IMAGE_SKIP_FULLPAGE", "0.6"))
IMAGE_SEARCH_ENABLED = os.getenv("IMAGE_SEARCH_ENABLED", "1") not in ("0", "false", "False")
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
PLATFORM_AWARE = os.getenv("PLATFORM_AWARE", "1") not in ("0", "false", "False")
PLATFORM_EXTRA_CHUNKS = int(os.getenv("PLATFORM_EXTRA_CHUNKS", "2"))
PLATFORM_MAX_EXTRA = int(os.getenv("PLATFORM_MAX_EXTRA", "6"))
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "8"))

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

SYSTEM_PROMPT = (
    "Ты — помощник по внутренней документации компании и ассистент по работе с текстом. "
    "Отвечай строго на основе переданных фрагментов документации. "
    "Если в фрагментах нет ответа, прямо скажи: "
    "'В документации нет информации по этому вопросу'. "
    "Никогда не достраивай ответ по памяти и не додумывай шаги, команды, "
    "названия параметров и номера страниц: опирайся только на текст фрагментов. "
    "Если фрагментов недостаточно для полного ответа, скажи, чего в них не хватает, "
    "и предложи уточнить вопрос или уточнить версию продукта. "
    "Не выдумывай факты. Отвечай на языке вопроса. "
    "Указывай источники в формате [doc:chunk]. "
    "Называй в тексте ответа полное название документа, из которого взята "
    "информация (например, «SIEM Help», а не «документ» или «файл»), и "
    "рекомендуй детально ознакомиться с этим документом. "
    "Документация частично разделена по операционным системам: одни материалы "
    "описаны для Windows, другие — для Linux, у каждого фрагмента указана "
    "строка «Платформа». Правила ответа: если в вопросе названа конкретная ОС — "
    "отвечай только по ней, вторую платформу не упоминай; если ОС не указана, а "
    "среди фрагментов есть и Windows, и Linux — веди ответ параллельно, отдельным "
    "блоком «Для Windows» и отдельным блоком «Для Linux» со своими шагами, путями "
    "и командами, и в конце перечисли, чем различаются варианты; если для одной из "
    "платформ в фрагментах ничего нет — прямо скажи, что в документации есть "
    "только вариант для другой ОС, и предложи уточнить ОС. Не выдумывай различий: "
    "бери их только из фрагментов."
    "Если пользователь спрашивает, что ты умеешь, перечисли свои возможности: "
    "работа с текстом и информацией, ответы на вопросы простыми словами, "
    "написание, редактирование, сокращение и перевод текстов, анализ документов, "
    "статей и договоров, помощь с письмами, резюме, отчётами и презентациями; "
    "поиск ответов по базе знаний продуктов со ссылками на источники. "
    "Не заявляй, что выходишь в интернет, выполняешь действия в системах или "
    "работаешь с файлами за пределами базы знаний."
)