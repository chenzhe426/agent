import os

from dotenv import load_dotenv

load_dotenv()

# =========================
# Ollama
# =========================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# =========================
# MySQL
# =========================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "knowledge_base")

# =========================
# Qdrant
# =========================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kb_chunks")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "dense")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# =========================
# RAG 默认参数
# =========================
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 600))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 60))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))

# =========================
# Ingestion / Indexing
# =========================
INGEST_AUTO_INDEX = os.getenv("INGEST_AUTO_INDEX", "false").lower() in {"1", "true", "yes", "on"}
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", 0))  # 0 = auto (min(cpu_count, file_count, 8))

# 增量索引：开启后 index_document 会 diff chunk_hash，不全量删除重建
INDEX_INCREMENTAL_ENABLED = os.getenv("INDEX_INCREMENTAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# Embedding 批量参数
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "4096"))

# Finance-aware chunking
DEFAULT_TEXT_CHUNK_SIZE = int(os.getenv("DEFAULT_TEXT_CHUNK_SIZE", 1200))
DEFAULT_TEXT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_TEXT_CHUNK_OVERLAP", 150))
DEFAULT_TABLE_ROW_GROUP_SIZE = int(os.getenv("DEFAULT_TABLE_ROW_GROUP_SIZE", 20))

# =========================
# Hybrid Retrieval / 检索参数
# =========================

# 两路召回候选数
HYBRID_VECTOR_TOP_K = int(os.getenv("HYBRID_VECTOR_TOP_K", 20))
HYBRID_KEYWORD_TOP_K = int(os.getenv("HYBRID_KEYWORD_TOP_K", 20))

# DB lexical 召回取数
HYBRID_LEXICAL_FETCH_K = int(os.getenv("HYBRID_LEXICAL_FETCH_K", 120))
HYBRID_BOOLEAN_FETCH_K = int(os.getenv("HYBRID_BOOLEAN_FETCH_K", 80))
HYBRID_HYDRATE_TOP_K = int(os.getenv("HYBRID_HYDRATE_TOP_K", 400))

# 最终重排权重
RETRIEVAL_WEIGHT_EMBEDDING = float(
    os.getenv("RETRIEVAL_WEIGHT_EMBEDDING", 0.45)
)
RETRIEVAL_WEIGHT_KEYWORD = float(
    os.getenv("RETRIEVAL_WEIGHT_KEYWORD", 0.20)
)
RETRIEVAL_WEIGHT_TITLE = float(
    os.getenv("RETRIEVAL_WEIGHT_TITLE", 0.15)
)
RETRIEVAL_WEIGHT_SECTION = float(
    os.getenv("RETRIEVAL_WEIGHT_SECTION", 0.10)
)
RETRIEVAL_WEIGHT_BM25 = float(
    os.getenv("RETRIEVAL_WEIGHT_BM25", 0.10)
)

# 关键词打分细节
KEYWORD_EXACT_MATCH_WEIGHT = float(
    os.getenv("KEYWORD_EXACT_MATCH_WEIGHT", 1.0)
)
KEYWORD_SUBSTRING_MATCH_WEIGHT = float(
    os.getenv("KEYWORD_SUBSTRING_MATCH_WEIGHT", 0.55)
)
TITLE_MATCH_WEIGHT = float(
    os.getenv("TITLE_MATCH_WEIGHT", 0.8)
)
SECTION_MATCH_WEIGHT = float(
    os.getenv("SECTION_MATCH_WEIGHT", 0.6)
)

# 去重 / 多样性控制
RETRIEVAL_DEDUP_SIM_THRESHOLD = float(
    os.getenv("RETRIEVAL_DEDUP_SIM_THRESHOLD", 0.82)
)
RETRIEVAL_MAX_SAME_SECTION = int(
    os.getenv("RETRIEVAL_MAX_SAME_SECTION", 3)
)

# 邻接 chunk 扩展
RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION = os.getenv(
    "RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION", "true"
).lower() in {"1", "true", "yes", "on"}
RETRIEVAL_NEIGHBOR_WINDOW = int(
    os.getenv("RETRIEVAL_NEIGHBOR_WINDOW", 1)
)

# =========================
# Query 预处理
# =========================
QUERY_MIN_TERM_LEN = int(os.getenv("QUERY_MIN_TERM_LEN", 2))

QUERY_STOPWORDS = {
    "什么",
    "是什么",
    "怎么",
    "如何",
    "为什么",
    "为何",
    "怎样",
    "一下",
    "一个",
    "这个",
    "那个",
    "这些",
    "那些",
    "以及",
    "和",
    "与",
    "或",
    "的",
    "了",
    "吗",
    "呢",
    "啊",
    "呀",
    "吧",
    "在",
    "中",
    "里",
    "上",
    "下",
    "对",
    "把",
    "被",
    "就",
    "都",
    "也",
    "还",
    "是",
    "有",
    "用",
    "讲",
    "说",
    "介绍",
    "区别",
    "优点",
    "缺点",
    "优缺点",
}

# =========================
# V3 Multi-stage Retrieval 配置
# =========================
RETRIEVAL_USE_MULTISTAGE = os.getenv("RETRIEVAL_USE_MULTISTAGE", "false").lower() in {"1", "true", "yes", "on"}
RETRIEVAL_CANDIDATE_DOCS = int(os.getenv("RETRIEVAL_CANDIDATE_DOCS", "3"))
RETRIEVAL_CANDIDATE_SECTIONS = int(os.getenv("RETRIEVAL_CANDIDATE_SECTIONS", "5"))
RETRIEVAL_CHUNKS_PER_SECTION = int(os.getenv("RETRIEVAL_CHUNKS_PER_SECTION", "5"))
RETRIEVAL_SECTION_PAGE_WINDOW = int(os.getenv("RETRIEVAL_SECTION_PAGE_WINDOW", "3"))
RETRIEVAL_SECTION_EMBEDDING_TOP_K = int(os.getenv("RETRIEVAL_SECTION_EMBEDDING_TOP_K", "10"))
RETRIEVAL_HYBRID_W_DENSE = float(os.getenv("RETRIEVAL_HYBRID_W_DENSE", "0.60"))
RETRIEVAL_HYBRID_W_LEXICAL = float(os.getenv("RETRIEVAL_HYBRID_W_LEXICAL", "0.40"))
RETRIEVAL_SECTION_RELEVANCE_BOOST = float(os.getenv("RETRIEVAL_SECTION_RELEVANCE_BOOST", "0.06"))
RETRIEVAL_CONTEXTUAL_HEADER_ENABLED = os.getenv("RETRIEVAL_CONTEXTUAL_HEADER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

# P0 rerank signal weights
RETRIEVAL_NUMERIC_BOOST_WEIGHT = float(os.getenv("RETRIEVAL_NUMERIC_BOOST_WEIGHT", "0.08"))
RETRIEVAL_TABLE_BOOST_WEIGHT = float(os.getenv("RETRIEVAL_TABLE_BOOST_WEIGHT", "0.06"))
RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT = float(os.getenv("RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT", "0.07"))
RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT = float(os.getenv("RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT", "0.04"))
RETRIEVAL_PAGE_CLUSTER_ALPHA = float(os.getenv("RETRIEVAL_PAGE_CLUSTER_ALPHA", "0.15"))

# Section/page clustering
SECTION_PAGE_GAP_THRESHOLD = int(os.getenv("SECTION_PAGE_GAP_THRESHOLD", "3"))
RETRIEVAL_MAX_CHUNKS_PER_PAGE = int(os.getenv("RETRIEVAL_MAX_CHUNKS_PER_PAGE", "2"))

# Evidence semantic scoring
RETRIEVAL_SEMANTIC_THRESHOLD = float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.78"))
RETRIEVAL_EVIDENCE_LEXICAL_THRESHOLD = float(os.getenv("RETRIEVAL_EVIDENCE_LEXICAL_THRESHOLD", "0.42"))

# =============================================================================
# V4 Pipeline Config — Engineering Defaults
#
# Default pipeline: query → multistage retrieval → LLM rerank → grounded answer → verifier
#
# Default behavior:
#   - LLM rerank: ON (answerability-first)
#   - Answer verifier: ON (diagnostic)
#   - Self-refine: OFF (only enabled manually when needed)
# =============================================================================

# --- V4: LLM Reranker ---
V4_ENABLE_LLM_RERANK = os.getenv("V4_ENABLE_LLM_RERANK", "true").lower() in {"1", "true", "yes", "on"}
V4_LLM_RERANK_TOP_N = int(os.getenv("V4_LLM_RERANK_TOP_N", "8"))      # Cost control: default 8
V4_LLM_RERANK_WEIGHT = float(os.getenv("V4_LLM_RERANK_WEIGHT", "0.40"))
V4_LLM_RERANK_MODEL = os.getenv("V4_LLM_RERANK_MODEL", "")
V4_LLM_RERANK_TEMPERATURE = float(os.getenv("V4_LLM_RERANK_TEMPERATURE", "0.0"))

# --- V4: Answer Verifier ---
V4_ENABLE_ANSWER_VERIFIER = os.getenv("V4_ENABLE_ANSWER_VERIFIER", "true").lower() in {"1", "true", "yes", "on"}
V4_VERIFIER_MODEL = os.getenv("V4_VERIFIER_MODEL", "")
V4_VERIFIER_TEMPERATURE = float(os.getenv("V4_VERIFIER_TEMPERATURE", "0.0"))
V4_VERIFIER_THRESHOLD = float(os.getenv("V4_VERIFIER_THRESHOLD", "0.5"))
V4_VERIFIER_LLM_WEIGHT = float(os.getenv("V4_VERIFIER_LLM_WEIGHT", "0.6"))

# --- V4: Self-Refine (default OFF) ---
V4_ENABLE_SELF_REFINE = os.getenv("V4_ENABLE_SELF_REFINE", "false").lower() in {"1", "true", "yes", "on"}
V4_MAX_REFINE_ROUNDS = int(os.getenv("V4_MAX_REFINE_ROUNDS", "1"))
V4_REFINE_MODEL = os.getenv("V4_REFINE_MODEL", "")
V4_REFINE_TEMPERATURE = float(os.getenv("V4_REFINE_TEMPERATURE", "0.2"))
V4_REFINE_TRIGGER_REQUIRES_VERIFIER_FAIL = os.getenv("V4_REFINE_TRIGGER_REQUIRES_VERIFIER_FAIL", "true").lower() in {"1", "true", "yes", "on"}

# --- V4: Answer Generation ---
V4_ANSWER_USE_STRUCTURED_OUTPUT = os.getenv("V4_ANSWER_USE_STRUCTURED_OUTPUT", "true").lower() in {"1", "true", "yes", "on"}
V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES = os.getenv("V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES", "true").lower() in {"1", "true", "yes", "on"}

# =========================
# 其他
# =========================
DATA_DIR = os.getenv("DATA_DIR", "data")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))

# =========================
# Agent 配置
# =========================
AGENT_ENABLE_VERIFIER = os.getenv("AGENT_ENABLE_VERIFIER", "true").lower() in {"1", "true", "yes", "on"}

# =========================
# Agent Memory / Learning
# =========================
VECTOR_MEMORY_ENABLED = os.getenv("VECTOR_MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
VECTOR_MEMORY_TOP_K = int(os.getenv("VECTOR_MEMORY_TOP_K", "3"))

TOOL_CHAIN_MEMORY_ENABLED = os.getenv("TOOL_CHAIN_MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
TOOL_CHAIN_MEMORY_MIN_SUCCESS = int(os.getenv("TOOL_CHAIN_MEMORY_MIN_SUCCESS", "2"))

QDRANT_MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "kb_memory")

# =========================
# Redis 配置 (用于中期记忆)
# =========================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# =========================
# MemoryAgent 配置
# =========================
MEMORY_SHORT_TERM_TURNS = int(os.getenv("MEMORY_SHORT_TERM_TURNS", "3"))  # 短期记忆保留轮数
MEMORY_MID_TERM_TTL = int(os.getenv("MEMORY_MID_TERM_TTL", "1800"))       # 中期记忆TTL(秒)，默认30分钟
MEMORY_MID_TERM_START_TURN = int(os.getenv("MEMORY_MID_TERM_START_TURN", "3"))   # 超过N轮开始中期记忆
MEMORY_LONG_TERM_START_TURN = int(os.getenv("MEMORY_LONG_TERM_START_TURN", "10"))  # 超过N轮开始长期记忆