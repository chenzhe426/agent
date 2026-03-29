"""
Retrieval configuration constants and regex patterns.
"""
from __future__ import annotations

import re
from typing import Any

import app.config as config


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


# --- Top-K ---
DEFAULT_TOP_K = _cfg("DEFAULT_TOP_K", 5)
HYBRID_VECTOR_TOP_K = int(_cfg("HYBRID_VECTOR_TOP_K", 20))
HYBRID_KEYWORD_TOP_K = int(_cfg("HYBRID_KEYWORD_TOP_K", 20))
HYBRID_LEXICAL_FETCH_K = int(_cfg("HYBRID_LEXICAL_FETCH_K", 120))
HYBRID_BOOLEAN_FETCH_K = int(_cfg("HYBRID_BOOLEAN_FETCH_K", 80))
HYBRID_HYDRATE_TOP_K = int(_cfg("HYBRID_HYDRATE_TOP_K", 160))

# --- Score fusion weights ---
RETRIEVAL_WEIGHT_EMBEDDING = float(_cfg("RETRIEVAL_WEIGHT_EMBEDDING", 0.45))
RETRIEVAL_WEIGHT_KEYWORD = float(_cfg("RETRIEVAL_WEIGHT_KEYWORD", 0.20))
RETRIEVAL_WEIGHT_TITLE = float(_cfg("RETRIEVAL_WEIGHT_TITLE", 0.15))
RETRIEVAL_WEIGHT_SECTION = float(_cfg("RETRIEVAL_WEIGHT_SECTION", 0.10))
RETRIEVAL_WEIGHT_BM25 = float(_cfg("RETRIEVAL_WEIGHT_BM25", 0.10))

# --- Ratio/numeric query weights (overrides for ratio queries) ---
RETRIEVAL_RATIO_EMBEDDING_WEIGHT = float(_cfg("RETRIEVAL_RATIO_EMBEDDING_WEIGHT", 0.10))
RETRIEVAL_RATIO_KEYWORD_WEIGHT = float(_cfg("RETRIEVAL_RATIO_KEYWORD_WEIGHT", 0.35))
RETRIEVAL_RATIO_BM25_WEIGHT = float(_cfg("RETRIEVAL_RATIO_BM25_WEIGHT", 0.35))
RETRIEVAL_RATIO_TITLE_WEIGHT = float(_cfg("RETRIEVAL_RATIO_TITLE_WEIGHT", 0.10))
RETRIEVAL_RATIO_SECTION_WEIGHT = float(_cfg("RETRIEVAL_RATIO_SECTION_WEIGHT", 0.10))

TITLE_MATCH_WEIGHT = float(_cfg("TITLE_MATCH_WEIGHT", 0.8))
SECTION_MATCH_WEIGHT = float(_cfg("SECTION_MATCH_WEIGHT", 0.6))
KEYWORD_EXACT_MATCH_WEIGHT = float(_cfg("KEYWORD_EXACT_MATCH_WEIGHT", 1.0))
KEYWORD_SUBSTRING_MATCH_WEIGHT = float(_cfg("KEYWORD_SUBSTRING_MATCH_WEIGHT", 0.55))

# --- Deduplication ---
RETRIEVAL_DEDUP_SIM_THRESHOLD = float(_cfg("RETRIEVAL_DEDUP_SIM_THRESHOLD", 0.82))
RETRIEVAL_MAX_SAME_SECTION = int(_cfg("RETRIEVAL_MAX_SAME_SECTION", 2))
RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION = bool(_cfg("RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION", True))
RETRIEVAL_NEIGHBOR_WINDOW = int(_cfg("RETRIEVAL_NEIGHBOR_WINDOW", 1))

# --- P0 Rerank signal weights ---
RETRIEVAL_NUMERIC_BOOST_WEIGHT = float(_cfg("RETRIEVAL_NUMERIC_BOOST_WEIGHT", 0.08))
RETRIEVAL_TABLE_BOOST_WEIGHT = float(_cfg("RETRIEVAL_TABLE_BOOST_WEIGHT", 0.06))
RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT = float(_cfg("RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT", 0.07))
RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT = float(_cfg("RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT", 0.04))
RETRIEVAL_PAGE_CLUSTER_ALPHA = float(_cfg("RETRIEVAL_PAGE_CLUSTER_ALPHA", 0.15))

# --- V3 Multi-stage ---
RETRIEVAL_USE_MULTISTAGE = bool(_cfg("RETRIEVAL_USE_MULTISTAGE", False))
RETRIEVAL_CANDIDATE_DOCS = int(_cfg("RETRIEVAL_CANDIDATE_DOCS", 3))
RETRIEVAL_CANDIDATE_SECTIONS = int(_cfg("RETRIEVAL_CANDIDATE_SECTIONS", 5))
RETRIEVAL_CHUNKS_PER_SECTION = int(_cfg("RETRIEVAL_CHUNKS_PER_SECTION", 5))
RETRIEVAL_SECTION_PAGE_WINDOW = int(_cfg("RETRIEVAL_SECTION_PAGE_WINDOW", 3))
RETRIEVAL_SECTION_EMBEDDING_TOP_K = int(_cfg("RETRIEVAL_SECTION_EMBEDDING_TOP_K", 10))
RETRIEVAL_HYBRID_W_DENSE = float(_cfg("RETRIEVAL_HYBRID_W_DENSE", 0.60))
RETRIEVAL_HYBRID_W_LEXICAL = float(_cfg("RETRIEVAL_HYBRID_W_LEXICAL", 0.40))
RETRIEVAL_SECTION_RELEVANCE_BOOST = float(_cfg("RETRIEVAL_SECTION_RELEVANCE_BOOST", 0.06))
RETRIEVAL_CONTEXTUAL_HEADER_ENABLED = bool(_cfg("RETRIEVAL_CONTEXTUAL_HEADER_ENABLED", True))

# --- Query processing ---
QUERY_MIN_TERM_LEN = int(_cfg("QUERY_MIN_TERM_LEN", 2))
QUERY_STOPWORDS = set(_cfg("QUERY_STOPWORDS", []))

# --- V4 LLM Reranker ---
V4_ENABLE_LLM_RERANK = bool(_cfg("V4_ENABLE_LLM_RERANK", True))

# --- Regex patterns ---
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./:-]*|[0-9]+(?:\.[0-9]+)?")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")
LEXICAL_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff]+")

# --- Query intent constants and patterns ---
QUERY_INTENT_NUMERIC = "numeric_fact"
QUERY_INTENT_LIST = "list_fact"
QUERY_INTENT_DESCRIPTIVE = "descriptive"
QUERY_INTENT_HYBRID = "hybrid"

_NUMERIC_FACT_PATTER = re.compile(
    r"\b(ratio|tax rate|effective tax|inventory|cash|revenue|profit|margin|"
    r"earnings|eps|ebitda|debt|liabilities|assets|current|liquidity|"
    r"dividend|percentage|%|percent|million|billion|amount|how much|how many|"
    r"total|cost|price|value|worth|balance|decline|increase|growth|driven)\b",
    re.IGNORECASE,
)

_LIST_FACT_PATTER = re.compile(
    r"\b(products|services|segments|subsidiaries|brands|offerings|"
    r"what does .* offer|what are .* products|what .* provide|"
    r"types of|categories|classifications|lines of business|"
    r"business.* overview|operating.* segment|revenue.* segment)\b",
    re.IGNORECASE,
)

_DESCRIPTIVE_PATTER = re.compile(
    r"\b(explain|discuss|describe|why|how did|how does|what is the reason|"
    r"what caused|management.* discussion|overview.* business|"
    r"trend|comparison|year over year|vs |versus)\b",
    re.IGNORECASE,
)

_NUMERIC_FINANCE_PATTERNS = [
    r"\$[\d,]+",
    r"\$\d+\.\d+",
    r"[\d,]+\.?\d*%",
    r"\d+%",
    r"\$\d+ billion",
    r"\$\d+ million",
    r"\d+ million",
    r"\d+ billion",
    r"(margin|gross|operating|net|EBITDA|EPS)[^\.]{0,100}\$[\d,]+",
]

# --- Financial table signal patterns ---
_TABLE_STRUCTURE_KEYWORDS = [
    "total", "assets", "liabilities", "equity", "inventory", "revenue",
    "tax", "rate", "cash", "current", "net income", "operating",
    "diluted", "shares", "eps", "dividend", "debt", "gross", "margin",
    "cost of sales", "operating income", "pretax", "provision",
    "balance sheet", "income statement", "cash flow",
    "current assets", "current liabilities", "long-term",
    "accounts receivable", "accounts payable", "property",
]

_NOISE_SECTION_PATTERNS = [
    r"table of contents", r"toc", r"sec filing", r"form 10-k",
    r"forward-looking", r"risk factor", r"item [0-9]",
    r"signature page", r"exhibit index", r"index of exhibits",
    r"schedule .{0,30}", r"attachment .{0,30}",
]

_CAUSAL_PHRASE_PATTERNS = [
    "driven by", "primarily due to", "attributable to", "resulted in",
    "increased by", "decreased by", "improved", "declined",
]
