"""
QA Specialist Agent - 带质量反馈的 ReAct 循环

ReAct 流程：
1. search → 召回文档
2. evaluate_recall → 评估召回质量
3. 如果召回质量不好：
   a. 检查 rerank 类型是否匹配问题类型
   b. 如果类型对但质量仍不好，迭代 rerank 权重
   c. 重新 rerank
4. generate_answer → 生成答案
5. evaluate_answer → 评估答案质量
6. 如果答案质量不好 → refine → 重新生成

问题类型配置：
- 理解类: 提高语义匹配权重（给分析段落、解释性内容加权重）
- 计算类: 提高表格/数字权重
- 定位类: 提高章节页数权重
- 分类问题: 规则先行，模型兜底
"""

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole
from app.services.reranker_service import rerank_with_llm, QUERY_INTENT_NUMERIC, QUERY_INTENT_DESCRIPTIVE, QUERY_INTENT_LIST, QUERY_INTENT_HYBRID


# =============================================================================
# 问题类型配置
# =============================================================================

"""
四种问题类型：

1. 理解类 (comprehension)
   - 问题特点：为什么、如何、什么原因、解释
   - 权重策略：提高语义匹配权重，给分析段落、解释性内容加权重
   - rerank侧重：段落内容相关性 > 关键词匹配

2. 计算类 (calculation)
   - 问题特点：多少、比率、金额、增长/下降、净利润、毛利率
   - 权重策略：提高表格/数字内容权重
   - rerank侧重：表格 > 数字 > 段落

3. 定位类 (location)
   - 问题特点：哪一页、哪个章节、第几章、在哪里、找到关于X的内容
   - 权重策略：提高章节页数权重
   - rerank侧重：section_path、page_start 匹配

4. 分类问题 (classification)
   - 问题特点：属于哪类、是不是、有没有、是否
   - 权重策略：规则先行，模型兜底
   - rerank侧重：标题匹配 + 语义分类
"""

RERANK_TYPE_CONFIG = {
    "comprehension": {
        "intent": QUERY_INTENT_DESCRIPTIVE,
        "default_weight": 0.4,  # 语义权重稍高
        "boost_keywords": [
            "为什么", "如何", "怎么样", "什么原因", "解释",
            "分析", "说明", "理解", "看法", "观点",
            "why", "how", "what is", "原因", "分析",
        ],
        "penalize_keywords": [
            "目录", "封面", "声明", "附录", "前言",
            "toc", "cover", "disclaimer", "appendix",
        ],
        "boost_chunk_types": ["paragraph", "section", "analysis"],
        "penalize_chunk_types": ["toc", "boilerplate", "cover", "table_of_contents"],
        "weight_iteration_step": 0.1,
        "max_iterations": 2,
        "min_answerable_ratio": 0.2,
        "score_factors": {
            "semantic_weight": 0.6,  # 语义匹配权重高
            "keyword_weight": 0.2,
            "position_weight": 0.2,
        },
    },
    "calculation": {
        "intent": QUERY_INTENT_NUMERIC,
        "default_weight": 0.5,  # 表格/数字权重高
        "boost_keywords": [
            "多少", "金额", "收入", "利润", "成本", "费用",
            "比率", "比例", "率", "增长", "下降", "增加", "减少",
            "营收", "净利润", "毛利率", "负债率",
            "how much", "amount", "revenue", "profit", "ratio", "rate",
        ],
        "penalize_keywords": [
            "背景", "风险", "概述", "前言", "目录", "历史",
            "background", "risk", "intro", "history",
        ],
        "boost_chunk_types": ["table", "table_cell", "numeric", "financial_statement"],
        "penalize_chunk_types": ["narrative", "boilerplate", "toc", "cover"],
        "weight_iteration_step": 0.15,
        "max_iterations": 3,
        "min_answerable_ratio": 0.3,
        "score_factors": {
            "numeric_weight": 0.5,   # 数字权重高
            "table_weight": 0.3,
            "semantic_weight": 0.2,
        },
    },
    "location": {
        "intent": QUERY_INTENT_DESCRIPTIVE,
        "default_weight": 0.3,
        "boost_keywords": [
            "哪一页", "哪个章节", "第几章", "在哪里",
            "找到", "定位", "位置", "页码", "章节",
            "which page", "which section", "find", "location",
        ],
        "penalize_keywords": [
            "多少", "为什么", "如何",
        ],
        "boost_chunk_types": ["section", "heading", "toc_entry"],
        "penalize_chunk_types": ["narrative", "table"],
        "weight_iteration_step": 0.1,
        "max_iterations": 2,
        "min_answerable_ratio": 0.15,
        "score_factors": {
            "position_weight": 0.5,  # 章节页数权重高
            "keyword_weight": 0.3,
            "semantic_weight": 0.2,
        },
    },
    "classification": {
        "intent": QUERY_INTENT_HYBRID,
        "default_weight": 0.35,
        "boost_keywords": [
            "是不是", "有没有", "是否", "属于", "算不算",
            "是", "为", "分类", "类别",
            "is it", "are there", "does it", "belongs to", "category",
        ],
        "penalize_keywords": [
            "多少", "如何", "为什么",
        ],
        "boost_chunk_types": ["heading", "section", "list"],
        "penalize_chunk_types": ["narrative", "table"],
        "weight_iteration_step": 0.1,
        "max_iterations": 2,
        "min_answerable_ratio": 0.25,
        "score_factors": {
            "title_weight": 0.4,     # 标题匹配权重高
            "semantic_weight": 0.4,
            "keyword_weight": 0.2,
        },
        "use_mixed_llm": True,  # 规则不够时用 LLM 分类
    },
    "hybrid": {
        "intent": QUERY_INTENT_HYBRID,
        "default_weight": 0.3,
        "boost_keywords": [],
        "penalize_keywords": ["目录", "声明"],
        "boost_chunk_types": [],
        "penalize_chunk_types": ["toc", "boilerplate"],
        "weight_iteration_step": 0.1,
        "max_iterations": 2,
        "min_answerable_ratio": 0.2,
        "score_factors": {
            "semantic_weight": 0.4,
            "keyword_weight": 0.3,
            "position_weight": 0.3,
        },
    },
}


# =============================================================================
# QA Agent 内部 State
# =============================================================================

class QAAgentState(TypedDict):
    """QA Agent 内部状态"""
    question: str
    rewritten_question: str
    session_id: str
    messages: list
    retrieved_chunks: list
    reranked_chunks: list
    current_answer: str
    query_type: str  # "calculation" / "comprehension" / "location" / "classification" / "hybrid"
    rerank_weight: float
    rerank_iterations: int
    recall_iterations: int  # 召回迭代次数（区别于 rerank）
    recall_quality: str  # "good" / "bad" / "unknown"
    answer_quality: str
    reasoning_trace: list
    step: int
    max_steps: int
    # 覆盖度相关
    coverage_diagnosis: dict
    expanded_queries: list
    missing_entities: list


# =============================================================================
# 工具定义
# =============================================================================

QA_TOOLS = [
    "kb_answer_question",
    "kb_assemble_context",
    "kb_generate_answer",
    "kb_get_chat_history",
    "kb_create_chat_session",
]


QA_SYSTEM_PROMPT = """你是一个专业的问答Agent，专注于知识库问答任务。

你的职责：
1. 使用RAG工具检索相关知识
2. 生成准确、有依据的答案
3. 对答案进行验证确保质量

可用工具：
- search: 搜索知识库获取相关文档
- rerank: 对已召回的文档进行重排（需要先search）
- assemble_context: 组装检索结果为上下文
- generate_answer: 基于上下文生成答案
- evaluate_recall: 评估召回质量
- evaluate_answer: 评估答案质量
- refine_answer: 修正低质量答案
- final: 返回最终答案

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名", "action_input": {"参数"}, "observation": ""}
"""


# =============================================================================
# 质量评估函数
# =============================================================================

# LLM 调参指导的系统提示
LLM_RERANK_GUIDANCE_PROMPT = """你是一个 RAG 系统专家，负责分析召回结果并指导 Rerank 参数调整。

问题类型：
- calculation: 计算类（需要表格/数字）
- comprehension: 理解类（需要分析段落）
- location: 定位类（需要章节位置）
- classification: 分类类（需要标题/分类）

你需要分析召回的 chunks，判断：
1. 当前召回是否回答了问题
2. 缺失什么类型的内容
3. 如何调整 rerank 参数

输出 JSON 格式：
{
    "quality": "good/bad/medium",
    "reason": "详细分析",
    "missing_content": ["缺失的内容类型"],
    "parameter_adjustments": {
        "suggested_weight": 0.0-1.0,
        "boost_types": ["建议 boost 的 chunk 类型"],
        "penalize_types": ["建议 penalize 的 chunk 类型"],
        "boost_keywords": ["建议 boost 的关键词"]
    },
    "suggested_action": "rerank/expand_query/generate_answer",
    "confidence": 0.0-1.0
}
"""


def evaluate_recall_quality(chunks: list, query_type: str) -> dict[str, Any]:
    """
    评估召回质量（规则 + LLM 智能分析）

    Returns:
        {
            "quality": "good" / "bad" / "medium",
            "directly_answerable_count": int,
            "total_count": int,
            "avg_score": float,
            "top_score": float,
            "answerable_ratio": float,
            "reason": str,
            "suggested_action": str,
            "parameter_adjustments": dict,  # LLM 给出的调参建议
        }
    """
    if not chunks:
        return {
            "quality": "bad",
            "directly_answerable_count": 0,
            "total_count": 0,
            "avg_score": 0.0,
            "top_score": 0.0,
            "answerable_ratio": 0.0,
            "reason": "无召回结果",
            "suggested_action": "expand_query",
            "parameter_adjustments": {},
        }

    total = len(chunks)
    directly_answerable = 0
    partially_useful = 0
    topically_related = 0
    noise = 0

    for chunk in chunks:
        answerability = chunk.get("llm_answerability", "topically_related")
        if answerability == "directly_answerable":
            directly_answerable += 1
        elif answerability == "partially_useful":
            partially_useful += 1
        elif answerability == "topically_related":
            topically_related += 1
        else:
            noise += 1

    config = RERANK_TYPE_CONFIG.get(query_type, RERANK_TYPE_CONFIG["hybrid"])
    min_ratio = config["min_answerable_ratio"]

    # 计算指标
    scores = [c.get("final_score", 0) for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0
    top_score = max(scores) if scores else 0
    answerable_ratio = directly_answerable / total if total > 0 else 0

    result = {
        "quality": "unknown",
        "directly_answerable_count": directly_answerable,
        "total_count": total,
        "avg_score": avg_score,
        "top_score": top_score,
        "answerable_ratio": answerable_ratio,
        "reason": "",
        "suggested_action": "generate_answer",
        "parameter_adjustments": {},
    }

    # ========== 规则快速判定 ==========

    # 好：直接可答比例达标
    if answerable_ratio >= min_ratio and top_score >= 0.4:
        result["quality"] = "good"
        result["reason"] = f"直接可答比例 {answerable_ratio:.1%} >= {min_ratio:.1%}，top_score={top_score:.3f}"
        result["suggested_action"] = "generate_answer"
        return result

    # 差：完全没有直接可答
    if directly_answerable == 0 or (avg_score < 0.2 and top_score < 0.3):
        result["quality"] = "bad"
        result["reason"] = f"无直接可答内容，avg_score={avg_score:.3f}，top_score={top_score:.3f}"
        result["suggested_action"] = "expand_query"
        result["parameter_adjustments"] = _get_default_adjustments(query_type)
        return result

    # 中等：有部分有用内容，rerank 可改善
    if answerable_ratio >= 0.1 or partially_useful >= 2 or avg_score >= 0.25:
        result["quality"] = "medium"
        result["reason"] = f"部分有用(直接:{directly_answerable}, 部分:{partially_useful})，建议 rerank"
        result["suggested_action"] = "rerank"
        result["parameter_adjustments"] = _get_default_adjustments(query_type)
        return result

    # 差：分数过低
    if avg_score < 0.25:
        result["quality"] = "bad"
        result["reason"] = f"分数过低 avg={avg_score:.3f}，建议迭代 rerank"
        result["suggested_action"] = "iterate_rerank"
        result["parameter_adjustments"] = _get_default_adjustments(query_type)
        return result

    # 默认中等
    result["quality"] = "medium"
    result["reason"] = f"需要改善，直接可答:{directly_answerable}/{total}"
    result["suggested_action"] = "rerank"
    result["parameter_adjustments"] = _get_default_adjustments(query_type)
    return result


def _get_default_adjustments(query_type: str) -> dict:
    """获取默认调参建议"""
    config = RERANK_TYPE_CONFIG.get(query_type, RERANK_TYPE_CONFIG["hybrid"])
    return {
        "suggested_weight": config.get("default_weight", 0.3),
        "boost_types": config.get("boost_chunk_types", []),
        "penalize_types": config.get("penalize_chunk_types", []),
        "boost_keywords": config.get("boost_keywords", []),
    }


def llm_guide_rerank_adjustment(
    chunks: list,
    question: str,
    query_type: str,
    current_weight: float,
) -> dict[str, Any]:
    """
    使用 LLM 分析召回结果，指导 rerank 参数调整

    Returns:
        {
            "quality": "good/bad/medium",
            "reason": str,
            "missing_content": list,
            "parameter_adjustments": {
                "suggested_weight": float,
                "boost_types": list,
                "penalize_types": list,
            },
            "suggested_action": str,
            "confidence": float,
        }
    """
    if not chunks:
        return {
            "quality": "bad",
            "reason": "无召回结果",
            "missing_content": ["任何相关内容"],
            "parameter_adjustments": _get_default_adjustments(query_type),
            "suggested_action": "expand_query",
            "confidence": 1.0,
        }

    # 构建召回结果摘要
    chunks_summary = []
    for i, chunk in enumerate(chunks[:10]):  # 只取 top 10
        chunk_type = chunk.get("chunk_type", "unknown")
        title = chunk.get("title", "")[:30]
        score = chunk.get("final_score", 0)
        answerability = chunk.get("llm_answerability", "unknown")
        preview = (chunk.get("chunk_text", "") or "")[:100]

        chunks_summary.append({
            "rank": i + 1,
            "type": chunk_type,
            "title": title,
            "score": f"{score:.3f}",
            "answerability": answerability,
            "preview": preview,
        })

    prompt = f"""问题类型: {query_type}
问题: {question}

召回结果（按 score 排序，取 top 10）:
{json.dumps(chunks_summary, ensure_ascii=False, indent=2)}

请分析以上召回结果，判断是否能够回答问题，并给出 rerank 参数调整建议。

输出 JSON 格式：
{{
    "quality": "good/bad/medium",
    "reason": "详细分析当前召回是否能回答问题",
    "missing_content": ["缺失的内容类型，如 '表格'、'数字'、'分析段落' 等"],
    "parameter_adjustments": {{
        "suggested_weight": 0.0-1.0,  // 建议的 LLM rerank 权重
        "boost_types": ["建议 boost 的 chunk 类型，如 'table'、'numeric'"],
        "penalize_types": ["建议 penalize 的 chunk 类型，如 'narrative'"]
    }},
    "suggested_action": "rerank/expand_query/generate_answer",
    "confidence": 0.0-1.0
}}
"""

    try:
        from app.agent.llm import get_chat_llm
        llm = get_chat_llm()

        response = llm.invoke([
            SystemMessage(content=LLM_RERANK_GUIDANCE_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw_content = response.content if hasattr(response, "content") else str(response)
        result = json.loads(raw_content)
        return result
    except Exception as e:
        # LLM 调用失败，使用默认调整
        return {
            "quality": "medium",
            "reason": f"LLM 分析失败: {e}，使用默认调整",
            "missing_content": [],
            "parameter_adjustments": _get_default_adjustments(query_type),
            "suggested_action": "rerank",
            "confidence": 0.3,
        }


def apply_rerank_adjustment(
    state: dict,
    adjustments: dict,
    config: dict,
) -> dict:
    """
    应用 LLM 给出的调参建议，更新 rerank 参数

    Returns:
        更新后的 state
    """
    suggested_weight = adjustments.get("suggested_weight", state.get("rerank_weight", 0.3))

    # 限制权重范围
    suggested_weight = max(0.2, min(0.8, suggested_weight))

    # 检查是否需要显著调整
    current_weight = state.get("rerank_weight", 0.3)
    weight_diff = abs(suggested_weight - current_weight)

    # 如果调整幅度超过 0.1，认为是显著调整
    if weight_diff > 0.1:
        state["rerank_weight"] = suggested_weight

    # 更新 boost/penalize 类型
    if adjustments.get("boost_types"):
        state["_boost_types"] = adjustments["boost_types"]
    if adjustments.get("penalize_types"):
        state["_penalize_types"] = adjustments["penalize_types"]

    return state


def should_iterate_rerank(
    current_weight: float,
    iterations: int,
    config: dict,
    eval_result: dict,
    coverage_diagnosis: dict = None,
    llm_guidance: dict = None,
) -> tuple[bool, float, str, str, dict]:
    """
    判断是否需要继续迭代 rerank 权重或扩展查询

    Args:
        llm_guidance: LLM 给出的调参建议

    Returns:
        (是否继续迭代, 下一个权重, 原因, 建议动作, 更新后的调参建议)
        建议动作: "iterate_rerank" / "expand_query" / "multi_signal_recall" / "generate_answer"
    """
    max_iterations = config.get("max_iterations", 3)
    step = config.get("weight_iteration_step", 0.15)

    # 获取调参建议
    param_adjustments = {}
    if llm_guidance and llm_guidance.get("parameter_adjustments"):
        param_adjustments = llm_guidance["parameter_adjustments"]

    # 已达最大迭代次数
    if iterations >= max_iterations:
        # 检查是否需要扩展查询
        if coverage_diagnosis and coverage_diagnosis.get("suggested_actions"):
            return False, current_weight, f"已达最大迭代次数 {max_iterations}", coverage_diagnosis["suggested_actions"][0], param_adjustments
        return False, current_weight, f"已达最大迭代次数 {max_iterations}", "generate_answer", param_adjustments

    # 召回质量已经是 good，不需要继续
    if eval_result["quality"] == "good":
        return False, current_weight, "召回质量已达标", "generate_answer", param_adjustments

    # ========== LLM 调参建议优先 ==========
    if llm_guidance:
        suggested_action = llm_guidance.get("suggested_action", "")
        suggested_weight = param_adjustments.get("suggested_weight", current_weight)

        # LLM 明确建议扩展查询
        if suggested_action == "expand_query":
            return False, current_weight, f"LLM 建议：{llm_guidance.get('reason', '扩展查询')}", "expand_query", param_adjustments

        # LLM 建议 rerank 且有明确的权重建议
        if suggested_action == "rerank" and suggested_weight != current_weight:
            # 直接应用 LLM 建议的权重
            next_weight = max(0.2, min(0.8, suggested_weight))
            if abs(next_weight - current_weight) > 0.05:  # 显著调整
                return True, next_weight, f"LLM 建议权重调整: {current_weight:.2f} → {next_weight:.2f}", "iterate_rerank", param_adjustments

    # ========== 覆盖度检查 ==========
    if coverage_diagnosis:
        coverage_score = coverage_diagnosis.get("coverage_score", 1.0)
        missing_entities = coverage_diagnosis.get("missing_entities", [])
        suggested_actions = coverage_diagnosis.get("suggested_actions", [])

        # 覆盖度严重不足，优先扩展查询
        if coverage_score < 0.3 and missing_entities:
            return False, current_weight, f"覆盖度不足 {coverage_score:.1%}，缺失实体: {missing_entities[:2]}", "expand_query", param_adjustments

        # 多信号召回建议
        if "multi_signal_recall" in suggested_actions and iterations >= 1:
            return False, current_weight, "多种信号召回", "multi_signal_recall", param_adjustments

    # ========== 默认迭代策略 ==========

    # 计算下一个权重
    next_weight = min(current_weight + step, 0.8)

    # 质量仍为 bad，增加权重
    if eval_result["quality"] == "bad":
        # 如果权重已经较高，尝试扩展查询
        if current_weight >= 0.6 and coverage_diagnosis and coverage_diagnosis.get("missing_entities"):
            return False, current_weight, f"权重已较高({current_weight:.2f})，尝试扩展查询", "expand_query", param_adjustments
        return True, next_weight, f"质量仍为 bad，尝试提高权重 {current_weight:.2f} → {next_weight:.2f}", "iterate_rerank", param_adjustments

    # 质量为 medium，可以再试一次
    if eval_result["quality"] == "medium":
        if current_weight < 0.5:
            return True, next_weight, f"质量为 medium，提升权重尝试 {current_weight:.2f} → {next_weight:.2f}", "iterate_rerank", param_adjustments
        else:
            return False, current_weight, "权重已达较高水平，质量改善有限", "generate_answer", param_adjustments

    return False, current_weight, "无需继续迭代", "generate_answer", param_adjustments


def evaluate_answer_quality(answer: str, chunks: list) -> dict[str, Any]:
    """
    评估答案质量

    Returns:
        {
            "quality": "good" / "bad" / "unknown",
            "has_evidence": bool,
            "has_substance": bool,
            "reason": str,
        }
    """
    if not answer or len(answer) < 10:
        return {
            "quality": "bad",
            "has_evidence": False,
            "has_substance": False,
            "reason": "答案过短或为空",
        }

    # 检查答案是否有实质性内容
    substance_keywords = ["是", "为", "有", "为", "在", "于", "等", "包括", "为"]
    substance_count = sum(1 for kw in substance_keywords if kw in answer)
    has_substance = substance_count >= 2 and len(answer) >= 20

    # 检查答案是否引用了召回内容（简单检查）
    has_evidence = False
    if chunks:
        # 检查答案中是否提到了文档标题或关键信息
        for chunk in chunks[:3]:  # 只检查 top 3
            title = chunk.get("title", "")
            if title and len(title) > 3 and title in answer:
                has_evidence = True
                break

        # 或者检查是否有数字/比率等具体信息（针对数字类问题）
        import re
        numbers = re.findall(r'\d+\.?\d*%?', answer)
        if len(numbers) >= 1:
            has_evidence = True

    if not has_substance:
        return {
            "quality": "bad",
            "has_evidence": has_evidence,
            "has_substance": False,
            "reason": "答案缺乏实质性内容",
        }

    if not has_evidence:
        return {
            "quality": "medium",
            "has_evidence": False,
            "has_substance": True,
            "reason": "答案有内容但可能缺乏证据支持",
        }

    return {
        "quality": "good",
        "has_evidence": has_evidence,
        "has_substance": True,
        "reason": "答案质量良好",
    }


def detect_query_intent(question: str) -> str:
    """
    检测问题类型（规则先行，模型兜底）

    Returns:
        "calculation" / "comprehension" / "location" / "classification" / "hybrid"
    """
    question_lower = question.lower()
    question_stripped = question_lower.strip()

    # ========== 定位类问题（优先匹配）==========
    # 问题特点：哪一页、哪个章节、第几章、在哪里、找到关于X的内容
    location_patterns = [
        r"哪一?[页章段落节]", r"第[一二三四五六七八九十\d]+[页章段]", r"在哪些",
        r"哪[里儿]", r"位置", r"页码", r"章节",
        r"which\s*(page|section|chapter)", r"where", r"find.*(location|page|section)",
    ]
    for pattern in location_patterns:
        import re
        if re.search(pattern, question_stripped):
            return "location"

    # ========== 计算类问题 ==========
    # 问题特点：多少、比率、金额、增长/下降
    calculation_keywords = [
        "多少", "几", "金额", "收入", "利润", "成本", "费用", "负债",
        "比率", "比例", "率", "增长", "下降", "增加", "减少",
        "营收", "净利润", "毛利率", "负债率", "收益率", "增长额",
        "how much", "how many", "amount", "revenue", "profit", "cost", "ratio", "rate",
        "growth", "increase", "decrease", "percent", "%",
    ]
    calc_count = sum(1 for kw in calculation_keywords if kw in question_lower)
    if calc_count >= 1:
        return "calculation"

    # ========== 分类问题 ==========
    # 问题特点：是不是、有没有、是否、属于哪类
    classification_keywords = [
        "是不是", "有没有", "是否", "算不算", "属于",
        "是.*还是", "是.*也是", "属于.*还是",
        "is it", "are there", "does it", "belongs to", "is.*or",
    ]
    for kw in classification_keywords:
        if kw in question_lower:
            return "classification"

    # ========== 理解类问题 ==========
    # 问题特点：为什么、如何、什么原因、解释、分析
    comprehension_keywords = [
        "为什么", "为何", "如何", "怎么样", "怎么", "什么原因",
        "解释", "说明", "分析", "理解", "看法", "观点", "原因",
        "why", "how", "what is", "what caused", "explain", "describe",
        "reason", "cause", "analysis",
    ]
    comp_count = sum(1 for kw in comprehension_keywords if kw in question_lower)
    if comp_count >= 1:
        return "comprehension"

    # ========== 兜底：混合/通用 ==========
    return "hybrid"


# =============================================================================
# 召回覆盖度诊断与多信号召回
# =============================================================================

def diagnose_recall_coverage(
    chunks: list,
    question: str,
    query_type: str,
) -> dict[str, Any]:
    """
    诊断召回覆盖度

    检查：
    1. 关键实体是否被召回
    2. 所需 chunk 类型是否覆盖
    3. 各召回信号贡献度

    Returns:
        {
            "coverage_score": float,        # 0-1 覆盖度评分
            "missing_entities": list,       # 缺失的关键实体
            "covered_entities": list,       # 已覆盖的关键实体
            "missing_chunk_types": list,   # 缺失的 chunk 类型
            "signal_contribution": dict,    # 各信号贡献度
            "suggested_actions": list,     # 建议动作
        }
    """
    import re

    if not chunks:
        return {
            "coverage_score": 0.0,
            "missing_entities": _extract_key_entities(question),
            "covered_entities": [],
            "missing_chunk_types": ["paragraph", "table", "section"],
            "signal_contribution": {},
            "suggested_actions": ["expand_query", "multi_signal_recall"],
        }

    # 提取问题中的关键实体
    key_entities = _extract_key_entities(question)
    covered_entities = []
    missing_entities = []

    # 检查实体覆盖
    for entity in key_entities:
        found = False
        for chunk in chunks:
            chunk_text = (chunk.get("chunk_text", "") or chunk.get("search_text", "")).lower()
            if entity.lower() in chunk_text:
                found = True
                break
        if found:
            covered_entities.append(entity)
        else:
            missing_entities.append(entity)

    # 检查 chunk 类型覆盖
    chunk_types_in_results = set(c.get("chunk_type", "unknown") for c in chunks)
    all_types = ["paragraph", "table", "table_cell", "section", "heading", "list"]
    missing_chunk_types = [t for t in all_types if t not in chunk_types_in_results]

    # 计算信号贡献度
    signal_contribution = {
        "vector_score": 0.0,
        "keyword_score": 0.0,
        "bm25_score": 0.0,
        "rerank_score": 0.0,
    }

    for chunk in chunks:
        signal_contribution["vector_score"] += chunk.get("embedding_score", 0) or 0
        signal_contribution["keyword_score"] += chunk.get("keyword_score", 0) or 0
        signal_contribution["bm25_score"] += chunk.get("bm25_score", 0) or 0
        signal_contribution["rerank_score"] += chunk.get("llm_combined_score", 0) or 0

    # 归一化
    n = len(chunks) if chunks else 1
    for k in signal_contribution:
        signal_contribution[k] /= n

    # 计算覆盖度评分
    entity_coverage = len(covered_entities) / len(key_entities) if key_entities else 0.5
    type_coverage = 1 - len(missing_chunk_types) / len(all_types)
    coverage_score = (entity_coverage * 0.6 + type_coverage * 0.4)

    # 生成建议动作
    suggested_actions = []
    if missing_entities:
        suggested_actions.append("expand_query")
    if len(missing_chunk_types) > 2:
        suggested_actions.append("multi_signal_recall")
    if signal_contribution["vector_score"] < 0.1:
        suggested_actions.append("boost_keyword_recall")
    if signal_contribution["bm25_score"] < 0.1:
        suggested_actions.append("boost_vector_recall")

    return {
        "coverage_score": coverage_score,
        "missing_entities": missing_entities,
        "covered_entities": covered_entities,
        "missing_chunk_types": missing_chunk_types,
        "signal_contribution": signal_contribution,
        "suggested_actions": suggested_actions,
    }


def _extract_key_entities(question: str) -> list[str]:
    """
    从问题中提取关键实体

    提取：
    - 公司/机构名
    - 数字/比率
    - 专有名词
    """
    import re

    entities = []

    # 提取中文实体（简单规则）
    chinese_patterns = [
        r'([A-Z]{2,}[A-Z0-9]*)',  # 大写英文缩写
        r'([\u4e00-\u9fa5]{2,}公司)',
        r'([\u4e00-\u9fa5]{2,}集团)',
        r'([\u4e00-\u9fa5]{2,}银行)',
        r'([\u4e00-\u9fa5]{2,}证券)',
        r'(第[一二三四五六七八九十\d]+[页章段])',
    ]

    for pattern in chinese_patterns:
        matches = re.findall(pattern, question)
        entities.extend(matches)

    # 提取数字和比率
    numbers = re.findall(r'\d+\.?\d*%?', question)
    for num in numbers:
        if len(num) >= 2:  # 过滤掉个位数
            entities.append(num)

    return list(set(entities))


def expand_query(question: str, missing_entities: list, query_type: str) -> list[str]:
    """
    扩展查询，生成多种查询变体

    Returns:
        扩展后的查询列表
    """
    queries = [question]  # 原始查询

    # 对定位类问题，提取位置信息
    if query_type == "location":
        import re
        location_match = re.search(r'第[一二三四五六七八九十\d]+[页章段]', question)
        if location_match:
            queries.append(f"文档{location_match.group()}")
            queries.append(f"在哪{question}")

    # 对计算类问题，提取实体并组合
    if query_type == "calculation" and missing_entities:
        for entity in missing_entities[:2]:  # 最多取2个实体
            queries.append(f"{entity} 财务数据")
            queries.append(f"{entity} 多少")

    # 对理解类问题，扩展同义词
    if query_type == "comprehension":
        synonyms = {
            "为什么": ["原因", "为何"],
            "如何": ["怎样", "怎么"],
            "原因": ["理由", "为何"],
        }
        for kw, syns in synonyms.items():
            if kw in question:
                for syn in syns:
                    queries.append(question.replace(kw, syn))

    # 对缺失实体，生成专门查询
    for entity in missing_entities[:3]:  # 最多3个
        if len(entity) >= 2:
            queries.append(entity)
            queries.append(f"关于{entity}")

    # 去重
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    return unique_queries[:5]  # 最多5个查询


def multi_signal_recall(
    queries: list[str],
    top_k: int = 20,
    session_id: str = "",
    governance_context: dict = None,
) -> list[dict[str, Any]]:
    """
    多信号召回

    使用多种召回信号：
    1. 原始查询向量召回
    2. 扩展查询向量召回
    3. 关键词/bm25 召回
    4. 实体召回（如有实体）

    Returns:
        合并后的候选 chunks
    """
    governance_context = governance_context or {}

    all_chunks = []
    seen_chunk_ids = set()

    for query in queries:
        # 调用搜索（优先通过 MCP）
        from app.tools.mcp_client import call_tool_mcp_or_local

        result = call_tool_mcp_or_local(
            tool_name="kb_search_knowledge_base",
            args={"query": query, "top_k": top_k},
            agent="qa",
            session_id=session_id,
            governance_context=governance_context,
        )

        chunks = result.get("chunks", [])
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk["_query_source"] = query  # 标记来源
                all_chunks.append(chunk)

    # 按 score 排序
    all_chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    return all_chunks


# =============================================================================
# ReAct Agent
# =============================================================================

class QAAgent:
    """
    QA Specialist Agent - 带质量反馈的 ReAct 循环
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_tool_map(self) -> dict:
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in QA_TOOLS if name in TOOL_MAP}

    def execute(
        self,
        question: str,
        session_id: str = "",
        rewritten_question: str = None,
        max_steps: int = 15,
        governance_context: dict[str, Any] = None,
    ) -> AgentResponse:
        """
        执行 QA 任务（ReAct 循环）

        Args:
            question: 原始问题
            session_id: 会话ID
            rewritten_question: 改写后的问题
            max_steps: 最大步数
            governance_context: 治理上下文

        Returns:
            AgentResponse
        """
        self._governance_context = governance_context or {}
        self._session_id = session_id

        tool_map = self._get_tool_map()
        query = rewritten_question if rewritten_question else question

        # 检测问题类型
        query_type = detect_query_intent(question)
        config = RERANK_TYPE_CONFIG.get(query_type, RERANK_TYPE_CONFIG["hybrid"])

        # 初始化状态
        state: QAAgentState = {
            "question": question,
            "rewritten_question": query,
            "session_id": session_id,
            "messages": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "current_answer": "",
            "query_type": query_type,
            "rerank_weight": config["default_weight"],
            "rerank_iterations": 0,
            "recall_iterations": 0,
            "recall_quality": "unknown",
            "answer_quality": "unknown",
            "reasoning_trace": [],
            "step": 0,
            "max_steps": max_steps,
            "coverage_diagnosis": {},
            "expanded_queries": [],
            "missing_entities": [],
        }

        # ========== ReAct 主循环 ==========
        step = 0
        while step < max_steps:
            step += 1
            state["step"] = step

            # ========== Step 1: Search（首次或 rewrite 后） ==========
            if step == 1 or state.get("_need_rewrite"):
                state["_need_rewrite"] = False

                trace_entry = {
                    "step": step,
                    "thought": f"[{query_type}] 执行搜索召回",
                    "action": "search",
                    "action_input": {"query": query, "top_k": 20},
                    "observation": "",
                }
                state["reasoning_trace"].append(trace_entry)

                result = self._execute_search({"query": query, "top_k": 20}, tool_map, state)
                state["retrieved_chunks"] = result.get("chunks", [])
                state["reranked_chunks"] = []
                state["rerank_iterations"] = 0
                state["rerank_weight"] = config["default_weight"]
                state["recall_quality"] = "unknown"

                trace_entry["observation"] = f"召回 {len(state['retrieved_chunks'])} 个文档"
                continue

            # ========== Step 2: Rerank + 评估（自动迭代闭环） ==========
            chunks_to_eval = state["reranked_chunks"] or state["retrieved_chunks"]

            # 如果还没 rerank，先 rerank
            if not state["reranked_chunks"] or state.get("_need_rerank"):
                state["_need_rerank"] = False

                trace_entry = {
                    "step": step,
                    "thought": f"[{query_type}] Rerank，迭代次数={state['rerank_iterations']}",
                    "action": "rerank",
                    "action_input": {"weight": state["rerank_weight"]},
                    "observation": "",
                }
                state["reasoning_trace"].append(trace_entry)

                reranked = self._execute_rerank(state)
                state["reranked_chunks"] = reranked
                state["rerank_iterations"] += 1

                trace_entry["observation"] = f"Rerank 完成(迭代{state['rerank_iterations']})，权重={state['rerank_weight']:.2f}"
                step += 1
                if step >= max_steps:
                    break

            # ========== Step 3: 评估召回质量 ==========
            eval_result = evaluate_recall_quality(
                state["reranked_chunks"] or state["retrieved_chunks"],
                state["query_type"]
            )
            state["recall_quality"] = eval_result["quality"]

            trace_entry = {
                "step": step,
                "thought": f"[{query_type}] 评估召回质量",
                "action": "evaluate_recall",
                "action_input": {},
                "observation": eval_result["reason"],
            }
            state["reasoning_trace"].append(trace_entry)

            # ========== 质量判定 + 自动迭代决策 ==========
            if eval_result["quality"] == "good":
                # 质量达标，进入答案生成
                state["recall_quality"] = "good"
            else:
                # 质量不达标，先诊断覆盖度
                chunks_to_eval = state["reranked_chunks"] or state["retrieved_chunks"]
                coverage_diag = diagnose_recall_coverage(
                    chunks_to_eval,
                    state["rewritten_question"],
                    state["query_type"],
                )
                state["coverage_diagnosis"] = coverage_diag
                state["missing_entities"] = coverage_diag.get("missing_entities", [])

                # ========== LLM 智能调参指导 ==========
                llm_guidance = {}
                if chunks_to_eval and state["rerank_iterations"] < config.get("max_iterations", 3):
                    # 调用 LLM 分析
                    llm_guidance = llm_guide_rerank_adjustment(
                        chunks_to_eval,
                        state["rewritten_question"],
                        state["query_type"],
                        state["rerank_weight"],
                    )
                    state["_llm_guidance"] = llm_guidance

                    trace_entry = {
                        "step": step,
                        "thought": f"[{query_type}] LLM 分析: {llm_guidance.get('reason', '分析中')[:80]}",
                        "action": "llm_guidance",
                        "action_input": {},
                        "observation": f"建议: {llm_guidance.get('suggested_action', 'unknown')}",
                    }
                    state["reasoning_trace"].append(trace_entry)

                # 检查是否需要迭代
                should_iterate, next_weight, reason, suggested_action, param_adjustments = should_iterate_rerank(
                    state["rerank_weight"],
                    state["rerank_iterations"],
                    config,
                    eval_result,
                    coverage_diag,
                    llm_guidance,
                )

                # 应用 LLM 调参建议的权重
                if param_adjustments and "suggested_weight" in param_adjustments:
                    suggested_weight = param_adjustments["suggested_weight"]
                    if abs(suggested_weight - state["rerank_weight"]) > 0.05:
                        next_weight = max(0.2, min(0.8, suggested_weight))
                        state["rerank_weight"] = next_weight

                if suggested_action == "expand_query" and state["missing_entities"]:
                    # 扩展查询，重新召回
                    trace_entry = {
                        "step": step,
                        "thought": f"[{query_type}] 覆盖度不足 {coverage_diag['coverage_score']:.1%}，扩展查询: {state['missing_entities'][:2]}",
                        "action": "expand_query",
                        "action_input": {"entities": state["missing_entities"]},
                        "observation": "",
                    }
                    state["reasoning_trace"].append(trace_entry)

                    # 扩展查询并重新召回
                    expanded_queries = expand_query(
                        state["rewritten_question"],
                        state["missing_entities"],
                        state["query_type"],
                    )
                    state["expanded_queries"] = expanded_queries

                    # 多信号召回
                    new_chunks = multi_signal_recall(
                        expanded_queries,
                        top_k=20,
                        session_id=state["session_id"],
                        governance_context=self._governance_context,
                    )

                    state["retrieved_chunks"] = new_chunks
                    state["reranked_chunks"] = []
                    state["recall_iterations"] += 1
                    state["rerank_iterations"] = 0
                    state["rerank_weight"] = config["default_weight"]

                    trace_entry["observation"] = f"多信号召回 {len(new_chunks)} 个 chunks"

                    step += 1
                    if step >= max_steps:
                        break
                    continue

                elif suggested_action == "multi_signal_recall":
                    trace_entry = {
                        "step": step,
                        "thought": f"[{query_type}] 多种信号召回",
                        "action": "multi_signal_recall",
                        "action_input": {},
                        "observation": "",
                    }
                    state["reasoning_trace"].append(trace_entry)

                    # 使用扩展查询进行多信号召回
                    queries = state.get("expanded_queries", [state["rewritten_question"]])
                    new_chunks = multi_signal_recall(
                        queries,
                        top_k=25,
                        session_id=state["session_id"],
                        governance_context=self._governance_context,
                    )

                    state["retrieved_chunks"] = new_chunks
                    state["reranked_chunks"] = []
                    state["recall_iterations"] += 1

                    trace_entry["observation"] = f"多信号召回 {len(new_chunks)} 个 chunks"

                    step += 1
                    if step >= max_steps:
                        break
                    continue

                elif should_iterate:
                    trace_entry = {
                        "step": step,
                        "thought": f"[{query_type}] 质量不达标，{reason}",
                        "action": "iterate_rerank",
                        "action_input": {"next_weight": next_weight},
                        "observation": "",
                    }
                    state["reasoning_trace"].append(trace_entry)

                    state["rerank_weight"] = next_weight
                    state["_need_rerank"] = True  # 下轮继续 rerank
                    step += 1
                    if step >= max_steps:
                        break
                    continue
                else:
                    # 不再迭代，接受当前质量
                    state["recall_quality"] = eval_result["quality"]

            # ========== Step 4: 生成答案 ==========
            trace_entry = {
                "step": step,
                "thought": f"[{query_type}] 生成答案",
                "action": "generate_answer",
                "action_input": {"question": query},
                "observation": "",
            }
            state["reasoning_trace"].append(trace_entry)

            result = self._execute_generate_answer({"question": query}, state, tool_map)
            state["current_answer"] = result.get("answer", "")
            trace_entry["observation"] = f"生成答案: {state['current_answer'][:50]}..."

            step += 1
            if step >= max_steps:
                break

            # ========== Step 5: 评估答案质量 ==========
            eval_answer = evaluate_answer_quality(
                state["current_answer"],
                state["reranked_chunks"] or state["retrieved_chunks"]
            )
            state["answer_quality"] = eval_answer["quality"]

            trace_entry = {
                "step": step,
                "thought": f"[{query_type}] 评估答案质量",
                "action": "evaluate_answer",
                "action_input": {},
                "observation": eval_answer["reason"],
            }
            state["reasoning_trace"].append(trace_entry)

            # ========== 答案质量判定 ==========
            if eval_answer["quality"] == "good":
                # 质量好，结束
                break
            elif eval_answer["quality"] == "medium" and state["rerank_iterations"] < config["max_iterations"]:
                # 质量一般，尝试重新 rerank
                state["rerank_weight"] = min(state["rerank_weight"] + config["weight_iteration_step"], 0.8)
                state["_need_rerank"] = True
                trace_entry = {
                    "step": step,
                    "thought": f"[{query_type}] 答案质量一般，尝试提高 rerank 权重到 {state['rerank_weight']:.2f}",
                    "action": "iterate_rerank",
                    "action_input": {},
                    "observation": "",
                }
                state["reasoning_trace"].append(trace_entry)
                step += 1
                if step >= max_steps:
                    break
                continue
            else:
                # 质量差或无法改善，接受当前结果
                break

        # ========== 返回结果 ==========
        return AgentResponse(
            agent=AgentRole.QA,
            task_type=TaskType.QA,
            result={
                "answer": state["current_answer"] or "已达到最大步数限制",
                "question": question,
                "rewritten_query": query,
                "retrieved_chunks": state["retrieved_chunks"],
                "reranked_chunks": state["reranked_chunks"],
                "query_type": state["query_type"],
                "rerank_weight": state["rerank_weight"],
                "rerank_iterations": state["rerank_iterations"],
                "recall_quality": state["recall_quality"],
                "answer_quality": state.get("answer_quality", "unknown"),
            },
            reasoning_trace=state["reasoning_trace"],
            success=bool(state["current_answer"]),
            error="max_steps_reached" if not state["current_answer"] else None,
        )
    def _execute_search(self, action_input: dict, tool_map: dict, state: QAAgentState) -> dict[str, Any]:
        """执行搜索"""
        query = action_input.get("query", state["rewritten_question"])
        top_k = action_input.get("top_k", 20)

        from app.tools.mcp_client import call_tool_mcp_or_local

        result = call_tool_mcp_or_local(
            tool_name="kb_search_knowledge_base",
            args={"query": query, "top_k": top_k},
            agent="qa",
            session_id=state["session_id"],
            governance_context=self._governance_context,
        )

        # 触发召回质量未知
        state["recall_quality"] = "unknown"
        return result

    def _execute_rerank(self, state: QAAgentState) -> list[dict[str, Any]]:
        """执行 Rerank"""
        chunks = state["retrieved_chunks"]
        if not chunks:
            return []

        query = state["rewritten_question"]
        query_type = state["query_type"]
        weight = state["rerank_weight"]

        # 获取当前类型配置
        config = RERANK_TYPE_CONFIG.get(query_type, RERANK_TYPE_CONFIG["hybrid"])

        # 临时覆盖权重
        from app.services import reranker_service as reranker
        original_weight = reranker.LLM_RERANK_WEIGHT
        reranker.LLM_RERANK_WEIGHT = weight

        try:
            reranked = rerank_with_llm(
                query=query,
                intent=config["intent"],
                candidates=chunks,
                top_n=min(10, len(chunks)),
            )
        finally:
            reranker.LLM_RERANK_WEIGHT = original_weight

        # 更新迭代次数
        state["rerank_iterations"] += 1

        return reranked

    def _execute_generate_answer(self, action_input: dict, state: QAAgentState, tool_map: dict) -> dict[str, Any]:
        """生成答案"""
        from app.tools.mcp_client import call_tool_mcp_or_local

        question = action_input.get("question", state["rewritten_question"])
        context = action_input.get("context", "")

        # 如果没有提供 context，尝试组装
        if not context:
            chunks = state["reranked_chunks"] or state["retrieved_chunks"]
            if chunks:
                assemble_result = call_tool_mcp_or_local(
                    tool_name="kb_assemble_context",
                    args={"hits": chunks, "max_chunks": 6},
                    agent="qa",
                    session_id=state["session_id"],
                    governance_context=self._governance_context,
                )
                context = assemble_result.get("context", "")

        # 调用生成答案
        result = call_tool_mcp_or_local(
            tool_name="kb_generate_answer",
            args={
                "question": question,
                "context": context,
            },
            agent="qa",
            session_id=state["session_id"],
            governance_context=self._governance_context,
        )

        return result

    def _parse_action_response(self, raw: str) -> dict[str, Any]:
        """解析 JSON 响应"""
        import re
        raw = (raw or "").strip()

        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        brace = re.search(r"(\{.*\})", raw, re.S)
        if brace:
            try:
                return json.loads(brace.group(1))
            except json.JSONDecodeError:
                pass

        return {"thought": raw, "action": "final", "action_input": {"answer": raw}}


# 全局实例
qa_agent = QAAgent()
