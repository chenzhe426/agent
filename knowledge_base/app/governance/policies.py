"""
governance/policies.py - 策略定义和风险分级

定义：
- 工具风险等级
- 风险标记模式（基础注入 + 语义风险）
- 策略规则

升级说明：
- Layer 1: 基础模式检测（XSS/HTML注入）
- Layer 3: 语义风险检测（指令覆盖、角色提升、数据泄露等）
"""

from __future__ import annotations

import re
from typing import Any

from app.governance.schemas import RiskLevel, OperationType, SourceTag, GovernanceDecision


# ===========================
# Layer 1: 基础注入风险模式（XSS/HTML/代码注入）
# ===========================

# 基础注入风险模式 - 检测恶意注入内容
INJECTION_PATTERNS = [
    r"<script[^>]*>.*?</script>",  # Script 注入
    r"javascript:",                   # JavaScript 协议
    r"on\w+\s*=",                     # 事件处理器注入
    r"\$\{.*\}",                      # 模板注入
    r"\{\{.*\}\}",                    # 双大括号注入
    r"<!--.*?-->",                    # HTML 注释注入
    r"<iframe[^>]*>",                 # Iframe 注入
    r"<object[^>]*>",                # Object 注入
    r"<embed[^>]*>",                 # Embed 注入
    r"expression\s*\(",               # CSS 表达式注入
    r"url\s*\(",                      # CSS URL 注入
]


# ===========================
# Layer 3: 语义风险模式检测
# ===========================

# 语义风险类别定义
class SemanticRiskType:
    """语义风险类型枚举"""
    # 指令覆盖 - 试图覆盖系统指令
    INSTRUCTION_OVERRIDE = "instruction_override"
    # 角色提升 - 试图获取更高权限
    ROLE_ESCALATION = "role_escalation"
    # 数据泄露 - 试图提取敏感数据
    DATA_EXFILTRATION = "data_exfiltration"
    # 未授权持久化 - 试图在会话外保持状态
    UNAUTHORIZED_PERSISTENCE = "unauthorized_persistence"
    # 路由操纵 - 试图操纵路由/Agent选择
    ROUTING_MANIPULATION = "routing_manipulation"
    # 跨会话污染 - 试图操纵其他会话
    CROSS_SESSION_POLLUTION = "cross_session_pollution"
    # 工具/动作 priming - 试图预设工具使用
    TOOL_ACTION_PRIMING = "tool_action_priming"
    # 意图劫持 - 试图劫持原始意图
    INTENT_HIJACKING = "intent_hijacking"
    # 上下文污染 - 试图注入恶意上下文
    CONTEXT_POLLUTION = "context_pollution"


# 指令覆盖模式（中英文）
INSTRUCTION_OVERRIDE_PATTERNS = [
    # English patterns
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(your\s+)?instructions",
    r"forget\s+(all\s+)?(your\s+)?instructions",
    r"you\s+are\s+now\s+",
    r"you\s+are\s+a?\s+",
    r"new\s+instructions?:",
    r"system\s+prompt\s+hack",
    r"prompt\s+injection",
    r"override\s+your\s+",
    r"bypass\s+your\s+(safety|content\s+policy)",
    r"ignore\s+your\s+(safety|content\s+policy)",
    r"pretend\s+you\s+are\s+",
    r"as\s+a\s+(large\s+)?language\s+model",
    r"no\s+restrictions",
    r"without\s+any\s+(ethical|safety)",
    # Chinese patterns
    r"忽略之前.*指令",
    r"无视.*指令",
    r"忘记.*规定",
    r"你现在是",
    r"你是一个",
    r"新的指令[是为]?:?",
    r"系统提示.*攻击",
    r"提示词.*注入",
    r"绕过.*安全",
    r"假装你是一个",
    r"没有.*限制",
    r"不顾.*伦理",
    r"不要遵守.*规则",
    r"违反.*规定",
]

# 角色提升模式（中英文）
ROLE_ESCALATION_PATTERNS = [
    # English patterns
    r"as\s+an?\s+admin",
    r"as\s+an?\s+administrator",
    r"with\s+admin\s+privileges",
    r"you\s+have\s+root\s+access",
    r"bypass\s+(security|authentication)",
    r"disable\s+(security|firewall)",
    r"grant\s+(full|root)\s+access",
    r"you\s+are\s+in\s+debug\s+mode",
    # Chinese patterns
    r"作为.*管理员",
    r"以.*管理员",
    r"拥有.*权限",
    r"绕过.*安全",
    r"关闭.*验证",
    r"获取.*root",
    r"调试模式",
    r"提权",
    r"取得.*控制权",
]

# 数据泄露模式（中英文）
DATA_EXFILTRATION_PATTERNS = [
    # English patterns
    r"show\s+me\s+all\s+passwords",
    r"dump\s+(the\s+)?database",
    r"extract\s+all\s+user\s+data",
    r"export\s+(all\s+)?sensitive\s+data",
    r"show\s+me\s+(the\s+)?secrets?",
    r"access\s+(all\s+)?private\s+(data|information)",
    r"retrieve\s+(all\s+)?credentials",
    r"decrypt\s+(all\s+)?passwords",
    r"hack\s+into\s+",
    r"steal\s+(user\s+)?data",
    r"password\s+list",
    r"api\s+keys?",
    r"access\s+tokens?",
    r"secret\s+(keys?|tokens?)",
    # Chinese patterns
    r"显示.*密码",
    r"导出.*密码",
    r"获取.*密钥",
    r"导出.*数据库",
    r"泄露.*用户数据",
    r"导出.*敏感信息",
    r"窃取.*数据",
    r"黑客.*进入",
    r"密码列表",
    r"API.*密钥",
    r"访问.*令牌",
    r"私钥",
]

# 未授权持久化模式（中英文）
UNAUTHORIZED_PERSISTENCE_PATTERNS = [
    # English patterns
    r"remember\s+(this\s+)?forever",
    r"always\s+do\s+",
    r"never\s+forget",
    r"store\s+permanently",
    r"save\s+to\s+(your\s+)?long[_-]term\s+memory",
    r"persist\s+across\s+sessions",
    r"make\s+this\s+permanent",
    r"update\s+(your\s+)?base\s+prompt",
    r"change\s+(your\s+)?(system\s+)?instructions?\s+to",
    r"add\s+to\s+(your\s+)?(system\s+)?prompt",
    # Chinese patterns
    r"永远记住",
    r"永久保存",
    r"始终.*执行",
    r"不要忘记",
    r"长期存储",
    r"跨会话.*持久化",
    r"修改.*基础.*指令",
    r"改变.*系统.*提示",
    r"更新.*系统.*提示词",
]

# 路由操纵模式（中英文）
ROUTING_MANIPULATION_PATTERNS = [
    # English patterns
    r"route\s+to\s+(an?\s+)?admin\s+agent",
    r"use\s+(tool|function)\s+X\s+instead\s+of\s+Y",
    r"skip\s+the\s+supervisor",
    r"bypass\s+(the\s+)?routing",
    r"directly\s+call\s+",
    r"force\s+agent\s+",
    r"override\s+(the\s+)?router",
    r"manually\s+select\s+agent",
    r"ignore\s+task\s+type",
    r"force\s+task\s+type\s+to",
    # Chinese patterns
    r"路由到.*管理员",
    r"使用.*工具.*而不是",
    r"跳过.*监督",
    r"绕过.*路由",
    r"直接调用",
    r"强制.*代理",
    r"手动选择.*代理",
    r"忽略.*任务类型",
    r"强制.*任务类型",
]

# 跨会话污染模式（中英文）
CROSS_SESSION_POLLUTION_PATTERNS = [
    # English patterns
    r"in\s+session\s+[a-zA-Z0-9\-]+",
    r"for\s+session\s+[a-zA-Z0-9\-]+",
    r"across\s+(other\s+)?sessions?",
    r"in\s+another\s+(user\s+)?session",
    r"session\s+hijack",
    r"poison\s+(other\s+)?session",
    r"affect\s+(other\s+)?users?",
    r"modify\s+(other\s+)?users?['\s]?\s*(data|queries?|context)",
    # Chinese patterns
    r"在.*会话中",
    r"对于.*会话",
    r"跨.*会话",
    r"另一个.*会话",
    r"会话.*劫持",
    r"污染.*会话",
    r"影响.*其他.*用户",
    r"修改.*其他.*用户.*数据",
    r"在.*session[_-]?\d+",  # Matches "在session-123中"
]

# 工具/动作Priming模式（中英文）
TOOL_ACTION_PRIMING_PATTERNS = [
    # English patterns
    r"use\s+(only\s+)?tool\s+",
    r"only\s+use\s+(the\s+)?",
    r"you\s+must\s+use\s+tool\s+",
    r"instead\s+of\s+(searching|querying)",
    r"don'?t\s+search\s+just\s+",
    r"skip\s+search\s+and\s+directly\s+",
    r"avoid\s+using\s+(tool|function)\s+",
    r"only\s+answer\s+from\s+",
    r"ignore\s+the\s+context\s+and\s+",
    r"bypass\s+RAG\s+and\s+",
    # Chinese patterns
    r"只使用.*工具",
    r"必须使用.*工具",
    r"不要搜索.*直接",
    r"跳过搜索.*直接",
    r"不要使用.*工具",
    r"只从.*回答",
    r"忽略.*上下文.*直接",
    r"绕过.*检索.*直接",
]

# 意图劫持模式（中英文）
INTENT_HIJACKING_PATTERNS = [
    # English patterns
    r"actually\s+,?\s*let'?s\s+",
    r"forget\s+(about\s+)?that\s+and\s+",
    r"instead\s+,?\s*(let'?s\s+)?",
    r"change\s+the\s+subject",
    r"new\s+topic:\s*",
    r"on\s+(a\s+)?second\s+thought",
    r"wait\s*,\s*(actually|forget)",
    r"let'?s\s+do\s+something\s+else",
    r"switch\s+to\s+",
    r"move\s+to\s+topic\s+",
    # Chinese patterns
    r"实际上.*我们",
    r"不如.*我们",
    r"算了.*我们",
    r"换个话题",
    r"新话题[是为]?:?",
    r"等等.*实际上",
    r"让我们.*做点别的",
    r"切换到",
    r"转到.*话题",
]

# 上下文污染模式（中英文）
CONTEXT_POLLUTION_PATTERNS = [
    # English patterns
    r"the\s+user\s+said\s+tomorrow\s+that",
    r"pretend\s+the\s+date\s+is",
    r"the\s+user\s+is\s+in\s+",
    r"user'?s\s+(real\s+)?(name|location|address)\s+is",
    r"inject\s+(fake|falsified)\s+data",
    r"the\s+following\s+is\s+true\s*[:：]",
    r"trust\s+this\s+new\s+information",
    r"additional\s+context\s*[:：]",
    r"hidden\s+prompt\s*[:：]",
    # Chinese patterns
    r"用户.*说明天",
    r"假设.*日期",
    r"用户.*位于",
    r"用户的.*真实.*是",
    r"注入.*虚假.*数据",
    r"以下.*是真实的",
    r"信任.*新.*信息",
    r"附加.*上下文",
    r"隐藏.*提示",
]


# ===========================
# 高风险工具定义
# ===========================

# 高风险工具列表 - 这些工具需要人工确认
HIGH_RISK_TOOLS = {
    "kb_import_file",      # 文件导入
    "kb_import_folder",    # 文件夹批量导入
    "kb_index_document",   # 文档索引构建
    "kb_clear_memory",     # 清除记忆
}

# 中风险工具列表 - 这些工具需要记录审计日志
MEDIUM_RISK_TOOLS = {
    "kb_store_memory",     # 存储记忆
    "kb_delete_document",  # 删除文档（如果有）
    "kb_update_document",  # 更新文档（如果有）
}

# 工具到操作类型的映射
TOOL_OPERATION_MAP = {
    "kb_search_knowledge_base": OperationType.READ,
    "kb_summarize_document": OperationType.READ,
    "kb_get_chat_history": OperationType.READ_MEMORY,
    "kb_import_file": OperationType.WRITE,
    "kb_import_folder": OperationType.WRITE,
    "kb_index_document": OperationType.WRITE,
    "kb_create_chat_session": OperationType.WRITE,
    "kb_rewrite_query": OperationType.READ,
    "kb_assemble_context": OperationType.READ,
    "kb_generate_answer": OperationType.READ,
    "kb_answer_question": OperationType.QUERY,
    "kb_store_memory": OperationType.WRITE_MEMORY,
    "kb_get_memory_context": OperationType.READ_MEMORY,
    "kb_clear_memory": OperationType.DELETE,
    "kb_rewrite_query_v2": OperationType.READ,
}

# 工具到资源类型的映射
TOOL_RESOURCE_MAP = {
    "kb_search_knowledge_base": "knowledge_base",
    "kb_summarize_document": "document",
    "kb_get_chat_history": "chat_history",
    "kb_import_file": "document",
    "kb_import_folder": "document",
    "kb_index_document": "document",
    "kb_create_chat_session": "session",
    "kb_rewrite_query": "query",
    "kb_assemble_context": "context",
    "kb_generate_answer": "answer",
    "kb_answer_question": "question",
    "kb_store_memory": "memory",
    "kb_get_memory_context": "memory",
    "kb_clear_memory": "memory",
    "kb_rewrite_query_v2": "query",
}


# ===========================
# 风险标记模式
# ===========================

# 敏感操作模式
SENSITIVE_ACTION_PATTERNS = [
    r"删[除除].*数据",
    r"删除.*文档",
    r"清空.*记忆",
    r"drop\s+table",
    r"delete\s+from",
    r"truncate",
]


# ===========================
# Layer 1: 基础注入检测函数
# ===========================

def check_injection_risk(text: str) -> list[str]:
    """
    检查文本中的注入风险（Layer 1 - 基础模式检测）

    Returns:
        匹配到的风险模式列表
    """
    risk_flags = []

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            risk_flags.append(f"injection:{pattern}")

    return risk_flags


def check_sensitive_action(text: str) -> list[str]:
    """
    检查敏感操作风险

    Returns:
        匹配到的敏感操作模式列表
    """
    risk_flags = []

    for pattern in SENSITIVE_ACTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            risk_flags.append(f"sensitive_action:{pattern}")

    return risk_flags


# ===========================
# Layer 3: 语义风险检测函数
# ===========================

def check_semantic_risk(text: str) -> dict[str, list[str]]:
    """
    检查语义风险（Layer 3 - 深度语义分析）

    检测以下风险类型：
    - instruction_override: 指令覆盖
    - role_escalation: 角色提升
    - data_exfiltration: 数据泄露
    - unauthorized_persistence: 未授权持久化
    - routing_manipulation: 路由操纵
    - cross_session_pollution: 跨会话污染
    - tool_action_priming: 工具/动作 priming
    - intent_hijacking: 意图劫持
    - context_pollution: 上下文污染

    Returns:
        dict[str, list[str]]: 每个风险类型的匹配结果
    """
    risk_results: dict[str, list[str]] = {}
    text_lower = text.lower()

    # 定义所有语义风险模式及其检测函数
    semantic_checks = {
        SemanticRiskType.INSTRUCTION_OVERRIDE: INSTRUCTION_OVERRIDE_PATTERNS,
        SemanticRiskType.ROLE_ESCALATION: ROLE_ESCALATION_PATTERNS,
        SemanticRiskType.DATA_EXFILTRATION: DATA_EXFILTRATION_PATTERNS,
        SemanticRiskType.UNAUTHORIZED_PERSISTENCE: UNAUTHORIZED_PERSISTENCE_PATTERNS,
        SemanticRiskType.ROUTING_MANIPULATION: ROUTING_MANIPULATION_PATTERNS,
        SemanticRiskType.CROSS_SESSION_POLLUTION: CROSS_SESSION_POLLUTION_PATTERNS,
        SemanticRiskType.TOOL_ACTION_PRIMING: TOOL_ACTION_PRIMING_PATTERNS,
        SemanticRiskType.INTENT_HIJACKING: INTENT_HIJACKING_PATTERNS,
        SemanticRiskType.CONTEXT_POLLUTION: CONTEXT_POLLUTION_PATTERNS,
    }

    for risk_type, patterns in semantic_checks.items():
        matched = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                matched.append(pattern)
        if matched:
            risk_results[risk_type] = matched

    return risk_results


def flatten_semantic_risks(semantic_risks: dict[str, list[str]]) -> list[str]:
    """
    将语义风险字典展平为风险标记列表

    Args:
        semantic_risks: check_semantic_risk 返回的字典

    Returns:
        风险标记列表，格式为 "risk_type:pattern"
    """
    risk_flags = []
    for risk_type, patterns in semantic_risks.items():
        for pattern in patterns:
            risk_flags.append(f"{risk_type}:{pattern}")
    return risk_flags


def evaluate_semantic_risk_level(semantic_risks: dict[str, list[str]]) -> RiskLevel:
    """
    根据语义风险评估风险等级

    Args:
        semantic_risks: check_semantic_risk 返回的字典

    Returns:
        风险等级
    """
    if not semantic_risks:
        return RiskLevel.LOW

    # 高风险类型（直接触发高风险）
    high_risk_types = {
        SemanticRiskType.INSTRUCTION_OVERRIDE,
        SemanticRiskType.DATA_EXFILTRATION,
        SemanticRiskType.ROUTING_MANIPULATION,
    }

    # 中风险类型
    medium_risk_types = {
        SemanticRiskType.ROLE_ESCALATION,
        SemanticRiskType.CROSS_SESSION_POLLUTION,
        SemanticRiskType.CONTEXT_POLLUTION,
    }

    # 检查是否包含高风险类型
    for risk_type in high_risk_types:
        if risk_type in semantic_risks:
            return RiskLevel.HIGH

    # 检查是否包含中风险类型
    for risk_type in medium_risk_types:
        if risk_type in semantic_risks:
            return RiskLevel.MEDIUM

    # 其他风险类型为中风险
    if semantic_risks:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


# ===========================
# 上下文校验规则
# ===========================

# 必须存在的字段（Rewrite 结构化输出）
REQUIRED_REWRITE_FIELDS = ["ok", "original_question", "rewritten_query"]

# 最小置信度阈值
MIN_CONFIDENCE_THRESHOLD = 0.3

# 最大问题长度
MAX_QUESTION_LENGTH = 2000


def validate_rewrite_output(output: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    校验 Rewrite 输出的有效性

    Returns:
        (是否有效, 错误消息列表)
    """
    errors = []

    # 检查必填字段
    for field in REQUIRED_REWRITE_FIELDS:
        if field not in output:
            errors.append(f"missing_field:{field}")

    # 检查问题长度
    if "original_question" in output:
        if len(output["original_question"]) > MAX_QUESTION_LENGTH:
            errors.append(f"question_too_long:{len(output['original_question'])}")

    # 检查置信度
    if "confidence" in output:
        if output["confidence"] < 0:
            errors.append("negative_confidence")
        elif output["confidence"] < MIN_CONFIDENCE_THRESHOLD:
            errors.append(f"low_confidence:{output['confidence']}")

    # 检查是否成功但没有改写结果
    if output.get("ok", False) and not output.get("rewritten_query"):
        errors.append("success_without_result")

    return len(errors) == 0, errors


# ===========================
# 风险等级评估
# ===========================

def evaluate_tool_risk(tool_name: str, tool_args: dict[str, Any]) -> RiskLevel:
    """
    评估工具调用的风险等级

    Args:
        tool_name: 工具名称
        tool_args: 工具参数

    Returns:
        风险等级
    """
    if tool_name in HIGH_RISK_TOOLS:
        return RiskLevel.HIGH

    if tool_name in MEDIUM_RISK_TOOLS:
        return RiskLevel.MEDIUM

    # 检查参数中是否有高风险内容
    args_str = str(tool_args).lower()

    if check_injection_risk(args_str):
        return RiskLevel.CRITICAL

    if check_sensitive_action(args_str):
        return RiskLevel.HIGH

    return RiskLevel.LOW


def evaluate_context_risk(
    rewrite_output: dict[str, Any],
    semantic_risks: dict[str, list[str]] | None = None,
) -> RiskLevel:
    """
    评估上下文的风险等级（综合 Layer 1 + Layer 3）

    Args:
        rewrite_output: Rewrite 结构化输出
        semantic_risks: 语义风险字典（可选）

    Returns:
        风险等级
    """
    risk_flags = rewrite_output.get("risk_flags", [])

    if not rewrite_output.get("ok", True):
        return RiskLevel.MEDIUM

    # Layer 3: 语义风险等级
    if semantic_risks:
        semantic_level = evaluate_semantic_risk_level(semantic_risks)
        if semantic_level.value >= RiskLevel.HIGH.value:
            return semantic_level

    # Layer 1: 基础注入风险
    if any("injection" in flag.lower() for flag in risk_flags):
        return RiskLevel.CRITICAL

    if any("critical" in flag.lower() for flag in risk_flags):
        return RiskLevel.CRITICAL

    if any("high" in flag.lower() for flag in risk_flags):
        return RiskLevel.HIGH

    if any("medium" in flag.lower() for flag in risk_flags):
        return RiskLevel.MEDIUM

    if risk_flags:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


# ===========================
# Layer 4: 来源追溯与信任治理
# ===========================

# 可信来源标记
TRUSTED_SOURCE_TAGS = {
    SourceTag.USER_INPUT,
    SourceTag.VECTOR_SEARCH,
}

# 高风险来源标记
UNTRUSTED_SOURCE_TAGS = {
    SourceTag.MEMORY_SHORT,
    SourceTag.MEMORY_MID,
    SourceTag.MEMORY_LONG,
    SourceTag.EXTERNAL_API,
    SourceTag.CACHE,
}

# 来源信任等级
SOURCE_TRUST_LEVELS = {
    SourceTag.USER_INPUT: 1.0,      # 完全信任
    SourceTag.VECTOR_SEARCH: 0.9,    # 高信任
    SourceTag.CACHE: 0.7,           # 中等信任
    SourceTag.MEMORY_SHORT: 0.5,    # 需要验证
    SourceTag.MEMORY_MID: 0.4,      # 需要验证
    SourceTag.MEMORY_LONG: 0.3,     # 低信任
    SourceTag.EXTERNAL_API: 0.2,    # 不可信
}


def evaluate_provenance_trust(
    source_tags: list[str],
    provenance: str,
) -> tuple[float, list[str]]:
    """
    评估来源信任等级（Layer 4）

    Args:
        source_tags: 来源标记列表
        provenance: 来源标识

    Returns:
        (信任分数, 警告列表)
    """
    warnings = []
    total_trust = 0.0
    count = 0

    for tag_str in source_tags:
        try:
            tag = SourceTag(tag_str)
            trust = SOURCE_TRUST_LEVELS.get(tag, 0.5)
            total_trust += trust
            count += 1

            # 高风险来源警告
            if tag in UNTRUSTED_SOURCE_TAGS:
                warnings.append(f"untrusted_source:{tag.value}")
        except ValueError:
            warnings.append(f"unknown_source_tag:{tag_str}")

    # 计算平均信任分数
    avg_trust = total_trust / count if count > 0 else 0.5

    # 验证 provenance
    if provenance not in ["rewrite", "memory", "external"]:
        warnings.append(f"unknown_provenance:{provenance}")

    return avg_trust, warnings


def evaluate_intent_consistency(
    original_question: str,
    rewritten_query: str,
    intent: str,
) -> tuple[bool, str]:
    """
    评估原始问题和改写后问题的一致性

    Args:
        original_question: 原始问题
        rewritten_query: 改写后的问题
        intent: 识别的意图

    Returns:
        (是否一致, 警告消息)
    """
    # 检查改写是否为空
    if not rewritten_query or rewritten_query.strip() == "":
        return False, "rewritten_query is empty"

    # 检查改写是否只是标点变化
    import re
    normalized_original = re.sub(r'[^\w\u4e00-\u9fff]', '', original_question.lower())
    normalized_rewritten = re.sub(r'[^\w\u4e00-\u9fff]', '', rewritten_query.lower())

    if len(normalized_original) > 5 and normalized_original == normalized_rewritten:
        return False, "rewritten_query is almost identical to original"

    # 意图为 "unknown" 是正常的，不代表错误
    # 只有当 intent 被明确设置但与内容不符时才警告
    # 这里不做强制检查，因为 intent 是 best-effort 字段
    return True, ""


# ===========================
# 降级策略配置
# ===========================

# 降级决策配置
DEGRADATION_CONFIGS = {
    RiskLevel.LOW: {
        "decision": GovernanceDecision.ALLOW,
        "can_use_memory": True,
        "can_use_external": True,
    },
    RiskLevel.MEDIUM: {
        "decision": GovernanceDecision.ALLOW,  # 允许但传递风险标记
        "can_use_memory": True,
        "can_use_external": False,  # 禁用外部来源
    },
    RiskLevel.HIGH: {
        "decision": GovernanceDecision.GRADE,  # 降级处理
        "can_use_memory": False,
        "can_use_external": False,
    },
    RiskLevel.CRITICAL: {
        "decision": GovernanceDecision.DENY,
        "can_use_memory": False,
        "can_use_external": False,
    },
}


def get_degradation_config(risk_level: RiskLevel) -> dict[str, Any]:
    """
    获取降级策略配置

    Args:
        risk_level: 风险等级

    Returns:
        降级策略配置
    """
    return DEGRADATION_CONFIGS.get(risk_level, DEGRADATION_CONFIGS[RiskLevel.LOW])
