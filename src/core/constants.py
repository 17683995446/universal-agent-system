"""
无上限上下文Agent系统 - 全局常量定义

本文件包含系统所有核心常量配置，遵循"第一性原理"设计：
- 所有常量有明确的物理/业务含义
- 常量值经过推导，非随意设定
- 支持环境变量覆盖，便于不同场景调优

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Final
import os


# =============================================================================
# 第一章：Token配额常量（窗口分区策略）
# =============================================================================
# 设计原则：
# 1. 系统指令区：必须永久保留，包含系统提示词、工具定义
# 2. 有效会话区：动态更新，包含用户对话、历史上下文
# 3. 预留推理区：防止模型输出超限，保证推理稳定性

# 系统指令区比例（弹性范围：15%-30%）
# - 下限15%：适用于简单场景，工具少、系统提示短
# - 上限30%：适用于复杂Agent，工具多、系统提示长
SYSTEM_ZONE_MIN_RATIO: Final[float] = float(os.getenv("SYSTEM_ZONE_MIN_RATIO", "0.15"))
SYSTEM_ZONE_MAX_RATIO: Final[float] = float(os.getenv("SYSTEM_ZONE_MAX_RATIO", "0.30"))
SYSTEM_ZONE_DEFAULT_RATIO: Final[float] = float(os.getenv("SYSTEM_ZONE_DEFAULT_RATIO", "0.20"))

# 有效会话区比例（固定60%）
# - 这是主要的用户交互区域
# - 包含：用户输入、历史对话、召回的记忆
SESSION_ZONE_RATIO: Final[float] = float(os.getenv("SESSION_ZONE_RATIO", "0.60"))

# 预留推理区比例（固定20%）
# - 防止模型输出Token超限导致截断
# - 根据模型不同可调整，但不得低于10%
RESERVE_ZONE_RATIO: Final[float] = float(os.getenv("RESERVE_ZONE_RATIO", "0.20"))
RESERVE_ZONE_MIN_RATIO: Final[float] = float(os.getenv("RESERVE_ZONE_MIN_RATIO", "0.10"))

# Token安全阈值（相对于窗口上限的百分比）
# - 当实际Token超过此阈值时触发预警
TOKEN_WARNING_THRESHOLD: Final[float] = float(os.getenv("TOKEN_WARNING_THRESHOLD", "0.85"))

# 严格安全阈值（必须保留的余量）
# - 无论如何裁剪，必须保留此比例的Token余量
TOKEN_HARD_LIMIT_RATIO: Final[float] = float(os.getenv("TOKEN_HARD_LIMIT_RATIO", "0.95"))


# =============================================================================
# 第二章：分片与重叠常量
# =============================================================================
# 设计原则：
# - 分片大小必须小于模型窗口，留出推理空间
# - 重叠区用于防止逻辑断裂，但会增加Token消耗

# 分片重叠缓冲区比例
# - 15%经验值：足够保留上下文衔接，不会过度消耗Token
CHUNK_OVERLAP_RATIO: Final[float] = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.15"))
CHUNK_OVERLAP_MIN_TOKENS: Final[int] = int(os.getenv("CHUNK_OVERLAP_MIN_TOKENS", "100"))

# 分片大小计算常量
# - 目标分片大小 = 模型窗口 * CHUNK_TARGET_RATIO
# - 实际分片大小 = 目标分片大小 - 重叠区
CHUNK_TARGET_RATIO: Final[float] = float(os.getenv("CHUNK_TARGET_RATIO", "0.70"))

# 最小分片大小（低于此值不分片，直接截断或压缩）
CHUNK_MIN_SIZE: Final[int] = int(os.getenv("CHUNK_MIN_SIZE", "500"))

# 最大分片数量（防止分片过多导致聚合困难）
CHUNK_MAX_COUNT: Final[int] = int(os.getenv("CHUNK_MAX_COUNT", "20"))


# =============================================================================
# 第三章：记忆管理常量
# =============================================================================
# 设计原则：
# - 瞬时记忆：随推理刷新，只保留当前必要信息
# - 短期记忆：Token配额制，LRU淘汰
# - 长期记忆：向量存储，语义检索按需召回

# 短期记忆Token配额上限
# - 替代传统的"轮次制"，改用Token配额制更灵活
SHORT_MEMORY_MAX_TOKENS: Final[int] = int(os.getenv("SHORT_MEMORY_MAX_TOKENS", "4000"))

# 短期记忆条目上限（双重限制）
SHORT_MEMORY_MAX_ITEMS: Final[int] = int(os.getenv("SHORT_MEMORY_MAX_ITEMS", "50"))

# 长期记忆召回参数
# Top-K召回数量
MEMORY_RECALL_TOP_K: Final[int] = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
# 语义相似度阈值（低于此值不召回）
MEMORY_SIMILARITY_THRESHOLD: Final[float] = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.70"))
# 召回Token配额上限（不超过会话区的30%）
MEMORY_RECALL_MAX_RATIO: Final[float] = float(os.getenv("MEMORY_RECALL_MAX_RATIO", "0.30"))

# 四维相关性评分权重
# - 语义相似度：向量检索的基础分数
# - 时间衰减：越近的记忆权重越高
# - 任务标签：同任务记忆加分
# - 行为权重：用户主动提及的记忆加分
RELEVANCE_SEMANTIC_WEIGHT: Final[float] = float(os.getenv("RELEVANCE_SEMANTIC_WEIGHT", "0.50"))
RELEVANCE_TIME_WEIGHT: Final[float] = float(os.getenv("RELEVANCE_TIME_WEIGHT", "0.25"))
RELEVANCE_TASK_WEIGHT: Final[float] = float(os.getenv("RELEVANCE_TASK_WEIGHT", "0.15"))
RELEVANCE_BEHAVIOR_WEIGHT: Final[float] = float(os.getenv("RELEVANCE_BEHAVIOR_WEIGHT", "0.10"))

# 时间衰减半衰期（小时）
# - 经过此时间后，时间权重衰减一半
MEMORY_TIME_HALFLIFE_HOURS: Final[float] = float(os.getenv("MEMORY_TIME_HALFLIFE_HOURS", "24.0"))


# =============================================================================
# 第四章：压缩与净化常量
# =============================================================================
# 设计原则：
# - 三级净化逐级触发，避免过度处理
# - 关键信息必须保留，禁止压缩丢失

# 一级净化：规则去噪
# - 空行连续数量超过此值时合并
DENOISE_MAX_EMPTY_LINES: Final[int] = int(os.getenv("DENOISE_MAX_EMPTY_LINES", "2"))
# - 重复内容检测窗口大小
DENOISE_REPEAT_WINDOW: Final[int] = int(os.getenv("DENOISE_REPEAT_WINDOW", "3"))

# 二级净化：结构化转换
# - 触发结构化转换的Token阈值
STRUCTURE_CONVERT_THRESHOLD: Final[int] = int(os.getenv("STRUCTURE_CONVERT_THRESHOLD", "2000"))

# 三级净化：小模型摘要
# - 触发摘要的Token阈值
SUMMARIZE_THRESHOLD: Final[int] = int(os.getenv("SUMMARIZE_THRESHOLD", "4000"))
# - 摘要后目标Token数（保留原文的此比例）
SUMMARIZE_TARGET_RATIO: Final[float] = float(os.getenv("SUMMARIZE_TARGET_RATIO", "0.30"))
# - 摘要最小保留Token数（无论如何压缩，至少保留此数量）
SUMMARIZE_MIN_TOKENS: Final[int] = int(os.getenv("SUMMARIZE_MIN_TOKENS", "500"))

# 关键信息保留规则
# - 以下类型的内容禁止压缩丢失
CRITICAL_INFO_PATTERNS: Final[list[str]] = [
    r"\b(参数|约束|条件|限制)\s*[:：]",  # 参数约束
    r"\b(结论|结果|输出|返回值)\s*[:：]",  # 结论结果
    r"\b(目标|目的|任务)\s*[:：]",  # 目标任务
    r"\b(步骤|流程|顺序)\s*\d+",  # 步骤流程
    r"\b(错误|异常|失败)\s*[:：]",  # 错误记录
    r"\b(待执行|未完成|pending)\s*[:：]",  # 待办事项
]

# 摘要完整性校验阈值
# - 关键信息保留率必须高于此值
INTEGRITY_MIN_PRESERVE_RATIO: Final[float] = float(os.getenv("INTEGRITY_MIN_PRESERVE_RATIO", "0.95"))


# =============================================================================
# 第五章：任务状态常量
# =============================================================================
# 设计原则：
# - 快照必须独立持久化，不依赖上下文窗口
# - 漂移检测定期执行，防止任务偏离

# 目标漂移检测间隔（轮次）
# - 每N轮对话执行一次目标对齐检查
DRIFT_CHECK_INTERVAL: Final[int] = int(os.getenv("DRIFT_CHECK_INTERVAL", "5"))

# 漂移检测相似度阈值
# - 当前意图与初始目标的语义相似度低于此值时报警
DRIFT_SIMILARITY_THRESHOLD: Final[float] = float(os.getenv("DRIFT_SIMILARITY_THRESHOLD", "0.60"))

# 快照自动保存间隔（轮次）
SNAPSHOT_AUTO_SAVE_INTERVAL: Final[int] = int(os.getenv("SNAPSHOT_AUTO_SAVE_INTERVAL", "3"))

# 快照保留数量上限（自动清理旧快照）
SNAPSHOT_MAX_RETAINED: Final[int] = int(os.getenv("SNAPSHOT_MAX_RETAINED", "10"))

# 任务状态字段定义（快照结构）
TASK_SNAPSHOT_FIELDS: Final[list[str]] = [
    "task_id",           # 任务唯一标识
    "session_id",        # 会话标识
    "created_at",        # 创建时间
    "updated_at",        # 更新时间
    "goal",              # 任务目标
    "completed_steps",   # 已完成步骤
    "pending_steps",     # 待执行步骤
    "variables",         # 关键变量
    "constraints",       # 约束条件
    "error_log",         # 错误记录
    "progress_ratio",    # 进度比例
]


# =============================================================================
# 第六章：模型与API常量
# =============================================================================

# 默认模型配置
DEFAULT_MODEL: Final[str] = os.getenv("DEFAULT_MODEL", "gpt-4")
DEFAULT_EMBEDDING_MODEL: Final[str] = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

# 模型窗口大小（Token）
# - 常用模型的上下文窗口大小
MODEL_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-haiku": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-opus": 200000,
}

# API调用超时配置（秒）
API_TIMEOUT_CONNECT: Final[float] = float(os.getenv("API_TIMEOUT_CONNECT", "10.0"))
API_TIMEOUT_READ: Final[float] = float(os.getenv("API_TIMEOUT_READ", "60.0"))

# 重试策略
API_RETRY_MAX_ATTEMPTS: Final[int] = int(os.getenv("API_RETRY_MAX_ATTEMPTS", "3"))
API_RETRY_BACKOFF_BASE: Final[float] = float(os.getenv("API_RETRY_BACKOFF_BASE", "2.0"))


# =============================================================================
# 第七章：日志与监控常量
# =============================================================================

# 日志级别
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")

# 关键指标监控阈值
METRICS_TOKEN_USAGE_THRESHOLD: Final[float] = float(os.getenv("METRICS_TOKEN_USAGE_THRESHOLD", "0.80"))
METRICS_MEMORY_RECALL_THRESHOLD: Final[float] = float(os.getenv("METRICS_MEMORY_RECALL_THRESHOLD", "0.50"))
METRICS_COMPRESSION_RATIO_THRESHOLD: Final[float] = float(os.getenv("METRICS_COMPRESSION_RATIO_THRESHOLD", "0.70"))


# =============================================================================
# 第八章：环境标识常量
# =============================================================================

# 运行环境
ENV_DEVELOPMENT: Final[str] = "development"
ENV_TESTING: Final[str] = "testing"
ENV_PRODUCTION: Final[str] = "production"

# 当前环境（从环境变量读取，默认为开发环境）
CURRENT_ENV: Final[str] = os.getenv("AGENT_ENV", ENV_DEVELOPMENT)

# 调试模式
DEBUG_MODE: Final[bool] = os.getenv("DEBUG_MODE", "false").lower() == "true"


# =============================================================================
# 常量校验函数
# =============================================================================

def validate_constants() -> list[str]:
    """
    校验常量配置的合法性
    
    Returns:
        错误信息列表，空列表表示校验通过
    """
    errors = []
    
    # 校验分区比例总和
    total_ratio = SYSTEM_ZONE_DEFAULT_RATIO + SESSION_ZONE_RATIO + RESERVE_ZONE_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        errors.append(f"Token分区比例总和必须为1.0，当前为{total_ratio}")
    
    # 校验系统区比例范围
    if not (0.1 <= SYSTEM_ZONE_MIN_RATIO <= SYSTEM_ZONE_MAX_RATIO <= 0.5):
        errors.append("系统区比例范围必须在0.1-0.5之间")
    
    # 校验预留区比例
    if RESERVE_ZONE_RATIO < RESERVE_ZONE_MIN_RATIO:
        errors.append(f"预留区比例不得低于最小值{RESERVE_ZONE_MIN_RATIO}")
    
    # 校验相关性权重总和
    relevance_total = (RELEVANCE_SEMANTIC_WEIGHT + RELEVANCE_TIME_WEIGHT + 
                       RELEVANCE_TASK_WEIGHT + RELEVANCE_BEHAVIOR_WEIGHT)
    if abs(relevance_total - 1.0) > 0.001:
        errors.append(f"相关性权重总和必须为1.0，当前为{relevance_total}")
    
    # 校验阈值范围
    if not (0 < TOKEN_WARNING_THRESHOLD < TOKEN_HARD_LIMIT_RATIO <= 1.0):
        errors.append("Token阈值配置不合理")
    
    if not (0 < MEMORY_SIMILARITY_THRESHOLD <= 1.0):
        errors.append("记忆相似度阈值必须在0-1之间")
    
    return errors


# 模块加载时自动校验
if __name__ == "__main__":
    validation_errors = validate_constants()
    if validation_errors:
        print("常量校验失败:")
        for error in validation_errors:
            print(f"  - {error}")
        exit(1)
    else:
        print("✅ 所有常量校验通过")
