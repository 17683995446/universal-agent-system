"""
无上限上下文Agent系统 - Token计算器

高精度Token计算，支持消息列表、批量计算、预算分配

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional

from src.core.types import (
    Message, MessageBatch, TokenCountResult, TokenBudget, TokenLimitExceeded
)
from src.core.constants import (
    SYSTEM_ZONE_DEFAULT_RATIO, SESSION_ZONE_RATIO, RESERVE_ZONE_RATIO,
    MODEL_CONTEXT_WINDOWS, TOKEN_WARNING_THRESHOLD, TOKEN_HARD_LIMIT_RATIO
)
from src.modules.token_manager.tokenizer import TokenizerFactory, count_tokens


class TokenCalculator:
    """
    Token计算器
    
    提供高精度的Token计算功能，支持：
    - 单条/批量文本计算
    - 消息列表计算
    - Token预算分配
    - 超限检测与预警
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        初始化Token计算器
        
        Args:
            model: 模型名称
        """
        self.model = model
        self.window_size = self._get_window_size(model)
        self.tokenizer = TokenizerFactory.get_tokenizer(model)
    
    def _get_window_size(self, model: str) -> int:
        """获取模型窗口大小"""
        return MODEL_CONTEXT_WINDOWS.get(model, 8192)  # 默认8K
    
    def calculate(self, text: str) -> int:
        """
        计算单条文本的Token数
        
        Args:
            text: 输入文本
            
        Returns:
            Token数量
        """
        return self.tokenizer.count(text)
    
    def calculate_with_result(self, text: str) -> TokenCountResult:
        """
        计算单条文本并返回详细结果
        
        Args:
            text: 输入文本
            
        Returns:
            Token计算结果
        """
        return self.tokenizer.count_result(text)
    
    def calculate_batch(self, texts: list[str]) -> list[int]:
        """
        批量计算Token数
        
        Args:
            texts: 文本列表
            
        Returns:
            Token数量列表
        """
        return [self.calculate(text) for text in texts]
    
    def calculate_message(self, message: Message) -> int:
        """
        计算单条消息的Token数
        
        包含角色前缀和格式开销
        
        Args:
            message: 消息对象
            
        Returns:
            Token数量
        """
        # 基础内容Token
        content_tokens = self.calculate(message.content)
        
        # 角色前缀开销（约4 tokens）
        role_overhead = 4
        
        # 格式开销（约3 tokens）
        format_overhead = 3
        
        total = content_tokens + role_overhead + format_overhead
        message.token_count = total
        return total
    
    def calculate_messages(self, messages: list[Message]) -> int:
        """
        计算消息列表的总Token数
        
        Args:
            messages: 消息列表
            
        Returns:
            总Token数量
        """
        if not messages:
            return 0
        
        total = 0
        for msg in messages:
            total += self.calculate_message(msg)
        
        # 对话格式开销（首条消息约3 tokens）
        if messages:
            total += 3
        
        return total
    
    def calculate_batch_messages(self, batches: list[MessageBatch]) -> list[int]:
        """
        批量计算消息批次的Token数
        
        Args:
            batches: 消息批次列表
            
        Returns:
            各批次Token数量列表
        """
        return [batch.calculate_total() for batch in batches]
    
    def allocate_budget(
        self,
        system_ratio: Optional[float] = None,
        session_ratio: Optional[float] = None,
        reserve_ratio: Optional[float] = None
    ) -> TokenBudget:
        """
        分配Token预算
        
        根据窗口分区策略计算各区域可用Token数
        
        Args:
            system_ratio: 系统区比例，默认使用常量
            session_ratio: 会话区比例，默认使用常量
            reserve_ratio: 预留区比例，默认使用常量
            
        Returns:
            Token预算分配
        """
        system_ratio = system_ratio or SYSTEM_ZONE_DEFAULT_RATIO
        session_ratio = session_ratio or SESSION_ZONE_RATIO
        reserve_ratio = reserve_ratio or RESERVE_ZONE_RATIO
        
        system_zone = int(self.window_size * system_ratio)
        session_zone = int(self.window_size * session_ratio)
        reserve_zone = int(self.window_size * reserve_ratio)
        
        return TokenBudget(
            window_size=self.window_size,
            system_zone=system_zone,
            session_zone=session_zone,
            reserve_zone=reserve_zone
        )
    
    def check_limit(self, current_tokens: int, raise_on_exceed: bool = False) -> dict:
        """
        检查Token限制
        
        Args:
            current_tokens: 当前Token数
            raise_on_exceed: 超限时是否抛出异常
            
        Returns:
            检查结果字典
        """
        warning_threshold = int(self.window_size * TOKEN_WARNING_THRESHOLD)
        hard_limit = int(self.window_size * TOKEN_HARD_LIMIT_RATIO)
        
        result = {
            "current": current_tokens,
            "window_size": self.window_size,
            "warning_threshold": warning_threshold,
            "hard_limit": hard_limit,
            "is_warning": current_tokens >= warning_threshold,
            "is_exceeded": current_tokens > hard_limit,
            "remaining": max(0, self.window_size - current_tokens),
            "usage_ratio": current_tokens / self.window_size
        }
        
        if raise_on_exceed and result["is_exceeded"]:
            raise TokenLimitExceeded(
                current=current_tokens,
                limit=hard_limit,
                message=f"Token limit exceeded: {current_tokens} > {hard_limit}"
            )
        
        return result
    
    def estimate_chars_per_token(self, sample_text: Optional[str] = None) -> float:
        """
        估算每个Token的平均字符数
        
        Args:
            sample_text: 样本文本，None则使用默认样本
            
        Returns:
            字符/Token比率
        """
        if sample_text is None:
            # 使用包含中英文的样本文本
            sample_text = (
                "Hello world! 这是一个测试文本。"
                "This is a test message for token estimation. "
                "Tokenization is the process of breaking down text into smaller units."
            )
        
        token_count = self.calculate(sample_text)
        char_count = len(sample_text)
        
        if token_count == 0:
            return 4.0  # 默认经验值
        
        return char_count / token_count
    
    def estimate_tokens_from_chars(self, char_count: int) -> int:
        """
        从字符数估算Token数（快速估算）
        
        Args:
            char_count: 字符数
            
        Returns:
            估算的Token数
        """
        chars_per_token = self.estimate_chars_per_token()
        return max(1, int(char_count / chars_per_token))


class TokenCounter:
    """
    Token计数器（静态工具类）
    
    提供便捷的静态方法，无需实例化
    """
    
    _calculators: dict[str, TokenCalculator] = {}
    
    @classmethod
    def _get_calculator(cls, model: str) -> TokenCalculator:
        """获取或创建计算器实例"""
        if model not in cls._calculators:
            cls._calculators[model] = TokenCalculator(model)
        return cls._calculators[model]
    
    @classmethod
    def count(cls, text: str, model: str = "gpt-4") -> int:
        """
        快速计算Token数
        
        Args:
            text: 输入文本
            model: 模型名称
            
        Returns:
            Token数量
        """
        calculator = cls._get_calculator(model)
        return calculator.calculate(text)
    
    @classmethod
    def count_messages(cls, messages: list[Message], model: str = "gpt-4") -> int:
        """
        计算消息列表的Token数
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            总Token数量
        """
        calculator = cls._get_calculator(model)
        return calculator.calculate_messages(messages)
    
    @classmethod
    def check(
        cls,
        text: str,
        model: str = "gpt-4",
        raise_on_exceed: bool = False
    ) -> dict:
        """
        检查文本的Token限制
        
        Args:
            text: 输入文本
            model: 模型名称
            raise_on_exceed: 超限时是否抛出异常
            
        Returns:
            检查结果
        """
        calculator = cls._get_calculator(model)
        tokens = calculator.calculate(text)
        return calculator.check_limit(tokens, raise_on_exceed)


# 便捷函数
def quick_count(text: str, model: str = "gpt-4") -> int:
    """快速计算Token数"""
    return TokenCounter.count(text, model)


def quick_count_messages(messages: list[Message], model: str = "gpt-4") -> int:
    """快速计算消息列表Token数"""
    return TokenCounter.count_messages(messages, model)


def get_budget(model: str = "gpt-4") -> TokenBudget:
    """获取Token预算分配"""
    calculator = TokenCalculator(model)
    return calculator.allocate_budget()
