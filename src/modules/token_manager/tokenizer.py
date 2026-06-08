"""
无上限上下文Agent系统 - Tokenizer模块

对接多模型原生Tokenizer，提供统一的Token计算接口

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Protocol
from abc import ABC, abstractmethod

from src.core.types import TokenCountResult, TokenizerBackend
from src.core.exceptions import TokenizerNotFound


class BaseTokenizer(ABC):
    """Tokenizer抽象基类"""
    
    def __init__(self, model: str, backend: TokenizerBackend):
        self.model = model
        self.backend = backend
        self._encoder: Optional[any] = None
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """将文本编码为Token ID列表"""
        pass
    
    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """将Token ID列表解码为文本"""
        pass
    
    def count(self, text: str) -> int:
        """计算文本的Token数量"""
        return len(self.encode(text))
    
    def count_result(self, text: str) -> TokenCountResult:
        """计算并返回详细结果"""
        try:
            token_count = self.count(text)
            return TokenCountResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                token_count=token_count,
                model=self.model,
                backend=self.backend
            )
        except Exception as e:
            return TokenCountResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                token_count=0,
                model=self.model,
                backend=self.backend,
                error=str(e)
            )


class TiktokenTokenizer(BaseTokenizer):
    """
    Tiktoken Tokenizer实现
    
    支持OpenAI GPT系列模型
    """
    
    # 模型到encoding的映射
    MODEL_ENCODING_MAP = {
        # GPT-4系列
        "gpt-4": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        # GPT-3.5系列
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        # 旧版GPT-3
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-001": "r50k_base",
        "text-curie-001": "r50k_base",
        "text-babbage-001": "r50k_base",
        "text-ada-001": "r50k_base",
    }
    
    def __init__(self, model: str):
        super().__init__(model, TokenizerBackend.TIKTOKEN)
        self._init_encoder()
    
    def _init_encoder(self) -> None:
        """初始化encoder"""
        try:
            import tiktoken
            
            # 获取encoding名称
            encoding_name = self.MODEL_ENCODING_MAP.get(self.model, "cl100k_base")
            
            # 尝试通过模型名获取encoding
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 模型名未注册，使用encoding名称
                self._encoder = tiktoken.get_encoding(encoding_name)
                
        except ImportError:
            raise TokenizerNotFound(
                f"tiktoken not installed. Install with: pip install tiktoken"
            )
    
    def encode(self, text: str) -> list[int]:
        """编码文本"""
        if self._encoder is None:
            raise TokenizerNotFound(self.model)
        return self._encoder.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        """解码Token"""
        if self._encoder is None:
            raise TokenizerNotFound(self.model)
        return self._encoder.decode(tokens)


class TransformersTokenizer(BaseTokenizer):
    """
    Transformers Tokenizer实现
    
    支持HuggingFace模型
    """
    
    def __init__(self, model: str, model_path: Optional[str] = None):
        super().__init__(model, TokenizerBackend.TRANSFORMERS)
        self.model_path = model_path or model
        self._init_tokenizer()
    
    def _init_tokenizer(self) -> None:
        """初始化tokenizer"""
        try:
            from transformers import AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
        except ImportError:
            raise TokenizerNotFound(
                f"transformers not installed. Install with: pip install transformers"
            )
        except Exception as e:
            raise TokenizerNotFound(f"Failed to load tokenizer for {self.model}: {e}")
    
    def encode(self, text: str) -> list[int]:
        """编码文本"""
        if self._tokenizer is None:
            raise TokenizerNotFound(self.model)
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, tokens: list[int]) -> str:
        """解码Token"""
        if self._tokenizer is None:
            raise TokenizerNotFound(self.model)
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class SimpleTokenizer(BaseTokenizer):
    """
    简单Tokenizer实现（兜底方案）
    
    基于字符数估算Token数，精度较低但无需依赖
    """
    
    # 平均每个Token的字符数（经验值）
    CHARS_PER_TOKEN = 4
    
    def __init__(self, model: str = "simple"):
        super().__init__(model, TokenizerBackend.CL100K_BASE)
    
    def encode(self, text: str) -> list[int]:
        """模拟编码（返回虚拟ID）"""
        token_count = self.count(text)
        return list(range(token_count))
    
    def decode(self, tokens: list[int]) -> str:
        """模拟解码"""
        return "[decoded text]"
    
    def count(self, text: str) -> int:
        """基于字符数估算Token数"""
        return max(1, len(text) // self.CHARS_PER_TOKEN)


class TokenizerFactory:
    """Tokenizer工厂类"""
    
    # 已注册的Tokenizer实例缓存
    _cache: dict[str, BaseTokenizer] = {}
    
    @classmethod
    def get_tokenizer(cls, model: str, backend: Optional[TokenizerBackend] = None) -> BaseTokenizer:
        """
        获取Tokenizer实例
        
        Args:
            model: 模型名称
            backend: 指定后端，None则自动选择
            
        Returns:
            Tokenizer实例
        """
        cache_key = f"{model}_{backend}"
        
        # 检查缓存
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # 根据后端创建实例
        if backend == TokenizerBackend.TIKTOKEN:
            tokenizer = TiktokenTokenizer(model)
        elif backend == TokenizerBackend.TRANSFORMERS:
            tokenizer = TransformersTokenizer(model)
        else:
            # 自动选择
            tokenizer = cls._auto_select(model)
        
        # 缓存实例
        cls._cache[cache_key] = tokenizer
        return tokenizer
    
    @classmethod
    def _auto_select(cls, model: str) -> BaseTokenizer:
        """自动选择合适的Tokenizer"""
        # 检查是否为OpenAI模型
        if any(name in model.lower() for name in ["gpt-", "text-davinci"]):
            try:
                return TiktokenTokenizer(model)
            except:
                pass
        
        # 尝试Transformers
        try:
            return TransformersTokenizer(model)
        except:
            pass
        
        # 使用兜底方案
        return SimpleTokenizer(model)
    
    @classmethod
    def clear_cache(cls) -> None:
        """清除缓存"""
        cls._cache.clear()


# 便捷函数
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    快速计算Token数
    
    Args:
        text: 输入文本
        model: 模型名称
        
    Returns:
        Token数量
    """
    tokenizer = TokenizerFactory.get_tokenizer(model)
    return tokenizer.count(text)


def count_tokens_batch(texts: list[str], model: str = "gpt-4") -> list[int]:
    """
    批量计算Token数
    
    Args:
        texts: 文本列表
        model: 模型名称
        
    Returns:
        Token数量列表
    """
    tokenizer = TokenizerFactory.get_tokenizer(model)
    return [tokenizer.count(text) for text in texts]
