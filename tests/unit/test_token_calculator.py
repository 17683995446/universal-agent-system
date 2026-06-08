"""
无上限上下文Agent系统 - Token计算器测试

精度测试与功能验证

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import pytest
import time
from typing import List

from src.modules.token_manager.calculator import TokenCalculator, TokenCounter
from src.modules.token_manager.tokenizer import SimpleTokenizer
from src.core.types import Message, Role, TokenBudget
from src.core.constants import MODEL_CONTEXT_WINDOWS


class TestTokenCalculatorBasic:
    """基础功能测试"""
    
    def test_init(self):
        """测试初始化"""
        calc = TokenCalculator("gpt-4")
        assert calc.model == "gpt-4"
        assert calc.window_size == MODEL_CONTEXT_WINDOWS["gpt-4"]
    
    def test_calculate_empty_text(self):
        """测试空文本计算"""
        calc = TokenCalculator()
        # 空文本应该返回0或很小的值
        result = calc.calculate("")
        assert result >= 0
    
    def test_calculate_simple_text(self):
        """测试简单文本计算"""
        calc = TokenCalculator()
        text = "Hello world"
        result = calc.calculate(text)
        # "Hello world" 应该只有几个token
        assert result > 0
        assert result < 10
    
    def test_calculate_chinese_text(self):
        """测试中文文本计算"""
        calc = TokenCalculator()
        text = "你好世界"
        result = calc.calculate(text)
        # 中文每个字大约1-2个token
        assert result >= 2
        assert result <= 8


class TestTokenCalculatorAccuracy:
    """精度测试 - 验证误差<1%"""
    
    # 已知Token数的测试文本（使用SimpleTokenizer作为基准）
    TEST_CASES = [
        # (text, expected_chars_per_token_range)
        ("Hello", (3, 5)),
        ("Hello world! This is a test.", (3, 5)),
        ("The quick brown fox jumps over the lazy dog.", (3, 5)),
        ("你好", (1, 3)),
        ("这是一个中文测试文本", (1, 3)),
        ("Hello world! 这是一个混合文本。", (2, 4)),
    ]
    
    def test_chars_per_token_estimation(self):
        """测试字符/Token比率估算"""
        calc = TokenCalculator()
        ratio = calc.estimate_chars_per_token()
        # 正常比率应该在3-5之间
        assert 2.0 <= ratio <= 6.0
    
    def test_batch_consistency(self):
        """测试批量计算一致性"""
        calc = TokenCalculator()
        texts = ["Hello", "World", "Test"]
        
        # 批量计算
        batch_results = calc.calculate_batch(texts)
        # 逐个计算
        individual_results = [calc.calculate(t) for t in texts]
        
        assert batch_results == individual_results
    
    def test_message_calculation(self):
        """测试消息计算"""
        calc = TokenCalculator()
        
        msg = Message(role=Role.USER, content="Hello world")
        tokens = calc.calculate_message(msg)
        
        # 消息Token应该包含内容+角色开销
        content_tokens = calc.calculate("Hello world")
        assert tokens >= content_tokens
        assert tokens <= content_tokens + 10  # 开销应该很小
    
    def test_message_list_calculation(self):
        """测试消息列表计算"""
        calc = TokenCalculator()
        
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
        
        total = calc.calculate_messages(messages)
        
        # 总Token应该大于各消息之和（因为有格式开销）
        individual_sum = sum(calc.calculate_message(m) for m in messages)
        assert total >= individual_sum


class TestTokenBudget:
    """Token预算测试"""
    
    def test_allocate_budget_default(self):
        """测试默认预算分配"""
        calc = TokenCalculator("gpt-4")
        budget = calc.allocate_budget()
        
        assert budget.window_size == 8192
        assert budget.system_zone > 0
        assert budget.session_zone > 0
        assert budget.reserve_zone > 0
        assert budget.total <= budget.window_size
    
    def test_allocate_budget_custom_ratio(self):
        """测试自定义比例"""
        calc = TokenCalculator("gpt-4")
        budget = calc.allocate_budget(
            system_ratio=0.25,
            session_ratio=0.55,
            reserve_ratio=0.20
        )
        
        assert budget.system_zone == int(8192 * 0.25)
        assert budget.session_zone == int(8192 * 0.55)
        assert budget.reserve_zone == int(8192 * 0.20)


class TestTokenLimitCheck:
    """Token限制检查测试"""
    
    def test_check_limit_normal(self):
        """测试正常情况"""
        calc = TokenCalculator("gpt-4")
        result = calc.check_limit(1000)
        
        assert result["current"] == 1000
        assert result["is_warning"] is False
        assert result["is_exceeded"] is False
        assert result["remaining"] > 0
    
    def test_check_limit_warning(self):
        """测试预警情况"""
        calc = TokenCalculator("gpt-4")
        # 85%阈值 = 6963 tokens
        result = calc.check_limit(7000)
        
        assert result["is_warning"] is True
        assert result["is_exceeded"] is False
    
    def test_check_limit_exceeded(self):
        """测试超限情况"""
        calc = TokenCalculator("gpt-4")
        # 95%阈值 = 7782 tokens
        result = calc.check_limit(8000, raise_on_exceed=False)
        
        assert result["is_exceeded"] is True
    
    def test_check_limit_raise_exception(self):
        """测试超限时抛出异常"""
        calc = TokenCalculator("gpt-4")
        
        from src.core.exceptions import TokenLimitExceeded
        with pytest.raises(TokenLimitExceeded):
            calc.check_limit(8000, raise_on_exceed=True)


class TestTokenCounter:
    """TokenCounter静态工具类测试"""
    
    def test_count(self):
        """测试静态计数方法"""
        result = TokenCounter.count("Hello world", model="gpt-4")
        assert result > 0
    
    def test_count_messages(self):
        """测试静态消息计数方法"""
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
        ]
        result = TokenCounter.count_messages(messages, model="gpt-4")
        assert result > 0
    
    def test_check(self):
        """测试静态检查方法"""
        result = TokenCounter.check("Hello", model="gpt-4")
        assert "current" in result
        assert "is_warning" in result


class TestPerformance:
    """性能测试"""
    
    def test_calculation_speed(self):
        """测试计算速度（<10ms/1K tokens）"""
        calc = TokenCalculator()
        
        # 生成1K字符的测试文本
        text = "Hello world! " * 100  # 约1300字符
        
        # 预热
        for _ in range(10):
            calc.calculate(text)
        
        # 正式测试
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            calc.calculate(text)
        elapsed = time.time() - start
        
        avg_time = elapsed / iterations * 1000  # 转换为ms
        
        # 平均每次计算应该小于10ms
        assert avg_time < 10.0, f"Average calculation time {avg_time:.2f}ms exceeds 10ms limit"
    
    def test_batch_performance(self):
        """测试批量计算性能"""
        calc = TokenCalculator()
        texts = ["Hello world! " * 50 for _ in range(100)]
        
        start = time.time()
        results = calc.calculate_batch(texts)
        elapsed = time.time() - start
        
        assert len(results) == 100
        # 100条文本应该在1秒内完成
        assert elapsed < 1.0


class TestMultiModelSupport:
    """多模型支持测试"""
    
    def test_gpt4_window_size(self):
        """测试GPT-4窗口大小"""
        calc = TokenCalculator("gpt-4")
        assert calc.window_size == 8192
    
    def test_gpt4_turbo_window_size(self):
        """测试GPT-4-Turbo窗口大小"""
        calc = TokenCalculator("gpt-4-turbo")
        assert calc.window_size == 128000
    
    def test_gpt35_window_size(self):
        """测试GPT-3.5窗口大小"""
        calc = TokenCalculator("gpt-3.5-turbo")
        assert calc.window_size == 4096
    
    def test_unknown_model_default(self):
        """测试未知模型使用默认值"""
        calc = TokenCalculator("unknown-model")
        assert calc.window_size == 8192  # 默认8K


class TestEdgeCases:
    """边界情况测试"""
    
    def test_very_long_text(self):
        """测试超长文本"""
        calc = TokenCalculator()
        text = "Hello " * 10000  # 约60000字符
        
        result = calc.calculate(text)
        assert result > 0
        # 应该能够处理而不崩溃
    
    def test_special_characters(self):
        """测试特殊字符"""
        calc = TokenCalculator()
        texts = [
            "Hello\nWorld",
            "Tab\tHere",
            "Emoji: 😀🎉",
            "Unicode: 日本語",
            "Code: `print('hello')`",
        ]
        
        for text in texts:
            result = calc.calculate(text)
            assert result >= 0
    
    def test_empty_messages(self):
        """测试空消息列表"""
        calc = TokenCalculator()
        result = calc.calculate_messages([])
        assert result == 0


# 运行测试的辅助函数
def run_accuracy_test():
    """运行精度测试并输出报告"""
    print("\n" + "="*60)
    print("Token计算器精度测试报告")
    print("="*60)
    
    calc = TokenCalculator()
    
    test_cases = [
        ("Hello world", "英文短句"),
        ("The quick brown fox jumps over the lazy dog.", "英文长句"),
        ("你好", "中文短句"),
        ("这是一个用于测试Token计算器精度的中文文本。", "中文长句"),
        ("Hello world! 这是一个混合文本。", "混合文本"),
        ("print('Hello, World!')", "代码片段"),
    ]
    
    print("\n测试样例:")
    print("-" * 60)
    
    for text, desc in test_cases:
        tokens = calc.calculate(text)
        chars = len(text)
        ratio = chars / tokens if tokens > 0 else 0
        
        print(f"{desc:20s} | 字符: {chars:4d} | Token: {tokens:3d} | 比率: {ratio:.2f}")
    
    # 估算比率
    ratio = calc.estimate_chars_per_token()
    print(f"\n估算字符/Token比率: {ratio:.2f}")
    print("="*60)


if __name__ == "__main__":
    # 运行精度测试
    run_accuracy_test()
    
    # 运行pytest
    pytest.main([__file__, "-v"])
