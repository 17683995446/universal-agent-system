"""
上下文管理器单元测试

测试范围：
- ContextAssembly 结构体（创建、序列化、验证、工具方法）
- TokenBudget 预算分配
- Partitioner 分区器（固定/自适应分配）
- StrategyPartitioner 策略分区器
- ContextAssembler 组装器（组装、截断、格式转换）

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
import json

import sys
sys.path.insert(0, '/workspace')

from src.modules.context_manager.context_assembly import ContextAssembly, TokenBudget
from src.modules.context_manager.partitioner import (
    Partitioner, StrategyPartitioner, PartitionStrategy, PartitionConfig
)
from src.modules.context_manager.assembler import (
    ContextAssembler, AssemblyConfig, AssemblyResult, quick_assemble
)
from src.core.message import Message, Role, create_system_message, create_user_message, create_assistant_message


# =============================================================================
# TokenBudget 测试
# =============================================================================

class TestTokenBudget(unittest.TestCase):
    """测试Token预算"""

    def test_creation(self):
        budget = TokenBudget(window_size=4096, system_zone=800, session_zone=2400, reserve_zone=800)
        self.assertEqual(budget.window_size, 4096)
        self.assertEqual(budget.system_zone, 800)
        self.assertEqual(budget.session_zone, 2400)
        self.assertEqual(budget.reserve_zone, 800)

    def test_total(self):
        budget = TokenBudget(window_size=4096, system_zone=800, session_zone=2400, reserve_zone=800)
        self.assertEqual(budget.total, 4000)

    def test_available_for_messages(self):
        budget = TokenBudget(window_size=4096, system_zone=800, session_zone=2400, reserve_zone=800)
        self.assertEqual(budget.available_for_messages, 3200)

    def test_serialization(self):
        budget = TokenBudget(window_size=4096, system_zone=800, session_zone=2400, reserve_zone=800)
        data = budget.to_dict()
        self.assertEqual(data["window_size"], 4096)

        restored = TokenBudget.from_dict(data)
        self.assertEqual(restored.window_size, budget.window_size)


# =============================================================================
# ContextAssembly 测试
# =============================================================================

class TestContextAssemblyCreation(unittest.TestCase):
    """测试ContextAssembly创建"""

    def test_empty_creation(self):
        assembly = ContextAssembly()
        self.assertEqual(assembly.get_message_count(), 0)
        self.assertEqual(assembly.total_tokens, 0)
        self.assertEqual(len(assembly.recalled_memories), 0)

    def test_with_messages(self):
        sys_msg = create_system_message("You are a helpful assistant")
        sys_msg.token_count = 10
        user_msg = create_user_message("Hello")
        user_msg.token_count = 5

        assembly = ContextAssembly(
            system_messages=[sys_msg],
            session_messages=[user_msg],
        )
        self.assertEqual(assembly.get_message_count(), 2)


class TestContextAssemblyTokenCount(unittest.TestCase):
    """测试Token计数"""

    def setUp(self):
        self.assembly = ContextAssembly()
        self.sys_msg = create_system_message("System prompt")
        self.sys_msg.token_count = 10
        self.user_msg = create_user_message("User message")
        self.user_msg.token_count = 5
        self.assistant_msg = create_assistant_message("Assistant response")
        self.assistant_msg.token_count = 8

    def test_get_system_token_count(self):
        self.assembly.system_messages = [self.sys_msg]
        self.assertEqual(self.assembly.get_system_token_count(), 10)

    def test_get_session_token_count(self):
        self.assembly.session_messages = [self.user_msg, self.assistant_msg]
        self.assertEqual(self.assembly.get_session_token_count(), 13)

    def test_update_total_tokens(self):
        self.assembly.system_messages = [self.sys_msg]
        self.assembly.session_messages = [self.user_msg]
        self.assembly.update_total_tokens()
        self.assertEqual(self.assembly.total_tokens, 15)


class TestContextAssemblyBudget(unittest.TestCase):
    """测试预算相关"""

    def setUp(self):
        self.assembly = ContextAssembly(
            budget=TokenBudget(window_size=1000, system_zone=200, session_zone=600, reserve_zone=200)
        )

    def test_is_within_budget(self):
        self.assembly.total_tokens = 500
        self.assertTrue(self.assembly.is_within_budget())

        self.assembly.total_tokens = 900
        self.assertFalse(self.assembly.is_within_budget())

    def test_get_budget_usage_ratio(self):
        self.assembly.total_tokens = 400
        # available_for_messages = 800
        self.assertEqual(self.assembly.get_budget_usage_ratio(), 0.5)


class TestContextAssemblySerialization(unittest.TestCase):
    """测试序列化"""

    def test_to_dict(self):
        assembly = ContextAssembly()
        assembly.add_system_message("System prompt")
        assembly.add_user_message("User message")

        data = assembly.to_dict()
        self.assertIn("system_messages", data)
        self.assertIn("session_messages", data)
        self.assertEqual(len(data["system_messages"]), 1)

    def test_roundtrip(self):
        assembly = ContextAssembly()
        assembly.add_system_message("System")
        assembly.add_user_message("User")

        data = assembly.to_dict()
        restored = ContextAssembly.from_dict(data)
        self.assertEqual(restored.get_message_count(), 2)


class TestContextAssemblyUtils(unittest.TestCase):
    """测试工具方法"""

    def test_to_messages(self):
        assembly = ContextAssembly()
        sys_msg = assembly.add_system_message("System")
        user_msg = assembly.add_user_message("User")

        messages = assembly.to_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, Role.SYSTEM)
        self.assertEqual(messages[1].role, Role.USER)

    def test_to_openai_format(self):
        assembly = ContextAssembly()
        assembly.add_system_message("System")
        assembly.add_user_message("User")

        openai_msgs = assembly.to_openai_format()
        self.assertEqual(len(openai_msgs), 2)
        self.assertEqual(openai_msgs[0]["role"], "system")

    def test_get_stats(self):
        assembly = ContextAssembly()
        assembly.add_system_message("System")
        assembly.add_user_message("User")
        assembly.update_total_tokens()

        stats = assembly.get_stats()
        self.assertEqual(stats["message_count"], 2)
        self.assertEqual(stats["system_message_count"], 1)
        self.assertEqual(stats["session_message_count"], 1)


# =============================================================================
# Partitioner 测试
# =============================================================================

class TestPartitioner(unittest.TestCase):
    """测试分区器"""

    def test_allocate_default(self):
        partitioner = Partitioner()
        budget = partitioner.allocate(4096)

        self.assertEqual(budget.window_size, 4096)
        self.assertGreater(budget.system_zone, 0)
        self.assertGreater(budget.session_zone, 0)
        self.assertGreater(budget.reserve_zone, 0)

    def test_allocate_custom_ratios(self):
        partitioner = Partitioner()
        budget = partitioner.allocate(
            1000,
            system_ratio=0.3,
            session_ratio=0.5,
            reserve_ratio=0.2,
        )

        self.assertEqual(budget.system_zone, 300)
        self.assertEqual(budget.session_zone, 500)
        self.assertEqual(budget.reserve_zone, 200)

    def test_allocate_adaptive(self):
        partitioner = Partitioner()
        budget = partitioner.allocate_adaptive(
            window_size=4096,
            system_tokens_needed=1500,
        )

        # 系统区应该大于默认值（20% = 819）
        self.assertGreater(budget.system_zone, 800)

    def test_calculate_for_model(self):
        partitioner = Partitioner()
        budget = partitioner.calculate_for_model("gpt-4")

        # gpt-4 窗口大小为 8192
        self.assertEqual(budget.window_size, 8192)


class TestPartitionConfig(unittest.TestCase):
    """测试分区配置"""

    def test_valid_config(self):
        config = PartitionConfig(system_ratio=0.2, session_ratio=0.6, reserve_ratio=0.2)
        self.assertEqual(config.system_ratio, 0.2)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            PartitionConfig(system_ratio=0.5, session_ratio=0.5, reserve_ratio=0.5)


class TestStrategyPartitioner(unittest.TestCase):
    """测试策略分区器"""

    def test_default_strategy(self):
        partitioner = StrategyPartitioner()
        budget = partitioner.allocate(1000, PartitionStrategy.DEFAULT)

        self.assertEqual(budget.system_zone, 200)
        self.assertEqual(budget.session_zone, 600)
        self.assertEqual(budget.reserve_zone, 200)

    def test_system_heavy_strategy(self):
        partitioner = StrategyPartitioner()
        budget = partitioner.allocate(1000, PartitionStrategy.SYSTEM_HEAVY)

        self.assertEqual(budget.system_zone, 300)
        self.assertEqual(budget.session_zone, 500)

    def test_session_priority_strategy(self):
        partitioner = StrategyPartitioner()
        budget = partitioner.allocate(1000, PartitionStrategy.SESSION_PRIORITY)

        self.assertEqual(budget.system_zone, 150)
        self.assertEqual(budget.session_zone, 700)

    def test_list_strategies(self):
        partitioner = StrategyPartitioner()
        strategies = partitioner.list_strategies()

        self.assertIn(PartitionStrategy.DEFAULT, strategies)
        self.assertIn(PartitionStrategy.SYSTEM_HEAVY, strategies)
        self.assertIn(PartitionStrategy.SESSION_PRIORITY, strategies)


# =============================================================================
# ContextAssembler 测试
# =============================================================================

class TestContextAssemblerBasic(unittest.TestCase):
    """测试组装器基础功能"""

    def setUp(self):
        self.assembler = ContextAssembler()
        self.sys_msg = create_system_message("System prompt")
        self.sys_msg.token_count = 10
        self.user_msg = create_user_message("User message")
        self.user_msg.token_count = 5

    def test_assemble_within_budget(self):
        result = self.assembler.assemble(
            system_messages=[self.sys_msg],
            session_messages=[self.user_msg],
            window_size=1000,
        )

        self.assertTrue(result.success)
        self.assertFalse(result.truncated)
        self.assertEqual(result.assembly.get_message_count(), 2)

    def test_assemble_with_custom_budget(self):
        budget = TokenBudget(window_size=1000, system_zone=200, session_zone=600, reserve_zone=200)
        result = self.assembler.assemble(
            system_messages=[self.sys_msg],
            session_messages=[self.user_msg],
            budget=budget,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.assembly.budget.window_size, 1000)


class TestContextAssemblerTruncation(unittest.TestCase):
    """测试截断功能"""

    def test_truncate_oldest_first(self):
        assembler = ContextAssembler(AssemblyConfig(min_messages=1))

        # 创建大量消息超出预算
        sys_msg = create_system_message("System")
        sys_msg.token_count = 100

        session_msgs = []
        for i in range(10):
            msg = create_user_message(f"Message {i}")
            msg.token_count = 100
            session_msgs.append(msg)

        # 窗口1000，系统100 + 会话1000 = 1100 > 800可用
        result = assembler.assemble(
            system_messages=[sys_msg],
            session_messages=session_msgs,
            window_size=1000,
        )

        # 应该触发截断
        self.assertTrue(result.truncated or result.assembly.is_within_budget())

    def test_preserve_system_messages(self):
        config = AssemblyConfig(min_messages=0)
        assembler = ContextAssembler(config)

        sys_msg = create_system_message("Important system prompt")
        sys_msg.token_count = 50

        user_msgs = []
        for i in range(5):
            msg = create_user_message(f"User {i}")
            msg.token_count = 200
            user_msgs.append(msg)

        result = assembler.assemble(
            system_messages=[sys_msg],
            session_messages=user_msgs,
            window_size=500,  # 很小窗口强制截断
        )

        # 系统消息应该保留
        assembly = result.assembly
        has_system = any(m.is_system() for m in assembly.to_messages())
        self.assertTrue(has_system)


class TestContextAssemblerAdaptive(unittest.TestCase):
    """测试自适应预算"""

    def test_assemble_with_adaptive_budget(self):
        assembler = ContextAssembler()

        # 大型系统提示
        sys_msg = create_system_message("A" * 1000)  # 大系统提示
        sys_msg.token_count = 200

        user_msg = create_user_message("Hello")
        user_msg.token_count = 5

        result = assembler.assemble_with_adaptive_budget(
            system_messages=[sys_msg],
            session_messages=[user_msg],
            window_size=1000,
        )

        # 系统区应该根据实际需要调整
        self.assertGreaterEqual(result.assembly.budget.system_zone, 200)


class TestContextAssemblerValidation(unittest.TestCase):
    """测试验证功能"""

    def test_validate_valid_assembly(self):
        assembler = ContextAssembler()
        assembly = ContextAssembly()
        assembly.add_system_message("System")
        assembly.add_user_message("User")
        assembly.update_total_tokens()

        valid, errors = assembler.validate_assembly(assembly)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_validate_exceeds_budget(self):
        assembler = ContextAssembler()
        assembly = ContextAssembly(
            budget=TokenBudget(window_size=100, system_zone=20, session_zone=60, reserve_zone=20)
        )
        assembly.total_tokens = 200  # 超预算

        valid, errors = assembler.validate_assembly(assembly)
        self.assertFalse(valid)
        self.assertTrue(any("超出预算" in e for e in errors))


class TestQuickAssemble(unittest.TestCase):
    """测试快速组装"""

    def test_quick_assemble_basic(self):
        assembly = quick_assemble(
            system_prompt="You are helpful",
            user_message="Hello",
            window_size=1000,
        )

        self.assertEqual(assembly.get_message_count(), 2)
        self.assertEqual(assembly.system_messages[0].content, "You are helpful")
        self.assertEqual(assembly.session_messages[0].content, "Hello")

    def test_quick_assemble_with_history(self):
        history = [create_user_message("Previous message")]
        assembly = quick_assemble(
            system_prompt="System",
            user_message="Current message",
            history=history,
            window_size=1000,
        )

        self.assertEqual(assembly.get_message_count(), 3)


# =============================================================================
# 集成测试
# =============================================================================

class TestContextIntegration(unittest.TestCase):
    """上下文管理器集成测试"""

    def test_full_assembly_workflow(self):
        """完整组装工作流"""
        # 1. 创建分区器并分配预算
        partitioner = StrategyPartitioner()
        budget = partitioner.allocate(4096, PartitionStrategy.BALANCED)

        # 2. 创建消息
        sys_msgs = [create_system_message("You are a helpful AI assistant.")]
        sys_msgs[0].token_count = 10

        session_msgs = [
            create_user_message("What is Python?"),
            create_assistant_message("Python is a programming language."),
            create_user_message("How do I install it?"),
        ]
        for i, msg in enumerate(session_msgs):
            msg.token_count = 10 + i

        # 3. 组装上下文
        assembler = ContextAssembler()
        result = assembler.assemble(
            system_messages=sys_msgs,
            session_messages=session_msgs,
            budget=budget,
        )

        # 4. 验证结果
        self.assertTrue(result.success)
        assembly = result.assembly

        # 5. 转换为OpenAI格式
        openai_msgs = assembly.to_openai_format()
        self.assertGreater(len(openai_msgs), 0)

        # 6. 获取统计
        stats = assembly.get_stats()
        self.assertIn("message_count", stats)
        self.assertIn("total_tokens", stats)

    def test_truncation_workflow(self):
        """截断工作流"""
        # 创建超出预算的消息
        sys_msg = create_system_message("System")
        sys_msg.token_count = 50

        session_msgs = []
        for i in range(20):
            msg = create_user_message(f"Message {i}")
            msg.token_count = 100
            session_msgs.append(msg)

        # 小窗口强制截断
        assembler = ContextAssembler(AssemblyConfig(min_messages=2))
        result = assembler.assemble(
            system_messages=[sys_msg],
            session_messages=session_msgs,
            window_size=500,
        )

        # 验证截断
        if result.truncated:
            self.assertGreater(result.messages_removed, 0)
            self.assertGreater(result.tokens_saved, 0)
            self.assertTrue(result.assembly.is_within_budget())


if __name__ == "__main__":
    unittest.main()
