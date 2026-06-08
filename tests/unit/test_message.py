"""
Message模块单元测试

测试范围：
- Message创建和字段初始化
- 序列化和反序列化
- 验证逻辑
- 工具方法
- OpenAI格式转换
- 消息列表操作

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import Any

import sys
sys.path.insert(0, '/workspace')

from src.core.message import (
    Message, Role, MessageType,
    create_system_message, create_user_message, create_assistant_message, create_tool_message,
    filter_messages_by_role, filter_messages_by_type, sort_messages_by_timestamp,
    get_total_token_count, validate_message_list,
    messages_to_openai_format, messages_to_json, messages_from_json
)
from src.core.utils.id_generator import generate_msg_id


class TestRoleAndMessageType(unittest.TestCase):
    """测试Role和MessageType枚举"""
    
    def test_role_values(self):
        """测试Role枚举值"""
        self.assertEqual(Role.SYSTEM.value, "system")
        self.assertEqual(Role.USER.value, "user")
        self.assertEqual(Role.ASSISTANT.value, "assistant")
        self.assertEqual(Role.TOOL.value, "tool")
    
    def test_message_type_values(self):
        """测试MessageType枚举值"""
        self.assertEqual(MessageType.TEXT.value, "text")
        self.assertEqual(MessageType.IMAGE.value, "image")
        self.assertEqual(MessageType.FILE.value, "file")
        self.assertEqual(MessageType.TOOL_CALL.value, "tool_call")
        self.assertEqual(MessageType.TOOL_RESULT.value, "tool_result")
    
    def test_role_from_string(self):
        """测试从字符串创建Role"""
        self.assertEqual(Role("system"), Role.SYSTEM)
        self.assertEqual(Role("user"), Role.USER)
        self.assertEqual(Role("assistant"), Role.ASSISTANT)
        self.assertEqual(Role("tool"), Role.TOOL)


class TestMessageCreation(unittest.TestCase):
    """测试Message创建和初始化"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        msg = Message(role=Role.USER, content="Hello")
        
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "Hello")
        self.assertIsNotNone(msg.msg_id)
        self.assertIsInstance(msg.timestamp, datetime)
        self.assertEqual(msg.token_count, 0)
        self.assertEqual(msg.msg_type, MessageType.TEXT)
        self.assertEqual(msg.metadata, {})
        self.assertIsNone(msg.parent_id)
        self.assertIsNone(msg.session_id)
    
    def test_creation_with_string_role(self):
        """测试使用字符串角色创建"""
        msg = Message(role="assistant", content="Hi")
        self.assertEqual(msg.role, Role.ASSISTANT)
    
    def test_creation_with_all_fields(self):
        """测试使用所有字段创建"""
        now = datetime.now()
        msg = Message(
            role=Role.SYSTEM,
            content="System prompt",
            msg_id="custom_id_123",
            timestamp=now,
            token_count=10,
            msg_type=MessageType.TEXT,
            metadata={"key": "value"},
            parent_id="parent_456",
            session_id="session_789"
        )
        
        self.assertEqual(msg.msg_id, "custom_id_123")
        self.assertEqual(msg.timestamp, now)
        self.assertEqual(msg.token_count, 10)
        self.assertEqual(msg.metadata, {"key": "value"})
        self.assertEqual(msg.parent_id, "parent_456")
        self.assertEqual(msg.session_id, "session_789")
    
    def test_creation_with_string_msg_type(self):
        """测试使用字符串消息类型创建"""
        msg = Message(role=Role.USER, content="Test", msg_type="image")
        self.assertEqual(msg.msg_type, MessageType.IMAGE)


class TestMessageSerialization(unittest.TestCase):
    """测试Message序列化和反序列化"""
    
    def setUp(self):
        """设置测试数据"""
        self.msg = Message(
            role=Role.ASSISTANT,
            content="Test message",
            msg_id="test_id_123",
            token_count=5,
            msg_type=MessageType.TEXT,
            metadata={"source": "test"},
            parent_id="parent_456",
            session_id="session_789"
        )
    
    def test_to_dict(self):
        """测试转换为字典"""
        data = self.msg.to_dict()
        
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Test message")
        self.assertEqual(data["msg_id"], "test_id_123")
        self.assertEqual(data["token_count"], 5)
        self.assertEqual(data["msg_type"], "text")
        self.assertEqual(data["metadata"], {"source": "test"})
        self.assertEqual(data["parent_id"], "parent_456")
        self.assertEqual(data["session_id"], "session_789")
        self.assertIsInstance(data["timestamp"], str)
    
    def test_to_json(self):
        """测试转换为JSON"""
        json_str = self.msg.to_json()
        data = json.loads(json_str)
        
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Test message")
    
    def test_to_json_with_indent(self):
        """测试带缩进的JSON转换"""
        json_str = self.msg.to_json(indent=2)
        self.assertIn("\n", json_str)  # 应该有换行
        self.assertIn("  ", json_str)  # 应该有缩进
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "role": "user",
            "content": "Hello from dict",
            "msg_id": "dict_id_123",
            "timestamp": "2026-05-25T10:30:00",
            "token_count": 3,
            "msg_type": "text",
            "metadata": {"test": True},
            "parent_id": "parent_id",
            "session_id": "session_id"
        }
        
        msg = Message.from_dict(data)
        
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "Hello from dict")
        self.assertEqual(msg.msg_id, "dict_id_123")
        self.assertEqual(msg.token_count, 3)
        self.assertEqual(msg.msg_type, MessageType.TEXT)
        self.assertEqual(msg.metadata, {"test": True})
        self.assertEqual(msg.parent_id, "parent_id")
        self.assertEqual(msg.session_id, "session_id")
        self.assertIsInstance(msg.timestamp, datetime)
    
    def test_from_dict_with_defaults(self):
        """测试从字典创建（使用默认值）"""
        data = {"role": "system", "content": "Minimal"}
        
        msg = Message.from_dict(data)
        
        self.assertEqual(msg.role, Role.SYSTEM)
        self.assertEqual(msg.content, "Minimal")
        self.assertIsNotNone(msg.msg_id)
        self.assertEqual(msg.token_count, 0)
        self.assertEqual(msg.msg_type, MessageType.TEXT)
    
    def test_from_dict_with_datetime_object(self):
        """测试从字典创建（datetime对象）"""
        now = datetime.now()
        data = {
            "role": "assistant",
            "content": "Test",
            "timestamp": now
        }
        
        msg = Message.from_dict(data)
        self.assertEqual(msg.timestamp, now)
    
    def test_from_json(self):
        """测试从JSON创建"""
        json_str = '{"role": "user", "content": "JSON test", "msg_id": "json_id"}'
        msg = Message.from_json(json_str)
        
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "JSON test")
        self.assertEqual(msg.msg_id, "json_id")
    
    def test_roundtrip_serialization(self):
        """测试往返序列化"""
        # 序列化
        data = self.msg.to_dict()
        # 反序列化
        msg2 = Message.from_dict(data)
        # 再次序列化
        data2 = msg2.to_dict()
        
        self.assertEqual(data, data2)


class TestMessageValidation(unittest.TestCase):
    """测试Message验证逻辑"""
    
    def test_valid_message(self):
        """测试有效消息"""
        msg = Message(role=Role.USER, content="Valid")
        valid, errors = msg.validate()
        
        self.assertTrue(valid)
        self.assertEqual(errors, [])
    
    def test_invalid_none_content(self):
        """测试无效消息（content为None）"""
        msg = Message(role=Role.USER, content=None)  # type: ignore
        valid, errors = msg.validate()
        
        self.assertFalse(valid)
        self.assertIn("content不能为None", errors)
    
    def test_invalid_non_string_content(self):
        """测试无效消息（content非字符串）"""
        msg = Message(role=Role.USER, content=12345)  # type: ignore
        valid, errors = msg.validate()
        
        self.assertFalse(valid)
        self.assertTrue(any("content必须是字符串" in e for e in errors))
    
    def test_invalid_negative_token_count(self):
        """测试无效消息（token_count为负数）"""
        msg = Message(role=Role.USER, content="Test", token_count=-1)
        valid, errors = msg.validate()
        
        self.assertFalse(valid)
        self.assertTrue(any("token_count不能为负数" in e for e in errors))
    
    def test_invalid_empty_msg_id(self):
        """测试无效消息（msg_id为空）"""
        msg = Message(role=Role.USER, content="Test", msg_id="")
        valid, errors = msg.validate()
        
        self.assertFalse(valid)
        self.assertIn("msg_id不能为空", errors)
    
    def test_is_valid_method(self):
        """测试is_valid快捷方法"""
        valid_msg = Message(role=Role.USER, content="Valid")
        self.assertTrue(valid_msg.is_valid())
        
        invalid_msg = Message(role=Role.USER, content=None)  # type: ignore
        self.assertFalse(invalid_msg.is_valid())


class TestMessageUtilityMethods(unittest.TestCase):
    """测试Message工具方法"""
    
    def setUp(self):
        """设置测试数据"""
        self.msg = Message(
            role=Role.ASSISTANT,
            content="Original content",
            token_count=10
        )
    
    def test_copy(self):
        """测试深拷贝"""
        msg_copy = self.msg.copy()
        
        # 内容相同
        self.assertEqual(msg_copy.content, self.msg.content)
        self.assertEqual(msg_copy.msg_id, self.msg.msg_id)
        
        # 但是不同对象
        self.assertIsNot(msg_copy, self.msg)
        
        # 修改副本不影响原对象
        msg_copy.content = "Modified"
        self.assertEqual(self.msg.content, "Original content")
    
    def test_update_token_count(self):
        """测试更新token计数"""
        result = self.msg.update_token_count(20)
        
        # 返回self（链式调用）
        self.assertIs(result, self.msg)
        
        # 值已更新
        self.assertEqual(self.msg.token_count, 20)
    
    def test_update_content(self):
        """测试更新内容"""
        result = self.msg.update_content("New content")
        
        # 返回self（链式调用）
        self.assertIs(result, self.msg)
        
        # 内容已更新
        self.assertEqual(self.msg.content, "New content")
        
        # token_count已重置
        self.assertEqual(self.msg.token_count, 0)
    
    def test_add_metadata(self):
        """测试添加元数据"""
        result = self.msg.add_metadata("key1", "value1")
        
        # 返回self（链式调用）
        self.assertIs(result, self.msg)
        
        # 元数据已添加
        self.assertEqual(self.msg.metadata["key1"], "value1")
        
        # 链式调用
        self.msg.add_metadata("key2", "value2").add_metadata("key3", "value3")
        self.assertEqual(self.msg.metadata["key2"], "value2")
        self.assertEqual(self.msg.metadata["key3"], "value3")
    
    def test_get_metadata(self):
        """测试获取元数据"""
        self.msg.metadata = {"existing": "value"}
        
        # 获取存在的键
        self.assertEqual(self.msg.get_metadata("existing"), "value")
        
        # 获取不存在的键
        self.assertIsNone(self.msg.get_metadata("nonexistent"))
        
        # 获取不存在的键（带默认值）
        self.assertEqual(self.msg.get_metadata("nonexistent", "default"), "default")
    
    def test_role_checks(self):
        """测试角色检查方法"""
        system_msg = Message(role=Role.SYSTEM, content="System")
        user_msg = Message(role=Role.USER, content="User")
        assistant_msg = Message(role=Role.ASSISTANT, content="Assistant")
        tool_msg = Message(role=Role.TOOL, content="Tool")
        
        self.assertTrue(system_msg.is_system())
        self.assertFalse(system_msg.is_user())
        
        self.assertTrue(user_msg.is_user())
        self.assertFalse(user_msg.is_assistant())
        
        self.assertTrue(assistant_msg.is_assistant())
        self.assertFalse(assistant_msg.is_tool())
        
        self.assertTrue(tool_msg.is_tool())
        self.assertFalse(tool_msg.is_system())
    
    def test_type_checks(self):
        """测试类型检查方法"""
        text_msg = Message(role=Role.USER, content="Text", msg_type=MessageType.TEXT)
        image_msg = Message(role=Role.USER, content="Image", msg_type=MessageType.IMAGE)
        tool_call_msg = Message(role=Role.ASSISTANT, content="Tool call", msg_type=MessageType.TOOL_CALL)
        tool_result_msg = Message(role=Role.TOOL, content="Tool result", msg_type=MessageType.TOOL_RESULT)
        
        self.assertTrue(text_msg.is_text())
        self.assertTrue(image_msg.is_image())
        self.assertTrue(tool_call_msg.is_tool_call())
        self.assertTrue(tool_result_msg.is_tool_result())
    
    def test_get_content_length(self):
        """测试获取内容长度"""
        msg = Message(role=Role.USER, content="Hello World")
        self.assertEqual(msg.get_content_length(), 11)
        
        msg_empty = Message(role=Role.USER, content="")
        self.assertEqual(msg_empty.get_content_length(), 0)
    
    def test_get_age_seconds(self):
        """测试获取消息年龄"""
        # 创建过去的消息
        past = datetime.now() - timedelta(seconds=5)
        msg = Message(role=Role.USER, content="Old", timestamp=past)
        
        age = msg.get_age_seconds()
        self.assertGreaterEqual(age, 5.0)
        self.assertLess(age, 6.0)


class TestOpenAIFormat(unittest.TestCase):
    """测试OpenAI格式转换"""
    
    def test_to_openai_format_basic(self):
        """测试基本OpenAI格式转换"""
        msg = Message(role=Role.USER, content="Hello")
        data = msg.to_openai_format()
        
        self.assertEqual(data["role"], "user")
        self.assertEqual(data["content"], "Hello")
    
    def test_to_openai_format_tool_call(self):
        """测试工具调用的OpenAI格式转换"""
        msg = Message(
            role=Role.ASSISTANT,
            content="Calling tool",
            msg_type=MessageType.TOOL_CALL,
            metadata={
                "tool_calls": [{"id": "call_123", "function": {"name": "test"}}]
            }
        )
        data = msg.to_openai_format()
        
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Calling tool")
        self.assertIn("tool_calls", data)
        self.assertEqual(len(data["tool_calls"]), 1)
    
    def test_to_openai_format_tool_result(self):
        """测试工具结果的OpenAI格式转换"""
        msg = Message(
            role=Role.TOOL,
            content="Tool output",
            msg_type=MessageType.TOOL_RESULT,
            metadata={"tool_call_id": "call_123"}
        )
        data = msg.to_openai_format()
        
        self.assertEqual(data["role"], "tool")
        self.assertEqual(data["content"], "Tool output")
        self.assertEqual(data["tool_call_id"], "call_123")
    
    def test_from_openai_format_basic(self):
        """测试从基本OpenAI格式创建"""
        data = {"role": "user", "content": "Hello from OpenAI"}
        msg = Message.from_openai_format(data)
        
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "Hello from OpenAI")
        self.assertEqual(msg.msg_type, MessageType.TEXT)
    
    def test_from_openai_format_tool_call(self):
        """测试从工具调用OpenAI格式创建"""
        data = {
            "role": "assistant",
            "content": "Calling",
            "tool_calls": [{"id": "call_123", "function": {"name": "test"}}]
        }
        msg = Message.from_openai_format(data)
        
        self.assertEqual(msg.role, Role.ASSISTANT)
        self.assertEqual(msg.msg_type, MessageType.TOOL_CALL)
        self.assertIn("tool_calls", msg.metadata)
    
    def test_from_openai_format_tool_result(self):
        """测试从工具结果OpenAI格式创建"""
        data = {
            "role": "tool",
            "content": "Result",
            "tool_call_id": "call_123"
        }
        msg = Message.from_openai_format(data)
        
        self.assertEqual(msg.role, Role.TOOL)
        self.assertEqual(msg.msg_type, MessageType.TOOL_RESULT)
        self.assertEqual(msg.metadata["tool_call_id"], "call_123")
    
    def test_roundtrip_openai_format(self):
        """测试OpenAI格式往返转换"""
        original = Message(role=Role.USER, content="Roundtrip test")
        openai_data = original.to_openai_format()
        restored = Message.from_openai_format(openai_data)
        
        self.assertEqual(restored.role, original.role)
        self.assertEqual(restored.content, original.content)


class TestMessageStringRepresentation(unittest.TestCase):
    """测试Message字符串表示"""
    
    def test_str_short_content(self):
        """测试短内容的字符串表示"""
        msg = Message(role=Role.USER, content="Hi")
        self.assertEqual(str(msg), "[user] Hi")
    
    def test_str_long_content(self):
        """测试长内容的字符串表示"""
        long_content = "a" * 100
        msg = Message(role=Role.USER, content=long_content)
        result = str(msg)
        
        self.assertIn("[user]", result)
        self.assertIn("...", result)
        # "[user] " = 7 chars, 50 chars content, "..." = 3 chars = 60 total
        self.assertEqual(len(result), 60)
    
    def test_repr(self):
        """测试repr表示"""
        msg = Message(role=Role.ASSISTANT, content="Test message")
        result = repr(msg)
        
        self.assertIn("Message", result)
        self.assertIn("assistant", result)
        self.assertIn(msg.msg_id, result)


class TestMessageEquality(unittest.TestCase):
    """测试Message相等性比较"""
    
    def test_equal_same_id(self):
        """测试相同ID的消息相等"""
        msg1 = Message(role=Role.USER, content="A", msg_id="same_id")
        msg2 = Message(role=Role.ASSISTANT, content="B", msg_id="same_id")
        
        self.assertEqual(msg1, msg2)
        self.assertEqual(hash(msg1), hash(msg2))
    
    def test_not_equal_different_id(self):
        """测试不同ID的消息不相等"""
        msg1 = Message(role=Role.USER, content="Same content", msg_id="id1")
        msg2 = Message(role=Role.USER, content="Same content", msg_id="id2")
        
        self.assertNotEqual(msg1, msg2)
    
    def test_not_equal_different_type(self):
        """测试与不同类型比较"""
        msg = Message(role=Role.USER, content="Test")
        
        self.assertNotEqual(msg, "not a message")
        self.assertNotEqual(msg, None)
        self.assertNotEqual(msg, 123)


class TestMessageFactoryFunctions(unittest.TestCase):
    """测试消息工厂函数"""
    
    def test_create_system_message(self):
        """测试创建系统消息"""
        msg = create_system_message("System prompt")
        
        self.assertEqual(msg.role, Role.SYSTEM)
        self.assertEqual(msg.content, "System prompt")
    
    def test_create_user_message(self):
        """测试创建用户消息"""
        msg = create_user_message("User input")
        
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.content, "User input")
    
    def test_create_assistant_message(self):
        """测试创建助手消息"""
        msg = create_assistant_message("Assistant response")
        
        self.assertEqual(msg.role, Role.ASSISTANT)
        self.assertEqual(msg.content, "Assistant response")
    
    def test_create_tool_message(self):
        """测试创建工具消息"""
        msg = create_tool_message("Tool output", "call_123")
        
        self.assertEqual(msg.role, Role.TOOL)
        self.assertEqual(msg.content, "Tool output")
        self.assertEqual(msg.msg_type, MessageType.TOOL_RESULT)
        self.assertEqual(msg.metadata["tool_call_id"], "call_123")
    
    def test_factory_with_additional_kwargs(self):
        """测试工厂函数带额外参数"""
        msg = create_user_message(
            "Hello",
            session_id="session_123",
            metadata={"source": "web"}
        )
        
        self.assertEqual(msg.session_id, "session_123")
        self.assertEqual(msg.metadata["source"], "web")


class TestMessageListOperations(unittest.TestCase):
    """测试消息列表操作"""
    
    def setUp(self):
        """设置测试数据"""
        self.messages = [
            Message(role=Role.SYSTEM, content="System", msg_id="1"),
            Message(role=Role.USER, content="User 1", msg_id="2"),
            Message(role=Role.ASSISTANT, content="Assistant 1", msg_id="3"),
            Message(role=Role.USER, content="User 2", msg_id="4"),
            Message(role=Role.ASSISTANT, content="Assistant 2", msg_id="5"),
            Message(role=Role.TOOL, content="Tool", msg_id="6"),
        ]
        
        # 设置token_count用于测试
        for i, msg in enumerate(self.messages):
            msg.token_count = (i + 1) * 10
    
    def test_filter_by_role(self):
        """测试按角色过滤"""
        user_msgs = filter_messages_by_role(self.messages, Role.USER)
        
        self.assertEqual(len(user_msgs), 2)
        self.assertTrue(all(m.role == Role.USER for m in user_msgs))
    
    def test_filter_by_type(self):
        """测试按类型过滤"""
        text_msgs = filter_messages_by_type(self.messages, MessageType.TEXT)
        
        self.assertEqual(len(text_msgs), 6)  # 所有都是TEXT类型
    
    def test_sort_by_timestamp(self):
        """测试按时间戳排序"""
        # 创建带特定时间戳的消息
        now = datetime.now()
        msgs = [
            Message(role=Role.USER, content="Second", timestamp=now),
            Message(role=Role.USER, content="First", timestamp=now - timedelta(hours=1)),
            Message(role=Role.USER, content="Third", timestamp=now + timedelta(hours=1)),
        ]
        
        sorted_msgs = sort_messages_by_timestamp(msgs)
        
        self.assertEqual(sorted_msgs[0].content, "First")
        self.assertEqual(sorted_msgs[1].content, "Second")
        self.assertEqual(sorted_msgs[2].content, "Third")
    
    def test_sort_by_timestamp_reverse(self):
        """测试按时间戳倒序排序"""
        now = datetime.now()
        msgs = [
            Message(role=Role.USER, content="Second", timestamp=now),
            Message(role=Role.USER, content="First", timestamp=now - timedelta(hours=1)),
        ]
        
        sorted_msgs = sort_messages_by_timestamp(msgs, reverse=True)
        
        self.assertEqual(sorted_msgs[0].content, "Second")
        self.assertEqual(sorted_msgs[1].content, "First")
    
    def test_get_total_token_count(self):
        """测试获取总Token数"""
        total = get_total_token_count(self.messages)
        
        # 10 + 20 + 30 + 40 + 50 + 60 = 210
        self.assertEqual(total, 210)
    
    def test_validate_message_list_valid(self):
        """测试验证有效消息列表"""
        valid, errors = validate_message_list(self.messages)
        
        self.assertTrue(valid)
        self.assertEqual(errors, [])
    
    def test_validate_message_list_invalid(self):
        """测试验证无效消息列表"""
        invalid_msgs = [
            Message(role=Role.USER, content="Valid"),
            Message(role=Role.USER, content=None),  # type: ignore
        ]
        
        valid, errors = validate_message_list(invalid_msgs)
        
        self.assertFalse(valid)
        self.assertTrue(any("Message[1]" in e for e in errors))
    
    def test_messages_to_openai_format(self):
        """测试消息列表转OpenAI格式"""
        openai_msgs = messages_to_openai_format(self.messages[:3])
        
        self.assertEqual(len(openai_msgs), 3)
        self.assertEqual(openai_msgs[0]["role"], "system")
        self.assertEqual(openai_msgs[1]["role"], "user")
        self.assertEqual(openai_msgs[2]["role"], "assistant")
    
    def test_messages_to_json(self):
        """测试消息列表转JSON"""
        json_str = messages_to_json(self.messages[:2])
        data = json.loads(json_str)
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["role"], "system")
        self.assertEqual(data[1]["role"], "user")
    
    def test_messages_from_json(self):
        """测试从JSON创建消息列表"""
        json_str = '[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]'
        msgs = messages_from_json(json_str)
        
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, Role.USER)
        self.assertEqual(msgs[1].role, Role.ASSISTANT)
    
    def test_messages_json_roundtrip(self):
        """测试消息列表JSON往返"""
        original = self.messages[:3]
        json_str = messages_to_json(original)
        restored = messages_from_json(json_str)
        
        self.assertEqual(len(restored), 3)
        for i, msg in enumerate(restored):
            self.assertEqual(msg.role, original[i].role)
            self.assertEqual(msg.content, original[i].content)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_content(self):
        """测试空内容"""
        msg = Message(role=Role.USER, content="")
        
        self.assertEqual(msg.content, "")
        self.assertTrue(msg.is_valid())
    
    def test_unicode_content(self):
        """测试Unicode内容"""
        msg = Message(role=Role.USER, content="你好世界 🌍 Привет мир")
        
        json_str = msg.to_json()
        restored = Message.from_json(json_str)
        
        self.assertEqual(restored.content, msg.content)
    
    def test_large_metadata(self):
        """测试大型元数据"""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        msg = Message(role=Role.USER, content="Test", metadata=large_metadata)
        
        self.assertEqual(len(msg.metadata), 1000)
    
    def test_nested_metadata(self):
        """测试嵌套元数据"""
        nested = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2"]
                }
            }
        }
        msg = Message(role=Role.USER, content="Test", metadata=nested)
        
        data = msg.to_dict()
        restored = Message.from_dict(data)
        
        self.assertEqual(restored.metadata["level1"]["level2"]["level3"], ["item1", "item2"])


if __name__ == "__main__":
    unittest.main()