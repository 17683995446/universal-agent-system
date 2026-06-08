"""
记忆模块单元测试

测试范围：
- MemoryEntry 结构体（创建、序列化、验证、工具方法）
- InstantMemory 瞬时记忆管理器
- ShortTermMemory 短期记忆管理器（LRU淘汰）
- LongTermMemory 长期记忆管理器（语义召回）
- RelevanceScorer 四维评分器
- MemoryFactory 记忆工厂

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
import json
import math
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, '/workspace')

from src.modules.memory_manager.memory_entry import (
    MemoryEntry, MemoryLevel, MemorySource,
    filter_by_level, filter_valid, filter_non_expired,
    sort_by_relevance, sort_by_access_time,
    get_total_tokens, entries_to_json, entries_from_json,
    validate_entry_list,
)
from src.modules.memory_manager.instant_memory import InstantMemory
from src.modules.memory_manager.short_term_memory import ShortTermMemory
from src.modules.memory_manager.long_term_memory import (
    LongTermMemory, RecallResult, RelevanceScorer,
)
from src.modules.memory_manager.factory import MemoryFactory


# =============================================================================
# MemoryEntry 测试
# =============================================================================

class TestMemoryLevelAndSource(unittest.TestCase):
    """测试枚举"""

    def test_memory_level_values(self):
        self.assertEqual(MemoryLevel.INSTANT.value, "instant")
        self.assertEqual(MemoryLevel.SHORT_TERM.value, "short")
        self.assertEqual(MemoryLevel.LONG_TERM.value, "long")

    def test_memory_source_values(self):
        self.assertEqual(MemorySource.USER_INPUT.value, "user_input")
        self.assertEqual(MemorySource.ASSISTANT_OUTPUT.value, "assistant")
        self.assertEqual(MemorySource.SYSTEM_INSTRUCTION.value, "system")
        self.assertEqual(MemorySource.TOOL_RESULT.value, "tool_result")
        self.assertEqual(MemorySource.EXTERNAL_IMPORT.value, "external")
        self.assertEqual(MemorySource.SUMMARIZATION.value, "summarization")

    def test_level_from_string(self):
        self.assertEqual(MemoryLevel("instant"), MemoryLevel.INSTANT)
        self.assertEqual(MemoryLevel("short"), MemoryLevel.SHORT_TERM)
        self.assertEqual(MemoryLevel("long"), MemoryLevel.LONG_TERM)


class TestMemoryEntryCreation(unittest.TestCase):
    """测试MemoryEntry创建"""

    def test_basic_creation(self):
        entry = MemoryEntry(content="Hello")
        self.assertEqual(entry.content, "Hello")
        self.assertEqual(entry.level, MemoryLevel.SHORT_TERM)
        self.assertIsNotNone(entry.memory_id)
        self.assertIsInstance(entry.created_at, datetime)
        self.assertEqual(entry.token_count, 0)
        self.assertEqual(entry.relevance_score, 0.0)
        self.assertEqual(entry.access_count, 0)
        self.assertEqual(entry.source, MemorySource.USER_INPUT)
        self.assertIsNone(entry.embedding)
        self.assertEqual(entry.metadata, {})
        self.assertIsNone(entry.ttl)

    def test_creation_with_all_fields(self):
        now = datetime.now()
        entry = MemoryEntry(
            content="Test",
            level=MemoryLevel.LONG_TERM,
            memory_id="mem_custom_123",
            created_at=now,
            token_count=10,
            relevance_score=0.85,
            access_count=5,
            source=MemorySource.ASSISTANT_OUTPUT,
            embedding=[0.1, 0.2, 0.3],
            metadata={"task": "test"},
            ttl=3600.0,
        )
        self.assertEqual(entry.level, MemoryLevel.LONG_TERM)
        self.assertEqual(entry.memory_id, "mem_custom_123")
        self.assertEqual(entry.token_count, 10)
        self.assertEqual(entry.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(entry.ttl, 3600.0)

    def test_creation_with_string_level(self):
        entry = MemoryEntry(content="Test", level="instant")
        self.assertEqual(entry.level, MemoryLevel.INSTANT)

    def test_creation_with_string_source(self):
        entry = MemoryEntry(content="Test", source="system")
        self.assertEqual(entry.source, MemorySource.SYSTEM_INSTRUCTION)


class TestMemoryEntrySerialization(unittest.TestCase):
    """测试序列化"""

    def setUp(self):
        self.entry = MemoryEntry(
            content="Test content",
            level=MemoryLevel.LONG_TERM,
            memory_id="mem_test_123",
            token_count=5,
            relevance_score=0.9,
            embedding=[0.1, 0.2],
            metadata={"key": "value"},
            ttl=7200.0,
        )

    def test_to_dict(self):
        data = self.entry.to_dict()
        self.assertEqual(data["content"], "Test content")
        self.assertEqual(data["level"], "long")
        self.assertEqual(data["memory_id"], "mem_test_123")
        self.assertEqual(data["token_count"], 5)
        self.assertEqual(data["relevance_score"], 0.9)
        self.assertEqual(data["embedding"], [0.1, 0.2])
        self.assertEqual(data["metadata"], {"key": "value"})
        self.assertEqual(data["ttl"], 7200.0)
        self.assertIsInstance(data["created_at"], str)

    def test_to_json(self):
        json_str = self.entry.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["content"], "Test content")

    def test_from_dict(self):
        data = {
            "content": "From dict",
            "level": "short",
            "memory_id": "mem_dict_1",
            "token_count": 3,
            "relevance_score": 0.5,
            "source": "assistant",
            "metadata": {"test": True},
            "ttl": 1800.0,
        }
        entry = MemoryEntry.from_dict(data)
        self.assertEqual(entry.content, "From dict")
        self.assertEqual(entry.level, MemoryLevel.SHORT_TERM)
        self.assertEqual(entry.source, MemorySource.ASSISTANT_OUTPUT)

    def test_from_json(self):
        json_str = '{"content": "JSON test", "level": "instant", "memory_id": "mem_json_1"}'
        entry = MemoryEntry.from_json(json_str)
        self.assertEqual(entry.content, "JSON test")
        self.assertEqual(entry.level, MemoryLevel.INSTANT)

    def test_roundtrip(self):
        data = self.entry.to_dict()
        entry2 = MemoryEntry.from_dict(data)
        data2 = entry2.to_dict()
        self.assertEqual(data, data2)


class TestMemoryEntryValidation(unittest.TestCase):
    """测试验证"""

    def test_valid_entry(self):
        entry = MemoryEntry(content="Valid")
        valid, errors = entry.validate()
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_empty_content(self):
        entry = MemoryEntry(content="")
        valid, errors = entry.validate()
        self.assertFalse(valid)
        self.assertTrue(any("content不能为空" in e for e in errors))

    def test_negative_token_count(self):
        entry = MemoryEntry(content="Test", token_count=-1)
        valid, errors = entry.validate()
        self.assertFalse(valid)
        self.assertTrue(any("token_count不能为负数" in e for e in errors))

    def test_relevance_out_of_range(self):
        entry = MemoryEntry(content="Test", relevance_score=1.5)
        valid, errors = entry.validate()
        self.assertFalse(valid)
        self.assertTrue(any("relevance_score" in e for e in errors))

    def test_empty_embedding(self):
        entry = MemoryEntry(content="Test", embedding=[])
        valid, errors = entry.validate()
        self.assertFalse(valid)
        self.assertTrue(any("embedding不能为空列表" in e for e in errors))

    def test_negative_ttl(self):
        entry = MemoryEntry(content="Test", ttl=-1.0)
        valid, errors = entry.validate()
        self.assertFalse(valid)
        self.assertTrue(any("ttl不能为负数" in e for e in errors))

    def test_is_valid(self):
        self.assertTrue(MemoryEntry(content="OK").is_valid())
        self.assertFalse(MemoryEntry(content="").is_valid())


class TestMemoryEntryUtility(unittest.TestCase):
    """测试工具方法"""

    def test_copy(self):
        entry = MemoryEntry(content="Original", metadata={"k": "v"})
        copy = entry.copy()
        self.assertEqual(copy.content, entry.content)
        self.assertIsNot(copy, entry)
        copy.content = "Modified"
        self.assertEqual(entry.content, "Original")

    def test_touch(self):
        entry = MemoryEntry(content="Test")
        old_access = entry.accessed_at
        old_count = entry.access_count
        entry.touch()
        self.assertGreaterEqual(entry.accessed_at, old_access)
        self.assertEqual(entry.access_count, old_count + 1)

    def test_chain_calls(self):
        entry = MemoryEntry(content="Test")
        result = entry.update_relevance(0.8).update_token_count(10).add_metadata("k", "v")
        self.assertIs(result, entry)
        self.assertEqual(entry.relevance_score, 0.8)
        self.assertEqual(entry.token_count, 10)
        self.assertEqual(entry.metadata["k"], "v")

    def test_get_metadata(self):
        entry = MemoryEntry(content="Test", metadata={"a": 1})
        self.assertEqual(entry.get_metadata("a"), 1)
        self.assertIsNone(entry.get_metadata("b"))
        self.assertEqual(entry.get_metadata("b", "default"), "default")

    def test_is_expired(self):
        # 无TTL = 不过期
        entry = MemoryEntry(content="Test")
        self.assertFalse(entry.is_expired())

        # 有TTL但未过期
        entry2 = MemoryEntry(content="Test", ttl=3600.0)
        self.assertFalse(entry2.is_expired())

        # 已过期
        past = datetime.now() - timedelta(hours=2)
        entry3 = MemoryEntry(content="Test", ttl=3600.0)
        entry3.created_at = past
        self.assertTrue(entry3.is_expired())

    def test_get_age_seconds(self):
        past = datetime.now() - timedelta(seconds=10)
        entry = MemoryEntry(content="Test", created_at=past)
        age = entry.get_age_seconds()
        self.assertGreaterEqual(age, 10.0)
        self.assertLess(age, 11.0)

    def test_time_decay(self):
        entry = MemoryEntry(content="Test")
        # 刚创建的，衰减应接近1.0
        decay = entry.compute_time_decay(24.0)
        self.assertGreater(decay, 0.99)

        # 24小时前的，衰减应为0.5
        past = datetime.now() - timedelta(hours=24)
        entry2 = MemoryEntry(content="Test", created_at=past)
        decay2 = entry2.compute_time_decay(24.0)
        self.assertAlmostEqual(decay2, 0.5, places=1)

    def test_level_checks(self):
        instant = MemoryEntry(content="T", level=MemoryLevel.INSTANT)
        short = MemoryEntry(content="T", level=MemoryLevel.SHORT_TERM)
        long = MemoryEntry(content="T", level=MemoryLevel.LONG_TERM)
        self.assertTrue(instant.is_instant())
        self.assertTrue(short.is_short_term())
        self.assertTrue(long.is_long_term())
        self.assertFalse(instant.is_long_term())

    def test_has_embedding(self):
        self.assertFalse(MemoryEntry(content="T").has_embedding())
        self.assertFalse(MemoryEntry(content="T", embedding=[]).has_embedding())
        self.assertTrue(MemoryEntry(content="T", embedding=[0.1]).has_embedding())

    def test_equality(self):
        e1 = MemoryEntry(content="A", memory_id="same")
        e2 = MemoryEntry(content="B", memory_id="same")
        self.assertEqual(e1, e2)
        self.assertEqual(hash(e1), hash(e2))

        e3 = MemoryEntry(content="A", memory_id="different")
        self.assertNotEqual(e1, e3)


class TestMemoryEntryListUtils(unittest.TestCase):
    """测试列表工具函数"""

    def setUp(self):
        self.entries = [
            MemoryEntry(content="Instant", level=MemoryLevel.INSTANT, token_count=5),
            MemoryEntry(content="Short", level=MemoryLevel.SHORT_TERM, token_count=10),
            MemoryEntry(content="Long", level=MemoryLevel.LONG_TERM, token_count=15),
        ]

    def test_filter_by_level(self):
        result = filter_by_level(self.entries, MemoryLevel.SHORT_TERM)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "Short")

    def test_filter_valid(self):
        invalid = MemoryEntry(content="")
        all_entries = self.entries + [invalid]
        result = filter_valid(all_entries)
        self.assertEqual(len(result), 3)

    def test_filter_non_expired(self):
        expired = MemoryEntry(content="Old", ttl=1.0)
        expired.created_at = datetime.now() - timedelta(hours=1)
        all_entries = self.entries + [expired]
        result = filter_non_expired(all_entries)
        self.assertEqual(len(result), 3)

    def test_sort_by_relevance(self):
        e1 = MemoryEntry(content="A", relevance_score=0.3)
        e2 = MemoryEntry(content="B", relevance_score=0.9)
        e3 = MemoryEntry(content="C", relevance_score=0.5)
        sorted_entries = sort_by_relevance([e1, e2, e3])
        self.assertEqual(sorted_entries[0].relevance_score, 0.9)
        self.assertEqual(sorted_entries[-1].relevance_score, 0.3)

    def test_get_total_tokens(self):
        self.assertEqual(get_total_tokens(self.entries), 30)

    def test_entries_json_roundtrip(self):
        json_str = entries_to_json(self.entries)
        restored = entries_from_json(json_str)
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored[0].content, "Instant")

    def test_validate_entry_list(self):
        valid, errors = validate_entry_list(self.entries)
        self.assertTrue(valid)

        invalid_list = self.entries + [MemoryEntry(content="")]
        valid, errors = validate_entry_list(invalid_list)
        self.assertFalse(valid)


# =============================================================================
# InstantMemory 测试
# =============================================================================

class TestInstantMemory(unittest.TestCase):
    """测试瞬时记忆管理器"""

    def setUp(self):
        self.memory = InstantMemory()

    def test_properties(self):
        self.assertEqual(self.memory.level, MemoryLevel.INSTANT)
        self.assertEqual(self.memory.size, 0)
        self.assertEqual(self.memory.total_tokens, 0)

    def test_add_and_get(self):
        entry = MemoryEntry(content="Hello")
        added = self.memory.add(entry)
        self.assertEqual(added.level, MemoryLevel.INSTANT)
        self.assertEqual(self.memory.size, 1)

        retrieved = self.memory.get(added.memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Hello")

    def test_add_invalid_raises(self):
        entry = MemoryEntry(content="")
        with self.assertRaises(Exception):
            self.memory.add(entry)

    def test_get_nonexistent(self):
        self.assertIsNone(self.memory.get("nonexistent"))

    def test_update(self):
        entry = self.memory.add(MemoryEntry(content="Original"))
        updated = self.memory.update(entry.memory_id, content="Updated")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "Updated")

    def test_update_nonexistent(self):
        self.assertIsNone(self.memory.update("nonexistent", content="X"))

    def test_remove(self):
        entry = self.memory.add(MemoryEntry(content="To remove"))
        self.assertTrue(self.memory.remove(entry.memory_id))
        self.assertFalse(self.memory.contains(entry.memory_id))

    def test_clear(self):
        self.memory.add(MemoryEntry(content="A"))
        self.memory.add(MemoryEntry(content="B"))
        count = self.memory.clear()
        self.assertEqual(count, 2)
        self.assertEqual(self.memory.size, 0)

    def test_max_items_eviction(self):
        memory = InstantMemory(max_items=2)
        e1 = memory.add(MemoryEntry(content="First"))
        e2 = memory.add(MemoryEntry(content="Second"))
        e3 = memory.add(MemoryEntry(content="Third"))
        # 超过上限，最早的被淘汰
        self.assertEqual(memory.size, 2)
        self.assertFalse(memory.contains(e1.memory_id))
        self.assertTrue(memory.contains(e3.memory_id))

    def test_set_and_get_context(self):
        self.memory.set_context("goal", "Complete task")
        value = self.memory.get_context("goal")
        self.assertEqual(value, "Complete task")
        self.assertIsNone(self.memory.get_context("nonexistent"))

    def test_snapshot_and_restore(self):
        self.memory.add(MemoryEntry(content="A"))
        self.memory.add(MemoryEntry(content="B"))
        snapshot = self.memory.snapshot()
        self.memory.clear()
        self.assertEqual(self.memory.size, 0)
        count = self.memory.restore(snapshot)
        self.assertEqual(count, 2)

    def test_health_check(self):
        self.memory.add(MemoryEntry(content="OK"))
        health = self.memory.health_check()
        self.assertEqual(health["level"], "instant")
        self.assertEqual(health["size"], 1)
        self.assertEqual(health["valid_count"], 1)

    def test_purge_expired(self):
        expired = MemoryEntry(content="Old", ttl=1.0)
        expired.created_at = datetime.now() - timedelta(hours=1)
        self.memory.add(expired)
        self.memory.add(MemoryEntry(content="Fresh"))
        purged = self.memory.purge_expired()
        self.assertEqual(purged, 1)
        self.assertEqual(self.memory.size, 1)


# =============================================================================
# ShortTermMemory 测试
# =============================================================================

class TestShortTermMemory(unittest.TestCase):
    """测试短期记忆管理器"""

    def setUp(self):
        self.memory = ShortTermMemory(max_tokens=100, max_items=10)

    def test_properties(self):
        self.assertEqual(self.memory.level, MemoryLevel.SHORT_TERM)
        self.assertEqual(self.memory.max_tokens, 100)
        self.assertEqual(self.memory.max_items, 10)
        self.assertEqual(self.memory.available_tokens, 100)
        self.assertAlmostEqual(self.memory.token_usage_ratio, 0.0)

    def test_add_and_get(self):
        entry = MemoryEntry(content="Hello", token_count=10)
        added = self.memory.add(entry)
        self.assertEqual(added.level, MemoryLevel.SHORT_TERM)
        self.assertEqual(self.memory.size, 1)
        self.assertEqual(self.memory.total_tokens, 10)

    def test_lru_eviction_on_token_overflow(self):
        """测试Token超限时LRU淘汰"""
        m = ShortTermMemory(max_tokens=50, max_items=100)
        e1 = m.add(MemoryEntry(content="First", token_count=20))
        e2 = m.add(MemoryEntry(content="Second", token_count=10))
        # 当前30 tokens，访问e1使其变新
        m.get(e1.memory_id)
        # 添加e3(15 tokens)，当前30+15=45 ≤ 50，无需淘汰
        e3 = m.add(MemoryEntry(content="Third", token_count=15))
        self.assertTrue(m.contains(e1.memory_id))
        self.assertTrue(m.contains(e2.memory_id))
        self.assertTrue(m.contains(e3.memory_id))
        # 添加e4(20 tokens)，当前45+20=65 > 50，需要淘汰
        # LRU顺序：e2(最旧) → e1(被访问过) → e3(最新)
        # 淘汰e2(10) → 55 > 50 → 淘汰e1(20) → 35 ≤ 50
        e4 = m.add(MemoryEntry(content="Fourth", token_count=20))
        self.assertFalse(m.contains(e2.memory_id))
        self.assertTrue(m.contains(e3.memory_id))
        self.assertTrue(m.contains(e4.memory_id))

    def test_lru_eviction_on_item_overflow(self):
        """测试条目数超限时LRU淘汰"""
        m = ShortTermMemory(max_tokens=10000, max_items=3)
        e1 = m.add(MemoryEntry(content="A", token_count=1))
        e2 = m.add(MemoryEntry(content="B", token_count=1))
        e3 = m.add(MemoryEntry(content="C", token_count=1))
        # 访问e1使其变新
        m.get(e1.memory_id)
        # 添加e4，超过max_items=3
        e4 = m.add(MemoryEntry(content="D", token_count=1))
        self.assertEqual(m.size, 3)
        # e2应该被淘汰（最久未访问）
        self.assertFalse(m.contains(e2.memory_id))

    def test_get_updates_lru(self):
        """测试get()更新LRU顺序"""
        m = ShortTermMemory(max_tokens=100, max_items=2)
        e1 = m.add(MemoryEntry(content="A", token_count=10))
        e2 = m.add(MemoryEntry(content="B", token_count=10))
        # 访问e1使其变新
        m.get(e1.memory_id)
        # 添加e3，e2应该被淘汰
        e3 = m.add(MemoryEntry(content="C", token_count=10))
        self.assertFalse(m.contains(e2.memory_id))
        self.assertTrue(m.contains(e1.memory_id))

    def test_evict(self):
        m = ShortTermMemory(max_tokens=10000, max_items=10)
        for i in range(5):
            m.add(MemoryEntry(content=f"Entry {i}", token_count=1))
        evicted = m.evict(2)
        self.assertEqual(evicted, 2)
        self.assertEqual(m.size, 3)

    def test_evict_oldest(self):
        m = ShortTermMemory(max_tokens=10000, max_items=100)
        m.add(MemoryEntry(content="Big", token_count=50))
        m.add(MemoryEntry(content="Small", token_count=10))
        evicted = m.evict_oldest(30)
        self.assertEqual(evicted, 1)
        self.assertEqual(m.size, 1)

    def test_trim_to_tokens(self):
        m = ShortTermMemory(max_tokens=10000, max_items=100)
        m.add(MemoryEntry(content="A", token_count=30))
        m.add(MemoryEntry(content="B", token_count=20))
        m.add(MemoryEntry(content="C", token_count=10))
        evicted = m.trim_to_tokens(25)
        self.assertEqual(evicted, 2)  # 淘汰A(30)和B(20)，保留C(10)
        self.assertEqual(m.total_tokens, 10)

    def test_get_lru_and_most_recent(self):
        m = ShortTermMemory(max_tokens=10000, max_items=100)
        e1 = m.add(MemoryEntry(content="First", token_count=1))
        e2 = m.add(MemoryEntry(content="Second", token_count=1))
        e3 = m.add(MemoryEntry(content="Third", token_count=1))

        lru = m.get_lru(2)
        self.assertEqual(len(lru), 2)
        self.assertEqual(lru[0].memory_id, e1.memory_id)

        recent = m.get_most_recent(2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].memory_id, e3.memory_id)

    def test_update(self):
        entry = self.memory.add(MemoryEntry(content="Old", token_count=10))
        updated = self.memory.update(entry.memory_id, content="New")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "New")

    def test_remove(self):
        entry = self.memory.add(MemoryEntry(content="Remove me"))
        self.assertTrue(self.memory.remove(entry.memory_id))
        self.assertEqual(self.memory.size, 0)

    def test_clear(self):
        self.memory.add(MemoryEntry(content="A", token_count=10))
        self.memory.add(MemoryEntry(content="B", token_count=10))
        count = self.memory.clear()
        self.assertEqual(count, 2)

    def test_health_check(self):
        self.memory.add(MemoryEntry(content="OK", token_count=5))
        health = self.memory.health_check()
        self.assertEqual(health["level"], "short")
        self.assertEqual(health["size"], 1)
        self.assertEqual(health["total_tokens"], 5)


# =============================================================================
# LongTermMemory 测试
# =============================================================================

class TestRelevanceScorer(unittest.TestCase):
    """测试四维相关性评分器"""

    def setUp(self):
        self.scorer = RelevanceScorer(
            semantic_weight=0.5,
            time_weight=0.25,
            task_weight=0.15,
            behavior_weight=0.10,
            time_halflife_hours=24.0,
        )

    def test_perfect_score(self):
        entry = MemoryEntry(content="Test", created_at=datetime.now())
        score = self.scorer.score(
            entry=entry,
            semantic_score=1.0,
            task_tags=["task1"],
            behavior_boost=1.0,
        )
        # 新条目 + 完美语义 + 行为满分
        # task维度取决于entry的tags
        self.assertGreater(score, 0.5)

    def test_zero_score(self):
        entry = MemoryEntry(content="Test", created_at=datetime.now())
        score = self.scorer.score(
            entry=entry,
            semantic_score=0.0,
            behavior_boost=0.0,
        )
        # 只有时间维度有分
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_task_match(self):
        entry = MemoryEntry(
            content="Test",
            metadata={"task_tags": ["coding", "python"]}
        )
        score_with_match = self.scorer.score(
            entry=entry,
            semantic_score=0.5,
            task_tags=["coding"],
        )
        score_without_match = self.scorer.score(
            entry=entry,
            semantic_score=0.5,
            task_tags=["cooking"],
        )
        self.assertGreater(score_with_match, score_without_match)


class TestLongTermMemory(unittest.TestCase):
    """测试长期记忆管理器"""

    def setUp(self):
        self.memory = LongTermMemory(top_k=5, similarity_threshold=0.0)

    def test_properties(self):
        self.assertEqual(self.memory.level, MemoryLevel.LONG_TERM)
        self.assertEqual(self.memory.size, 0)

    def test_add_and_get(self):
        entry = MemoryEntry(content="Python basics", level=MemoryLevel.LONG_TERM)
        added = self.memory.add(entry)
        self.assertEqual(added.level, MemoryLevel.LONG_TERM)

        retrieved = self.memory.get(added.memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Python basics")

    def test_recall_by_text(self):
        """测试基于文本的召回"""
        self.memory.add(MemoryEntry(content="Python programming language"))
        self.memory.add(MemoryEntry(content="Cooking Italian recipes"))
        self.memory.add(MemoryEntry(content="Machine learning with Python"))

        result = self.memory.recall("Python tutorial")
        # "Python" 应该匹配到包含 Python 的条目
        self.assertGreater(result.count, 0)

    def test_recall_with_embedding(self):
        """测试基于向量的召回"""
        self.memory.add(MemoryEntry(
            content="AI research",
            embedding=[1.0, 0.0, 0.0],
        ))
        self.memory.add(MemoryEntry(
            content="Cooking",
            embedding=[0.0, 1.0, 0.0],
        ))
        self.memory.add(MemoryEntry(
            content="Sports",
            embedding=[0.0, 0.0, 1.0],
        ))

        result = self.memory.recall(
            "AI",
            query_embedding=[0.9, 0.1, 0.0],
            similarity_threshold=0.5,
        )
        self.assertGreaterEqual(result.count, 1)
        self.assertEqual(result.recalled[0].content, "AI research")

    def test_recall_top_k(self):
        """测试Top-K限制"""
        for i in range(10):
            self.memory.add(MemoryEntry(content=f"Entry {i} about Python"))
        result = self.memory.recall("Python", top_k=3)
        self.assertLessEqual(result.count, 3)

    def test_recall_expired_filtered(self):
        """测试过期条目不参与召回"""
        expired = MemoryEntry(content="Python old", ttl=1.0)
        expired.created_at = datetime.now() - timedelta(hours=1)
        self.memory.add(expired)
        self.memory.add(MemoryEntry(content="Python new"))

        result = self.memory.recall("Python")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.recalled[0].content, "Python new")

    def test_recall_by_tags(self):
        """测试按标签召回"""
        self.memory.add(MemoryEntry(
            content="Task A result",
            metadata={"task_tags": ["coding", "python"]},
        ))
        self.memory.add(MemoryEntry(
            content="Task B result",
            metadata={"task_tags": ["writing", "english"]},
        ))

        result = self.memory.recall_by_tags(["coding"])
        self.assertEqual(result.count, 1)
        self.assertEqual(result.recalled[0].content, "Task A result")

    def test_update_invalidates_embedding(self):
        """测试更新内容后embedding失效"""
        entry = self.memory.add(MemoryEntry(
            content="Original",
            embedding=[0.1, 0.2, 0.3],
        ))
        self.memory.update(entry.memory_id, content="Updated")
        updated = self.memory.get(entry.memory_id)
        self.assertIsNone(updated.embedding)

    def test_remove(self):
        entry = self.memory.add(MemoryEntry(content="Remove me"))
        self.assertTrue(self.memory.remove(entry.memory_id))
        self.assertEqual(self.memory.size, 0)

    def test_clear(self):
        self.memory.add(MemoryEntry(content="A"))
        self.memory.add(MemoryEntry(content="B"))
        count = self.memory.clear()
        self.assertEqual(count, 2)

    def test_health_check(self):
        self.memory.add(MemoryEntry(content="OK"))
        health = self.memory.health_check()
        self.assertEqual(health["level"], "long")
        self.assertEqual(health["size"], 1)

    def test_cosine_similarity(self):
        # 相同向量
        sim = LongTermMemory._cosine_similarity([1, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(sim, 1.0)

        # 正交向量
        sim = LongTermMemory._cosine_similarity([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(sim, 0.0)

        # 相似向量
        sim = LongTermMemory._cosine_similarity([1, 0, 0], [0.9, 0.1, 0])
        self.assertGreater(sim, 0.9)

        # 空向量
        sim = LongTermMemory._cosine_similarity([], [])
        self.assertEqual(sim, 0.0)

        # 不同长度
        sim = LongTermMemory._cosine_similarity([1, 0], [1, 0, 0])
        self.assertEqual(sim, 0.0)

    def test_text_similarity(self):
        sim = LongTermMemory._text_similarity("python code", "write python code here")
        self.assertGreater(sim, 0.0)

        sim = LongTermMemory._text_similarity("python", "java programming")
        self.assertEqual(sim, 0.0)

        sim = LongTermMemory._text_similarity("", "content")
        self.assertEqual(sim, 0.0)


# =============================================================================
# RecallResult 测试
# =============================================================================

class TestRecallResult(unittest.TestCase):
    """测试召回结果"""

    def test_empty_result(self):
        result = RecallResult(query="test")
        self.assertEqual(result.count, 0)
        self.assertEqual(result.total_tokens, 0)

    def test_with_entries(self):
        entries = [
            MemoryEntry(content="A", token_count=10),
            MemoryEntry(content="B", token_count=20),
        ]
        result = RecallResult(query="test", recalled=entries, total_count=5)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.total_tokens, 30)
        self.assertEqual(result.total_count, 5)


# =============================================================================
# MemoryFactory 测试
# =============================================================================

class TestMemoryFactory(unittest.TestCase):
    """测试记忆工厂"""

    def setUp(self):
        self.factory = MemoryFactory()

    def test_create_instant(self):
        memory = self.factory.create_instant()
        self.assertIsInstance(memory, InstantMemory)
        self.assertEqual(memory.level, MemoryLevel.INSTANT)

    def test_create_short_term(self):
        memory = self.factory.create_short_term()
        self.assertIsInstance(memory, ShortTermMemory)
        self.assertEqual(memory.level, MemoryLevel.SHORT_TERM)

    def test_create_long_term(self):
        memory = self.factory.create_long_term()
        self.assertIsInstance(memory, LongTermMemory)
        self.assertEqual(memory.level, MemoryLevel.LONG_TERM)

    def test_create_by_level(self):
        instant = self.factory.create_by_level(MemoryLevel.INSTANT)
        short = self.factory.create_by_level(MemoryLevel.SHORT_TERM)
        long = self.factory.create_by_level(MemoryLevel.LONG_TERM)
        self.assertIsInstance(instant, InstantMemory)
        self.assertIsInstance(short, ShortTermMemory)
        self.assertIsInstance(long, LongTermMemory)

    def test_create_all(self):
        all_memories = self.factory.create_all()
        self.assertEqual(len(all_memories), 3)
        self.assertIn(MemoryLevel.INSTANT, all_memories)
        self.assertIn(MemoryLevel.SHORT_TERM, all_memories)
        self.assertIn(MemoryLevel.LONG_TERM, all_memories)

    def test_custom_config(self):
        factory = MemoryFactory(
            short_term_max_tokens=500,
            short_term_max_items=20,
        )
        memory = factory.create_short_term()
        self.assertEqual(memory.max_tokens, 500)
        self.assertEqual(memory.max_items, 20)


# =============================================================================
# 集成测试
# =============================================================================

class TestMemoryIntegration(unittest.TestCase):
    """三层记忆集成测试"""

    def test_three_layer_workflow(self):
        """模拟完整的三层记忆工作流"""
        factory = MemoryFactory()
        instant = factory.create_instant()
        short_term = factory.create_short_term(max_tokens=200, max_items=10)
        long_term = factory.create_long_term(top_k=3, similarity_threshold=0.0)

        # 1. 推理开始：设置瞬时上下文
        instant.set_context("current_goal", "Help user with Python")

        # 2. 用户输入 → 短期记忆
        msg1 = MemoryEntry(content="How do I read a file in Python?", token_count=10)
        short_term.add(msg1)

        # 3. 助手回复 → 短期记忆
        msg2 = MemoryEntry(
            content="Use open() function with 'r' mode",
            token_count=8,
            source=MemorySource.ASSISTANT_OUTPUT,
        )
        short_term.add(msg2)

        # 4. 重要信息 → 长期记忆
        important = MemoryEntry(
            content="User is learning Python, prefers code examples",
            level=MemoryLevel.LONG_TERM,
            metadata={"task_tags": ["python", "learning"]},
            token_count=12,
        )
        long_term.add(important)

        # 5. 验证各层状态
        self.assertEqual(instant.size, 1)
        self.assertEqual(short_term.size, 2)
        self.assertEqual(long_term.size, 1)

        # 6. 召回长期记忆
        result = long_term.recall("Python learning")
        self.assertGreater(result.count, 0)

        # 7. 推理结束：清空瞬时记忆
        instant.clear()
        self.assertEqual(instant.size, 0)

        # 8. 短期记忆仍在
        self.assertEqual(short_term.size, 2)

        # 9. 长期记忆仍在
        self.assertEqual(long_term.size, 1)

    def test_message_promotion_workflow(self):
        """测试消息从短期到长期的晋升流程"""
        short = ShortTermMemory(max_tokens=1000, max_items=100)
        long = LongTermMemory(top_k=5, similarity_threshold=0.0)

        # 添加多条短期记忆
        for i in range(10):
            short.add(MemoryEntry(
                content=f"Conversation turn {i}",
                token_count=5,
            ))

        # 模拟：将重要的短期记忆晋升为长期记忆
        important_entries = short.get_most_recent(3)
        for entry in important_entries:
            promoted = entry.copy()
            promoted.level = MemoryLevel.LONG_TERM
            long.add(promoted)

        self.assertEqual(long.size, 3)

        # 短期记忆淘汰后，长期记忆仍保留
        short.trim_to_tokens(20)
        self.assertLess(short.size, 10)
        self.assertEqual(long.size, 3)


if __name__ == "__main__":
    unittest.main()
