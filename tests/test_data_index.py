"""阶段63: Agent数据索引系统 - 单元测试"""
import pytest
from src.agent.data_index import (
    InvertedIndex, BTreeIndex, CompositeIndex, IndexManager, create_index_manager,
)

class TestInvertedIndex:
    def test_add_search(self):
        idx = InvertedIndex("test")
        idx.add("apple", "doc1")
        idx.add("apple", "doc2")
        assert idx.search("apple") == {"doc1", "doc2"}

    def test_remove(self):
        idx = InvertedIndex("test")
        idx.add("apple", "doc1")
        idx.remove("apple", "doc1")
        assert idx.search("apple") == set()

    def test_search_multi(self):
        idx = InvertedIndex("test")
        idx.add("a", "d1")
        idx.add("a", "d2")
        idx.add("b", "d2")
        idx.add("b", "d3")
        assert idx.search_multi(["a", "b"], match_all=True) == {"d2"}
        assert idx.search_multi(["a", "b"], match_all=False) == {"d1", "d2", "d3"}

    def test_stats(self):
        idx = InvertedIndex("test")
        idx.add("a", "d1")
        stats = idx.get_stats()
        assert stats["terms"] == 1
        assert stats["total_entries"] == 1

class TestBTreeIndex:
    def test_add_search(self):
        idx = BTreeIndex("test")
        idx.add(10, "doc1")
        idx.add(20, "doc2")
        assert idx.search(10) == {"doc1"}

    def test_range_search(self):
        idx = BTreeIndex("test")
        idx.add(1, "d1")
        idx.add(5, "d2")
        idx.add(10, "d3")
        result = idx.range_search(2, 8)
        assert 5 in result
        assert 1 not in result
        assert 10 not in result

    def test_remove(self):
        idx = BTreeIndex("test")
        idx.add(10, "doc1")
        idx.remove(10, "doc1")
        assert idx.search(10) == set()

class TestCompositeIndex:
    def test_add_search(self):
        idx = CompositeIndex("test", ["city", "age"])
        idx.add(("beijing", 30), "u1")
        idx.add(("beijing", 30), "u2")
        assert idx.search(("beijing", 30)) == {"u1", "u2"}

    def test_prefix_search(self):
        idx = CompositeIndex("test", ["city", "age"])
        idx.add(("beijing", 30), "u1")
        idx.add(("beijing", 40), "u2")
        idx.add(("shanghai", 30), "u3")
        result = idx.prefix_search(("beijing",))
        assert result == {"u1", "u2"}

class TestIndexManager:
    def test_create_inverted(self):
        m = IndexManager()
        idx = m.create_inverted_index("inv")
        assert isinstance(idx, InvertedIndex)

    def test_create_btree(self):
        m = IndexManager()
        idx = m.create_btree_index("bt")
        assert isinstance(idx, BTreeIndex)

    def test_create_composite(self):
        m = IndexManager()
        idx = m.create_composite_index("comp", ["a", "b"])
        assert isinstance(idx, CompositeIndex)

    def test_create_index_manager(self):
        m = create_index_manager()
        assert isinstance(m, IndexManager)
