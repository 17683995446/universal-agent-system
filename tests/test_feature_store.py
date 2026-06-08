"""阶段49: Agent特征标记系统 - 单元测试"""
import pytest
from src.agent.feature_store import (
    Feature, FeatureSet, FeatureStore, create_feature_store,
)

class TestFeatureSet:
    def test_set_get_feature(self):
        fs = FeatureSet(entity_id="e1")
        fs.set_feature("name", "Alice")
        assert fs.get_feature("name") == "Alice"

    def test_remove_feature(self):
        fs = FeatureSet(entity_id="e1")
        fs.set_feature("name", "Alice")
        assert fs.remove_feature("name")
        assert fs.get_feature("name") is None

    def test_tags(self):
        fs = FeatureSet(entity_id="e1")
        fs.add_tag("vip")
        fs.add_tag("active")
        assert fs.has_tag("vip")
        assert fs.has_all_tags(["vip", "active"])
        assert not fs.has_any_tag(["banned"])

    def test_to_dict(self):
        fs = FeatureSet(entity_id="e1")
        fs.set_feature("name", "Alice")
        fs.add_tag("vip")
        d = fs.to_dict()
        assert d["features"]["name"] == "Alice"
        assert "vip" in d["tags"]

class TestFeatureStore:
    def test_set_and_get(self):
        store = FeatureStore()
        store.set_feature("e1", "name", "Alice")
        assert store.get_feature("e1", "name") == "Alice"

    def test_tags(self):
        store = FeatureStore()
        store.add_tag("e1", "vip")
        store.add_tag("e2", "vip")
        store.add_tag("e2", "active")
        assert set(store.query_by_tag("vip")) == {"e1", "e2"}

    def test_query_by_tags_all(self):
        store = FeatureStore()
        store.add_tag("e1", "vip")
        store.add_tag("e1", "active")
        store.add_tag("e2", "vip")
        result = store.query_by_tags(["vip", "active"], match_all=True)
        assert result == ["e1"]

    def test_query_by_tags_any(self):
        store = FeatureStore()
        store.add_tag("e1", "vip")
        store.add_tag("e2", "active")
        result = store.query_by_tags(["vip", "active"], match_all=False)
        assert set(result) == {"e1", "e2"}

    def test_query_by_feature(self):
        store = FeatureStore()
        store.set_feature("e1", "role", "admin")
        store.set_feature("e2", "role", "user")
        result = store.query_by_feature("role", "admin")
        assert result == ["e1"]

    def test_delete_entity(self):
        store = FeatureStore()
        store.set_feature("e1", "name", "Alice")
        store.add_tag("e1", "vip")
        assert store.delete_entity("e1")
        assert store.get("e1") is None
        assert store.query_by_tag("vip") == []

    def test_stats(self):
        store = FeatureStore()
        store.set_feature("e1", "name", "A")
        store.add_tag("e1", "vip")
        stats = store.get_stats()
        assert stats["total_entities"] == 1
        assert stats["unique_tags"] == 1

    def test_create_feature_store(self):
        s = create_feature_store()
        assert isinstance(s, FeatureStore)
