"""阶段79: Agent推荐引擎系统 - 单元测试"""
import pytest
from src.agent.recommendation import (
    User, Item, Interaction,
    UserProfile, CollaborativeFiltering, ContentBasedRecommender,
    RecommendationEngine, create_recommendation_engine,
)


class TestUserProfile:
    def test_update_from_interaction(self):
        p = UserProfile("u1")
        item = Item(item_id="i1", features={"action": 1.0, "sci-fi": 0.5}, category="movie")
        p.update_from_interaction(item, 5.0)
        assert p.interaction_count == 1
        assert p.preferences["action"] == 5.0
        assert p.category_prefs["movie"] == 5.0

    def test_get_top_preferences(self):
        p = UserProfile("u1")
        item = Item(item_id="i1", features={"action": 1.0, "drama": 0.5})
        p.update_from_interaction(item, 1.0)
        prefs = p.get_top_preferences(2)
        assert len(prefs) == 2
        assert prefs[0][0] == "action"


class TestCollaborativeFiltering:
    def test_add_interaction(self):
        cf = CollaborativeFiltering()
        cf.add_interaction("u1", "i1", 5.0)
        assert "i1" in cf._user_items["u1"]
        assert "u1" in cf._item_users["i1"]
        assert cf._user_ratings["u1"]["i1"] == 5.0

    def test_get_similar_users(self):
        cf = CollaborativeFiltering()
        cf.add_interaction("u1", "i1")
        cf.add_interaction("u1", "i2")
        cf.add_interaction("u2", "i1")
        cf.add_interaction("u2", "i2")
        cf.add_interaction("u3", "i3")
        similar = cf.get_similar_users("u1", top_k=2)
        assert len(similar) == 1
        assert similar[0][0] == "u2"

    def test_recommend(self):
        cf = CollaborativeFiltering()
        cf.add_interaction("u1", "i1")
        cf.add_interaction("u1", "i2")
        cf.add_interaction("u2", "i1")
        cf.add_interaction("u2", "i2")
        cf.add_interaction("u2", "i3")  # u2 有 i3，u1 没有
        recs = cf.recommend("u1", top_k=1)
        assert len(recs) == 1
        assert recs[0][0] == "i3"


class TestContentBasedRecommender:
    def test_add_and_recommend(self):
        rec = ContentBasedRecommender()
        rec.add_item(Item(item_id="i1", features={"action": 1.0}))
        rec.add_item(Item(item_id="i2", features={"action": 0.8, "drama": 0.5}))
        rec.record_interaction("u1", "i1", 5.0)
        recs = rec.recommend("u1", top_k=2)
        assert len(recs) == 2
        # i1 自身相似度最高（完全相同），i2 次之
        assert recs[0][0] == "i1"
        assert recs[1][0] == "i2"

    def test_compute_similarity(self):
        rec = ContentBasedRecommender()
        prefs = {"action": 1.0, "drama": 0.5}
        features = {"action": 1.0, "drama": 0.5}
        score = rec._compute_similarity(prefs, features)
        assert score > 0.9  # 几乎相同


class TestRecommendationEngine:
    def test_add_user_and_item(self):
        engine = RecommendationEngine()
        user = engine.add_user("u1", {"age": 25})
        item = engine.add_item("i1", {"action": 1.0}, category="movie")
        assert engine._users["u1"] == user
        assert engine._items["i1"] == item

    def test_record_interaction(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_item("i1", category="movie")
        assert engine.record_interaction("u1", "i1", 5.0)
        assert "i1" in engine._users["u1"].history
        assert engine._items["i1"].popularity == 1
        assert not engine.record_interaction("u1", "nonexistent")

    def test_recommend_cf(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_user("u2")
        engine.add_item("i1")
        engine.add_item("i2")
        engine.add_item("i3")
        engine.record_interaction("u1", "i1")
        engine.record_interaction("u1", "i2")
        engine.record_interaction("u2", "i1")
        engine.record_interaction("u2", "i2")
        engine.record_interaction("u2", "i3")
        recs = engine.recommend("u1", method="cf", top_k=1)
        assert len(recs) == 1

    def test_recommend_content(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_item("i1", features={"action": 1.0})
        engine.add_item("i2", features={"action": 0.8})
        engine.record_interaction("u1", "i1", 5.0)
        recs = engine.recommend("u1", method="content", top_k=1)
        assert len(recs) == 1

    def test_recommend_hybrid(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_user("u2")
        engine.add_item("i1", features={"action": 1.0})
        engine.add_item("i2", features={"action": 0.8})
        engine.add_item("i3", features={"drama": 1.0})
        engine.record_interaction("u1", "i1")
        engine.record_interaction("u2", "i1")
        engine.record_interaction("u2", "i2")
        recs = engine.recommend("u1", method="hybrid", top_k=2)
        assert len(recs) <= 2

    def test_get_popular_items(self):
        engine = RecommendationEngine()
        engine.add_item("i1")
        engine.add_item("i2")
        engine.add_user("u1")
        engine.record_interaction("u1", "i1")
        engine.record_interaction("u1", "i1")  # i1 被交互两次
        engine.record_interaction("u1", "i2")
        popular = engine.get_popular_items(top_k=2)
        assert len(popular) == 2
        assert popular[0][0] == "i1"

    def test_get_similar_items(self):
        engine = RecommendationEngine()
        engine.add_item("i1", features={"action": 1.0, "drama": 0.5})
        engine.add_item("i2", features={"action": 0.9, "drama": 0.4})
        engine.add_item("i3", features={"comedy": 1.0})
        similar = engine.get_similar_items("i1", top_k=2)
        assert len(similar) >= 1
        assert similar[0][0] == "i2"

    def test_get_user_profile(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_item("i1", features={"action": 1.0}, category="movie")
        engine.record_interaction("u1", "i1", 5.0)
        profile = engine.get_user_profile("u1")
        assert profile is not None
        assert profile["interaction_count"] == 1

    def test_get_stats(self):
        engine = RecommendationEngine()
        engine.add_user("u1")
        engine.add_item("i1")
        engine.record_interaction("u1", "i1")
        stats = engine.get_stats()
        assert stats["user_count"] == 1
        assert stats["item_count"] == 1
        assert stats["interaction_count"] == 1

    def test_create_recommendation_engine(self):
        engine = create_recommendation_engine()
        assert isinstance(engine, RecommendationEngine)
