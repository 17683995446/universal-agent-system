"""
阶段79: Agent推荐引擎系统

遵循"图难于其易，为大于其细"的原则，实现用户画像、物品特征、协同过滤、推荐排序。
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class User:
    user_id: str
    features: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # 交互过的物品ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "feature_count": len(self.features),
            "history_count": len(self.history),
        }


@dataclass
class Item:
    item_id: str
    features: Dict[str, float] = field(default_factory=dict)
    category: str = ""
    popularity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "popularity": self.popularity,
        }


@dataclass
class Interaction:
    user_id: str
    item_id: str
    rating: float = 1.0
    timestamp: float = 0.0


class UserProfile:
    """用户画像"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences: Dict[str, float] = defaultdict(float)
        self.category_prefs: Dict[str, float] = defaultdict(float)
        self.interaction_count = 0

    def update_from_interaction(self, item: Item, rating: float = 1.0) -> None:
        self.interaction_count += 1
        # 更新物品特征偏好
        for feature, value in item.features.items():
            self.preferences[feature] += rating * value
        # 更新类别偏好
        if item.category:
            self.category_prefs[item.category] += rating

    def get_top_preferences(self, top_k: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.preferences.items(), key=lambda x: -x[1])[:top_k]


class CollaborativeFiltering:
    """协同过滤"""

    def __init__(self):
        self._user_items: Dict[str, Set[str]] = defaultdict(set)
        self._item_users: Dict[str, Set[str]] = defaultdict(set)
        self._user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_interaction(self, user_id: str, item_id: str, rating: float = 1.0) -> None:
        self._user_items[user_id].add(item_id)
        self._item_users[item_id].add(user_id)
        self._user_ratings[user_id][item_id] = rating

    def get_similar_users(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """基于Jaccard相似度找相似用户"""
        if user_id not in self._user_items:
            return []
        target_items = self._user_items[user_id]
        similarities = []
        for other_id, other_items in self._user_items.items():
            if other_id == user_id:
                continue
            intersection = len(target_items & other_items)
            union = len(target_items | other_items)
            if union > 0:
                similarity = intersection / union
                if similarity > 0:
                    similarities.append((other_id, similarity))
        return sorted(similarities, key=lambda x: -x[1])[:top_k]

    def recommend(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """基于用户的协同过滤推荐"""
        if user_id not in self._user_items:
            return []
        similar_users = self.get_similar_users(user_id, top_k=10)
        scores: Dict[str, float] = defaultdict(float)
        for similar_user, similarity in similar_users:
            for item_id in self._user_items[similar_user]:
                if item_id not in self._user_items[user_id]:
                    rating = self._user_ratings[similar_user].get(item_id, 1.0)
                    scores[item_id] += similarity * rating
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


class ContentBasedRecommender:
    """基于内容的推荐"""

    def __init__(self):
        self._items: Dict[str, Item] = {}
        self._user_profiles: Dict[str, UserProfile] = {}

    def add_item(self, item: Item) -> None:
        self._items[item.item_id] = item

    def get_user_profile(self, user_id: str) -> UserProfile:
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserProfile(user_id)
        return self._user_profiles[user_id]

    def record_interaction(self, user_id: str, item_id: str, rating: float = 1.0) -> None:
        item = self._items.get(item_id)
        if not item:
            return
        profile = self.get_user_profile(user_id)
        profile.update_from_interaction(item, rating)

    def recommend(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        profile = self._user_profiles.get(user_id)
        if not profile:
            return []
        scores = []
        for item_id, item in self._items.items():
            # 计算用户画像与物品特征的相似度
            score = self._compute_similarity(profile.preferences, item.features)
            scores.append((item_id, score))
        return sorted(scores, key=lambda x: -x[1])[:top_k]

    def _compute_similarity(self, prefs: Dict[str, float], features: Dict[str, float]) -> float:
        """计算余弦相似度"""
        if not prefs or not features:
            return 0.0
        dot_product = sum(prefs.get(f, 0) * v for f, v in features.items())
        norm_prefs = math.sqrt(sum(v ** 2 for v in prefs.values()))
        norm_features = math.sqrt(sum(v ** 2 for v in features.values()))
        if norm_prefs == 0 or norm_features == 0:
            return 0.0
        return dot_product / (norm_prefs * norm_features)


class RecommendationEngine:
    """推荐引擎"""

    def __init__(self):
        self._cf = CollaborativeFiltering()
        self._content = ContentBasedRecommender()
        self._items: Dict[str, Item] = {}
        self._users: Dict[str, User] = {}

    def add_user(self, user_id: str, features: Optional[Dict[str, float]] = None) -> User:
        user = User(user_id=user_id, features=features or {})
        self._users[user_id] = user
        return user

    def add_item(self, item_id: str, features: Optional[Dict[str, float]] = None,
                 category: str = "") -> Item:
        item = Item(item_id=item_id, features=features or {}, category=category)
        self._items[item_id] = item
        self._content.add_item(item)
        return item

    def record_interaction(self, user_id: str, item_id: str, rating: float = 1.0) -> bool:
        if user_id not in self._users or item_id not in self._items:
            return False
        self._cf.add_interaction(user_id, item_id, rating)
        self._content.record_interaction(user_id, item_id, rating)
        self._users[user_id].history.append(item_id)
        self._items[item_id].popularity += 1
        return True

    def recommend(self, user_id: str, top_k: int = 10, method: str = "hybrid") -> List[Tuple[str, float]]:
        if method == "cf":
            return self._cf.recommend(user_id, top_k)
        elif method == "content":
            return self._content.recommend(user_id, top_k)
        elif method == "hybrid":
            cf_results = dict(self._cf.recommend(user_id, top_k * 2))
            content_results = dict(self._content.recommend(user_id, top_k * 2))
            # 混合排序
            all_items = set(cf_results.keys()) | set(content_results.keys())
            merged = []
            for item_id in all_items:
                cf_score = cf_results.get(item_id, 0)
                content_score = content_results.get(item_id, 0)
                # 加权融合
                final_score = cf_score * 0.6 + content_score * 0.4
                # 加入流行度因子
                item = self._items.get(item_id)
                if item:
                    final_score *= (1 + math.log(1 + item.popularity) / 10)
                merged.append((item_id, final_score))
            return sorted(merged, key=lambda x: -x[1])[:top_k]
        return []

    def get_popular_items(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """获取热门物品"""
        items = [(item_id, item.popularity) for item_id, item in self._items.items()]
        return sorted(items, key=lambda x: -x[1])[:top_k]

    def get_similar_items(self, item_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """基于特征相似度找相似物品"""
        target = self._items.get(item_id)
        if not target:
            return []
        similarities = []
        for other_id, other in self._items.items():
            if other_id == item_id:
                continue
            score = self._content._compute_similarity(target.features, other.features)
            if score > 0:
                similarities.append((other_id, score))
        return sorted(similarities, key=lambda x: -x[1])[:top_k]

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        profile = self._content._user_profiles.get(user_id)
        if not profile:
            return None
        return {
            "user_id": user_id,
            "interaction_count": profile.interaction_count,
            "top_preferences": profile.get_top_preferences(5),
            "category_preferences": sorted(profile.category_prefs.items(), key=lambda x: -x[1])[:5],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "user_count": len(self._users),
            "item_count": len(self._items),
            "interaction_count": sum(len(u.history) for u in self._users.values()),
        }


def create_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine()
