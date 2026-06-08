"""
阶段49: Agent特征标记系统

遵循"图难于其易，为大于其细"的原则，实现特征定义、标记、查询、聚合统计。
"""
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

@dataclass
class Feature:
    name: str
    value: Any = None
    feature_type: str = "string"  # string, number, boolean, list
    description: str = ""
    created_at: float = field(default_factory=time.time)

@dataclass
class FeatureSet:
    entity_id: str
    features: Dict[str, Feature] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    updated_at: float = field(default_factory=time.time)

    def set_feature(self, name: str, value: Any, ftype: str = "string",
                    description: str = "") -> None:
        self.features[name] = Feature(name=name, value=value, feature_type=ftype,
                                       description=description)
        self.updated_at = time.time()

    def get_feature(self, name: str, default: Any = None) -> Any:
        f = self.features.get(name)
        return f.value if f else default

    def remove_feature(self, name: str) -> bool:
        if name in self.features:
            del self.features[name]
            self.updated_at = time.time()
            return True
        return False

    def add_tag(self, tag: str) -> None:
        self.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        self.tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def has_all_tags(self, tags: List[str]) -> bool:
        return all(t in self.tags for t in tags)

    def has_any_tag(self, tags: List[str]) -> bool:
        return any(t in self.tags for t in tags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "features": {k: v.value for k, v in self.features.items()},
            "tags": list(self.tags),
            "updated_at": self.updated_at,
        }

class FeatureStore:
    """特征存储"""
    def __init__(self):
        self._sets: Dict[str, FeatureSet] = {}
        self._feature_index: Dict[str, Set[str]] = defaultdict(set)  # feature_name -> entity_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> entity_ids
        self._lock = threading.RLock()

    def get_or_create(self, entity_id: str) -> FeatureSet:
        with self._lock:
            if entity_id not in self._sets:
                self._sets[entity_id] = FeatureSet(entity_id=entity_id)
            return self._sets[entity_id]

    def get(self, entity_id: str) -> Optional[FeatureSet]:
        return self._sets.get(entity_id)

    def set_feature(self, entity_id: str, name: str, value: Any,
                    ftype: str = "string") -> None:
        with self._lock:
            fs = self.get_or_create(entity_id)
            fs.set_feature(name, value, ftype)
            self._feature_index[name].add(entity_id)

    def get_feature(self, entity_id: str, name: str, default: Any = None) -> Any:
        fs = self._sets.get(entity_id)
        return fs.get_feature(name, default) if fs else default

    def add_tag(self, entity_id: str, tag: str) -> None:
        with self._lock:
            fs = self.get_or_create(entity_id)
            fs.add_tag(tag)
            self._tag_index[tag].add(entity_id)

    def remove_tag(self, entity_id: str, tag: str) -> None:
        with self._lock:
            fs = self._sets.get(entity_id)
            if fs:
                fs.remove_tag(tag)
                self._tag_index[tag].discard(entity_id)

    def query_by_tag(self, tag: str) -> List[str]:
        with self._lock:
            return list(self._tag_index.get(tag, set()))

    def query_by_tags(self, tags: List[str], match_all: bool = True) -> List[str]:
        with self._lock:
            if not tags:
                return list(self._sets.keys())
            if match_all:
                result = None
                for tag in tags:
                    ids = self._tag_index.get(tag, set())
                    result = set(ids) if result is None else result & ids
                return list(result) if result else []
            else:
                result = set()
                for tag in tags:
                    result.update(self._tag_index.get(tag, set()))
                return list(result)

    def query_by_feature(self, name: str, value: Any) -> List[str]:
        with self._lock:
            candidates = self._feature_index.get(name, set())
            return [eid for eid in candidates
                    if self._sets[eid].get_feature(name) == value]

    def get_all_entity_ids(self) -> List[str]:
        return list(self._sets.keys())

    def delete_entity(self, entity_id: str) -> bool:
        with self._lock:
            fs = self._sets.pop(entity_id, None)
            if not fs:
                return False
            for name in fs.features:
                self._feature_index[name].discard(entity_id)
            for tag in fs.tags:
                self._tag_index[tag].discard(entity_id)
            return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_entities": len(self._sets),
                "total_features": sum(len(fs.features) for fs in self._sets.values()),
                "total_tags": sum(len(fs.tags) for fs in self._sets.values()),
                "unique_feature_names": len(self._feature_index),
                "unique_tags": len(self._tag_index),
            }

def create_feature_store() -> FeatureStore:
    return FeatureStore()
