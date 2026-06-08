"""
阶段63: Agent数据索引系统

遵循"图难于其易，为大于其细"的原则，实现倒排索引、B树索引、复合索引、索引管理。
"""
import bisect
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

@dataclass
class IndexEntry:
    key: Any
    value_ids: Set[str] = field(default_factory=set)

    def add(self, value_id: str) -> None:
        self.value_ids.add(value_id)

    def remove(self, value_id: str) -> None:
        self.value_ids.discard(value_id)

class InvertedIndex:
    """倒排索引"""
    def __init__(self, index_name: str):
        self.index_name = index_name
        self._index: Dict[Any, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add(self, term: Any, doc_id: str) -> None:
        with self._lock:
            self._index[term].add(doc_id)

    def remove(self, term: Any, doc_id: str) -> None:
        with self._lock:
            self._index[term].discard(doc_id)
            if not self._index[term]:
                del self._index[term]

    def search(self, term: Any) -> Set[str]:
        with self._lock:
            return set(self._index.get(term, set()))

    def search_multi(self, terms: List[Any], match_all: bool = True) -> Set[str]:
        with self._lock:
            if not terms:
                return set()
            result = None
            for term in terms:
                docs = self._index.get(term, set())
                if result is None:
                    result = set(docs)
                elif match_all:
                    result &= docs
                else:
                    result |= docs
            return result or set()

    def get_terms(self) -> List[Any]:
        with self._lock:
            return list(self._index.keys())

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "terms": len(self._index),
                "total_entries": sum(len(docs) for docs in self._index.values()),
            }

class BTreeIndex:
    """B树索引（简化版）"""
    def __init__(self, index_name: str, order: int = 4):
        self.index_name = index_name
        self.order = order
        self._keys: List[Any] = []
        self._values: Dict[Any, Set[str]] = {}
        self._lock = threading.RLock()

    def add(self, key: Any, value_id: str) -> None:
        with self._lock:
            if key not in self._values:
                self._values[key] = set()
                bisect.insort(self._keys, key)
            self._values[key].add(value_id)

    def remove(self, key: Any, value_id: str) -> None:
        with self._lock:
            if key in self._values:
                self._values[key].discard(value_id)
                if not self._values[key]:
                    del self._values[key]
                    self._keys.remove(key)

    def search(self, key: Any) -> Set[str]:
        with self._lock:
            return set(self._values.get(key, set()))

    def range_search(self, start: Any, end: Any) -> Dict[Any, Set[str]]:
        with self._lock:
            left = bisect.bisect_left(self._keys, start)
            right = bisect.bisect_right(self._keys, end)
            result = {}
            for i in range(left, right):
                k = self._keys[i]
                result[k] = set(self._values[k])
            return result

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "keys": len(self._keys),
                "total_entries": sum(len(v) for v in self._values.values()),
            }

class CompositeIndex:
    """复合索引"""
    def __init__(self, index_name: str, fields: List[str]):
        self.index_name = index_name
        self.fields = fields
        self._index: Dict[Tuple, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add(self, values: Tuple, doc_id: str) -> None:
        with self._lock:
            self._index[values].add(doc_id)

    def remove(self, values: Tuple, doc_id: str) -> None:
        with self._lock:
            self._index[values].discard(doc_id)
            if not self._index[values]:
                del self._index[values]

    def search(self, values: Tuple) -> Set[str]:
        with self._lock:
            return set(self._index.get(values, set()))

    def prefix_search(self, prefix: Tuple) -> Set[str]:
        with self._lock:
            result = set()
            for key, docs in self._index.items():
                if key[:len(prefix)] == prefix:
                    result.update(docs)
            return result

class IndexManager:
    """索引管理器"""
    def __init__(self):
        self._indexes: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def create_inverted_index(self, name: str) -> InvertedIndex:
        with self._lock:
            idx = InvertedIndex(name)
            self._indexes[name] = idx
            return idx

    def create_btree_index(self, name: str, order: int = 4) -> BTreeIndex:
        with self._lock:
            idx = BTreeIndex(name, order)
            self._indexes[name] = idx
            return idx

    def create_composite_index(self, name: str, fields: List[str]) -> CompositeIndex:
        with self._lock:
            idx = CompositeIndex(name, fields)
            self._indexes[name] = idx
            return idx

    def get_index(self, name: str) -> Optional[Any]:
        return self._indexes.get(name)

    def remove_index(self, name: str) -> bool:
        with self._lock:
            return self._indexes.pop(name, None) is not None

    def list_indexes(self) -> List[str]:
        return list(self._indexes.keys())

def create_index_manager() -> IndexManager:
    return IndexManager()
