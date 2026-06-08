"""
阶段52: Agent分布式计数器

遵循"图难于其易，为大于其细"的原则，实现原子递增/递减、批量更新、限流计数。
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class CounterValue:
    value: int = 0
    min_value: int = 0
    max_value: int = 0
    total_increments: int = 0
    total_decrements: int = 0
    last_updated: float = field(default_factory=time.time)

class DistributedCounter:
    """分布式计数器（本地模拟）"""
    def __init__(self, counter_id: str, initial: int = 0):
        self.counter_id = counter_id
        self._value = initial
        self._min_value = initial
        self._max_value = initial
        self._increments = 0
        self._decrements = 0
        self._lock = threading.RLock()

    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            self._increments += 1
            if self._value > self._max_value:
                self._max_value = self._value
            return self._value

    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            self._decrements += 1
            if self._value < self._min_value:
                self._min_value = self._value
            return self._value

    def get(self) -> int:
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> int:
        with self._lock:
            old = self._value
            self._value = value
            self._min_value = value
            self._max_value = value
            return old

    def compare_and_set(self, expected: int, new_value: int) -> bool:
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def get_value(self) -> CounterValue:
        with self._lock:
            return CounterValue(
                value=self._value,
                min_value=self._min_value,
                max_value=self._max_value,
                total_increments=self._increments,
                total_decrements=self._decrements,
                last_updated=time.time(),
            )

class CounterManager:
    """计数器管理器"""
    def __init__(self):
        self._counters: Dict[str, DistributedCounter] = {}
        self._lock = threading.RLock()

    def get_counter(self, counter_id: str, initial: int = 0) -> DistributedCounter:
        with self._lock:
            if counter_id not in self._counters:
                self._counters[counter_id] = DistributedCounter(counter_id, initial)
            return self._counters[counter_id]

    def remove_counter(self, counter_id: str) -> bool:
        with self._lock:
            return self._counters.pop(counter_id, None) is not None

    def get_all_values(self) -> Dict[str, int]:
        with self._lock:
            return {k: v.get() for k, v in self._counters.items()}

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_counters": len(self._counters),
                "total_values": sum(v.get() for v in self._counters.values()),
            }

def create_counter_manager() -> CounterManager:
    return CounterManager()
