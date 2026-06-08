"""
阶段67: Agent数据窗口计算系统

遵循"图难于其易，为大于其细"的原则，实现滚动窗口、滑动窗口、会话窗口、窗口聚合。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class WindowEvent:
    data: Any
    timestamp: float = field(default_factory=time.time)

class Window:
    """窗口基类"""
    def __init__(self, window_id: str):
        self.window_id = window_id
        self._events: List[WindowEvent] = []

    def add(self, event: WindowEvent) -> None:
        self._events.append(event)

    def get_events(self) -> List[WindowEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()

    def size(self) -> int:
        return len(self._events)

    def apply(self, agg_func: Callable[[List[Any]], Any]) -> Any:
        return agg_func([e.data for e in self._events])

class TumblingWindow(Window):
    """滚动窗口"""
    def __init__(self, window_id: str, size_ms: float):
        super().__init__(window_id)
        self.size_ms = size_ms
        self._start_time: Optional[float] = None

    def add(self, event: WindowEvent) -> bool:
        if self._start_time is None:
            self._start_time = event.timestamp
        if event.timestamp - self._start_time >= self.size_ms:
            return False
        self._events.append(event)
        return True

    def is_full(self, current_time: float) -> bool:
        if self._start_time is None:
            return False
        return current_time - self._start_time >= self.size_ms

class SlidingWindow(Window):
    """滑动窗口"""
    def __init__(self, window_id: str, size_ms: float, slide_ms: float):
        super().__init__(window_id)
        self.size_ms = size_ms
        self.slide_ms = slide_ms

    def add(self, event: WindowEvent) -> None:
        self._events.append(event)
        self._evict_old(event.timestamp)

    def _evict_old(self, current_time: float) -> None:
        cutoff = current_time - self.size_ms
        self._events = [e for e in self._events if e.timestamp >= cutoff]

class SessionWindow(Window):
    """会话窗口"""
    def __init__(self, window_id: str, gap_ms: float):
        super().__init__(window_id)
        self.gap_ms = gap_ms
        self._last_time: Optional[float] = None

    def add(self, event: WindowEvent) -> bool:
        if self._last_time is not None and event.timestamp - self._last_time > self.gap_ms:
            return False
        self._events.append(event)
        self._last_time = event.timestamp
        return True

class WindowManager:
    """窗口管理器"""
    def __init__(self):
        self._windows: Dict[str, Window] = {}

    def create_tumbling(self, window_id: str, size_ms: float) -> TumblingWindow:
        w = TumblingWindow(window_id, size_ms)
        self._windows[window_id] = w
        return w

    def create_sliding(self, window_id: str, size_ms: float, slide_ms: float) -> SlidingWindow:
        w = SlidingWindow(window_id, size_ms, slide_ms)
        self._windows[window_id] = w
        return w

    def create_session(self, window_id: str, gap_ms: float) -> SessionWindow:
        w = SessionWindow(window_id, gap_ms)
        self._windows[window_id] = w
        return w

    def get_window(self, window_id: str) -> Optional[Window]:
        return self._windows.get(window_id)

    def remove_window(self, window_id: str) -> bool:
        return self._windows.pop(window_id, None) is not None

    def list_windows(self) -> List[str]:
        return list(self._windows.keys())

def create_window_manager() -> WindowManager:
    return WindowManager()
