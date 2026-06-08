"""阶段67: Agent数据窗口计算系统 - 单元测试"""
import time
import pytest
from src.agent.window_computation import (
    WindowEvent, Window, TumblingWindow, SlidingWindow, SessionWindow,
    WindowManager, create_window_manager,
)

class TestTumblingWindow:
    def test_add_within_window(self):
        w = TumblingWindow("w1", size_ms=1000)
        assert w.add(WindowEvent(data=1, timestamp=0))
        assert w.add(WindowEvent(data=2, timestamp=500))
        assert w.size() == 2

    def test_add_outside_window(self):
        w = TumblingWindow("w1", size_ms=100)
        w.add(WindowEvent(data=1, timestamp=0))
        assert not w.add(WindowEvent(data=2, timestamp=200))

    def test_is_full(self):
        w = TumblingWindow("w1", size_ms=100)
        w.add(WindowEvent(data=1, timestamp=0))
        assert w.is_full(150)

    def test_apply(self):
        w = TumblingWindow("w1", size_ms=1000)
        w.add(WindowEvent(data=1))
        w.add(WindowEvent(data=2))
        w.add(WindowEvent(data=3))
        assert w.apply(sum) == 6

class TestSlidingWindow:
    def test_eviction(self):
        w = SlidingWindow("w1", size_ms=100, slide_ms=50)
        w.add(WindowEvent(data=1, timestamp=0))
        w.add(WindowEvent(data=2, timestamp=50))
        w.add(WindowEvent(data=3, timestamp=150))
        assert w.size() == 2  # first evicted

class TestSessionWindow:
    def test_session_gap(self):
        w = SessionWindow("w1", gap_ms=100)
        assert w.add(WindowEvent(data=1, timestamp=0))
        assert w.add(WindowEvent(data=2, timestamp=50))
        assert not w.add(WindowEvent(data=3, timestamp=200))

class TestWindowManager:
    def test_create_tumbling(self):
        m = WindowManager()
        w = m.create_tumbling("w1", 1000)
        assert isinstance(w, TumblingWindow)

    def test_create_sliding(self):
        m = WindowManager()
        w = m.create_sliding("w1", 1000, 500)
        assert isinstance(w, SlidingWindow)

    def test_create_session(self):
        m = WindowManager()
        w = m.create_session("w1", 100)
        assert isinstance(w, SessionWindow)

    def test_create_window_manager(self):
        m = create_window_manager()
        assert isinstance(m, WindowManager)
