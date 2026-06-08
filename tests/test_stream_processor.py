"""阶段66: Agent数据流处理系统 - 单元测试"""
import time
import pytest
from src.agent.stream_processor import (
    StreamStatus, StreamEvent, Stream, StreamProcessor, create_stream_processor,
)

class TestStream:
    def test_emit_subscribe(self):
        s = Stream("s1")
        received = []
        s.subscribe(lambda e: received.append(e.data))
        s.emit("hello")
        assert received == ["hello"]

    def test_pause(self):
        s = Stream("s1")
        received = []
        s.subscribe(lambda e: received.append(e.data))
        s.pause()
        s.emit("hello")
        assert received == []

    def test_close(self):
        s = Stream("s1")
        received = []
        s.subscribe(lambda e: received.append(e.data))
        s.close()
        s.emit("hello")
        assert received == []

    def test_get_events(self):
        s = Stream("s1")
        s.emit("a")
        s.emit("b")
        events = s.get_events(2)
        assert len(events) == 2
        assert events[0].data == "a"

class TestStreamProcessor:
    def test_create_stream(self):
        p = StreamProcessor()
        s = p.create_stream("s1")
        assert isinstance(s, Stream)

    def test_map_stream(self):
        p = StreamProcessor()
        source = p.create_stream("source")
        target = p.map_stream("source", "target", lambda x: x * 2)
        source.emit(5)
        time.sleep(0.01)
        assert target.get_events(1)[0].data == 10

    def test_filter_stream(self):
        p = StreamProcessor()
        source = p.create_stream("source")
        target = p.filter_stream("source", "target", lambda x: x > 5)
        source.emit(3)
        source.emit(10)
        time.sleep(0.01)
        assert len(target.get_events(2)) == 1
        assert target.get_events(1)[0].data == 10

    def test_merge_streams(self):
        p = StreamProcessor()
        s1 = p.create_stream("s1")
        s2 = p.create_stream("s2")
        merged = p.merge_streams(["s1", "s2"], "merged")
        s1.emit("a")
        s2.emit("b")
        time.sleep(0.01)
        assert len(merged.get_events(2)) == 2

    def test_split_stream(self):
        p = StreamProcessor()
        source = p.create_stream("source")
        branches = p.split_stream("source", {
            "even": lambda x: x % 2 == 0,
            "odd": lambda x: x % 2 == 1,
        })
        source.emit(1)
        source.emit(2)
        source.emit(3)
        time.sleep(0.01)
        assert len(branches["even"].get_events(3)) == 1
        assert len(branches["odd"].get_events(3)) == 2

    def test_remove_stream(self):
        p = StreamProcessor()
        p.create_stream("s1")
        assert p.remove_stream("s1")
        assert p.get_stream("s1") is None

    def test_create_stream_processor(self):
        p = create_stream_processor()
        assert isinstance(p, StreamProcessor)
