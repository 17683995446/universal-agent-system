"""阶段69: Agent数据血缘追踪系统 - 单元测试"""
import pytest
from src.agent.data_lineage import (
    LineageType, LineageNode, LineageEdge, DataLineage, LineageTracker, create_lineage_tracker,
)

class TestDataLineage:
    def test_add_node(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source1", LineageType.SOURCE))
        assert "n1" in dl._nodes

    def test_add_edge(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source1", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform1", LineageType.TRANSFORM))
        dl.add_edge(LineageEdge("n1", "n2"))
        assert "n2" in dl._downstream["n1"]
        assert "n1" in dl._upstream["n2"]

    def test_get_downstream(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform", LineageType.TRANSFORM))
        dl.add_node(LineageNode("n3", "sink", LineageType.SINK))
        dl.add_edge(LineageEdge("n1", "n2"))
        dl.add_edge(LineageEdge("n2", "n3"))
        downstream = dl.get_downstream("n1")
        assert len(downstream) == 2

    def test_get_upstream(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform", LineageType.TRANSFORM))
        dl.add_edge(LineageEdge("n1", "n2"))
        upstream = dl.get_upstream("n2")
        assert len(upstream) == 1
        assert upstream[0].node_id == "n1"

    def test_get_lineage_path(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform", LineageType.TRANSFORM))
        dl.add_node(LineageNode("n3", "sink", LineageType.SINK))
        dl.add_edge(LineageEdge("n1", "n2"))
        dl.add_edge(LineageEdge("n2", "n3"))
        path = dl.get_lineage_path("n1", "n3")
        assert path == ["n1", "n2", "n3"]

    def test_get_impact(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform", LineageType.TRANSFORM))
        dl.add_edge(LineageEdge("n1", "n2"))
        impact = dl.get_impact("n1")
        assert impact["direct_downstream"] == 1
        assert impact["total_downstream"] == 1

    def test_stats(self):
        dl = DataLineage()
        dl.add_node(LineageNode("n1", "source", LineageType.SOURCE))
        dl.add_node(LineageNode("n2", "transform", LineageType.TRANSFORM))
        dl.add_node(LineageNode("n3", "sink", LineageType.SINK))
        stats = dl.get_stats()
        assert stats["nodes"] == 3
        assert stats["sources"] == 1
        assert stats["transforms"] == 1
        assert stats["sinks"] == 1

class TestLineageTracker:
    def test_track_source(self):
        t = LineageTracker()
        nid = t.track_source("db1")
        assert len(nid) == 8

    def test_track_transform(self):
        t = LineageTracker()
        src = t.track_source("db1")
        tx = t.track_transform("clean", [src])
        assert tx in t.get_lineage()._nodes

    def test_track_sink(self):
        t = LineageTracker()
        src = t.track_source("db1")
        sink = t.track_sink("report", [src])
        assert sink in t.get_lineage()._nodes

    def test_create_lineage_tracker(self):
        t = create_lineage_tracker()
        assert isinstance(t, LineageTracker)
