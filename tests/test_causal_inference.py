"""阶段86: Agent因果推断系统 - 单元测试"""
import pytest
from src.agent.causal_inference import (
    CausalVariable, CausalEdge, Intervention, CounterfactualQuery,
    CausalGraph, StructuralCausalModel, CausalInferenceEngine, create_causal_inference_engine,
)


class TestCausalGraph:
    def test_add_variable(self):
        g = CausalGraph()
        v = g.add_variable("X")
        assert v.name == "X"
        assert "X" in g._variables

    def test_add_edge(self):
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        assert g.add_edge("X", "Y")
        assert "Y" in g._variables["X"].children
        assert "X" in g._variables["Y"].parents

    def test_add_edge_missing(self):
        g = CausalGraph()
        g.add_variable("X")
        assert not g.add_edge("X", "Z")

    def test_get_parents_children(self):
        g = CausalGraph()
        g.add_variable("X")
        g.add_variable("Y")
        g.add_edge("X", "Y")
        assert g.get_parents("Y") == ["X"]
        assert g.get_children("X") == ["Y"]

    def test_get_ancestors(self):
        g = CausalGraph()
        for v in ["Z", "X", "Y"]:
            g.add_variable(v)
        g.add_edge("Z", "X")
        g.add_edge("X", "Y")
        assert g.get_ancestors("Y") == {"Z", "X"}

    def test_get_descendants(self):
        g = CausalGraph()
        for v in ["Z", "X", "Y"]:
            g.add_variable(v)
        g.add_edge("Z", "X")
        g.add_edge("X", "Y")
        assert g.get_descendants("Z") == {"X", "Y"}

    def test_get_backdoor_adjustment_set(self):
        g = CausalGraph()
        for v in ["Z", "X", "Y"]:
            g.add_variable(v)
        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")
        backdoor = g.get_backdoor_adjustment_set("X", "Y")
        assert "Z" in backdoor
        assert "X" not in backdoor
        assert "Y" not in backdoor


class TestStructuralCausalModel:
    def test_intervene(self):
        g = CausalGraph()
        for v in ["X", "Y"]:
            g.add_variable(v)
        g.add_edge("X", "Y")
        scm = StructuralCausalModel(g)
        scm.set_function("Y", lambda parents: parents["X"] * 2)
        result = scm.intervene(Intervention("X", 5))
        assert result["X"] == 5
        assert result["Y"] == 10

    def test_counterfactual(self):
        g = CausalGraph()
        for v in ["X", "Y"]:
            g.add_variable(v)
        g.add_edge("X", "Y")
        scm = StructuralCausalModel(g)
        scm.set_function("Y", lambda parents: parents["X"] * 2)
        query = CounterfactualQuery(
            target="Y",
            intervention=Intervention("X", 10),
            evidence={"X": 5},
        )
        result = scm.counterfactual(query)
        assert result == 20

    def test_topological_sort(self):
        g = CausalGraph()
        for v in ["Z", "X", "Y"]:
            g.add_variable(v)
        g.add_edge("Z", "X")
        g.add_edge("X", "Y")
        scm = StructuralCausalModel(g)
        order = scm._topological_sort()
        assert order.index("Z") < order.index("X") < order.index("Y")


class TestCausalInferenceEngine:
    def test_build_graph(self):
        engine = CausalInferenceEngine()
        g = engine.build_graph(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")])
        assert len(g._variables) == 3
        assert len(g._edges) == 2
        assert engine.scm is not None

    def test_estimate_ate(self):
        engine = CausalInferenceEngine()
        data = [
            {"T": 1, "Y": 10}, {"T": 1, "Y": 12},
            {"T": 0, "Y": 5}, {"T": 0, "Y": 7},
        ]
        ate = engine.estimate_ate("T", "Y", data)
        # treated avg = 11, control avg = 6, ate = 5
        assert ate == 5.0

    def test_estimate_cate(self):
        engine = CausalInferenceEngine()
        data = [
            {"T": 1, "Y": 10, "Z": "A"}, {"T": 0, "Y": 5, "Z": "A"},
            {"T": 1, "Y": 20, "Z": "B"}, {"T": 0, "Y": 15, "Z": "B"},
        ]
        cate = engine.estimate_cate("T", "Y", "Z", data)
        assert cate["A"] == 5.0
        assert cate["B"] == 5.0

    def test_identify_backdoor(self):
        engine = CausalInferenceEngine()
        engine.build_graph(["Z", "X", "Y"], [("Z", "X"), ("Z", "Y"), ("X", "Y")])
        backdoor = engine.identify_backdoor("X", "Y")
        assert "Z" in backdoor

    def test_do_calculus(self):
        engine = CausalInferenceEngine()
        engine.build_graph(["X", "Y"], [("X", "Y")])
        engine.scm.set_function("Y", lambda p: p["X"] * 3)
        result = engine.do_calculus("X", 4)
        assert result["X"] == 4
        assert result["Y"] == 12

    def test_get_stats(self):
        engine = CausalInferenceEngine()
        engine.build_graph(["X", "Y"], [("X", "Y")])
        stats = engine.get_stats()
        assert stats["variable_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["has_scm"] is True

    def test_create_causal_inference_engine(self):
        engine = create_causal_inference_engine()
        assert isinstance(engine, CausalInferenceEngine)
