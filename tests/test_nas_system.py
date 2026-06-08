"""
阶段97测试: 神经架构搜索系统
"""
import pytest
from src.agent.nas_system import (
    Architecture, SearchSpace, ArchitectureEvaluator,
    RandomSearchNAS, EvolutionaryNAS, NASEngine,
    create_nas_engine,
)


class TestArchitecture:
    def test_to_dict(self):
        a = Architecture(name="test", layers=[{"op": "linear"}], score=0.5)
        d = a.to_dict()
        assert d["name"] == "test"
        assert d["layers"] == 1
        assert d["score"] == 0.5

    def test_copy(self):
        a = Architecture(name="orig", layers=[{"op": "conv"}])
        b = a.copy()
        assert b.name == a.name
        assert b.layers == a.layers
        assert b is not a


class TestSearchSpace:
    def test_sample_random(self):
        ss = SearchSpace()
        arch = ss.sample_random()
        assert 1 <= len(arch.layers) <= ss.max_layers
        assert all("op" in layer and "size" in layer for layer in arch.layers)

    def test_sample_neighbors(self):
        ss = SearchSpace()
        arch = ss.sample_random()
        neighbors = ss.sample_neighbors(arch, n=3)
        assert len(neighbors) == 3
        assert all(isinstance(n, Architecture) for n in neighbors)

    def test_crossover(self):
        ss = SearchSpace()
        p1 = Architecture(name="p1", layers=[{"op": "linear"}, {"op": "relu"}])
        p2 = Architecture(name="p2", layers=[{"op": "conv"}, {"op": "pool"}])
        child = ss.crossover(p1, p2)
        assert isinstance(child, Architecture)
        assert len(child.layers) > 0


class TestArchitectureEvaluator:
    def test_evaluate_default(self):
        ev = ArchitectureEvaluator()
        arch = Architecture(name="a", layers=[{"op": "linear", "size": 128}])
        score = ev.evaluate(arch)
        assert score >= 0.0
        assert arch.score == score

    def test_evaluate_custom(self):
        ev = ArchitectureEvaluator()
        arch = Architecture(name="a", layers=[])
        score = ev.evaluate(arch, evaluator_fn=lambda a: 5.0)
        assert score == 5.0

    def test_cache(self):
        ev = ArchitectureEvaluator()
        arch = Architecture(name="a", layers=[{"op": "relu", "size": 64}])
        s1 = ev.evaluate(arch)
        s2 = ev.evaluate(arch)
        assert s1 == s2

    def test_get_stats(self):
        ev = ArchitectureEvaluator()
        ev.evaluate(Architecture(name="a", layers=[]))
        assert ev.get_stats()["evaluated"] == 1


class TestRandomSearchNAS:
    def test_search(self):
        ss = SearchSpace()
        ev = ArchitectureEvaluator()
        rs = RandomSearchNAS(ss, ev)
        best = rs.search(n_samples=5)
        assert isinstance(best, Architecture)
        assert best.score >= 0.0


class TestEvolutionaryNAS:
    def test_evolve(self):
        ss = SearchSpace()
        ev = ArchitectureEvaluator()
        ea = EvolutionaryNAS(ss, ev, population_size=4)
        best = ea.evolve(n_generations=2)
        assert isinstance(best, Architecture)
        assert best.score >= 0.0

    def test_initialize(self):
        ss = SearchSpace()
        ev = ArchitectureEvaluator()
        ea = EvolutionaryNAS(ss, ev, population_size=3)
        ea.initialize()
        assert len(ea.population) == 3


class TestNASEngine:
    def test_search_random(self):
        engine = NASEngine()
        best = engine.search_random(n_samples=5)
        assert isinstance(best, Architecture)

    def test_search_evolutionary(self):
        engine = NASEngine()
        best = engine.search_evolutionary(n_generations=2)
        assert isinstance(best, Architecture)

    def test_get_stats(self):
        engine = NASEngine()
        engine.search_random(n_samples=3)
        stats = engine.get_stats()
        assert stats["random_search_history"] == 3
        assert stats["evaluator"]["evaluated"] >= 3


class TestCreateNASEngine:
    def test_factory(self):
        engine = create_nas_engine()
        assert isinstance(engine, NASEngine)
