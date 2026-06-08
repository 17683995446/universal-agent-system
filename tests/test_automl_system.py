"""
阶段91测试: AutoML自动化系统
"""
import pytest
from src.agent.automl_system import (
    HPTrial, HyperParameterSearch, ModelCandidate, ModelSelector,
    FeatureEngineer, AutoMLEngine, create_automl_engine,
)


class TestHPTrial:
    def test_to_dict(self):
        t = HPTrial(params={"lr": 0.01}, score=0.85, status="completed")
        d = t.to_dict()
        assert d["params"] == {"lr": 0.01}
        assert d["score"] == 0.85
        assert d["status"] == "completed"


class TestHyperParameterSearch:
    def test_grid_search(self):
        space = {"a": [1, 2], "b": [10, 20]}
        hp = HyperParameterSearch(space)
        best = hp.grid_search(lambda p: p["a"] * 10 + p["b"])
        assert best.score == 40  # a=2, b=20
        assert best.params == {"a": 2, "b": 20}
        assert len(hp.trials) == 4

    def test_random_search(self):
        space = {"x": [1, 2, 3]}
        hp = HyperParameterSearch(space)
        best = hp.random_search(lambda p: p["x"], n_trials=5)
        assert best is not None
        assert len(hp.trials) == 5

    def test_get_best_trial(self):
        space = {"x": [1, 2]}
        hp = HyperParameterSearch(space)
        hp.grid_search(lambda p: p["x"])
        best = hp.get_best_trial()
        assert best.score == 2

    def test_get_stats(self):
        space = {"x": [1, 2]}
        hp = HyperParameterSearch(space)
        hp.grid_search(lambda p: p["x"])
        stats = hp.get_stats()
        assert stats["total_trials"] == 2
        assert stats["completed"] == 2
        assert stats["best_score"] == 2


class TestModelCandidate:
    def test_to_dict(self):
        c = ModelCandidate(name="rf", score=0.9, params={"n": 10})
        d = c.to_dict()
        assert d["name"] == "rf"
        assert d["score"] == 0.9


class TestModelSelector:
    def test_register_and_select(self):
        ms = ModelSelector()
        ms.register("model_a", None)
        ms.register("model_b", None)
        ms.evaluate("model_a", lambda: 0.8)
        ms.evaluate("model_b", lambda: 0.9)
        best = ms.select_best()
        assert best.name == "model_b"
        assert best.score == 0.9

    def test_rank(self):
        ms = ModelSelector()
        ms.register("a", None)
        ms.register("b", None)
        ms.evaluate("a", lambda: 0.7)
        ms.evaluate("b", lambda: 0.95)
        ranked = ms.rank()
        assert ranked[0].name == "b"
        assert ranked[1].name == "a"

    def test_get_stats(self):
        ms = ModelSelector()
        ms.register("x", None)
        ms.evaluate("x", lambda: 0.5)
        stats = ms.get_stats()
        assert stats["candidates"] == 1
        assert stats["best"] == "x"


class TestFeatureEngineer:
    def test_normalize(self):
        fe = FeatureEngineer()
        result = fe.apply([1.0, 2.0, 3.0], ["normalize"])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    def test_standardize(self):
        fe = FeatureEngineer()
        result = fe.apply([1.0, 2.0, 3.0], ["standardize"])
        assert sum(result) == pytest.approx(0.0, abs=1e-6)

    def test_auto_transform_normal(self):
        fe = FeatureEngineer()
        result, transforms = fe.auto_transform([1.0, 100.0, 200.0], target_type="normal")
        assert len(result) == 3
        assert "standardize" in transforms

    def test_auto_transform_log(self):
        fe = FeatureEngineer()
        result, transforms = fe.auto_transform([1.0, 10.0, 100.0], target_type="log")
        assert len(result) == 3
        assert "log" in transforms

    def test_register_custom(self):
        fe = FeatureEngineer()
        fe.register("double", lambda d: [v * 2 for v in d])
        result = fe.apply([1.0, 2.0], ["double"])
        assert result == [2.0, 4.0]


class TestAutoMLEngine:
    def test_search_hyperparameters(self):
        engine = AutoMLEngine()
        space = {"lr": [0.01, 0.1]}
        best = engine.search_hyperparameters(space, lambda p: p["lr"] * 10, method="grid")
        assert best.score == pytest.approx(1.0)

    def test_select_model(self):
        engine = AutoMLEngine()
        best = engine.select_model(["a", "b"], lambda n: 0.9 if n == "a" else 0.8)
        assert best.name == "a"

    def test_engineer_features(self):
        engine = AutoMLEngine()
        result, transforms = engine.engineer_features([10.0, 20.0, 30.0])
        assert len(result) == 3
        assert len(transforms) > 0

    def test_run_pipeline(self):
        engine = AutoMLEngine()
        result = engine.run_pipeline(
            data=[1.0, 2.0, 3.0],
            param_space={"lr": [0.01, 0.1]},
            evaluator=lambda p: p["lr"] * 100,
        )
        assert "features" in result
        assert "best_params" in result
        assert "best_score" in result
        assert result["best_score"] > 0

    def test_get_stats(self):
        engine = AutoMLEngine()
        engine.search_hyperparameters({"x": [1]}, lambda p: 1.0, method="grid")
        stats = engine.get_stats()
        assert stats["pipeline_steps"] >= 1
        assert stats["hp_trials"] >= 1


class TestCreateAutoMLEngine:
    def test_factory(self):
        engine = create_automl_engine()
        assert isinstance(engine, AutoMLEngine)
