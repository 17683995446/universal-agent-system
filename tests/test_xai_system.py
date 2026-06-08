"""
阶段92测试: 可解释AI系统
"""
import pytest
from src.agent.xai_system import (
    FeatureImportance, Explanation, FeatureImportanceExtractor,
    ShapApproximator, LimeApproximator, DecisionExplainer,
    XAIEngine, create_xai_engine,
)


class TestFeatureImportance:
    def test_to_dict(self):
        fi = FeatureImportance("age", 0.85, "positive")
        d = fi.to_dict()
        assert d["feature_name"] == "age"
        assert d["importance"] == 0.85
        assert d["direction"] == "positive"


class TestExplanation:
    def test_top_features(self):
        fis = [
            FeatureImportance("a", 0.1),
            FeatureImportance("b", 0.9),
            FeatureImportance("c", 0.5),
        ]
        exp = Explanation(prediction=1.0, feature_importances=fis, confidence=0.8, method="test")
        top = exp.top_features(2)
        assert len(top) == 2
        assert top[0].feature_name == "b"

    def test_to_dict(self):
        exp = Explanation(prediction=1, confidence=0.9, method="test")
        d = exp.to_dict()
        assert d["prediction"] == 1
        assert d["confidence"] == 0.9


class TestFeatureImportanceExtractor:
    def test_extract(self):
        extractor = FeatureImportanceExtractor(["x", "y"])
        model = lambda v: v[0] * 2 + v[1] * 0.5
        importances = extractor.extract(model, [1.0, 2.0], n_permutations=5)
        assert len(importances) == 2
        assert all(isinstance(i.importance, float) for i in importances)


class TestShapApproximator:
    def test_approximate(self):
        shap = ShapApproximator(["x", "y"])
        model = lambda v: sum(v)
        shap_vals = shap.approximate(model, [1.0, 2.0], [[0.0, 0.0], [1.0, 1.0]], n_samples=10)
        assert len(shap_vals) == 2
        assert "x" in shap_vals
        assert "y" in shap_vals


class TestLimeApproximator:
    def test_explain(self):
        lime = LimeApproximator(["x", "y"])
        model = lambda v: v[0] * 3 + v[1] * 1
        importances = lime.explain(model, [1.0, 1.0], n_samples=20)
        assert len(importances) == 2
        assert all(isinstance(i.importance, float) for i in importances)


class TestDecisionExplainer:
    def test_explain_decision_match(self):
        explainer = DecisionExplainer()
        rules = [{"condition": lambda p: p > 0.5, "explanation": "高置信度"}]
        result = explainer.explain_decision(0.8, rules)
        assert result == "高置信度"

    def test_explain_decision_no_match(self):
        explainer = DecisionExplainer()
        rules = [{"condition": lambda p: p > 0.5, "explanation": "高置信度"}]
        result = explainer.explain_decision(0.3, rules)
        assert "综合判断" in result

    def test_generate_counterfactual_found(self):
        explainer = DecisionExplainer()
        model = lambda v: 1 if v[0] > 1.5 else 0
        cf = explainer.generate_counterfactual([1.0], model, target=1, max_changes=1)
        assert cf is not None
        assert model(cf) == 1

    def test_generate_counterfactual_already_target(self):
        explainer = DecisionExplainer()
        model = lambda v: 1 if v[0] > 1.5 else 0
        cf = explainer.generate_counterfactual([2.0], model, target=1)
        assert cf == [2.0]


class TestXAIEngine:
    def test_explain_feature_importance(self):
        engine = XAIEngine(["x", "y"])
        model = lambda v: v[0] * 2 + v[1]
        exp = engine.explain_feature_importance(model, [1.0, 2.0])
        assert exp.method == "permutation_importance"
        assert len(exp.feature_importances) == 2
        assert len(engine._history) == 1

    def test_explain_shap(self):
        engine = XAIEngine(["x", "y"])
        model = lambda v: sum(v)
        exp = engine.explain_shap(model, [1.0, 2.0], [[0.0, 0.0]])
        assert exp.method == "shap_approx"
        assert len(exp.feature_importances) == 2

    def test_explain_lime(self):
        engine = XAIEngine(["x", "y"])
        model = lambda v: v[0] * 3 + v[1]
        exp = engine.explain_lime(model, [1.0, 1.0])
        assert exp.method == "lime_approx"
        assert len(exp.feature_importances) == 2

    def test_explain_with_rules(self):
        engine = XAIEngine(["x"])
        rules = [{"condition": lambda p: p > 0, "explanation": "正类"}]
        result = engine.explain_with_rules(1, rules)
        assert result == "正类"

    def test_counterfactual(self):
        engine = XAIEngine(["x"])
        model = lambda v: 1 if v[0] > 0.5 else 0
        cf = engine.counterfactual([0.0], model, target=1)
        assert cf is not None
        assert model(cf) == 1

    def test_get_stats(self):
        engine = XAIEngine(["x"])
        model = lambda v: v[0]
        engine.explain_lime(model, [1.0])
        stats = engine.get_stats()
        assert stats["total_explanations"] == 1
        assert "lime_approx" in stats["methods_used"]


class TestCreateXAIEngine:
    def test_factory(self):
        engine = create_xai_engine(["a", "b"])
        assert isinstance(engine, XAIEngine)
        assert engine.feature_names == ["a", "b"]
