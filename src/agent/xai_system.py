"""
阶段92: Agent 可解释AI系统 (XAI)

遵循"图难于其易，为大于其细"的原则，实现特征重要性、SHAP近似、LIME近似、决策解释。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    importance: float
    direction: str = "positive"  # positive, negative

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "importance": round(self.importance, 4),
            "direction": self.direction,
        }


@dataclass
class Explanation:
    """解释结果"""
    prediction: Any
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    confidence: float = 0.0
    method: str = ""

    def top_features(self, n: int = 3) -> List[FeatureImportance]:
        return sorted(self.feature_importances, key=lambda f: abs(f.importance), reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "top_features": [f.to_dict() for f in self.top_features()],
        }


class FeatureImportanceExtractor:
    """特征重要性提取器（基于置换）"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def extract(self, model_fn: Callable[[List[float]], float],
                baseline: List[float],
                n_permutations: int = 10) -> List[FeatureImportance]:
        """通过置换法估计特征重要性"""
        baseline_pred = model_fn(baseline)
        importances = []

        for i, name in enumerate(self.feature_names):
            scores = []
            for _ in range(n_permutations):
                permuted = baseline.copy()
                permuted[i] = random.choice([0.0, 1.0, -1.0, 0.5])
                permuted_pred = model_fn(permuted)
                scores.append(abs(permuted_pred - baseline_pred))

            avg_impact = sum(scores) / len(scores) if scores else 0.0
            direction = "positive" if (model_fn(baseline) - model_fn([baseline[j] if j != i else 0.0 for j in range(len(baseline))])) > 0 else "negative"
            importances.append(FeatureImportance(
                feature_name=name,
                importance=avg_impact,
                direction=direction,
            ))

        return importances


class ShapApproximator:
    """SHAP值近似器"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def approximate(self, model_fn: Callable[[List[float]], float],
                    instance: List[float],
                    background: List[List[float]],
                    n_samples: int = 20) -> Dict[str, float]:
        """近似计算SHAP值"""
        shap_values = {}
        baseline = [sum(col) / len(col) for col in zip(*background)] if background else [0.0] * len(instance)
        base_pred = model_fn(baseline)

        for i, name in enumerate(self.feature_names):
            contributions = []
            for _ in range(n_samples):
                # 随机 coalition：部分特征用instance值，部分用baseline值
                coalition = baseline.copy()
                for j in range(len(instance)):
                    if j == i or random.random() > 0.5:
                        coalition[j] = instance[j]
                pred_with = model_fn(coalition)
                coalition[i] = baseline[i]
                pred_without = model_fn(coalition)
                contributions.append(pred_with - pred_without)

            shap_values[name] = sum(contributions) / len(contributions) if contributions else 0.0

        return shap_values


class LimeApproximator:
    """LIME近似器（局部线性解释）"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain(self, model_fn: Callable[[List[float]], float],
                instance: List[float],
                n_samples: int = 30) -> List[FeatureImportance]:
        """局部线性近似解释"""
        # 在instance附近采样
        samples = []
        predictions = []
        for _ in range(n_samples):
            perturbed = [v + random.gauss(0, 0.1) for v in instance]
            samples.append(perturbed)
            predictions.append(model_fn(perturbed))

        # 简单线性回归（单变量近似）
        importances = []
        for i, name in enumerate(self.feature_names):
            # 计算该特征与预测的相关性作为近似权重
            x_vals = [s[i] for s in samples]
            x_mean = sum(x_vals) / len(x_vals)
            y_mean = sum(predictions) / len(predictions)

            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, predictions))
            den = sum((x - x_mean) ** 2 for x in x_vals)
            weight = num / den if den != 0 else 0.0

            direction = "positive" if weight > 0 else "negative"
            importances.append(FeatureImportance(
                feature_name=name,
                importance=abs(weight),
                direction=direction,
            ))

        return importances


class DecisionExplainer:
    """决策解释器"""

    def explain_decision(self, prediction: Any,
                         rules: List[Dict[str, Any]]) -> str:
        """基于规则解释决策"""
        matched = []
        for rule in rules:
            if rule.get("condition")(prediction):
                matched.append(rule.get("explanation", ""))
        if matched:
            return "; ".join(matched)
        return "基于模型综合判断"

    def generate_counterfactual(self, instance: List[float],
                                 model_fn: Callable[[List[float]], Any],
                                 target: Any,
                                 max_changes: int = 3) -> Optional[List[float]]:
        """生成反事实解释（简化版）"""
        current = instance.copy()
        if model_fn(current) == target:
            return current

        for _ in range(50):
            candidate = current.copy()
            n_changes = random.randint(1, min(max_changes, len(instance)))
            indices = random.sample(range(len(instance)), n_changes)
            for idx in indices:
                candidate[idx] += random.gauss(0, 0.5)
            if model_fn(candidate) == target:
                return candidate

        return None


class XAIEngine:
    """可解释AI引擎"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.importance_extractor = FeatureImportanceExtractor(feature_names)
        self.shap_approx = ShapApproximator(feature_names)
        self.lime_approx = LimeApproximator(feature_names)
        self.decision_explainer = DecisionExplainer()
        self._history: List[Explanation] = []

    def explain_feature_importance(self, model_fn: Callable[[List[float]], float],
                                    baseline: List[float]) -> Explanation:
        """特征重要性解释"""
        importances = self.importance_extractor.extract(model_fn, baseline)
        pred = model_fn(baseline)
        exp = Explanation(
            prediction=pred,
            feature_importances=importances,
            confidence=0.8,
            method="permutation_importance",
        )
        self._history.append(exp)
        return exp

    def explain_shap(self, model_fn: Callable[[List[float]], float],
                     instance: List[float],
                     background: List[List[float]]) -> Explanation:
        """SHAP解释"""
        shap_vals = self.shap_approx.approximate(model_fn, instance, background)
        importances = [
            FeatureImportance(name, abs(val), "positive" if val > 0 else "negative")
            for name, val in shap_vals.items()
        ]
        pred = model_fn(instance)
        exp = Explanation(
            prediction=pred,
            feature_importances=importances,
            confidence=0.85,
            method="shap_approx",
        )
        self._history.append(exp)
        return exp

    def explain_lime(self, model_fn: Callable[[List[float]], float],
                     instance: List[float]) -> Explanation:
        """LIME解释"""
        importances = self.lime_approx.explain(model_fn, instance)
        pred = model_fn(instance)
        exp = Explanation(
            prediction=pred,
            feature_importances=importances,
            confidence=0.75,
            method="lime_approx",
        )
        self._history.append(exp)
        return exp

    def explain_with_rules(self, prediction: Any,
                           rules: List[Dict[str, Any]]) -> str:
        """规则解释"""
        return self.decision_explainer.explain_decision(prediction, rules)

    def counterfactual(self, instance: List[float],
                       model_fn: Callable[[List[float]], Any],
                       target: Any) -> Optional[List[float]]:
        """反事实解释"""
        return self.decision_explainer.generate_counterfactual(instance, model_fn, target)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_explanations": len(self._history),
            "methods_used": list(set(e.method for e in self._history)),
        }


def create_xai_engine(feature_names: List[str]) -> XAIEngine:
    return XAIEngine(feature_names)
