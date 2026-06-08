"""
阶段94: Agent 模型安全与对抗防御系统

遵循"图难于其易，为大于其细"的原则，实现对抗样本检测、输入净化、模型鲁棒性评估、防御策略。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math


@dataclass
class AdversarialSample:
    """对抗样本"""
    original: List[float]
    perturbed: List[float]
    perturbation: List[float]
    label: Any
    predicted_label: Any
    attack_method: str = ""

    def perturbation_norm(self, norm: str = "l2") -> float:
        if norm == "l2":
            return math.sqrt(sum(p ** 2 for p in self.perturbation))
        elif norm == "linf":
            return max(abs(p) for p in self.perturbation)
        elif norm == "l1":
            return sum(abs(p) for p in self.perturbation)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation_norm_l2": round(self.perturbation_norm("l2"), 4),
            "label": self.label,
            "predicted_label": self.predicted_label,
            "attack_method": self.attack_method,
        }


class AdversarialDetector:
    """对抗样本检测器"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect_statistical(self, input_data: List[float]) -> bool:
        """基于统计特征的检测"""
        if not input_data:
            return False
        mean = sum(input_data) / len(input_data)
        std = math.sqrt(sum((v - mean) ** 2 for v in input_data) / len(input_data)) if input_data else 0.0
        # 如果标准差异常大，可能是对抗样本
        return std > self.threshold * 10

    def detect_consistency(self, model_fn: Callable[[List[float]], Any],
                           input_data: List[float],
                           n_perturbations: int = 5) -> bool:
        """基于一致性检测：多次微小扰动后预测是否稳定"""
        original_pred = model_fn(input_data)
        consistent = 0
        for _ in range(n_perturbations):
            perturbed = [v + random.gauss(0, 0.01) for v in input_data]
            if model_fn(perturbed) == original_pred:
                consistent += 1
        # 如果一致性低，可能是对抗样本
        return consistent < n_perturbations * 0.8

    def detect(self, input_data: List[float],
               model_fn: Optional[Callable[[List[float]], Any]] = None) -> Tuple[bool, str]:
        """综合检测"""
        if self.detect_statistical(input_data):
            return True, "statistical_anomaly"
        if model_fn and self.detect_consistency(model_fn, input_data):
            return True, "inconsistent_predictions"
        return False, "clean"


class InputPurifier:
    """输入净化器"""

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def clip(self, input_data: List[float]) -> List[float]:
        """裁剪到合法范围"""
        return [max(self.clip_min, min(self.clip_max, v)) for v in input_data]

    def smooth(self, input_data: List[float], window: int = 3) -> List[float]:
        """平滑处理"""
        result = []
        half = window // 2
        for i in range(len(input_data)):
            neighbors = []
            for j in range(max(0, i - half), min(len(input_data), i + half + 1)):
                neighbors.append(input_data[j])
            result.append(sum(neighbors) / len(neighbors))
        return result

    def quantize(self, input_data: List[float], levels: int = 256) -> List[float]:
        """量化（减少微小扰动影响）"""
        step = (self.clip_max - self.clip_min) / levels
        return [round((v - self.clip_min) / step) * step + self.clip_min for v in input_data]

    def purify(self, input_data: List[float], methods: List[str] = None) -> List[float]:
        """综合净化"""
        result = input_data.copy()
        methods = methods or ["clip", "smooth", "quantize"]
        for method in methods:
            if method == "clip":
                result = self.clip(result)
            elif method == "smooth":
                result = self.smooth(result)
            elif method == "quantize":
                result = self.quantize(result)
        return result


class RobustnessEvaluator:
    """模型鲁棒性评估器"""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def evaluate_under_attack(self, model_fn: Callable[[List[float]], Any],
                              test_data: List[List[float]],
                              labels: List[Any],
                              attack_fn: Callable[[List[float]], List[float]]) -> Dict[str, float]:
        """评估模型在攻击下的准确率"""
        correct = 0
        total = len(test_data)
        for x, y in zip(test_data, labels):
            perturbed = attack_fn(x)
            pred = model_fn(perturbed)
            if pred == y:
                correct += 1
        accuracy = correct / total if total > 0 else 0.0
        self.results.append({"attack": "custom", "accuracy": accuracy})
        return {"accuracy": accuracy, "total": total}

    def fgsm_attack(self, model_fn: Callable[[List[float]], Any],
                    input_data: List[float],
                    epsilon: float = 0.1) -> List[float]:
        """FGSM攻击（简化版，使用数值梯度近似）"""
        grad = []
        pred = model_fn(input_data)
        for i in range(len(input_data)):
            perturbed = input_data.copy()
            perturbed[i] += 0.01
            new_pred = model_fn(perturbed)
            # 简化：如果预测变化，梯度非零
            g = 1.0 if new_pred != pred else -1.0
            grad.append(g)

        # 生成对抗样本
        adv = [input_data[i] + epsilon * math.copysign(1.0, grad[i]) for i in range(len(input_data))]
        return adv

    def random_attack(self, input_data: List[float],
                      epsilon: float = 0.1) -> List[float]:
        """随机扰动攻击"""
        return [v + random.uniform(-epsilon, epsilon) for v in input_data]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "evaluations": len(self.results),
            "avg_accuracy": sum(r["accuracy"] for r in self.results) / len(self.results) if self.results else 0.0,
        }


class DefenseStrategy:
    """防御策略管理"""

    def __init__(self):
        self.detector = AdversarialDetector()
        self.purifier = InputPurifier()
        self.evaluator = RobustnessEvaluator()
        self._defense_history: List[Dict[str, Any]] = []

    def defend(self, input_data: List[float],
               model_fn: Callable[[List[float]], Any],
               methods: List[str] = None) -> Tuple[List[float], bool]:
        """防御流程：检测 -> 净化 -> 预测"""
        is_adversarial, reason = self.detector.detect(input_data, model_fn)
        if is_adversarial:
            purified = self.purifier.purify(input_data, methods)
            self._defense_history.append({
                "detected": True,
                "reason": reason,
                "purified": True,
            })
            return purified, True
        self._defense_history.append({"detected": False})
        return input_data, False

    def evaluate_robustness(self, model_fn: Callable[[List[float]], Any],
                            test_data: List[List[float]],
                            labels: List[Any]) -> Dict[str, float]:
        """评估模型鲁棒性"""
        # 原始准确率
        orig_correct = sum(1 for x, y in zip(test_data, labels) if model_fn(x) == y)
        orig_acc = orig_correct / len(test_data) if test_data else 0.0

        # 对抗攻击后准确率
        adv_correct = 0
        for x, y in zip(test_data, labels):
            adv_x = self.evaluator.random_attack(x, epsilon=0.1)
            if model_fn(adv_x) == y:
                adv_correct += 1
        adv_acc = adv_correct / len(test_data) if test_data else 0.0

        return {
            "original_accuracy": orig_acc,
            "adversarial_accuracy": adv_acc,
            "robustness_drop": orig_acc - adv_acc,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "defenses_triggered": len([d for d in self._defense_history if d.get("detected")]),
            "total_requests": len(self._defense_history),
        }


class ModelSecuritySystem:
    """模型安全系统"""

    def __init__(self):
        self.defense = DefenseStrategy()
        self._alerts: List[Dict[str, Any]] = []

    def predict_secure(self, input_data: List[float],
                       model_fn: Callable[[List[float]], Any]) -> Tuple[Any, bool]:
        """安全预测"""
        purified, was_attacked = self.defense.defend(input_data, model_fn)
        pred = model_fn(purified)
        if was_attacked:
            self._alerts.append({"input": input_data, "prediction": pred})
        return pred, was_attacked

    def evaluate(self, model_fn: Callable[[List[float]], Any],
                 test_data: List[List[float]],
                 labels: List[Any]) -> Dict[str, float]:
        """安全评估"""
        return self.defense.evaluate_robustness(model_fn, test_data, labels)

    def get_alerts(self) -> List[Dict[str, Any]]:
        return self._alerts

    def get_stats(self) -> Dict[str, Any]:
        return {
            "alerts": len(self._alerts),
            "defense": self.defense.get_stats(),
        }


def create_model_security_system() -> ModelSecuritySystem:
    return ModelSecuritySystem()
