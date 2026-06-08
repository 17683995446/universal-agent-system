"""
阶段74: Agent机器学习A/B测试系统

遵循"图难于其易，为大于其细"的原则，实现实验分组、流量分配、指标收集、显著性检验。
"""
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Variant:
    variant_id: str
    name: str
    weight: float = 0.5
    traffic: int = 0
    conversions: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def conversion_rate(self) -> float:
        return self.conversions / self.traffic if self.traffic > 0 else 0.0

    def mean_metric(self, key: str) -> float:
        vals = self.metrics.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "weight": self.weight,
            "traffic": self.traffic,
            "conversions": self.conversions,
            "conversion_rate": round(self.conversion_rate(), 4),
        }


@dataclass
class Experiment:
    experiment_id: str
    name: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    variants: Dict[str, Variant] = field(default_factory=dict)
    target_metric: str = "conversion"
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "variant_count": len(self.variants),
            "target_metric": self.target_metric,
        }


class TrafficRouter:
    """流量路由器"""

    def __init__(self):
        self._salt = str(uuid.uuid4())

    def assign(self, user_id: str, variants: List[Variant]) -> Optional[Variant]:
        if not variants:
            return None
        # 一致性哈希分配
        hash_input = f"{user_id}:{self._salt}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        total_weight = sum(v.weight for v in variants)
        if total_weight <= 0:
            return variants[0]
        point = (hash_val % 10000) / 10000.0 * total_weight
        cumulative = 0.0
        for v in variants:
            cumulative += v.weight
            if point <= cumulative:
                return v
        return variants[-1]

    def assign_variant_id(self, user_id: str, variant_ids: List[str]) -> str:
        if not variant_ids:
            return ""
        hash_input = f"{user_id}:{self._salt}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        idx = hash_val % len(variant_ids)
        return variant_ids[idx]


class SignificanceTest:
    """显著性检验（简化版Z检验）"""

    @staticmethod
    def z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
        """返回 (z_score, p_value)"""
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        se = ((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2) ** 0.5
        if se == 0:
            return 0.0, 1.0
        z = (p1 - p2) / se
        # 简化的p值估计（双侧检验）
        p_value = min(1.0, 2 * (1 - min(1.0, abs(z) / 3.5)))
        return z, p_value

    @staticmethod
    def t_test_mean(mean1: float, var1: float, n1: int,
                    mean2: float, var2: float, n2: int) -> Tuple[float, float]:
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        pooled_se = ((var1 / n1) + (var2 / n2)) ** 0.5
        if pooled_se == 0:
            return 0.0, 1.0
        t = (mean1 - mean2) / pooled_se
        p_value = min(1.0, 2 * (1 - min(1.0, abs(t) / 3.5)))
        return t, p_value


class ExperimentManager:
    """实验管理器"""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._router = TrafficRouter()

    def create(self, name: str, target_metric: str = "conversion") -> Experiment:
        exp_id = str(uuid.uuid4())[:8]
        exp = Experiment(experiment_id=exp_id, name=name, target_metric=target_metric)
        self._experiments[exp_id] = exp
        return exp

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def add_variant(self, experiment_id: str, name: str, weight: float = 0.5) -> Optional[Variant]:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None
        variant_id = f"{experiment_id}_{name}"
        variant = Variant(variant_id=variant_id, name=name, weight=weight)
        exp.variants[variant_id] = variant
        return variant

    def start(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp or len(exp.variants) < 2:
            return False
        exp.status = ExperimentStatus.RUNNING
        return True

    def pause(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.PAUSED
        return True

    def stop(self, experiment_id: str) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = time.time()
        return True

    def assign_user(self, experiment_id: str, user_id: str) -> Optional[str]:
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None
        variants = list(exp.variants.values())
        variant = self._router.assign(user_id, variants)
        if variant:
            variant.traffic += 1
            return variant.variant_id
        return None

    def record_conversion(self, experiment_id: str, variant_id: str, value: float = 1.0) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        variant = exp.variants.get(variant_id)
        if not variant:
            return False
        variant.conversions += int(value)
        metric_key = exp.target_metric
        if metric_key not in variant.metrics:
            variant.metrics[metric_key] = []
        variant.metrics[metric_key].append(value)
        return True

    def record_metric(self, experiment_id: str, variant_id: str, metric: str, value: float) -> bool:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False
        variant = exp.variants.get(variant_id)
        if not variant:
            return False
        if metric not in variant.metrics:
            variant.metrics[metric] = []
        variant.metrics[metric].append(value)
        return True

    def get_results(self, experiment_id: str) -> Dict[str, Any]:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}
        variants = list(exp.variants.values())
        if len(variants) < 2:
            return {"experiment": exp.to_dict(), "comparisons": []}

        control = variants[0]
        comparisons = []
        for treatment in variants[1:]:
            p_control = control.conversion_rate()
            p_treatment = treatment.conversion_rate()
            z, p_value = SignificanceTest.z_test(
                p_control, control.traffic, p_treatment, treatment.traffic
            )
            comparisons.append({
                "control": control.name,
                "treatment": treatment.name,
                "control_rate": round(p_control, 4),
                "treatment_rate": round(p_treatment, 4),
                "relative_uplift": round((p_treatment - p_control) / p_control, 4) if p_control > 0 else 0.0,
                "z_score": round(z, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            })
        return {
            "experiment": exp.to_dict(),
            "variants": [v.to_dict() for v in variants],
            "comparisons": comparisons,
        }

    def get_best_variant(self, experiment_id: str) -> Optional[str]:
        results = self.get_results(experiment_id)
        comparisons = results.get("comparisons", [])
        if not comparisons:
            return None
        best = max(comparisons, key=lambda x: x["treatment_rate"])
        return best["treatment"]

    def list_experiments(self) -> List[str]:
        return list(self._experiments.keys())

    def delete(self, experiment_id: str) -> bool:
        return self._experiments.pop(experiment_id, None) is not None


def create_experiment_manager() -> ExperimentManager:
    return ExperimentManager()
