"""
阶段91: Agent AutoML自动化系统

遵循"图难于其易，为大于其细"的原则，实现自动化机器学习流程：
超参搜索、模型选择、特征工程自动化、AutoML编排器。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


@dataclass
class HPTrial:
    """超参搜索 trial"""
    params: Dict[str, Any]
    score: float = 0.0
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "score": self.score,
            "status": self.status,
        }


class HyperParameterSearch:
    """超参数搜索（网格搜索 + 随机搜索）"""

    def __init__(self, param_space: Dict[str, List[Any]]):
        self.param_space = param_space
        self.trials: List[HPTrial] = []

    def grid_search(self, evaluator: Callable[[Dict[str, Any]], float]) -> HPTrial:
        """网格搜索"""
        keys = list(self.param_space.keys())
        values_list = [self.param_space[k] for k in keys]

        best_trial = None
        best_score = float("-inf")

        def recurse(idx: int, current: Dict[str, Any]):
            nonlocal best_trial, best_score
            if idx == len(keys):
                trial = HPTrial(params=current.copy())
                trial.status = "running"
                try:
                    score = evaluator(current)
                    trial.score = score
                    trial.status = "completed"
                except Exception:
                    trial.status = "failed"
                    score = float("-inf")
                self.trials.append(trial)
                if score > best_score:
                    best_score = score
                    best_trial = trial
                return
            for v in values_list[idx]:
                current[keys[idx]] = v
                recurse(idx + 1, current)

        recurse(0, {})
        return best_trial if best_trial else HPTrial(params={})

    def random_search(self, evaluator: Callable[[Dict[str, Any]], float],
                      n_trials: int = 10) -> HPTrial:
        """随机搜索"""
        best_trial = None
        best_score = float("-inf")

        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in self.param_space.items()}
            trial = HPTrial(params=params)
            trial.status = "running"
            try:
                score = evaluator(params)
                trial.score = score
                trial.status = "completed"
            except Exception:
                trial.status = "failed"
                score = float("-inf")
            self.trials.append(trial)
            if score > best_score:
                best_score = score
                best_trial = trial

        return best_trial if best_trial else HPTrial(params={})

    def get_best_trial(self) -> Optional[HPTrial]:
        completed = [t for t in self.trials if t.status == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda t: t.score)

    def get_stats(self) -> Dict[str, Any]:
        completed = [t for t in self.trials if t.status == "completed"]
        return {
            "total_trials": len(self.trials),
            "completed": len(completed),
            "failed": len([t for t in self.trials if t.status == "failed"]),
            "best_score": max((t.score for t in completed), default=None),
        }


@dataclass
class ModelCandidate:
    """模型候选"""
    name: str
    score: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "params": self.params,
        }


class ModelSelector:
    """模型选择器"""

    def __init__(self):
        self.candidates: List[ModelCandidate] = []

    def register(self, name: str, model_fn: Callable, default_params: Dict[str, Any] = None):
        """注册模型候选"""
        self.candidates.append(ModelCandidate(
            name=name,
            params=default_params or {},
        ))

    def evaluate(self, name: str, evaluator: Callable[[], float]) -> float:
        """评估指定模型"""
        for c in self.candidates:
            if c.name == name:
                c.score = evaluator()
                return c.score
        return 0.0

    def select_best(self) -> Optional[ModelCandidate]:
        """选择最佳模型"""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.score)

    def rank(self) -> List[ModelCandidate]:
        """模型排名"""
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "candidates": len(self.candidates),
            "best": self.select_best().name if self.select_best() else None,
        }


class FeatureEngineer:
    """自动化特征工程"""

    def __init__(self):
        self.transformers: Dict[str, Callable[[List[float]], List[float]]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.transformers["normalize"] = self._normalize
        self.transformers["standardize"] = self._standardize
        self.transformers["log"] = self._log_transform
        self.transformers["square"] = self._square

    def register(self, name: str, fn: Callable[[List[float]], List[float]]):
        self.transformers[name] = fn

    def apply(self, data: List[float], transforms: List[str]) -> List[float]:
        """依次应用变换"""
        result = data.copy()
        for t in transforms:
            if t in self.transformers:
                result = self.transformers[t](result)
        return result

    def auto_transform(self, data: List[float], target_type: str = "normal") -> Tuple[List[float], List[str]]:
        """根据目标分布自动选择变换"""
        transforms = []
        result = data.copy()

        if target_type == "normal":
            # 简单启发式：如果数据范围大，先做标准化
            min_val, max_val = min(data), max(data)
            if max_val - min_val > 10:
                result = self._standardize(result)
                transforms.append("standardize")
            else:
                result = self._normalize(result)
                transforms.append("normalize")
        elif target_type == "log":
            result = self._log_transform(result)
            transforms.append("log")

        return result, transforms

    @staticmethod
    def _normalize(data: List[float]) -> List[float]:
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [0.5] * len(data)
        return [(v - min_val) / (max_val - min_val) for v in data]

    @staticmethod
    def _standardize(data: List[float]) -> List[float]:
        mean = sum(data) / len(data)
        std = (sum((v - mean) ** 2 for v in data) / len(data)) ** 0.5
        if std == 0:
            return [0.0] * len(data)
        return [(v - mean) / std for v in data]

    @staticmethod
    def _log_transform(data: List[float]) -> List[float]:
        import math
        return [math.log(v + 1) for v in data]

    @staticmethod
    def _square(data: List[float]) -> List[float]:
        return [v ** 2 for v in data]


class AutoMLEngine:
    """AutoML 编排引擎"""

    def __init__(self):
        self.hp_search = HyperParameterSearch({})
        self.model_selector = ModelSelector()
        self.feature_engineer = FeatureEngineer()
        self._history: List[Dict[str, Any]] = []

    def search_hyperparameters(self, param_space: Dict[str, List[Any]],
                                evaluator: Callable[[Dict[str, Any]], float],
                                method: str = "random",
                                n_trials: int = 10) -> HPTrial:
        """超参搜索"""
        self.hp_search = HyperParameterSearch(param_space)
        if method == "grid":
            best = self.hp_search.grid_search(evaluator)
        else:
            best = self.hp_search.random_search(evaluator, n_trials)
        self._history.append({
            "step": "hyperparameter_search",
            "method": method,
            "best_score": best.score if best else None,
        })
        return best

    def select_model(self, candidates: List[str],
                     evaluator: Callable[[str], float]) -> Optional[ModelCandidate]:
        """模型选择"""
        for name in candidates:
            self.model_selector.register(name, None)
        for name in candidates:
            self.model_selector.evaluate(name, lambda n=name: evaluator(n))
        best = self.model_selector.select_best()
        self._history.append({
            "step": "model_selection",
            "candidates": candidates,
            "best_model": best.name if best else None,
        })
        return best

    def engineer_features(self, data: List[float],
                          transforms: Optional[List[str]] = None) -> Tuple[List[float], List[str]]:
        """特征工程"""
        if transforms:
            result = self.feature_engineer.apply(data, transforms)
            used = transforms
        else:
            result, used = self.feature_engineer.auto_transform(data)
        self._history.append({
            "step": "feature_engineering",
            "transforms": used,
        })
        return result, used

    def run_pipeline(self, data: List[float],
                     param_space: Dict[str, List[Any]],
                     evaluator: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """运行完整 AutoML 流程"""
        # 1. 特征工程
        features, transforms = self.engineer_features(data)
        # 2. 超参搜索
        best_hp = self.search_hyperparameters(param_space, evaluator, method="random", n_trials=5)
        # 3. 汇总结果
        return {
            "features": features,
            "transforms": transforms,
            "best_params": best_hp.params if best_hp else {},
            "best_score": best_hp.score if best_hp else 0.0,
            "history": self._history,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pipeline_steps": len(self._history),
            "hp_trials": len(self.hp_search.trials),
            "models": len(self.model_selector.candidates),
        }


def create_automl_engine() -> AutoMLEngine:
    return AutoMLEngine()
