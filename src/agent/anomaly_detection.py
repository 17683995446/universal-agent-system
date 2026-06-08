"""
阶段88: Agent异常检测系统

遵循"图难于其易，为大于其细"的原则，实现统计异常检测、孤立森林、密度估计、集成检测。
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "score": round(self.score, 4),
            "method": self.method,
        }


class StatisticalDetector:
    """统计异常检测器（基于均值和标准差）"""

    def __init__(self, threshold_sigma: float = 3.0):
        self.threshold_sigma = threshold_sigma
        self._mean = 0.0
        self._std = 0.0
        self._fitted = False

    def fit(self, values: List[float]) -> None:
        if not values:
            return
        self._mean = sum(values) / len(values)
        variance = sum((v - self._mean) ** 2 for v in values) / len(values)
        self._std = math.sqrt(variance) if variance > 0 else 1.0
        self._fitted = True

    def detect(self, value: float) -> AnomalyResult:
        if not self._fitted:
            return AnomalyResult(is_anomaly=False, score=0.0, method="statistical")
        z_score = abs(value - self._mean) / self._std if self._std > 0 else 0.0
        is_anomaly = z_score > self.threshold_sigma
        # 归一化分数到0-1
        score = min(1.0, z_score / (self.threshold_sigma * 2))
        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            method="statistical",
            details={"mean": self._mean, "std": self._std, "z_score": z_score},
        )

    def detect_batch(self, values: List[float]) -> List[AnomalyResult]:
        return [self.detect(v) for v in values]


class IsolationTreeNode:
    """孤立树节点"""

    def __init__(self, left=None, right=None, split_attr: int = -1,
                 split_value: float = 0.0, size: int = 0, is_leaf: bool = False):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_value = split_value
        self.size = size
        self.is_leaf = is_leaf


class IsolationForestDetector:
    """孤立森林异常检测器（简化版）"""

    def __init__(self, n_trees: int = 10, sample_size: int = 256, max_depth: int = 8):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self._trees: List[IsolationTreeNode] = []
        self._fitted = False

    def fit(self, data: List[List[float]]) -> None:
        if not data:
            return
        self._trees = []
        import random
        for _ in range(self.n_trees):
            sample = random.sample(data, min(self.sample_size, len(data)))
            tree = self._build_tree(sample, 0)
            self._trees.append(tree)
        self._fitted = True

    def _build_tree(self, data: List[List[float]], depth: int) -> IsolationTreeNode:
        if len(data) <= 1 or depth >= self.max_depth:
            return IsolationTreeNode(size=len(data), is_leaf=True)

        import random
        n_features = len(data[0])
        split_attr = random.randint(0, n_features - 1)
        values = [d[split_attr] for d in data]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return IsolationTreeNode(size=len(data), is_leaf=True)

        split_value = random.uniform(min_val, max_val)
        left_data = [d for d in data if d[split_attr] < split_value]
        right_data = [d for d in data if d[split_attr] >= split_value]

        left = self._build_tree(left_data, depth + 1)
        right = self._build_tree(right_data, depth + 1)
        return IsolationTreeNode(left=left, right=right,
                                 split_attr=split_attr, split_value=split_value,
                                 size=len(data))

    def detect(self, point: List[float]) -> AnomalyResult:
        if not self._fitted or not self._trees:
            return AnomalyResult(is_anomaly=False, score=0.0, method="isolation_forest")

        avg_depth = sum(self._path_length(point, tree) for tree in self._trees) / len(self._trees)
        # 异常分数：路径越短越异常
        score = 1.0 - min(1.0, avg_depth / self.max_depth)
        is_anomaly = score > 0.6
        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            method="isolation_forest",
            details={"avg_path_length": avg_depth},
        )

    def _path_length(self, point: List[float], tree: IsolationTreeNode, depth: int = 0) -> float:
        if tree.is_leaf:
            return depth + self._c(tree.size)
        if point[tree.split_attr] < tree.split_value:
            return self._path_length(point, tree.left, depth + 1)
        else:
            return self._path_length(point, tree.right, depth + 1)

    def _c(self, size: int) -> float:
        if size <= 1:
            return 0.0
        return 2.0 * (math.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size


class DensityBasedDetector:
    """基于密度的异常检测器（简化版LOF）"""

    def __init__(self, k: int = 5, threshold: float = 2.0):
        self.k = k
        self.threshold = threshold
        self._data: List[List[float]] = []
        self._fitted = False

    def fit(self, data: List[List[float]]) -> None:
        self._data = data
        self._fitted = True

    def detect(self, point: List[float]) -> AnomalyResult:
        if not self._fitted or len(self._data) < self.k:
            return AnomalyResult(is_anomaly=False, score=0.0, method="density")

        # 计算k近邻距离
        distances = [(self._euclidean(point, d), d) for d in self._data]
        distances.sort(key=lambda x: x[0])
        k_distances = [d for d, _ in distances[:self.k]]
        avg_k_distance = sum(k_distances) / len(k_distances) if k_distances else 1.0

        # 局部密度分数
        if avg_k_distance == 0:
            score = 0.0
        else:
            score = min(1.0, 1.0 / (1.0 + avg_k_distance))

        is_anomaly = score > self.threshold / 10  # 缩放阈值
        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=score,
            method="density",
            details={"avg_k_distance": avg_k_distance},
        )

    def _euclidean(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class EnsembleDetector:
    """集成异常检测器"""

    def __init__(self):
        self._detectors: List[Any] = []
        self._weights: List[float] = []

    def add_detector(self, detector: Any, weight: float = 1.0) -> None:
        self._detectors.append(detector)
        self._weights.append(weight)

    def detect(self, value: Any) -> AnomalyResult:
        if not self._detectors:
            return AnomalyResult(is_anomaly=False, score=0.0, method="ensemble")

        results = []
        for detector in self._detectors:
            if hasattr(detector, "detect"):
                result = detector.detect(value)
                results.append(result)

        if not results:
            return AnomalyResult(is_anomaly=False, score=0.0, method="ensemble")

        # 加权投票
        total_weight = sum(self._weights)
        weighted_score = sum(r.score * w for r, w in zip(results, self._weights)) / total_weight
        anomaly_votes = sum(1 for r in results if r.is_anomaly)
        is_anomaly = anomaly_votes >= len(results) / 2

        return AnomalyResult(
            is_anomaly=is_anomaly,
            score=weighted_score,
            method="ensemble",
            details={"detectors": [r.method for r in results], "votes": anomaly_votes},
        )


class AnomalyDetectionSystem:
    """异常检测系统"""

    def __init__(self):
        self.statistical = StatisticalDetector()
        self.isolation_forest = IsolationForestDetector()
        self.density = DensityBasedDetector()
        self.ensemble = EnsembleDetector()
        self._history: List[Tuple[Any, AnomalyResult]] = []

    def fit(self, data: List[Any], data_type: str = "univariate") -> None:
        """拟合模型"""
        if data_type == "univariate":
            values = [float(v) for v in data]
            self.statistical.fit(values)
            # 将一维数据转为二维用于孤立森林
            self.isolation_forest.fit([[v] for v in values])
            self.density.fit([[v] for v in values])
        elif data_type == "multivariate":
            self.isolation_forest.fit(data)
            self.density.fit(data)

    def detect(self, value: Any, method: str = "ensemble") -> AnomalyResult:
        """检测异常"""
        point = value if isinstance(value, list) else [float(value)]
        scalar = float(value) if not isinstance(value, list) else None
        if method == "statistical":
            result = self.statistical.detect(scalar if scalar is not None else point[0])
        elif method == "isolation_forest":
            result = self.isolation_forest.detect(point)
        elif method == "density":
            result = self.density.detect(point)
        elif method == "ensemble":
            # 确保集成检测器已配置
            if not self.ensemble._detectors:
                self.ensemble.add_detector(self.statistical, 0.3)
                self.ensemble.add_detector(self.isolation_forest, 0.4)
                self.ensemble.add_detector(self.density, 0.3)
            # 集成检测器需要为每个检测器传递正确格式的值
            results = []
            weights = []
            # statistical
            r = self.statistical.detect(scalar if scalar is not None else point[0])
            results.append(r)
            weights.append(0.3)
            # isolation forest
            r = self.isolation_forest.detect(point)
            results.append(r)
            weights.append(0.4)
            # density
            r = self.density.detect(point)
            results.append(r)
            weights.append(0.3)
            total_weight = sum(weights)
            weighted_score = sum(r.score * w for r, w in zip(results, weights)) / total_weight
            anomaly_votes = sum(1 for r in results if r.is_anomaly)
            is_anomaly = anomaly_votes >= len(results) / 2
            result = AnomalyResult(
                is_anomaly=is_anomaly,
                score=weighted_score,
                method="ensemble",
                details={"detectors": [r.method for r in results], "votes": anomaly_votes},
            )
        else:
            result = AnomalyResult(is_anomaly=False, score=0.0, method="unknown")

        self._history.append((value, result))
        return result

    def detect_batch(self, values: List[Any], method: str = "ensemble") -> List[AnomalyResult]:
        return [self.detect(v, method) for v in values]

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._history)
        anomalies = sum(1 for _, r in self._history if r.is_anomaly)
        return {
            "total_checked": total,
            "anomalies_found": anomalies,
            "anomaly_rate": round(anomalies / total, 4) if total > 0 else 0.0,
        }

    def get_anomalies(self) -> List[Tuple[Any, AnomalyResult]]:
        return [(v, r) for v, r in self._history if r.is_anomaly]


def create_anomaly_detection_system() -> AnomalyDetectionSystem:
    return AnomalyDetectionSystem()
