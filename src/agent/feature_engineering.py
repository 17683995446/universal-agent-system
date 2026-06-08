"""
阶段71: Agent机器学习特征工程系统

遵循"图难于其易，为大于其细"的原则，实现特征提取、特征转换、特征选择、特征缩放。
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

class FeatureTransformer:
    """特征转换器"""

    @staticmethod
    def normalize(values: List[float], method: str = "minmax") -> List[float]:
        """归一化"""
        if not values or len(values) < 2:
            return values
        if method == "minmax":
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [0.5] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]
        elif method == "zscore":
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 1
            if std == 0:
                return [0.0] * len(values)
            return [(v - mean) / std for v in values]
        return values

    @staticmethod
    def bucketize(values: List[float], bins: int = 5) -> List[int]:
        """分桶"""
        if not values:
            return []
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0] * len(values)
        bin_size = (max_val - min_val) / bins
        result = []
        for v in values:
            idx = min(int((v - min_val) / bin_size), bins - 1)
            result.append(idx)
        return result

    @staticmethod
    def one_hot_encode(values: List[str]) -> Dict[str, List[int]]:
        """One-Hot编码"""
        categories = sorted(set(values))
        result = {cat: [] for cat in categories}
        for v in values:
            for cat in categories:
                result[cat].append(1 if v == cat else 0)
        return result

    @staticmethod
    def log_transform(values: List[float], offset: float = 1.0) -> List[float]:
        """对数变换"""
        return [math.log(v + offset) for v in values]

    @staticmethod
    def polynomial_features(values: List[float], degree: int = 2) -> List[List[float]]:
        """多项式特征"""
        return [[v ** d for d in range(1, degree + 1)] for v in values]

class FeatureExtractor:
    """特征提取器"""

    @staticmethod
    def statistical_features(values: List[float]) -> Dict[str, float]:
        """统计特征"""
        if not values:
            return {}
        result = {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) > 1:
            result["std"] = statistics.stdev(values)
            result["variance"] = statistics.variance(values)
        return result

    @staticmethod
    def text_features(text: str) -> Dict[str, Any]:
        """文本特征"""
        words = text.split()
        return {
            "length": len(text),
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "unique_words": len(set(words)),
        }

    @staticmethod
    def time_features(timestamp: float) -> Dict[str, int]:
        """时间特征"""
        import time as t
        tm = t.localtime(timestamp)
        return {
            "hour": tm.tm_hour,
            "day_of_week": tm.tm_wday,
            "day_of_month": tm.tm_mday,
            "month": tm.tm_mon,
            "is_weekend": 1 if tm.tm_wday >= 5 else 0,
        }

class FeatureSelector:
    """特征选择器"""

    @staticmethod
    def variance_threshold(features: Dict[str, List[float]], threshold: float = 0.0) -> List[str]:
        """方差阈值选择"""
        selected = []
        for name, values in features.items():
            if len(values) > 1:
                var = statistics.variance(values)
                if var >= threshold:
                    selected.append(name)
            elif threshold == 0:
                selected.append(name)
        return selected

    @staticmethod
    def correlation_filter(features: Dict[str, List[float]], target: List[float],
                           top_k: int = 5) -> List[str]:
        """相关性过滤"""
        correlations = []
        for name, values in features.items():
            if len(values) == len(target) and len(values) > 1:
                corr = FeatureSelector._pearson_correlation(values, target)
                correlations.append((name, abs(corr)))
        correlations.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in correlations[:top_k]]

    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> float:
        """皮尔逊相关系数"""
        n = len(x)
        if n == 0:
            return 0.0
        mean_x, mean_y = statistics.mean(x), statistics.mean(y)
        std_x = statistics.stdev(x) if n > 1 else 1
        std_y = statistics.stdev(y) if n > 1 else 1
        if std_x == 0 or std_y == 0:
            return 0.0
        return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / ((n - 1) * std_x * std_y)

class FeaturePipeline:
    """特征处理流水线"""
    def __init__(self):
        self._steps: List[Tuple[str, Callable]] = []

    def add_step(self, name: str, transform: Callable) -> "FeaturePipeline":
        self._steps.append((name, transform))
        return self

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data)
        for name, transform in self._steps:
            result = transform(result)
        return result

    def fit_transform(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.transform(d) for d in data_list]

def create_feature_pipeline() -> FeaturePipeline:
    return FeaturePipeline()
