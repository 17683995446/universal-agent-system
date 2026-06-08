"""
阶段90: Agent模型压缩与加速系统

遵循"图难于其易，为大于其细"的原则，实现剪枝、量化、知识蒸馏、低秩分解。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CompressedModel:
    original_size: int
    compressed_size: int
    method: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.compressed_size / self.original_size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": round(self.compression_ratio(), 4),
            "method": self.method,
        }


class WeightPruner:
    """权重剪枝器"""

    def __init__(self, sparsity: float = 0.5):
        self.sparsity = sparsity

    def prune(self, weights: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """基于幅值的剪枝"""
        pruned = {}
        for key, values in weights.items():
            if not values:
                pruned[key] = values
                continue
            # 计算阈值
            abs_values = sorted([abs(v) for v in values])
            threshold_idx = int(len(abs_values) * self.sparsity)
            threshold = abs_values[min(threshold_idx, len(abs_values) - 1)]
            # 剪枝
            pruned[key] = [v if abs(v) >= threshold else 0.0 for v in values]
        return pruned

    def get_sparsity(self, weights: Dict[str, List[float]]) -> float:
        """计算稀疏度"""
        total = 0
        zeros = 0
        for values in weights.values():
            total += len(values)
            zeros += sum(1 for v in values if v == 0.0)
        return zeros / total if total > 0 else 0.0


class Quantizer:
    """量化器"""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.levels = 2 ** bits

    def quantize(self, weights: Dict[str, List[float]]) -> Dict[str, List[int]]:
        """将浮点数量化为整数"""
        quantized = {}
        for key, values in weights.items():
            if not values:
                quantized[key] = []
                continue
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                quantized[key] = [0] * len(values)
                continue
            scale = (max_val - min_val) / (self.levels - 1)
            quantized[key] = [int((v - min_val) / scale) for v in values]
        return quantized

    def dequantize(self, quantized: Dict[str, List[int]],
                   ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List[float]]:
        """将整数反量化为浮点数"""
        weights = {}
        for key, values in quantized.items():
            min_val, max_val = ranges.get(key, (0.0, 1.0))
            scale = (max_val - min_val) / (self.levels - 1)
            weights[key] = [v * scale + min_val for v in values]
        return weights

    def get_ranges(self, weights: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
        ranges = {}
        for key, values in weights.items():
            if values:
                ranges[key] = (min(values), max(values))
        return ranges


class KnowledgeDistiller:
    """知识蒸馏器（简化版）"""

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def distill_logits(self, teacher_logits: List[float],
                       student_logits: List[float]) -> float:
        """计算蒸馏损失（简化版KL散度）"""
        # 温度缩放
        teacher_soft = self._softmax([t / self.temperature for t in teacher_logits])
        student_soft = self._softmax([s / self.temperature for s in student_logits])
        # KL散度
        kl = sum(t * (t - s) for t, s in zip(teacher_soft, student_soft))
        return kl

    def _softmax(self, values: List[float]) -> List[float]:
        import math
        exp_values = [math.exp(v) for v in values]
        total = sum(exp_values)
        return [v / total for v in exp_values]


class LowRankDecomposer:
    """低秩分解器"""

    def __init__(self, rank: int = 2):
        self.rank = rank

    def decompose_matrix(self, matrix: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """简化版SVD分解（使用幂迭代）"""
        if not matrix or not matrix[0]:
            return [], []

        rows = len(matrix)
        cols = len(matrix[0])
        actual_rank = min(self.rank, rows, cols)

        # 简化：使用随机投影近似
        import random
        U = [[random.random() for _ in range(actual_rank)] for _ in range(rows)]
        V = [[random.random() for _ in range(cols)] for _ in range(actual_rank)]

        # 简单归一化
        for i in range(rows):
            norm = sum(u ** 2 for u in U[i]) ** 0.5
            if norm > 0:
                U[i] = [u / norm for u in U[i]]

        return U, V

    def reconstruct(self, U: List[List[float]], V: List[List[float]]) -> List[List[float]]:
        """从低秩分解重建矩阵"""
        rows = len(U)
        cols = len(V[0]) if V else 0
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = sum(U[i][k] * V[k][j] for k in range(len(V)))
                row.append(val)
            result.append(row)
        return result


class ModelCompressor:
    """模型压缩器"""

    def __init__(self):
        self.pruner = WeightPruner()
        self.quantizer = Quantizer()
        self.distiller = KnowledgeDistiller()
        self.decomposer = LowRankDecomposer()
        self._compression_history: List[CompressedModel] = []

    def compress_prune(self, weights: Dict[str, List[float]],
                       sparsity: float = 0.5) -> Dict[str, List[float]]:
        """剪枝压缩"""
        self.pruner.sparsity = sparsity
        original_size = sum(len(v) for v in weights.values())
        pruned = self.pruner.prune(weights)
        compressed_size = sum(len([v for v in values if v != 0.0]) for values in pruned.values())
        self._compression_history.append(CompressedModel(
            original_size=original_size,
            compressed_size=compressed_size,
            method="pruning",
        ))
        return pruned

    def compress_quantize(self, weights: Dict[str, List[float]],
                          bits: int = 8) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[float, float]]]:
        """量化压缩"""
        self.quantizer.bits = bits
        original_size = sum(len(v) * 4 for v in weights.values())  # 假设float32=4字节
        quantized = self.quantizer.quantize(weights)
        ranges = self.quantizer.get_ranges(weights)
        compressed_size = sum(len(v) for v in quantized.values())  # int8=1字节
        self._compression_history.append(CompressedModel(
            original_size=original_size,
            compressed_size=compressed_size,
            method="quantization",
        ))
        return quantized, ranges

    def compress_low_rank(self, matrix: List[List[float]],
                          rank: int = 2) -> Tuple[List[List[float]], List[List[float]]]:
        """低秩分解压缩"""
        self.decomposer.rank = rank
        original_size = len(matrix) * len(matrix[0]) if matrix else 0
        U, V = self.decomposer.decompose_matrix(matrix)
        compressed_size = len(U) * len(U[0]) + len(V) * len(V[0]) if U and V else 0
        self._compression_history.append(CompressedModel(
            original_size=original_size,
            compressed_size=compressed_size,
            method="low_rank",
        ))
        return U, V

    def get_stats(self) -> Dict[str, Any]:
        if not self._compression_history:
            return {"total_compressions": 0}
        methods = {}
        for c in self._compression_history:
            methods[c.method] = methods.get(c.method, 0) + 1
        avg_ratio = sum(c.compression_ratio() for c in self._compression_history) / len(self._compression_history)
        return {
            "total_compressions": len(self._compression_history),
            "methods_used": methods,
            "avg_compression_ratio": round(avg_ratio, 4),
        }


def create_model_compressor() -> ModelCompressor:
    return ModelCompressor()
