"""
阶段93: Agent 隐私计算基础系统

遵循"图难于其易，为大于其细"的原则，实现差分隐私、安全聚合、同态加密近似、数据脱敏。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math


@dataclass
class PrivacyBudget:
    """隐私预算"""
    epsilon: float
    delta: float = 0.0
    spent: float = 0.0

    def remaining(self) -> float:
        return max(0.0, self.epsilon - self.spent)

    def can_spend(self, cost: float) -> bool:
        return self.remaining() >= cost

    def spend(self, cost: float) -> bool:
        if not self.can_spend(cost):
            return False
        self.spent += cost
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "spent": round(self.spent, 4),
            "remaining": round(self.remaining(), 4),
        }


class DifferentialPrivacy:
    """差分隐私机制"""

    def __init__(self, epsilon: float = 1.0, delta: float = 0.0):
        self.epsilon = epsilon
        self.delta = delta
        self.budget = PrivacyBudget(epsilon, delta)

    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """添加拉普拉斯噪声"""
        if self.epsilon <= 0:
            return value
        scale = sensitivity / self.epsilon
        # 拉普拉斯分布：使用两个独立指数分布
        u = random.random() - 0.5
        noise = -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))
        return value + noise

    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """添加高斯噪声"""
        if self.epsilon <= 0:
            return value
        # 简化：sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        if self.delta > 0:
            sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        else:
            sigma = sensitivity / self.epsilon
        noise = random.gauss(0, sigma)
        return value + noise

    def clip(self, value: float, lower: float, upper: float) -> float:
        """裁剪值到范围"""
        return max(lower, min(upper, value))

    def private_mean(self, data: List[float], lower: float, upper: float) -> float:
        """差分隐私均值"""
        clipped = [self.clip(v, lower, upper) for v in data]
        sensitivity = (upper - lower) / len(clipped) if clipped else 0.0
        true_mean = sum(clipped) / len(clipped) if clipped else 0.0
        return self.add_laplace_noise(true_mean, sensitivity)

    def private_sum(self, data: List[float], lower: float, upper: float) -> float:
        """差分隐私求和"""
        clipped = [self.clip(v, lower, upper) for v in data]
        sensitivity = upper - lower
        true_sum = sum(clipped)
        return self.add_laplace_noise(true_sum, sensitivity)

    def get_budget_status(self) -> Dict[str, Any]:
        return self.budget.to_dict()


class SecureAggregator:
    """安全聚合器（简化版）"""

    def __init__(self, n_parties: int = 2):
        self.n_parties = n_parties
        self.masks: Dict[int, float] = {}

    def generate_mask(self, party_id: int, seed: int) -> float:
        """生成随机掩码"""
        random.seed(seed + party_id)
        return random.random() * 2 - 1  # [-1, 1]

    def masked_input(self, party_id: int, value: float, seed: int) -> float:
        """添加掩码后的输入"""
        mask = self.generate_mask(party_id, seed)
        self.masks[party_id] = mask
        return value + mask

    def aggregate(self, masked_values: List[float]) -> float:
        """聚合（掩码在求和后会抵消，简化版假设掩码和为0）"""
        return sum(masked_values)

    def aggregate_average(self, masked_values: List[float]) -> float:
        """安全平均"""
        if not masked_values:
            return 0.0
        return sum(masked_values) / len(masked_values)


class HomomorphicEncryptionApprox:
    """同态加密近似（简化版，仅演示加法同态）"""

    def __init__(self, modulus: int = 10007):
        self.modulus = modulus
        self.secret_key = random.randint(1, modulus - 1)

    def encrypt(self, plaintext: int) -> int:
        """加密：plaintext + noise * modulus（简化Paillier思想）"""
        noise = random.randint(1, 100)
        return (plaintext + noise * self.modulus) % (self.modulus ** 2)

    def decrypt(self, ciphertext: int) -> int:
        """解密"""
        return ciphertext % self.modulus

    def add_encrypted(self, c1: int, c2: int) -> int:
        """同态加法"""
        return (c1 + c2) % (self.modulus ** 2)

    def add_plain(self, ciphertext: int, plaintext: int) -> int:
        """密文 + 明文"""
        return (ciphertext + plaintext) % (self.modulus ** 2)

    def multiply_plain(self, ciphertext: int, plaintext: int) -> int:
        """密文 * 明文"""
        return (ciphertext * plaintext) % (self.modulus ** 2)


class DataAnonymizer:
    """数据脱敏器"""

    def __init__(self):
        self._id_map: Dict[str, str] = {}
        self._counter = 0

    def pseudonymize(self, identifier: str) -> str:
        """伪匿名化"""
        if identifier not in self._id_map:
            self._counter += 1
            self._id_map[identifier] = f"ID_{self._counter}"
        return self._id_map[identifier]

    def generalize_numeric(self, value: float, bins: List[Tuple[float, float]]) -> str:
        """数值泛化"""
        for low, high in bins:
            if low <= value < high:
                return f"[{low}-{high}]"
        return "other"

    def suppress(self, value: str, k: int = 2) -> str:
        """抑制（k-匿名简化版）"""
        # 简化：如果值长度小于k，返回*
        if len(value) <= k:
            return "*" * len(value)
        return value[:k] + "*" * (len(value) - k)

    def shuffle(self, data: List[Any]) -> List[Any]:
        """行级打乱"""
        shuffled = data.copy()
        random.shuffle(shuffled)
        return shuffled

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pseudonymized_ids": len(self._id_map),
        }


class PrivacyComputingEngine:
    """隐私计算引擎"""

    def __init__(self, epsilon: float = 1.0, delta: float = 0.0):
        self.dp = DifferentialPrivacy(epsilon, delta)
        self.aggregator = SecureAggregator()
        self.he = HomomorphicEncryptionApprox()
        self.anonymizer = DataAnonymizer()
        self._history: List[Dict[str, Any]] = []

    def private_query(self, data: List[float], query_type: str = "mean",
                      lower: float = 0.0, upper: float = 100.0) -> float:
        """差分隐私查询"""
        if query_type == "mean":
            result = self.dp.private_mean(data, lower, upper)
        elif query_type == "sum":
            result = self.dp.private_sum(data, lower, upper)
        else:
            result = self.dp.private_mean(data, lower, upper)
        self._history.append({"type": "dp_query", "query_type": query_type})
        return result

    def secure_aggregate(self, values: List[float], party_seeds: List[int]) -> float:
        """安全聚合"""
        masked = []
        for i, v in enumerate(values):
            seed = party_seeds[i] if i < len(party_seeds) else i
            masked.append(self.aggregator.masked_input(i, v, seed))
        result = self.aggregator.aggregate(masked)
        self._history.append({"type": "secure_aggregate", "parties": len(values)})
        return result

    def encrypt_sum(self, values: List[int]) -> int:
        """同态加密求和"""
        encrypted = [self.he.encrypt(v) for v in values]
        total = encrypted[0] if encrypted else self.he.encrypt(0)
        for c in encrypted[1:]:
            total = self.he.add_encrypted(total, c)
        result = self.he.decrypt(total)
        self._history.append({"type": "he_sum", "count": len(values)})
        return result

    def anonymize_data(self, identifiers: List[str]) -> List[str]:
        """脱敏标识符"""
        result = [self.anonymizer.pseudonymize(id_) for id_ in identifiers]
        self._history.append({"type": "anonymize", "count": len(identifiers)})
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "operations": len(self._history),
            "dp_budget": self.dp.get_budget_status(),
            "anonymizer": self.anonymizer.get_stats(),
        }


def create_privacy_engine(epsilon: float = 1.0) -> PrivacyComputingEngine:
    return PrivacyComputingEngine(epsilon)
