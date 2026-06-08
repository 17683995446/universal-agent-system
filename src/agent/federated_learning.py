"""
阶段85: Agent联邦学习基础系统

遵循"图难于其易，为大于其细"的原则，实现客户端管理、本地训练、模型聚合、差分隐私。
"""
import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ClientModel:
    """客户端模型"""
    client_id: str
    weights: Dict[str, List[float]] = field(default_factory=dict)
    sample_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "sample_count": self.sample_count,
            "metrics": self.metrics,
        }


@dataclass
class GlobalModel:
    """全局模型"""
    model_id: str
    weights: Dict[str, List[float]] = field(default_factory=dict)
    version: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "metrics": self.metrics,
        }


class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.local_data: List[Any] = []
        self.local_model: Optional[ClientModel] = None

    def load_data(self, data: List[Any]) -> None:
        self.local_data = data

    def local_train(self, global_weights: Dict[str, List[float]],
                    epochs: int = 1) -> ClientModel:
        """本地训练（简化版：对权重进行微小扰动模拟训练）"""
        local_weights = copy.deepcopy(global_weights)
        for key in local_weights:
            for i in range(len(local_weights[key])):
                # 模拟本地更新：添加基于数据量的微小变化
                local_weights[key][i] += 0.01 * (hash(self.client_id) % 10 - 5) / 100

        self.local_model = ClientModel(
            client_id=self.client_id,
            weights=local_weights,
            sample_count=len(self.local_data),
            metrics={"loss": 0.5, "accuracy": 0.8},
        )
        return self.local_model

    def get_stats(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "data_size": len(self.local_data),
            "has_model": self.local_model is not None,
        }


class AggregationStrategy:
    """聚合策略"""

    @staticmethod
    def fed_avg(client_models: List[ClientModel]) -> Dict[str, List[float]]:
        """FedAvg聚合"""
        if not client_models:
            return {}

        total_samples = sum(m.sample_count for m in client_models)
        if total_samples == 0:
            return client_models[0].weights

        # 获取所有权重键
        keys = client_models[0].weights.keys()
        aggregated = {}

        for key in keys:
            # 加权平均
            aggregated[key] = []
            weight_len = len(client_models[0].weights[key])
            for i in range(weight_len):
                weighted_sum = sum(
                    m.weights[key][i] * m.sample_count / total_samples
                    for m in client_models
                )
                aggregated[key].append(weighted_sum)

        return aggregated

    @staticmethod
    def simple_avg(client_models: List[ClientModel]) -> Dict[str, List[float]]:
        """简单平均"""
        if not client_models:
            return {}
        keys = client_models[0].weights.keys()
        aggregated = {}
        n = len(client_models)
        for key in keys:
            aggregated[key] = []
            weight_len = len(client_models[0].weights[key])
            for i in range(weight_len):
                avg = sum(m.weights[key][i] for m in client_models) / n
                aggregated[key].append(avg)
        return aggregated


class PrivacyPreserver:
    """隐私保护器（差分隐私简化版）"""

    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale

    def add_noise(self, weights: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """向权重添加高斯噪声"""
        import random
        noisy = {}
        for key, values in weights.items():
            noisy[key] = [v + random.gauss(0, self.noise_scale) for v in values]
        return noisy

    def clip_weights(self, weights: Dict[str, List[float]],
                     max_norm: float = 1.0) -> Dict[str, List[float]]:
        """权重裁剪"""
        # 计算L2范数
        all_values = []
        for values in weights.values():
            all_values.extend(values)
        norm = sum(v ** 2 for v in all_values) ** 0.5
        if norm > max_norm:
            scale = max_norm / norm
            clipped = {}
            for key, values in weights.items():
                clipped[key] = [v * scale for v in values]
            return clipped
        return weights


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, aggregation: str = "fed_avg"):
        self.global_model: Optional[GlobalModel] = None
        self.clients: Dict[str, FederatedClient] = {}
        self._aggregation = aggregation
        self.privacy = PrivacyPreserver()
        self.round_history: List[Dict[str, Any]] = []

    def initialize_model(self, weights: Dict[str, List[float]]) -> GlobalModel:
        model_id = str(uuid.uuid4())[:8]
        self.global_model = GlobalModel(model_id=model_id, weights=weights, version=0)
        return self.global_model

    def register_client(self, client_id: str) -> FederatedClient:
        client = FederatedClient(client_id)
        self.clients[client_id] = client
        return client

    def run_round(self, client_ids: Optional[List[str]] = None,
                  epochs: int = 1, apply_privacy: bool = False) -> GlobalModel:
        """执行一轮联邦学习"""
        if not self.global_model:
            raise RuntimeError("Global model not initialized")

        selected_clients = client_ids or list(self.clients.keys())
        client_models = []

        for cid in selected_clients:
            client = self.clients.get(cid)
            if not client:
                continue
            local_model = client.local_train(self.global_model.weights, epochs)
            if apply_privacy:
                local_model.weights = self.privacy.add_noise(local_model.weights)
            client_models.append(local_model)

        if not client_models:
            return self.global_model

        # 聚合
        if self._aggregation == "fed_avg":
            new_weights = AggregationStrategy.fed_avg(client_models)
        else:
            new_weights = AggregationStrategy.simple_avg(client_models)

        self.global_model.weights = new_weights
        self.global_model.version += 1

        # 记录历史
        self.round_history.append({
            "round": self.global_model.version,
            "participants": len(client_models),
            "total_samples": sum(m.sample_count for m in client_models),
        })

        return self.global_model

    def evaluate(self, client_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估全局模型"""
        if not self.global_model:
            return {}
        selected = client_ids or list(self.clients.keys())
        results = {}
        for cid in selected:
            client = self.clients.get(cid)
            if client and client.local_model:
                results[cid] = client.local_model.metrics
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "client_count": len(self.clients),
            "model_version": self.global_model.version if self.global_model else 0,
            "rounds_completed": len(self.round_history),
            "has_global_model": self.global_model is not None,
        }


class FederatedLearningSystem:
    """联邦学习系统"""

    def __init__(self):
        self.server = FederatedServer()
        self._clients_data: Dict[str, List[Any]] = {}

    def setup(self, initial_weights: Dict[str, List[float]]) -> None:
        self.server.initialize_model(initial_weights)

    def add_client(self, client_id: str, data: List[Any]) -> FederatedClient:
        client = self.server.register_client(client_id)
        client.load_data(data)
        return client

    def train(self, rounds: int = 5, clients_per_round: Optional[int] = None,
              apply_privacy: bool = False) -> List[Dict[str, Any]]:
        """训练多轮"""
        all_clients = list(self.server.clients.keys())
        for r in range(rounds):
            if clients_per_round and clients_per_round < len(all_clients):
                import random
                selected = random.sample(all_clients, clients_per_round)
            else:
                selected = all_clients
            self.server.run_round(selected, apply_privacy=apply_privacy)
        return self.server.round_history

    def get_global_model(self) -> Optional[GlobalModel]:
        return self.server.global_model

    def get_stats(self) -> Dict[str, Any]:
        return self.server.get_stats()


def create_federated_learning_system() -> FederatedLearningSystem:
    return FederatedLearningSystem()
