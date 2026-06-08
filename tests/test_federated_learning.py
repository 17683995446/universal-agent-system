"""阶段85: Agent联邦学习基础系统 - 单元测试"""
import pytest
from src.agent.federated_learning import (
    ClientModel, GlobalModel,
    FederatedClient, AggregationStrategy, PrivacyPreserver,
    FederatedServer, FederatedLearningSystem, create_federated_learning_system,
)


class TestFederatedClient:
    def test_load_data(self):
        c = FederatedClient("c1")
        c.load_data([1, 2, 3])
        assert len(c.local_data) == 3

    def test_local_train(self):
        c = FederatedClient("c1")
        c.load_data([1, 2])
        weights = {"w": [0.5, 0.5]}
        model = c.local_train(weights)
        assert model.client_id == "c1"
        assert model.sample_count == 2
        assert "w" in model.weights

    def test_get_stats(self):
        c = FederatedClient("c1")
        c.load_data([1, 2, 3])
        stats = c.get_stats()
        assert stats["data_size"] == 3
        assert not stats["has_model"]


class TestAggregationStrategy:
    def test_fed_avg(self):
        m1 = ClientModel("c1", weights={"w": [1.0, 2.0]}, sample_count=10)
        m2 = ClientModel("c2", weights={"w": [3.0, 4.0]}, sample_count=30)
        result = AggregationStrategy.fed_avg([m1, m2])
        # (1*10 + 3*30) / 40 = 2.5, (2*10 + 4*30) / 40 = 3.5
        assert result["w"][0] == 2.5
        assert result["w"][1] == 3.5

    def test_fed_avg_empty(self):
        assert AggregationStrategy.fed_avg([]) == {}

    def test_simple_avg(self):
        m1 = ClientModel("c1", weights={"w": [1.0, 2.0]})
        m2 = ClientModel("c2", weights={"w": [3.0, 4.0]})
        result = AggregationStrategy.simple_avg([m1, m2])
        assert result["w"][0] == 2.0
        assert result["w"][1] == 3.0


class TestPrivacyPreserver:
    def test_add_noise(self):
        p = PrivacyPreserver(noise_scale=0.1)
        weights = {"w": [1.0, 2.0, 3.0]}
        noisy = p.add_noise(weights)
        assert len(noisy["w"]) == 3
        # 噪声应该使值发生变化
        assert noisy["w"] != weights["w"]

    def test_clip_weights(self):
        p = PrivacyPreserver()
        weights = {"w": [3.0, 4.0]}  # norm = 5
        clipped = p.clip_weights(weights, max_norm=1.0)
        # 应该被缩放到范数为1
        norm = sum(v ** 2 for v in clipped["w"]) ** 0.5
        assert abs(norm - 1.0) < 0.001

    def test_clip_weights_no_clip(self):
        p = PrivacyPreserver()
        weights = {"w": [0.3, 0.4]}  # norm = 0.5 < 1
        clipped = p.clip_weights(weights, max_norm=1.0)
        assert clipped["w"] == weights["w"]


class TestFederatedServer:
    def test_initialize_model(self):
        s = FederatedServer()
        model = s.initialize_model({"w": [0.5]})
        assert model.version == 0
        assert s.global_model is not None

    def test_register_client(self):
        s = FederatedServer()
        c = s.register_client("c1")
        assert c.client_id == "c1"
        assert "c1" in s.clients

    def test_run_round(self):
        s = FederatedServer()
        s.initialize_model({"w": [0.0, 0.0]})
        c = s.register_client("c1")
        c.load_data([1, 2, 3])
        model = s.run_round()
        assert model.version == 1
        assert len(s.round_history) == 1

    def test_run_round_no_model(self):
        s = FederatedServer()
        with pytest.raises(RuntimeError):
            s.run_round()

    def test_evaluate(self):
        s = FederatedServer()
        s.initialize_model({"w": [0.0]})
        c = s.register_client("c1")
        c.load_data([1])
        s.run_round()
        results = s.evaluate()
        assert "c1" in results

    def test_get_stats(self):
        s = FederatedServer()
        s.initialize_model({"w": [0.0]})
        s.register_client("c1")
        stats = s.get_stats()
        assert stats["client_count"] == 1
        assert stats["has_global_model"] is True


class TestFederatedLearningSystem:
    def test_setup_and_add_client(self):
        fl = FederatedLearningSystem()
        fl.setup({"w": [0.5]})
        fl.add_client("c1", [1, 2, 3])
        assert fl.get_global_model() is not None
        assert fl.get_stats()["client_count"] == 1

    def test_train(self):
        fl = FederatedLearningSystem()
        fl.setup({"w": [0.0, 0.0]})
        fl.add_client("c1", [1, 2])
        fl.add_client("c2", [3, 4, 5])
        history = fl.train(rounds=3)
        assert len(history) == 3
        assert fl.get_global_model().version == 3

    def test_train_with_privacy(self):
        fl = FederatedLearningSystem()
        fl.setup({"w": [0.0]})
        fl.add_client("c1", [1])
        history = fl.train(rounds=2, apply_privacy=True)
        assert len(history) == 2

    def test_train_subset(self):
        fl = FederatedLearningSystem()
        fl.setup({"w": [0.0]})
        for i in range(5):
            fl.add_client(f"c{i}", [i])
        history = fl.train(rounds=2, clients_per_round=2)
        assert len(history) == 2

    def test_create_federated_learning_system(self):
        fl = create_federated_learning_system()
        assert isinstance(fl, FederatedLearningSystem)
