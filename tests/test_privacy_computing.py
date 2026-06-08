"""
阶段93测试: 隐私计算基础系统
"""
import pytest
from src.agent.privacy_computing import (
    PrivacyBudget, DifferentialPrivacy, SecureAggregator,
    HomomorphicEncryptionApprox, DataAnonymizer,
    PrivacyComputingEngine, create_privacy_engine,
)


class TestPrivacyBudget:
    def test_remaining(self):
        pb = PrivacyBudget(epsilon=10.0, spent=3.0)
        assert pb.remaining() == 7.0

    def test_can_spend(self):
        pb = PrivacyBudget(epsilon=5.0)
        assert pb.can_spend(3.0) is True
        assert pb.can_spend(6.0) is False

    def test_spend(self):
        pb = PrivacyBudget(epsilon=5.0)
        assert pb.spend(2.0) is True
        assert pb.spent == 2.0
        assert pb.spend(4.0) is False

    def test_to_dict(self):
        pb = PrivacyBudget(epsilon=1.0, delta=0.01, spent=0.5)
        d = pb.to_dict()
        assert d["epsilon"] == 1.0
        assert d["delta"] == 0.01
        assert d["spent"] == 0.5
        assert d["remaining"] == 0.5


class TestDifferentialPrivacy:
    def test_add_laplace_noise(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        noisy = dp.add_laplace_noise(10.0, sensitivity=1.0)
        assert isinstance(noisy, float)
        # 噪声不应该太大
        assert abs(noisy - 10.0) < 20.0

    def test_add_gaussian_noise(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=0.01)
        noisy = dp.add_gaussian_noise(10.0, sensitivity=1.0)
        assert isinstance(noisy, float)

    def test_clip(self):
        dp = DifferentialPrivacy()
        assert dp.clip(150.0, 0.0, 100.0) == 100.0
        assert dp.clip(-10.0, 0.0, 100.0) == 0.0
        assert dp.clip(50.0, 0.0, 100.0) == 50.0

    def test_private_mean(self):
        dp = DifferentialPrivacy(epsilon=10.0)
        data = [10.0, 20.0, 30.0]
        result = dp.private_mean(data, 0.0, 100.0)
        assert 0.0 <= result <= 100.0
        assert abs(result - 20.0) < 10.0  # 噪声较小

    def test_private_sum(self):
        dp = DifferentialPrivacy(epsilon=10.0)
        data = [10.0, 20.0, 30.0]
        result = dp.private_sum(data, 0.0, 100.0)
        assert abs(result - 60.0) < 50.0

    def test_get_budget_status(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        status = dp.get_budget_status()
        assert status["epsilon"] == 1.0


class TestSecureAggregator:
    def test_generate_mask(self):
        agg = SecureAggregator(n_parties=2)
        m1 = agg.generate_mask(0, seed=42)
        m2 = agg.generate_mask(1, seed=42)
        assert isinstance(m1, float)
        assert -1.0 <= m1 <= 1.0

    def test_masked_input(self):
        agg = SecureAggregator()
        masked = agg.masked_input(0, 10.0, seed=42)
        assert masked != 10.0  # 加了掩码
        assert 0 in agg.masks

    def test_aggregate(self):
        agg = SecureAggregator()
        result = agg.aggregate([1.0, 2.0, 3.0])
        assert result == 6.0

    def test_aggregate_average(self):
        agg = SecureAggregator()
        result = agg.aggregate_average([10.0, 20.0, 30.0])
        assert result == 20.0


class TestHomomorphicEncryptionApprox:
    def test_encrypt_decrypt(self):
        he = HomomorphicEncryptionApprox(modulus=10007)
        plaintext = 42
        encrypted = he.encrypt(plaintext)
        decrypted = he.decrypt(encrypted)
        assert decrypted == plaintext

    def test_add_encrypted(self):
        he = HomomorphicEncryptionApprox(modulus=10007)
        c1 = he.encrypt(10)
        c2 = he.encrypt(20)
        c_sum = he.add_encrypted(c1, c2)
        result = he.decrypt(c_sum)
        assert result == 30

    def test_add_plain(self):
        he = HomomorphicEncryptionApprox(modulus=10007)
        c = he.encrypt(10)
        c2 = he.add_plain(c, 5)
        assert he.decrypt(c2) == 15

    def test_multiply_plain(self):
        he = HomomorphicEncryptionApprox(modulus=10007)
        c = he.encrypt(10)
        c2 = he.multiply_plain(c, 3)
        assert he.decrypt(c2) == 30


class TestDataAnonymizer:
    def test_pseudonymize(self):
        anon = DataAnonymizer()
        p1 = anon.pseudonymize("alice")
        p2 = anon.pseudonymize("bob")
        p3 = anon.pseudonymize("alice")
        assert p1 != "alice"
        assert p1 == p3  # 同一输入得到同一输出
        assert p1 != p2

    def test_generalize_numeric(self):
        anon = DataAnonymizer()
        bins = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        assert anon.generalize_numeric(5.0, bins) == "[0.0-10.0]"
        assert anon.generalize_numeric(15.0, bins) == "[10.0-20.0]"
        assert anon.generalize_numeric(35.0, bins) == "other"

    def test_suppress(self):
        anon = DataAnonymizer()
        assert anon.suppress("hello", k=2) == "he***"
        assert anon.suppress("ab", k=3) == "**"

    def test_shuffle(self):
        anon = DataAnonymizer()
        data = [1, 2, 3, 4, 5]
        shuffled = anon.shuffle(data)
        assert sorted(shuffled) == data

    def test_get_stats(self):
        anon = DataAnonymizer()
        anon.pseudonymize("x")
        stats = anon.get_stats()
        assert stats["pseudonymized_ids"] == 1


class TestPrivacyComputingEngine:
    def test_private_query_mean(self):
        engine = PrivacyComputingEngine(epsilon=10.0)
        data = [10.0, 20.0, 30.0]
        result = engine.private_query(data, "mean", 0.0, 100.0)
        assert abs(result - 20.0) < 10.0

    def test_private_query_sum(self):
        engine = PrivacyComputingEngine(epsilon=10.0)
        data = [10.0, 20.0, 30.0]
        result = engine.private_query(data, "sum", 0.0, 100.0)
        assert abs(result - 60.0) < 30.0

    def test_secure_aggregate(self):
        engine = PrivacyComputingEngine()
        values = [1.0, 2.0, 3.0]
        result = engine.secure_aggregate(values, [0, 1, 2])
        assert isinstance(result, float)

    def test_encrypt_sum(self):
        engine = PrivacyComputingEngine()
        result = engine.encrypt_sum([10, 20, 30])
        assert result == 60

    def test_anonymize_data(self):
        engine = PrivacyComputingEngine()
        result = engine.anonymize_data(["alice", "bob", "alice"])
        assert len(result) == 3
        assert result[0] == result[2]  # alice 映射到同一ID

    def test_get_stats(self):
        engine = PrivacyComputingEngine(epsilon=1.0)
        engine.private_query([1.0, 2.0], "mean")
        stats = engine.get_stats()
        assert stats["operations"] >= 1
        assert "dp_budget" in stats


class TestCreatePrivacyEngine:
    def test_factory(self):
        engine = create_privacy_engine(epsilon=0.5)
        assert isinstance(engine, PrivacyComputingEngine)
        assert engine.dp.epsilon == 0.5
