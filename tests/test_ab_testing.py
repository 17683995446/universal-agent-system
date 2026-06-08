"""阶段74: Agent机器学习A/B测试系统 - 单元测试"""
import pytest
from src.agent.ab_testing import (
    ExperimentStatus, Variant, Experiment,
    TrafficRouter, SignificanceTest, ExperimentManager, create_experiment_manager,
)


class TestVariant:
    def test_conversion_rate(self):
        v = Variant(variant_id="v1", name="control", traffic=100, conversions=10)
        assert v.conversion_rate() == 0.1

    def test_conversion_rate_zero_traffic(self):
        v = Variant(variant_id="v1", name="control")
        assert v.conversion_rate() == 0.0

    def test_mean_metric(self):
        v = Variant(variant_id="v1", name="control")
        v.metrics["revenue"] = [10.0, 20.0, 30.0]
        assert v.mean_metric("revenue") == 20.0

    def test_to_dict(self):
        v = Variant(variant_id="v1", name="control", traffic=100, conversions=10)
        d = v.to_dict()
        assert d["name"] == "control"
        assert d["conversion_rate"] == 0.1


class TestTrafficRouter:
    def test_assign_consistency(self):
        router = TrafficRouter()
        v1 = Variant(variant_id="a", name="control", weight=0.5)
        v2 = Variant(variant_id="b", name="treatment", weight=0.5)
        # 同一用户应该被分配到同一组
        result1 = router.assign("user_123", [v1, v2])
        result2 = router.assign("user_123", [v1, v2])
        assert result1 == result2

    def test_assign_distribution(self):
        router = TrafficRouter()
        v1 = Variant(variant_id="a", name="control", weight=0.5)
        v2 = Variant(variant_id="b", name="treatment", weight=0.5)
        counts = {"a": 0, "b": 0}
        for i in range(1000):
            v = router.assign(f"user_{i}", [v1, v2])
            counts[v.variant_id] += 1
        # 大致均匀分布
        assert 400 < counts["a"] < 600
        assert 400 < counts["b"] < 600

    def test_assign_empty(self):
        router = TrafficRouter()
        assert router.assign("user_1", []) is None


class TestSignificanceTest:
    def test_z_test_same_rate(self):
        z, p = SignificanceTest.z_test(0.1, 100, 0.1, 100)
        assert z == 0.0
        assert p == 1.0

    def test_z_test_different_rate(self):
        z, p = SignificanceTest.z_test(0.1, 1000, 0.15, 1000)
        assert z < 0  # control rate lower
        assert p < 1.0

    def test_z_test_zero_sample(self):
        z, p = SignificanceTest.z_test(0.0, 0, 0.1, 100)
        assert z == 0.0
        assert p == 1.0


class TestExperimentManager:
    def test_create(self):
        mgr = ExperimentManager()
        exp = mgr.create("test_exp", "conversion")
        assert exp.name == "test_exp"
        assert exp.status == ExperimentStatus.DRAFT

    def test_get(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        assert mgr.get(exp.experiment_id) == exp

    def test_add_variant(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        v = mgr.add_variant(exp.experiment_id, "control", 0.5)
        assert v is not None
        assert v.name == "control"

    def test_start_requires_two_variants(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        assert not mgr.start(exp.experiment_id)  # 只有一个变体
        mgr.add_variant(exp.experiment_id, "treatment")
        assert mgr.start(exp.experiment_id)

    def test_pause_and_stop(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "a")
        mgr.add_variant(exp.experiment_id, "b")
        mgr.start(exp.experiment_id)
        assert mgr.pause(exp.experiment_id)
        assert exp.status == ExperimentStatus.PAUSED
        assert mgr.stop(exp.experiment_id)
        assert exp.status == ExperimentStatus.COMPLETED

    def test_assign_user(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        mgr.start(exp.experiment_id)
        variant_id = mgr.assign_user(exp.experiment_id, "user_1")
        assert variant_id is not None
        assert variant_id.startswith(exp.experiment_id)

    def test_assign_user_not_running(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        # 未启动
        assert mgr.assign_user(exp.experiment_id, "user_1") is None

    def test_record_conversion(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        mgr.start(exp.experiment_id)
        variant_id = mgr.assign_user(exp.experiment_id, "user_1")
        assert mgr.record_conversion(exp.experiment_id, variant_id, 1.0)

    def test_record_metric(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        mgr.start(exp.experiment_id)
        variant_id = mgr.assign_user(exp.experiment_id, "user_1")
        assert mgr.record_metric(exp.experiment_id, variant_id, "revenue", 100.0)

    def test_get_results(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        mgr.start(exp.experiment_id)
        # 模拟流量和转化
        for i in range(100):
            vid = mgr.assign_user(exp.experiment_id, f"user_{i}")
            mgr.record_conversion(exp.experiment_id, vid, 1.0 if i % 2 == 0 else 0.0)
        results = mgr.get_results(exp.experiment_id)
        assert "experiment" in results
        assert "comparisons" in results
        assert len(results["comparisons"]) == 1

    def test_get_best_variant(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        mgr.add_variant(exp.experiment_id, "control")
        mgr.add_variant(exp.experiment_id, "treatment")
        mgr.start(exp.experiment_id)
        # treatment 转化率更高
        for i in range(100):
            vid = mgr.assign_user(exp.experiment_id, f"user_{i}")
            if "treatment" in vid:
                mgr.record_conversion(exp.experiment_id, vid, 1.0)
            else:
                mgr.record_conversion(exp.experiment_id, vid, 0.0)
        best = mgr.get_best_variant(exp.experiment_id)
        assert best == "treatment"

    def test_list_and_delete(self):
        mgr = ExperimentManager()
        exp = mgr.create("test")
        assert exp.experiment_id in mgr.list_experiments()
        assert mgr.delete(exp.experiment_id)
        assert exp.experiment_id not in mgr.list_experiments()

    def test_create_experiment_manager(self):
        mgr = create_experiment_manager()
        assert isinstance(mgr, ExperimentManager)
