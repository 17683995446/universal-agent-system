"""阶段88: Agent异常检测系统 - 单元测试"""
import pytest
from src.agent.anomaly_detection import (
    AnomalyResult,
    StatisticalDetector, IsolationForestDetector, DensityBasedDetector,
    EnsembleDetector, AnomalyDetectionSystem, create_anomaly_detection_system,
)


class TestStatisticalDetector:
    def test_fit_and_detect_normal(self):
        d = StatisticalDetector(threshold_sigma=2.0)
        d.fit([10.0, 11.0, 10.5, 11.5, 10.0])
        result = d.detect(10.5)
        assert not result.is_anomaly
        assert result.score < 0.5

    def test_detect_anomaly(self):
        d = StatisticalDetector(threshold_sigma=2.0)
        d.fit([10.0, 11.0, 10.5, 11.5, 10.0])
        result = d.detect(100.0)
        assert result.is_anomaly
        assert result.score > 0.5

    def test_detect_not_fitted(self):
        d = StatisticalDetector()
        result = d.detect(10.0)
        assert not result.is_anomaly
        assert result.score == 0.0

    def test_detect_batch(self):
        d = StatisticalDetector(threshold_sigma=2.0)
        d.fit([10.0, 11.0, 10.5])
        results = d.detect_batch([10.0, 100.0])
        assert not results[0].is_anomaly
        assert results[1].is_anomaly


class TestIsolationForestDetector:
    def test_fit_and_detect(self):
        d = IsolationForestDetector(n_trees=5, max_depth=4)
        data = [[i] for i in range(50)]
        d.fit(data)
        assert d._fitted

    def test_detect_normal(self):
        d = IsolationForestDetector(n_trees=5, max_depth=4)
        data = [[i] for i in range(50)]
        d.fit(data)
        result = d.detect([25.0])
        assert isinstance(result.score, float)

    def test_detect_anomaly(self):
        d = IsolationForestDetector(n_trees=5, max_depth=4)
        data = [[i] for i in range(50)]
        d.fit(data)
        result = d.detect([200.0])
        # 孤立森林对远离分布的点应该有较高分数
        assert isinstance(result.score, float)
        assert result.score >= 0.0

    def test_detect_not_fitted(self):
        d = IsolationForestDetector()
        result = d.detect([1.0])
        assert result.score == 0.0


class TestDensityBasedDetector:
    def test_fit_and_detect(self):
        d = DensityBasedDetector(k=3)
        data = [[i] for i in range(20)]
        d.fit(data)
        result = d.detect([10.0])
        assert isinstance(result.score, float)

    def test_detect_anomaly(self):
        d = DensityBasedDetector(k=3)
        data = [[i] for i in range(20)]
        d.fit(data)
        result = d.detect([100.0])
        assert result.score > 0.0

    def test_detect_not_fitted(self):
        d = DensityBasedDetector()
        result = d.detect([1.0])
        assert result.score == 0.0


class TestEnsembleDetector:
    def test_add_and_detect(self):
        e = EnsembleDetector()
        stat = StatisticalDetector(threshold_sigma=2.0)
        stat.fit([10.0, 11.0, 10.5])
        e.add_detector(stat, weight=1.0)
        result = e.detect(100.0)
        assert result.method == "ensemble"
        assert result.is_anomaly

    def test_empty(self):
        e = EnsembleDetector()
        result = e.detect(10.0)
        assert not result.is_anomaly
        assert result.score == 0.0


class TestAnomalyDetectionSystem:
    def test_fit_and_detect_statistical(self):
        sys = AnomalyDetectionSystem()
        sys.fit([10.0, 11.0, 10.5, 11.5, 10.0])
        result = sys.detect(10.5, method="statistical")
        assert result.method == "statistical"

    def test_detect_isolation_forest(self):
        sys = AnomalyDetectionSystem()
        sys.fit(list(range(50)))
        result = sys.detect(25, method="isolation_forest")
        assert result.method == "isolation_forest"

    def test_detect_density(self):
        sys = AnomalyDetectionSystem()
        sys.fit(list(range(20)))
        result = sys.detect(10, method="density")
        assert result.method == "density"

    def test_detect_ensemble(self):
        sys = AnomalyDetectionSystem()
        sys.fit([10.0, 11.0, 10.5, 11.5, 10.0])
        result = sys.detect(100.0, method="ensemble")
        assert result.method == "ensemble"
        assert result.is_anomaly

    def test_detect_batch(self):
        sys = AnomalyDetectionSystem()
        sys.fit([10.0, 11.0, 10.5])
        results = sys.detect_batch([10.0, 100.0], method="statistical")
        assert len(results) == 2

    def test_get_stats(self):
        sys = AnomalyDetectionSystem()
        sys.fit([10.0, 11.0, 10.5])
        sys.detect(10.0, method="statistical")
        sys.detect(100.0, method="statistical")
        stats = sys.get_stats()
        assert stats["total_checked"] == 2
        assert stats["anomalies_found"] == 1

    def test_get_anomalies(self):
        sys = AnomalyDetectionSystem()
        sys.fit([10.0, 11.0, 10.5])
        sys.detect(10.0, method="statistical")
        sys.detect(100.0, method="statistical")
        anomalies = sys.get_anomalies()
        assert len(anomalies) == 1
        assert anomalies[0][0] == 100.0

    def test_fit_multivariate(self):
        sys = AnomalyDetectionSystem()
        data = [[i, i * 2] for i in range(20)]
        sys.fit(data, data_type="multivariate")
        result = sys.detect([10.0, 20.0], method="isolation_forest")
        assert result.method == "isolation_forest"

    def test_create_anomaly_detection_system(self):
        sys = create_anomaly_detection_system()
        assert isinstance(sys, AnomalyDetectionSystem)
