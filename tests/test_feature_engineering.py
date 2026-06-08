"""阶段71: Agent机器学习特征工程系统 - 单元测试"""
import pytest
from src.agent.feature_engineering import (
    FeatureTransformer, FeatureExtractor, FeatureSelector, FeaturePipeline, create_feature_pipeline,
)

class TestFeatureTransformer:
    def test_normalize_minmax(self):
        result = FeatureTransformer.normalize([1, 2, 3, 4, 5])
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_normalize_zscore(self):
        result = FeatureTransformer.normalize([1, 2, 3, 4, 5], method="zscore")
        assert abs(result[2]) < 0.01  # middle value near 0

    def test_bucketize(self):
        result = FeatureTransformer.bucketize([1, 2, 3, 4, 5], bins=2)
        assert all(0 <= r <= 1 for r in result)

    def test_one_hot_encode(self):
        result = FeatureTransformer.one_hot_encode(["a", "b", "a"])
        assert result["a"] == [1, 0, 1]
        assert result["b"] == [0, 1, 0]

    def test_log_transform(self):
        result = FeatureTransformer.log_transform([0, 1, 2])
        assert result[0] == 0.0  # log(1) = 0

    def test_polynomial_features(self):
        result = FeatureTransformer.polynomial_features([2], degree=3)
        assert result[0] == [2, 4, 8]

class TestFeatureExtractor:
    def test_statistical_features(self):
        result = FeatureExtractor.statistical_features([1, 2, 3, 4, 5])
        assert result["mean"] == 3.0
        assert result["min"] == 1
        assert result["max"] == 5

    def test_text_features(self):
        result = FeatureExtractor.text_features("hello world hello")
        assert result["word_count"] == 3
        assert result["unique_words"] == 2

    def test_time_features(self):
        import time
        ts = time.mktime((2024, 1, 15, 10, 0, 0, 0, 0, 0))
        result = FeatureExtractor.time_features(ts)
        assert result["hour"] == 10
        assert result["month"] == 1

class TestFeatureSelector:
    def test_variance_threshold(self):
        features = {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 1, 1, 1, 1],
        }
        result = FeatureSelector.variance_threshold(features, threshold=0.1)
        assert "a" in result
        assert "b" not in result

    def test_correlation_filter(self):
        features = {
            "x1": [1, 2, 3, 4, 5],
            "x2": [5, 4, 3, 2, 1],
            "x3": [1, 1, 1, 1, 1],
        }
        target = [1, 2, 3, 4, 5]
        result = FeatureSelector.correlation_filter(features, target, top_k=2)
        assert "x1" in result

class TestFeaturePipeline:
    def test_transform(self):
        p = FeaturePipeline()
        p.add_step("add_x", lambda d: {**d, "x2": d["x"] * 2})
        result = p.transform({"x": 5})
        assert result["x2"] == 10

    def test_fit_transform(self):
        p = FeaturePipeline()
        p.add_step("add_one", lambda d: {**d, "y": d["x"] + 1})
        results = p.fit_transform([{"x": 1}, {"x": 2}])
        assert results[0]["y"] == 2
        assert results[1]["y"] == 3

    def test_create_feature_pipeline(self):
        p = create_feature_pipeline()
        assert isinstance(p, FeaturePipeline)
