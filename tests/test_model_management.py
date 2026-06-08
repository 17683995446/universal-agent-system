"""阶段72: Agent机器学习模型管理系统 - 单元测试"""
import pytest
from src.agent.model_management import (
    ModelStatus, ModelFramework, ModelVersion, ModelArtifact, ModelRegistry, create_model_registry,
)

class TestModelRegistry:
    def test_register(self):
        r = ModelRegistry()
        m = r.register("classifier", ModelFramework.SKLEARN)
        assert len(m.artifact_id) == 8
        assert m.name == "classifier"

    def test_add_version(self):
        r = ModelRegistry()
        m = r.register("classifier")
        m.add_version(ModelVersion(version="1.0", metrics={"accuracy": 0.9}))
        assert "1.0" in m.versions
        assert m.current_version == "1.0"

    def test_get_version(self):
        r = ModelRegistry()
        m = r.register("classifier")
        m.add_version(ModelVersion(version="1.0"))
        assert m.get_version("1.0") is not None
        assert m.get_version("2.0") is None

    def test_promote_version(self):
        r = ModelRegistry()
        m = r.register("classifier")
        m.add_version(ModelVersion(version="1.0"))
        assert r.promote_version(m.artifact_id, "1.0", ModelStatus.PRODUCTION)
        assert m.get_version("1.0").status == ModelStatus.PRODUCTION
        assert m.current_version == "1.0"

    def test_compare_versions(self):
        r = ModelRegistry()
        m = r.register("classifier")
        m.add_version(ModelVersion(version="1.0", metrics={"acc": 0.8}))
        m.add_version(ModelVersion(version="2.0", metrics={"acc": 0.9}))
        diff = r.compare_versions(m.artifact_id, "1.0", "2.0")
        assert abs(diff["metrics_diff"]["acc"] - 0.1) < 0.001

    def test_get_by_name(self):
        r = ModelRegistry()
        r.register("model_a")
        r.register("model_a")
        assert len(r.get_by_name("model_a")) == 2

    def test_delete(self):
        r = ModelRegistry()
        m = r.register("temp")
        assert r.delete(m.artifact_id)
        assert r.get(m.artifact_id) is None

    def test_stats(self):
        r = ModelRegistry()
        m = r.register("classifier")
        m.add_version(ModelVersion(version="1.0"))
        r.promote_version(m.artifact_id, "1.0", ModelStatus.PRODUCTION)
        stats = r.get_stats()
        assert stats["total_models"] == 1
        assert stats["production_models"] == 1

    def test_create_model_registry(self):
        r = create_model_registry()
        assert isinstance(r, ModelRegistry)
