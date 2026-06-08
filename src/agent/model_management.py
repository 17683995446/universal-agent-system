"""
阶段72: Agent机器学习模型管理系统

遵循"图难于其易，为大于其细"的原则，实现模型注册、版本管理、元数据、生命周期。
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class ModelStatus(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class ModelFramework(str, Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM = "custom"

@dataclass
class ModelVersion:
    version: str
    model_data: Any = None
    status: ModelStatus = ModelStatus.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

@dataclass
class ModelArtifact:
    artifact_id: str
    name: str
    framework: ModelFramework = ModelFramework.CUSTOM
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    current_version: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_version(self, version: ModelVersion) -> None:
        self.versions[version.version] = version
        if not self.current_version:
            self.current_version = version.version

    def get_version(self, version: str) -> Optional[ModelVersion]:
        return self.versions.get(version)

    def get_current(self) -> Optional[ModelVersion]:
        return self.versions.get(self.current_version)

    def set_current(self, version: str) -> bool:
        if version in self.versions:
            self.current_version = version
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "framework": self.framework.value,
            "version_count": len(self.versions),
            "current_version": self.current_version,
        }

class ModelRegistry:
    """模型注册表"""
    def __init__(self):
        self._models: Dict[str, ModelArtifact] = {}

    def register(self, name: str, framework: ModelFramework = ModelFramework.CUSTOM,
                 description: str = "", tags: Optional[List[str]] = None) -> ModelArtifact:
        artifact_id = str(uuid.uuid4())[:8]
        artifact = ModelArtifact(
            artifact_id=artifact_id,
            name=name,
            framework=framework,
            description=description,
            tags=tags or [],
        )
        self._models[artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Optional[ModelArtifact]:
        return self._models.get(artifact_id)

    def get_by_name(self, name: str) -> List[ModelArtifact]:
        return [m for m in self._models.values() if m.name == name]

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def delete(self, artifact_id: str) -> bool:
        return self._models.pop(artifact_id, None) is not None

    def promote_version(self, artifact_id: str, version: str,
                        target_status: ModelStatus) -> bool:
        artifact = self._models.get(artifact_id)
        if not artifact:
            return False
        mv = artifact.get_version(version)
        if not mv:
            return False
        mv.status = target_status
        if target_status == ModelStatus.PRODUCTION:
            artifact.set_current(version)
        return True

    def compare_versions(self, artifact_id: str, v1: str, v2: str) -> Dict[str, Any]:
        artifact = self._models.get(artifact_id)
        if not artifact:
            return {}
        mv1 = artifact.get_version(v1)
        mv2 = artifact.get_version(v2)
        if not mv1 or not mv2:
            return {}
        return {
            "version1": v1,
            "version2": v2,
            "metrics_diff": {k: mv2.metrics.get(k, 0) - mv1.metrics.get(k, 0)
                             for k in set(mv1.metrics) | set(mv2.metrics)},
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_models": len(self._models),
            "total_versions": sum(len(m.versions) for m in self._models.values()),
            "production_models": sum(
                1 for m in self._models.values()
                if any(v.status == ModelStatus.PRODUCTION for v in m.versions.values())
            ),
        }

def create_model_registry() -> ModelRegistry:
    return ModelRegistry()
