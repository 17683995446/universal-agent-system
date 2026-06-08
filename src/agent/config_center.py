"""
阶段32: Agent配置中心

遵循"图难于其易，为大于其细"的原则，实现集中式配置管理、版本控制和灰度发布。
"""

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class ConfigStatus(str, Enum):
    """配置状态"""
    DRAFT = "draft"           # 草稿
    PUBLISHED = "published"   # 已发布
    ROLLED_BACK = "rolled_back"  # 已回滚
    DEPRECATED = "deprecated"    # 已废弃


class ConfigScope(str, Enum):
    """配置作用域"""
    GLOBAL = "global"         # 全局
    APPLICATION = "application"  # 应用级
    SERVICE = "service"       # 服务级
    INSTANCE = "instance"     # 实例级


class RolloutStrategy(str, Enum):
    """灰度发布策略"""
    ALL_AT_ONCE = "all_at_once"      # 全量发布
    PERCENTAGE = "percentage"        # 百分比
    CANARY = "canary"                # 金丝雀
    BLUE_GREEN = "blue_green"        # 蓝绿部署
    A_B_TEST = "a_b_test"            # A/B测试


@dataclass
class ConfigValue:
    """配置值"""
    key: str
    value: Any
    value_type: str = "string"
    description: str = ""
    encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": "***encrypted***" if self.encrypted else self.value,
            "value_type": self.value_type,
            "description": self.description,
            "encrypted": self.encrypted,
            "metadata": self.metadata
        }


@dataclass
class ConfigVersion:
    """配置版本"""
    version_id: str
    config_id: str
    version_number: int
    data: Dict[str, Any]
    checksum: str
    created_at: datetime
    created_by: str
    comment: str = ""
    status: ConfigStatus = ConfigStatus.DRAFT
    parent_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version_id": self.version_id,
            "config_id": self.config_id,
            "version_number": self.version_number,
            "data": self.data,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "comment": self.comment,
            "status": self.status.value,
            "parent_version": self.parent_version
        }


@dataclass
class ConfigInstance:
    """配置实例"""
    instance_id: str
    config_id: str
    scope: ConfigScope
    target_id: str  # 应用ID/服务ID/实例ID
    current_version: Optional[str] = None
    previous_version: Optional[str] = None
    applied_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "instance_id": self.instance_id,
            "config_id": self.config_id,
            "scope": self.scope.value,
            "target_id": self.target_id,
            "current_version": self.current_version,
            "previous_version": self.previous_version,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "metadata": self.metadata
        }


@dataclass
class RolloutTarget:
    """灰度发布目标"""
    target_id: str
    target_type: str  # instance, service, region, etc.
    weight: float = 1.0  # 权重 (0-1)
    criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutPlan:
    """灰度发布计划"""
    plan_id: str
    config_id: str
    version_id: str
    strategy: RolloutStrategy
    targets: List[RolloutTarget] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, cancelled
    progress: float = 0.0  # 0-100
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "plan_id": self.plan_id,
            "config_id": self.config_id,
            "version_id": self.version_id,
            "strategy": self.strategy.value,
            "targets": [
                {
                    "target_id": t.target_id,
                    "target_type": t.target_type,
                    "weight": t.weight,
                    "criteria": t.criteria
                }
                for t in self.targets
            ],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    event_id: str
    config_id: str
    version_id: str
    change_type: str  # create, update, delete, publish, rollback
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    operator: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "config_id": self.config_id,
            "version_id": self.version_id,
            "change_type": self.change_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "operator": self.operator
        }


class ConfigStorageBackend:
    """配置存储后端基类"""

    def save(self, key: str, value: Dict[str, Any]) -> bool:
        """保存配置"""
        raise NotImplementedError

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """加载配置"""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """删除配置"""
        raise NotImplementedError

    def list_keys(self, prefix: str = "") -> List[str]:
        """列出所有键"""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        raise NotImplementedError


class MemoryConfigStorage(ConfigStorageBackend):
    """内存配置存储"""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save(self, key: str, value: Dict[str, Any]) -> bool:
        with self._lock:
            self._data[key] = value.copy()
            return True

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        with self._lock:
            return [k for k in self._data.keys() if k.startswith(prefix)]

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data


class FileConfigStorage(ConfigStorageBackend):
    """文件配置存储"""

    def __init__(self, base_path: str = "./config_data"):
        self._base_path = base_path
        self._lock = threading.RLock()
        import os
        os.makedirs(base_path, exist_ok=True)

    def _get_file_path(self, key: str) -> str:
        import os
        # 将键转换为安全的文件名
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self._base_path, f"{safe_key}.json")

    def save(self, key: str, value: Dict[str, Any]) -> bool:
        import os
        with self._lock:
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, default=str)
                return True
            except Exception:
                return False

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        import os
        with self._lock:
            file_path = self._get_file_path(key)
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception:
                pass
            return None

    def delete(self, key: str) -> bool:
        import os
        with self._lock:
            file_path = self._get_file_path(key)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
            except Exception:
                pass
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        import os
        with self._lock:
            keys = []
            try:
                for filename in os.listdir(self._base_path):
                    if filename.endswith('.json'):
                        key = filename[:-5].replace("_", "/")
                        if key.startswith(prefix):
                            keys.append(key)
            except Exception:
                pass
            return keys

    def exists(self, key: str) -> bool:
        import os
        with self._lock:
            return os.path.exists(self._get_file_path(key))


class ConfigCenter:
    """配置中心 - 核心管理类"""

    def __init__(self, storage: Optional[ConfigStorageBackend] = None):
        self._storage = storage or MemoryConfigStorage()
        self._lock = threading.RLock()
        self._version_counter: Dict[str, int] = {}
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self._instances: Dict[str, ConfigInstance] = {}
        self._rollout_plans: Dict[str, RolloutPlan] = {}

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return str(uuid.uuid4())

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """计算数据校验和"""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_next_version_number(self, config_id: str) -> int:
        """获取下一个版本号"""
        with self._lock:
            current = self._version_counter.get(config_id, 0)
            self._version_counter[config_id] = current + 1
            return current + 1

    def create_config(
        self,
        config_id: str,
        data: Dict[str, Any],
        created_by: str = "system",
        comment: str = ""
    ) -> ConfigVersion:
        """创建配置"""
        with self._lock:
            version_number = self._get_next_version_number(config_id)
            version_id = self._generate_id()

            version = ConfigVersion(
                version_id=version_id,
                config_id=config_id,
                version_number=version_number,
                data=data.copy(),
                checksum=self._calculate_checksum(data),
                created_at=datetime.now(),
                created_by=created_by,
                comment=comment,
                status=ConfigStatus.DRAFT
            )

            # 保存到存储
            key = f"config:{config_id}:version:{version_id}"
            self._storage.save(key, version.to_dict())

            # 保存版本索引
            versions_key = f"config:{config_id}:versions"
            versions = self._storage.load(versions_key) or {"versions": []}
            versions["versions"].append(version_id)
            self._storage.save(versions_key, versions)

            # 触发事件
            self._notify_listeners(ConfigChangeEvent(
                event_id=self._generate_id(),
                config_id=config_id,
                version_id=version_id,
                change_type="create",
                new_value=data,
                operator=created_by
            ))

            return version

    def get_version(self, config_id: str, version_id: str) -> Optional[ConfigVersion]:
        """获取特定版本"""
        key = f"config:{config_id}:version:{version_id}"
        data = self._storage.load(key)
        if data:
            return self._dict_to_version(data)
        return None

    def get_latest_version(self, config_id: str) -> Optional[ConfigVersion]:
        """获取最新版本"""
        versions_key = f"config:{config_id}:versions"
        versions_data = self._storage.load(versions_key)
        if versions_data and versions_data.get("versions"):
            latest_version_id = versions_data["versions"][-1]
            return self.get_version(config_id, latest_version_id)
        return None

    def get_published_version(self, config_id: str) -> Optional[ConfigVersion]:
        """获取已发布版本"""
        published_key = f"config:{config_id}:published"
        published_data = self._storage.load(published_key)
        if published_data:
            version_id = published_data.get("version_id")
            if version_id:
                return self.get_version(config_id, version_id)
        return None

    def publish_version(
        self,
        config_id: str,
        version_id: str,
        operator: str = "system"
    ) -> bool:
        """发布版本"""
        with self._lock:
            version = self.get_version(config_id, version_id)
            if not version:
                return False

            # 更新版本状态
            version.status = ConfigStatus.PUBLISHED
            key = f"config:{config_id}:version:{version_id}"
            self._storage.save(key, version.to_dict())

            # 保存发布信息
            published_key = f"config:{config_id}:published"
            self._storage.save(published_key, {
                "version_id": version_id,
                "published_at": datetime.now().isoformat(),
                "published_by": operator
            })

            # 触发事件
            self._notify_listeners(ConfigChangeEvent(
                event_id=self._generate_id(),
                config_id=config_id,
                version_id=version_id,
                change_type="publish",
                new_value=version.data,
                operator=operator
            ))

            return True

    def rollback(
        self,
        config_id: str,
        to_version_id: str,
        operator: str = "system"
    ) -> bool:
        """回滚到指定版本"""
        with self._lock:
            target_version = self.get_version(config_id, to_version_id)
            if not target_version:
                return False

            # 标记为已回滚
            current_published = self.get_published_version(config_id)
            if current_published:
                current_published.status = ConfigStatus.ROLLED_BACK
                key = f"config:{config_id}:version:{current_published.version_id}"
                self._storage.save(key, current_published.to_dict())

            # 重新发布目标版本
            target_version.status = ConfigStatus.PUBLISHED
            key = f"config:{config_id}:version:{to_version_id}"
            self._storage.save(key, target_version.to_dict())

            # 更新发布信息
            published_key = f"config:{config_id}:published"
            self._storage.save(published_key, {
                "version_id": to_version_id,
                "published_at": datetime.now().isoformat(),
                "published_by": operator,
                "rolled_back_from": current_published.version_id if current_published else None
            })

            # 触发事件
            self._notify_listeners(ConfigChangeEvent(
                event_id=self._generate_id(),
                config_id=config_id,
                version_id=to_version_id,
                change_type="rollback",
                old_value=current_published.data if current_published else None,
                new_value=target_version.data,
                operator=operator
            ))

            return True

    def list_versions(self, config_id: str) -> List[ConfigVersion]:
        """列出所有版本"""
        versions_key = f"config:{config_id}:versions"
        versions_data = self._storage.load(versions_key)
        versions = []
        if versions_data and versions_data.get("versions"):
            for version_id in versions_data["versions"]:
                version = self.get_version(config_id, version_id)
                if version:
                    versions.append(version)
        return sorted(versions, key=lambda v: v.version_number)

    def register_instance(
        self,
        config_id: str,
        scope: ConfigScope,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConfigInstance:
        """注册配置实例"""
        with self._lock:
            instance_id = self._generate_id()
            instance = ConfigInstance(
                instance_id=instance_id,
                config_id=config_id,
                scope=scope,
                target_id=target_id,
                metadata=metadata or {}
            )
            self._instances[instance_id] = instance

            # 保存实例信息
            key = f"instance:{instance_id}"
            self._storage.save(key, instance.to_dict())

            return instance

    def get_instance(self, instance_id: str) -> Optional[ConfigInstance]:
        """获取配置实例"""
        return self._instances.get(instance_id)

    def apply_config_to_instance(
        self,
        instance_id: str,
        version_id: str,
        operator: str = "system"
    ) -> bool:
        """将配置应用到实例"""
        with self._lock:
            instance = self._instances.get(instance_id)
            if not instance:
                return False

            version = self.get_version(instance.config_id, version_id)
            if not version:
                return False

            # 更新实例
            instance.previous_version = instance.current_version
            instance.current_version = version_id
            instance.applied_at = datetime.now()

            # 保存
            key = f"instance:{instance_id}"
            self._storage.save(key, instance.to_dict())

            # 触发事件
            self._notify_listeners(ConfigChangeEvent(
                event_id=self._generate_id(),
                config_id=instance.config_id,
                version_id=version_id,
                change_type="apply",
                operator=operator
            ))

            return True

    def create_rollout_plan(
        self,
        config_id: str,
        version_id: str,
        strategy: RolloutStrategy,
        targets: List[RolloutTarget],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RolloutPlan:
        """创建灰度发布计划"""
        with self._lock:
            plan_id = self._generate_id()
            plan = RolloutPlan(
                plan_id=plan_id,
                config_id=config_id,
                version_id=version_id,
                strategy=strategy,
                targets=targets,
                metadata=metadata or {}
            )
            self._rollout_plans[plan_id] = plan

            # 保存
            key = f"rollout:{plan_id}"
            self._storage.save(key, plan.to_dict())

            return plan

    def get_rollout_plan(self, plan_id: str) -> Optional[RolloutPlan]:
        """获取灰度发布计划"""
        return self._rollout_plans.get(plan_id)

    def start_rollout(self, plan_id: str) -> bool:
        """开始灰度发布"""
        with self._lock:
            plan = self._rollout_plans.get(plan_id)
            if not plan or plan.status != "pending":
                return False

            plan.status = "running"
            plan.start_time = datetime.now()

            # 保存
            key = f"rollout:{plan_id}"
            self._storage.save(key, plan.to_dict())

            return True

    def update_rollout_progress(self, plan_id: str, progress: float) -> bool:
        """更新灰度发布进度"""
        with self._lock:
            plan = self._rollout_plans.get(plan_id)
            if not plan or plan.status != "running":
                return False

            plan.progress = min(100.0, max(0.0, progress))

            if plan.progress >= 100.0:
                plan.status = "completed"
                plan.end_time = datetime.now()

            # 保存
            key = f"rollout:{plan_id}"
            self._storage.save(key, plan.to_dict())

            return True

    def cancel_rollout(self, plan_id: str) -> bool:
        """取消灰度发布"""
        with self._lock:
            plan = self._rollout_plans.get(plan_id)
            if not plan or plan.status not in ["pending", "running"]:
                return False

            plan.status = "cancelled"
            plan.end_time = datetime.now()

            # 保存
            key = f"rollout:{plan_id}"
            self._storage.save(key, plan.to_dict())

            return True

    def add_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> None:
        """添加配置变更监听器"""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ConfigChangeEvent], None]) -> bool:
        """移除配置变更监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)
            return True
        return False

    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        """通知所有监听器"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                # 忽略监听器错误
                pass

    def _dict_to_version(self, data: Dict[str, Any]) -> ConfigVersion:
        """将字典转换为版本对象"""
        return ConfigVersion(
            version_id=data["version_id"],
            config_id=data["config_id"],
            version_number=data["version_number"],
            data=data["data"],
            checksum=data["checksum"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            comment=data.get("comment", ""),
            status=ConfigStatus(data["status"]),
            parent_version=data.get("parent_version")
        )

    def get_config_for_target(
        self,
        target_id: str,
        scope: ConfigScope
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """获取目标的所有配置"""
        configs = []
        for instance in self._instances.values():
            if instance.target_id == target_id and instance.scope == scope:
                if instance.current_version:
                    version = self.get_version(instance.config_id, instance.current_version)
                    if version:
                        configs.append((instance.config_id, version.data))
        return configs


class ConfigTemplate:
    """配置模板"""

    def __init__(self, template_id: str, schema: Dict[str, Any]):
        self._template_id = template_id
        self._schema = schema
        self._lock = threading.RLock()

    @property
    def template_id(self) -> str:
        return self._template_id

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证数据是否符合模板"""
        errors = []

        # 检查必需字段
        required = self._schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # 检查字段类型
        properties = self._schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"Field '{field}' should be of type {expected_type}")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True

    def apply_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认值"""
        result = data.copy()
        properties = self._schema.get("properties", {})
        for field, prop in properties.items():
            if field not in result and "default" in prop:
                result[field] = prop["default"]
        return result


# 便捷函数
def create_config_center(storage: Optional[ConfigStorageBackend] = None) -> ConfigCenter:
    """创建配置中心"""
    return ConfigCenter(storage)


def create_memory_storage() -> MemoryConfigStorage:
    """创建内存存储"""
    return MemoryConfigStorage()


def create_file_storage(base_path: str = "./config_data") -> FileConfigStorage:
    """创建文件存储"""
    return FileConfigStorage(base_path)
