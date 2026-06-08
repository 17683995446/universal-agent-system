"""
阶段32测试: Agent配置中心

遵循"图难于其易，为大于其细"的原则，全面测试配置中心系统。
"""

import os
import pytest
import tempfile
import shutil
from datetime import datetime

from src.agent.config_center import (
    ConfigStatus,
    ConfigScope,
    RolloutStrategy,
    ConfigValue,
    ConfigVersion,
    ConfigInstance,
    ConfigChangeEvent,
    RolloutPlan,
    RolloutTarget,
    MemoryConfigStorage,
    FileConfigStorage,
    ConfigCenter,
    ConfigTemplate,
    create_config_center,
    create_memory_storage,
    create_file_storage,
)


class TestConfigStatus:
    """测试配置状态枚举"""

    def test_config_status_values(self):
        """测试配置状态值"""
        assert ConfigStatus.DRAFT.value == "draft"
        assert ConfigStatus.PUBLISHED.value == "published"
        assert ConfigStatus.ROLLED_BACK.value == "rolled_back"
        assert ConfigStatus.DEPRECATED.value == "deprecated"


class TestConfigScope:
    """测试配置作用域枚举"""

    def test_config_scope_values(self):
        """测试配置作用域值"""
        assert ConfigScope.GLOBAL.value == "global"
        assert ConfigScope.APPLICATION.value == "application"
        assert ConfigScope.SERVICE.value == "service"
        assert ConfigScope.INSTANCE.value == "instance"


class TestRolloutStrategy:
    """测试灰度发布策略枚举"""

    def test_rollout_strategy_values(self):
        """测试灰度发布策略值"""
        assert RolloutStrategy.ALL_AT_ONCE.value == "all_at_once"
        assert RolloutStrategy.PERCENTAGE.value == "percentage"
        assert RolloutStrategy.CANARY.value == "canary"
        assert RolloutStrategy.BLUE_GREEN.value == "blue_green"
        assert RolloutStrategy.A_B_TEST.value == "a_b_test"


class TestConfigValue:
    """测试配置值"""

    def test_config_value_creation(self):
        """测试配置值创建"""
        value = ConfigValue(
            key="test_key",
            value="test_value",
            value_type="string",
            description="Test description"
        )
        assert value.key == "test_key"
        assert value.value == "test_value"
        assert value.value_type == "string"
        assert value.description == "Test description"
        assert value.encrypted is False

    def test_config_value_encrypted(self):
        """测试加密配置值"""
        value = ConfigValue(
            key="secret",
            value="password123",
            encrypted=True
        )
        data = value.to_dict()
        assert data["value"] == "***encrypted***"
        assert data["encrypted"] is True


class TestConfigVersion:
    """测试配置版本"""

    def test_config_version_creation(self):
        """测试配置版本创建"""
        version = ConfigVersion(
            version_id="v1",
            config_id="config1",
            version_number=1,
            data={"key": "value"},
            checksum="abc123",
            created_at=datetime.now(),
            created_by="user1",
            comment="Initial version"
        )
        assert version.version_id == "v1"
        assert version.config_id == "config1"
        assert version.version_number == 1
        assert version.status == ConfigStatus.DRAFT

    def test_config_version_to_dict(self):
        """测试配置版本转字典"""
        version = ConfigVersion(
            version_id="v1",
            config_id="config1",
            version_number=1,
            data={"key": "value"},
            checksum="abc123",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            created_by="user1"
        )
        data = version.to_dict()
        assert data["version_id"] == "v1"
        assert data["status"] == "draft"
        assert data["created_at"] == "2024-01-01T12:00:00"


class TestConfigInstance:
    """测试配置实例"""

    def test_config_instance_creation(self):
        """测试配置实例创建"""
        instance = ConfigInstance(
            instance_id="i1",
            config_id="config1",
            scope=ConfigScope.SERVICE,
            target_id="service1"
        )
        assert instance.instance_id == "i1"
        assert instance.config_id == "config1"
        assert instance.scope == ConfigScope.SERVICE
        assert instance.target_id == "service1"
        assert instance.current_version is None

    def test_config_instance_to_dict(self):
        """测试配置实例转字典"""
        instance = ConfigInstance(
            instance_id="i1",
            config_id="config1",
            scope=ConfigScope.APPLICATION,
            target_id="app1",
            current_version="v1"
        )
        data = instance.to_dict()
        assert data["instance_id"] == "i1"
        assert data["scope"] == "application"
        assert data["current_version"] == "v1"


class TestRolloutPlan:
    """测试灰度发布计划"""

    def test_rollout_plan_creation(self):
        """测试灰度发布计划创建"""
        plan = RolloutPlan(
            plan_id="p1",
            config_id="config1",
            version_id="v1",
            strategy=RolloutStrategy.CANARY
        )
        assert plan.plan_id == "p1"
        assert plan.strategy == RolloutStrategy.CANARY
        assert plan.status == "pending"
        assert plan.progress == 0.0

    def test_rollout_plan_with_targets(self):
        """测试带目标的灰度发布计划"""
        target = RolloutTarget(
            target_id="instance1",
            target_type="instance",
            weight=0.5
        )
        plan = RolloutPlan(
            plan_id="p1",
            config_id="config1",
            version_id="v1",
            strategy=RolloutStrategy.PERCENTAGE,
            targets=[target]
        )
        assert len(plan.targets) == 1
        assert plan.targets[0].target_id == "instance1"


class TestConfigChangeEvent:
    """测试配置变更事件"""

    def test_config_change_event_creation(self):
        """测试配置变更事件创建"""
        event = ConfigChangeEvent(
            event_id="e1",
            config_id="config1",
            version_id="v1",
            change_type="create",
            operator="user1"
        )
        assert event.event_id == "e1"
        assert event.change_type == "create"
        assert event.operator == "user1"

    def test_config_change_event_to_dict(self):
        """测试配置变更事件转字典"""
        event = ConfigChangeEvent(
            event_id="e1",
            config_id="config1",
            version_id="v1",
            change_type="update",
            old_value={"key": "old"},
            new_value={"key": "new"}
        )
        data = event.to_dict()
        assert data["change_type"] == "update"
        assert data["old_value"] == {"key": "old"}
        assert data["new_value"] == {"key": "new"}


class TestMemoryConfigStorage:
    """测试内存配置存储"""

    def test_save_and_load(self):
        """测试保存和加载"""
        storage = MemoryConfigStorage()
        data = {"key": "value", "number": 42}

        assert storage.save("test_key", data) is True
        loaded = storage.load("test_key")
        assert loaded == data

    def test_load_nonexistent(self):
        """测试加载不存在的键"""
        storage = MemoryConfigStorage()
        assert storage.load("nonexistent") is None

    def test_delete(self):
        """测试删除"""
        storage = MemoryConfigStorage()
        storage.save("key1", {"data": 1})

        assert storage.delete("key1") is True
        assert storage.load("key1") is None
        assert storage.delete("key1") is False

    def test_list_keys(self):
        """测试列出键"""
        storage = MemoryConfigStorage()
        storage.save("prefix/key1", {})
        storage.save("prefix/key2", {})
        storage.save("other/key3", {})

        keys = storage.list_keys("prefix/")
        assert len(keys) == 2
        assert "prefix/key1" in keys
        assert "prefix/key2" in keys

    def test_exists(self):
        """测试键是否存在"""
        storage = MemoryConfigStorage()
        storage.save("key1", {})

        assert storage.exists("key1") is True
        assert storage.exists("nonexistent") is False


class TestFileConfigStorage:
    """测试文件配置存储"""

    def setup_method(self):
        """设置测试目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileConfigStorage(self.temp_dir)

    def teardown_method(self):
        """清理测试目录"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load(self):
        """测试保存和加载"""
        data = {"key": "value", "nested": {"a": 1}}

        assert self.storage.save("test_key", data) is True
        loaded = self.storage.load("test_key")
        assert loaded["key"] == "value"
        assert loaded["nested"]["a"] == 1

    def test_load_nonexistent(self):
        """测试加载不存在的键"""
        assert self.storage.load("nonexistent") is None

    def test_delete(self):
        """测试删除"""
        self.storage.save("key1", {"data": 1})

        assert self.storage.delete("key1") is True
        assert self.storage.load("key1") is None
        assert self.storage.delete("key1") is False

    def test_exists(self):
        """测试键是否存在"""
        self.storage.save("key1", {})

        assert self.storage.exists("key1") is True
        assert self.storage.exists("nonexistent") is False


class TestConfigCenter:
    """测试配置中心"""

    def setup_method(self):
        """设置配置中心"""
        self.center = ConfigCenter(MemoryConfigStorage())

    def test_create_config(self):
        """测试创建配置"""
        version = self.center.create_config(
            config_id="app.config",
            data={"timeout": 30, "retries": 3},
            created_by="admin",
            comment="Initial config"
        )

        assert version.config_id == "app.config"
        assert version.version_number == 1
        assert version.data["timeout"] == 30
        assert version.status == ConfigStatus.DRAFT

    def test_create_multiple_versions(self):
        """测试创建多个版本"""
        v1 = self.center.create_config("app.config", {"v": 1})
        v2 = self.center.create_config("app.config", {"v": 2})

        assert v1.version_number == 1
        assert v2.version_number == 2

    def test_get_version(self):
        """测试获取版本"""
        version = self.center.create_config("app.config", {"key": "value"})
        retrieved = self.center.get_version("app.config", version.version_id)

        assert retrieved is not None
        assert retrieved.version_id == version.version_id
        assert retrieved.data == {"key": "value"}

    def test_get_latest_version(self):
        """测试获取最新版本"""
        self.center.create_config("app.config", {"v": 1})
        self.center.create_config("app.config", {"v": 2})

        latest = self.center.get_latest_version("app.config")
        assert latest is not None
        assert latest.data["v"] == 2

    def test_publish_version(self):
        """测试发布版本"""
        version = self.center.create_config("app.config", {"key": "value"})

        assert self.center.publish_version("app.config", version.version_id, "admin") is True

        published = self.center.get_published_version("app.config")
        assert published is not None
        assert published.version_id == version.version_id
        assert published.status == ConfigStatus.PUBLISHED

    def test_publish_nonexistent_version(self):
        """测试发布不存在的版本"""
        assert self.center.publish_version("app.config", "nonexistent") is False

    def test_rollback(self):
        """测试回滚"""
        v1 = self.center.create_config("app.config", {"v": 1})
        v2 = self.center.create_config("app.config", {"v": 2})

        self.center.publish_version("app.config", v2.version_id)

        # 回滚到v1
        assert self.center.rollback("app.config", v1.version_id, "admin") is True

        published = self.center.get_published_version("app.config")
        assert published.version_id == v1.version_id

        # 重新获取v2检查状态
        v2_updated = self.center.get_version("app.config", v2.version_id)
        assert v2_updated.status == ConfigStatus.ROLLED_BACK

    def test_list_versions(self):
        """测试列出所有版本"""
        self.center.create_config("app.config", {"v": 1})
        self.center.create_config("app.config", {"v": 2})
        self.center.create_config("app.config", {"v": 3})

        versions = self.center.list_versions("app.config")
        assert len(versions) == 3
        assert versions[0].version_number == 1
        assert versions[2].version_number == 3

    def test_register_instance(self):
        """测试注册实例"""
        instance = self.center.register_instance(
            config_id="app.config",
            scope=ConfigScope.SERVICE,
            target_id="service1",
            metadata={"region": "us-east"}
        )

        assert instance.config_id == "app.config"
        assert instance.scope == ConfigScope.SERVICE
        assert instance.target_id == "service1"
        assert instance.metadata["region"] == "us-east"

    def test_get_instance(self):
        """测试获取实例"""
        instance = self.center.register_instance(
            config_id="app.config",
            scope=ConfigScope.SERVICE,
            target_id="service1"
        )

        retrieved = self.center.get_instance(instance.instance_id)
        assert retrieved is not None
        assert retrieved.instance_id == instance.instance_id

    def test_apply_config_to_instance(self):
        """测试应用配置到实例"""
        version = self.center.create_config("app.config", {"timeout": 30})
        instance = self.center.register_instance(
            config_id="app.config",
            scope=ConfigScope.SERVICE,
            target_id="service1"
        )

        assert self.center.apply_config_to_instance(
            instance.instance_id,
            version.version_id
        ) is True

        updated = self.center.get_instance(instance.instance_id)
        assert updated.current_version == version.version_id
        assert updated.applied_at is not None

    def test_create_rollout_plan(self):
        """测试创建灰度发布计划"""
        version = self.center.create_config("app.config", {"v": 2})
        target = RolloutTarget(
            target_id="instance1",
            target_type="instance",
            weight=0.1
        )

        plan = self.center.create_rollout_plan(
            config_id="app.config",
            version_id=version.version_id,
            strategy=RolloutStrategy.CANARY,
            targets=[target]
        )

        assert plan.plan_id is not None
        assert plan.strategy == RolloutStrategy.CANARY
        assert len(plan.targets) == 1

    def test_start_and_update_rollout(self):
        """测试开始和更新灰度发布"""
        version = self.center.create_config("app.config", {"v": 2})
        plan = self.center.create_rollout_plan(
            config_id="app.config",
            version_id=version.version_id,
            strategy=RolloutStrategy.PERCENTAGE,
            targets=[]
        )

        assert self.center.start_rollout(plan.plan_id) is True

        plan = self.center.get_rollout_plan(plan.plan_id)
        assert plan.status == "running"

        # 更新进度
        assert self.center.update_rollout_progress(plan.plan_id, 50.0) is True
        plan = self.center.get_rollout_plan(plan.plan_id)
        assert plan.progress == 50.0

        # 完成
        assert self.center.update_rollout_progress(plan.plan_id, 100.0) is True
        plan = self.center.get_rollout_plan(plan.plan_id)
        assert plan.status == "completed"
        assert plan.end_time is not None

    def test_cancel_rollout(self):
        """测试取消灰度发布"""
        version = self.center.create_config("app.config", {"v": 2})
        plan = self.center.create_rollout_plan(
            config_id="app.config",
            version_id=version.version_id,
            strategy=RolloutStrategy.ALL_AT_ONCE,
            targets=[]
        )

        assert self.center.cancel_rollout(plan.plan_id) is True

        plan = self.center.get_rollout_plan(plan.plan_id)
        assert plan.status == "cancelled"

    def test_add_and_remove_listener(self):
        """测试添加和移除监听器"""
        events = []

        def listener(event):
            events.append(event)

        self.center.add_listener(listener)

        self.center.create_config("app.config", {"key": "value"})

        assert len(events) == 1
        assert events[0].change_type == "create"

        # 移除监听器
        assert self.center.remove_listener(listener) is True
        self.center.create_config("app.config", {"key": "value2"})
        assert len(events) == 1  # 不再接收事件

    def test_get_config_for_target(self):
        """测试获取目标的配置"""
        version = self.center.create_config("app.config", {"timeout": 30})
        instance = self.center.register_instance(
            config_id="app.config",
            scope=ConfigScope.SERVICE,
            target_id="service1"
        )
        self.center.apply_config_to_instance(instance.instance_id, version.version_id)

        configs = self.center.get_config_for_target("service1", ConfigScope.SERVICE)
        assert len(configs) == 1
        assert configs[0][0] == "app.config"
        assert configs[0][1]["timeout"] == 30


class TestConfigTemplate:
    """测试配置模板"""

    def test_template_validation_success(self):
        """测试模板验证成功"""
        schema = {
            "type": "object",
            "required": ["name", "port"],
            "properties": {
                "name": {"type": "string"},
                "port": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        }
        template = ConfigTemplate("app_schema", schema)

        data = {"name": "myapp", "port": 8080, "enabled": True}
        valid, errors = template.validate(data)

        assert valid is True
        assert len(errors) == 0

    def test_template_validation_missing_required(self):
        """测试模板验证缺少必需字段"""
        schema = {
            "required": ["name"],
            "properties": {
                "name": {"type": "string"}
            }
        }
        template = ConfigTemplate("app_schema", schema)

        data = {"port": 8080}
        valid, errors = template.validate(data)

        assert valid is False
        assert any("Missing required field: name" in e for e in errors)

    def test_template_validation_wrong_type(self):
        """测试模板验证类型错误"""
        schema = {
            "properties": {
                "port": {"type": "integer"},
                "name": {"type": "string"}
            }
        }
        template = ConfigTemplate("app_schema", schema)

        data = {"port": "not_a_number", "name": 123}
        valid, errors = template.validate(data)

        assert valid is False
        assert len(errors) == 2

    def test_apply_defaults(self):
        """测试应用默认值"""
        schema = {
            "properties": {
                "timeout": {"type": "integer", "default": 30},
                "retries": {"type": "integer", "default": 3}
            }
        }
        template = ConfigTemplate("app_schema", schema)

        data = {"timeout": 60}  # 只提供timeout
        result = template.apply_defaults(data)

        assert result["timeout"] == 60
        assert result["retries"] == 3  # 使用默认值


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_config_center(self):
        """测试创建配置中心"""
        center = create_config_center()
        assert isinstance(center, ConfigCenter)

    def test_create_memory_storage(self):
        """测试创建内存存储"""
        storage = create_memory_storage()
        assert isinstance(storage, MemoryConfigStorage)

    def test_create_file_storage(self):
        """测试创建文件存储"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = create_file_storage(temp_dir)
            assert isinstance(storage, FileConfigStorage)
        finally:
            shutil.rmtree(temp_dir)


class TestIntegration:
    """集成测试"""

    def test_full_config_lifecycle(self):
        """测试完整配置生命周期"""
        center = create_config_center()

        # 1. 创建配置
        v1 = center.create_config(
            "app.config",
            {"timeout": 30, "retries": 3},
            comment="Initial"
        )

        # 2. 发布配置
        center.publish_version("app.config", v1.version_id)

        # 3. 注册实例
        instance = center.register_instance(
            "app.config",
            ConfigScope.SERVICE,
            "service1"
        )

        # 4. 应用配置
        center.apply_config_to_instance(instance.instance_id, v1.version_id)

        # 5. 创建新版本
        v2 = center.create_config(
            "app.config",
            {"timeout": 60, "retries": 5},
            comment="Increased timeout"
        )

        # 6. 创建灰度发布计划
        target = RolloutTarget(target_id="service1", target_type="service")
        plan = center.create_rollout_plan(
            "app.config",
            v2.version_id,
            RolloutStrategy.CANARY,
            [target]
        )

        # 7. 执行灰度发布
        center.start_rollout(plan.plan_id)
        center.update_rollout_progress(plan.plan_id, 100.0)

        # 8. 验证
        plan = center.get_rollout_plan(plan.plan_id)
        assert plan.status == "completed"

        # 9. 回滚（如果需要）
        center.rollback("app.config", v1.version_id)
        published = center.get_published_version("app.config")
        assert published.version_id == v1.version_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
