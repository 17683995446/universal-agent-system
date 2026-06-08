# =============================================================================
# 插件系统测试 - 阶段26测试
# =============================================================================

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.agent.plugin_system import (
    PluginManager,
    PluginRegistry,
    PluginLoader,
    Plugin,
    PluginMetadata,
    PluginContext,
    PluginStatus,
    Hook,
)


# =============================================================================
# 测试插件类
# =============================================================================

class DummyPlugin(Plugin):
    """测试用插件"""
    PLUGIN_NAME = "DummyPlugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "A test plugin"
    PLUGIN_AUTHOR = "Test Author"

    def __init__(self):
        super().__init__()
        self.init_called = False
        self.shutdown_called = False

    def on_init(self) -> bool:
        self.init_called = True
        return True

    def on_shutdown(self) -> bool:
        self.shutdown_called = True
        return True


class FailingPlugin(Plugin):
    """会失败的测试插件"""
    PLUGIN_NAME = "FailingPlugin"
    PLUGIN_VERSION = "1.0.0"

    def on_init(self) -> bool:
        return False

    def on_shutdown(self) -> bool:
        return False


# =============================================================================
# PluginStatus 测试
# =============================================================================

class TestPluginStatus:
    """插件状态枚举测试"""

    def test_enum_values(self):
        assert PluginStatus.UNLOADED.value == "unloaded"
        assert PluginStatus.LOADING.value == "loading"
        assert PluginStatus.LOADED.value == "loaded"
        assert PluginStatus.INITIALIZING.value == "initializing"
        assert PluginStatus.ACTIVE.value == "active"
        assert PluginStatus.ERROR.value == "error"
        assert PluginStatus.DISABLED.value == "disabled"
        assert PluginStatus.UNLOADING.value == "unloading"


# =============================================================================
# PluginMetadata 测试
# =============================================================================

class TestPluginMetadata:
    """插件元数据测试"""

    def test_creation(self):
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            hooks=["hook1", "hook2"],
        )
        assert metadata.name == "TestPlugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "A test plugin"
        assert metadata.author == "Test Author"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.hooks == ["hook1", "hook2"]

    def test_to_dict(self):
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="A test plugin",
        )
        result = metadata.to_dict()
        assert result["name"] == "TestPlugin"
        assert result["version"] == "1.0.0"
        assert result["description"] == "A test plugin"


# =============================================================================
# PluginContext 测试
# =============================================================================

class TestPluginContext:
    """插件上下文测试"""

    def test_creation(self):
        ctx = PluginContext(plugin_id="test_plugin")
        assert ctx.plugin_id == "test_plugin"
        assert ctx.config == {}
        assert ctx.shared_data == {}

    def test_config(self):
        ctx = PluginContext(
            plugin_id="test_plugin",
            config={"key1": "value1", "key2": 123},
        )
        assert ctx.get_config("key1") == "value1"
        assert ctx.get_config("key2") == 123
        assert ctx.get_config("nonexistent", "default") == "default"

    def test_shared_data(self):
        ctx = PluginContext(plugin_id="test_plugin")
        ctx.set_shared_data("shared_key", "shared_value")
        assert ctx.get_shared_data("shared_key") == "shared_value"
        assert ctx.get_shared_data("nonexistent", "default") == "default"


# =============================================================================
# Plugin 测试
# =============================================================================

class TestPluginBase:
    """插件基类测试"""

    def test_initial_state(self):
        plugin = DummyPlugin()
        assert plugin.status == PluginStatus.UNLOADED
        assert plugin.metadata is None
        assert plugin.context is None

    def test_initialize_success(self):
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")

        result = plugin.initialize(metadata, context)

        assert result is True
        assert plugin.status == PluginStatus.ACTIVE
        assert plugin.metadata == metadata
        assert plugin.context == context
        assert plugin.init_called is True

    def test_initialize_failure(self):
        plugin = FailingPlugin()
        metadata = PluginMetadata(name="FailingPlugin", version="1.0.0")
        context = PluginContext(plugin_id="failing_plugin")

        result = plugin.initialize(metadata, context)

        assert result is False
        assert plugin.status == PluginStatus.ERROR

    def test_shutdown_success(self):
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)

        result = plugin.shutdown()

        assert result is True
        assert plugin.status == PluginStatus.UNLOADED
        assert plugin.shutdown_called is True

    def test_get_name(self):
        plugin = DummyPlugin()
        assert plugin.get_name() == "unknown"  # 未初始化时

        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)

        assert plugin.get_name() == "DummyPlugin"


# =============================================================================
# Hook 测试
# =============================================================================

class TestHook:
    """钩子测试"""

    def test_creation(self):
        hook = Hook("test_hook")
        assert hook.name == "test_hook"
        assert hook.has_handlers() is False

    def test_register_handler(self):
        hook = Hook("test_hook")
        handler = Mock()

        hook.register(handler)
        assert hook.has_handlers() is True

    def test_register_duplicate(self):
        hook = Hook("test_hook")
        handler = Mock()

        hook.register(handler)
        hook.register(handler)  # 重复注册
        assert len(hook._handlers) == 1

    def test_unregister_handler(self):
        hook = Hook("test_hook")
        handler = Mock()

        hook.register(handler)
        result = hook.unregister(handler)

        assert result is True
        assert hook.has_handlers() is False

    def test_unregister_nonexistent(self):
        hook = Hook("test_hook")
        handler = Mock()

        result = hook.unregister(handler)
        assert result is False

    def test_execute(self):
        hook = Hook("test_hook")
        handler1 = Mock(return_value="result1")
        handler2 = Mock(return_value="result2")

        hook.register(handler1)
        hook.register(handler2)
        results = hook.execute("arg1", kwarg1="value1")

        assert results == ["result1", "result2"]
        handler1.assert_called_once_with("arg1", kwarg1="value1")
        handler2.assert_called_once_with("arg1", kwarg1="value1")

    def test_execute_with_exception(self):
        hook = Hook("test_hook")
        handler1 = Mock(side_effect=Exception("error"))
        handler2 = Mock(return_value="result2")

        hook.register(handler1)
        hook.register(handler2)
        results = hook.execute()

        assert results == ["result2"]  # 只有第二个处理器成功

    def test_execute_first(self):
        hook = Hook("test_hook")
        handler1 = Mock(return_value="result1")
        handler2 = Mock(return_value="result2")

        hook.register(handler1)
        hook.register(handler2)
        result = hook.execute_first()

        assert result == "result1"

    def test_execute_first_no_handlers(self):
        hook = Hook("test_hook")
        result = hook.execute_first()
        assert result is None


# =============================================================================
# PluginRegistry 测试
# =============================================================================

class TestPluginRegistry:
    """插件注册表测试"""

    def test_creation(self):
        registry = PluginRegistry()
        assert registry.get_all_plugins() == {}

    def test_register_plugin(self):
        registry = PluginRegistry()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")

        result = registry.register_plugin("test_plugin", plugin, metadata)

        assert result is True
        assert registry.has_plugin("test_plugin") is True
        assert registry.get_plugin("test_plugin") == plugin
        assert registry.get_metadata("test_plugin") == metadata

    def test_register_duplicate(self):
        registry = PluginRegistry()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")

        registry.register_plugin("test_plugin", plugin, metadata)
        result = registry.register_plugin("test_plugin", plugin, metadata)

        assert result is False

    def test_unregister_plugin(self):
        registry = PluginRegistry()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")

        registry.register_plugin("test_plugin", plugin, metadata)
        result = registry.unregister_plugin("test_plugin")

        assert result is True
        assert registry.has_plugin("test_plugin") is False

    def test_unregister_nonexistent(self):
        registry = PluginRegistry()
        result = registry.unregister_plugin("nonexistent")
        assert result is False

    def test_get_plugin_ids(self):
        registry = PluginRegistry()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")

        registry.register_plugin("plugin1", plugin, metadata)
        registry.register_plugin("plugin2", plugin, metadata)

        ids = registry.get_plugin_ids()
        assert sorted(ids) == ["plugin1", "plugin2"]

    def test_get_or_create_hook(self):
        registry = PluginRegistry()
        hook = registry.get_or_create_hook("test_hook")
        assert hook.name == "test_hook"

        # 再次获取应该是同一个
        hook2 = registry.get_or_create_hook("test_hook")
        assert hook is hook2

    def test_get_hook(self):
        registry = PluginRegistry()
        registry.get_or_create_hook("test_hook")
        hook = registry.get_hook("test_hook")
        assert hook is not None

    def test_get_nonexistent_hook(self):
        registry = PluginRegistry()
        hook = registry.get_hook("nonexistent")
        assert hook is None


# =============================================================================
# PluginManager 测试
# =============================================================================

class TestPluginManager:
    """插件管理器测试"""

    def test_creation(self):
        manager = PluginManager()
        assert manager.registry is not None

    def test_set_plugin_config(self):
        manager = PluginManager()
        manager.set_plugin_config("test_plugin", {"key": "value"})
        # 配置会在加载时使用

    def test_get_plugin_status_nonexistent(self):
        manager = PluginManager()
        status = manager.get_plugin_status("nonexistent")
        assert status is None

    def test_get_active_plugins(self):
        manager = PluginManager()
        # 创建并初始化插件
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)

        manager.registry.register_plugin("test_plugin", plugin, metadata)

        active = manager.get_active_plugins()
        assert "test_plugin" in active

    def test_enable_plugin(self):
        manager = PluginManager()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)
        plugin._status = PluginStatus.DISABLED

        manager.registry.register_plugin("test_plugin", plugin, metadata)
        manager.set_plugin_config("test_plugin", {})

        result = manager.enable_plugin("test_plugin")
        assert result is True
        assert plugin.status == PluginStatus.ACTIVE

    def test_disable_plugin(self):
        manager = PluginManager()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)

        manager.registry.register_plugin("test_plugin", plugin, metadata)

        result = manager.disable_plugin("test_plugin")
        assert result is True
        assert plugin.status == PluginStatus.DISABLED

    def test_register_and_execute_hook(self):
        manager = PluginManager()
        handler = Mock(return_value="result")

        manager.register_hook("test_hook", handler)
        results = manager.execute_hook("test_hook", "arg1")

        assert results == ["result"]
        handler.assert_called_once_with("arg1")

    def test_unregister_hook(self):
        manager = PluginManager()
        handler = Mock()

        manager.register_hook("test_hook", handler)
        result = manager.unregister_hook("test_hook", handler)

        assert result is True
        assert manager.execute_hook("test_hook") == []

    def test_execute_hook_first(self):
        manager = PluginManager()
        handler = Mock(return_value="result")

        manager.register_hook("test_hook", handler)
        result = manager.execute_hook_first("test_hook")

        assert result == "result"

    def test_execute_nonexistent_hook(self):
        manager = PluginManager()
        results = manager.execute_hook("nonexistent_hook")
        assert results == []

    def test_get_manager_stats(self):
        manager = PluginManager()
        plugin = DummyPlugin()
        metadata = PluginMetadata(name="DummyPlugin", version="1.0.0")
        context = PluginContext(plugin_id="test_plugin")
        plugin.initialize(metadata, context)

        manager.registry.register_plugin("test_plugin", plugin, metadata)
        manager.register_hook("test_hook", Mock())

        stats = manager.get_manager_stats()
        assert stats["total_plugins"] == 1
        assert stats["active_plugins"] == 1
        assert "test_plugin" in stats["plugin_ids"]
        assert "test_hook" in stats["hooks"]


# =============================================================================
# 导出测试
# =============================================================================

class TestExportPluginSystem:
    """导出测试"""

    def test_plugin_manager_exported(self):
        from src.agent import PluginManager
        assert PluginManager is not None

    def test_plugin_registry_exported(self):
        from src.agent import PluginRegistry
        assert PluginRegistry is not None

    def test_plugin_loader_exported(self):
        from src.agent import PluginLoader
        assert PluginLoader is not None

    def test_plugin_exported(self):
        from src.agent import Plugin
        assert Plugin is not None

    def test_plugin_metadata_exported(self):
        from src.agent import PluginMetadata
        assert PluginMetadata is not None

    def test_plugin_context_exported(self):
        from src.agent import PluginContext
        assert PluginContext is not None

    def test_plugin_status_exported(self):
        from src.agent import PluginStatus
        assert PluginStatus.ACTIVE is not None

    def test_hook_exported(self):
        from src.agent import Hook
        assert Hook is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
