# =============================================================================
# Agent插件系统 - 阶段26: Agent Plugin System
# =============================================================================
#
# 功能：实现Agent插件系统，支持动态扩展和模块化架构
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Generic

# 配置日志
logger = logging.getLogger(__name__)


# =============================================================================
# 插件状态
# =============================================================================

class PluginStatus(str, Enum):
    """插件状态"""
    UNLOADED = "unloaded"       # 未加载
    LOADING = "loading"         # 加载中
    LOADED = "loaded"           # 已加载
    INITIALIZING = "initializing"  # 初始化中
    ACTIVE = "active"           # 活跃
    ERROR = "error"             # 错误
    DISABLED = "disabled"       # 已禁用
    UNLOADING = "unloading"     # 卸载中


# =============================================================================
# 插件元数据
# =============================================================================

@dataclass
class PluginMetadata:
    """
    插件元数据
    
    描述插件的基本信息
    """
    name: str                           # 插件名称
    version: str                        # 版本号
    description: str = ""               # 描述
    author: str = ""                    # 作者
    dependencies: list[str] = field(default_factory=list)  # 依赖列表
    entry_point: str = ""               # 入口点（类名）
    config_schema: dict[str, Any] = field(default_factory=dict)  # 配置模式
    hooks: list[str] = field(default_factory=list)  # 提供的钩子

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "hooks": self.hooks,
        }


# =============================================================================
# 插件上下文
# =============================================================================

@dataclass
class PluginContext:
    """
    插件上下文
    
    提供给插件的运行时环境
    """
    plugin_id: str
    config: dict[str, Any] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)

    def set_shared_data(self, key: str, value: Any) -> None:
        """设置共享数据"""
        self.shared_data[key] = value

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """获取共享数据"""
        return self.shared_data.get(key, default)


# =============================================================================
# 插件基类
# =============================================================================

class Plugin(ABC):
    """
    插件基类
    
    所有插件必须继承此类
    """

    def __init__(self):
        self._metadata: Optional[PluginMetadata] = None
        self._context: Optional[PluginContext] = None
        self._status: PluginStatus = PluginStatus.UNLOADED

    @property
    def metadata(self) -> Optional[PluginMetadata]:
        """获取插件元数据"""
        return self._metadata

    @property
    def context(self) -> Optional[PluginContext]:
        """获取插件上下文"""
        return self._context

    @property
    def status(self) -> PluginStatus:
        """获取插件状态"""
        return self._status

    def initialize(self, metadata: PluginMetadata, context: PluginContext) -> bool:
        """
        初始化插件
        
        Args:
            metadata: 插件元数据
            context: 插件上下文
        
        Returns:
            是否初始化成功
        """
        self._metadata = metadata
        self._context = context
        self._status = PluginStatus.INITIALIZING

        try:
            result = self.on_init()
            if result:
                self._status = PluginStatus.ACTIVE
            else:
                self._status = PluginStatus.ERROR
            return result
        except Exception as e:
            logger.error(f"Plugin {metadata.name} initialization failed: {e}")
            self._status = PluginStatus.ERROR
            return False

    def shutdown(self) -> bool:
        """
        关闭插件
        
        Returns:
            是否关闭成功
        """
        self._status = PluginStatus.UNLOADING

        try:
            result = self.on_shutdown()
            self._status = PluginStatus.UNLOADED
            return result
        except Exception as e:
            logger.error(f"Plugin {self._metadata.name if self._metadata else 'unknown'} shutdown failed: {e}")
            self._status = PluginStatus.ERROR
            return False

    @abstractmethod
    def on_init(self) -> bool:
        """
        插件初始化回调
        
        子类必须实现此方法
        
        Returns:
            是否初始化成功
        """
        pass

    @abstractmethod
    def on_shutdown(self) -> bool:
        """
        插件关闭回调
        
        子类必须实现此方法
        
        Returns:
            是否关闭成功
        """
        pass

    def get_name(self) -> str:
        """获取插件名称"""
        return self._metadata.name if self._metadata else "unknown"


# =============================================================================
# 插件钩子
# =============================================================================

T = TypeVar('T')


class Hook(Generic[T]):
    """
    钩子类
    
    用于实现插件间的扩展点
    """

    def __init__(self, name: str):
        self._name = name
        self._handlers: list[Callable[..., T]] = []
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        """获取钩子名称"""
        return self._name

    def register(self, handler: Callable[..., T]) -> None:
        """
        注册钩子处理器
        
        Args:
            handler: 处理函数
        """
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
                logger.debug(f"Registered handler for hook '{self._name}'")

    def unregister(self, handler: Callable[..., T]) -> bool:
        """
        注销钩子处理器
        
        Args:
            handler: 处理函数
        
        Returns:
            是否成功注销
        """
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
                logger.debug(f"Unregistered handler for hook '{self._name}'")
                return True
            return False

    def execute(self, *args, **kwargs) -> list[T]:
        """
        执行所有钩子处理器
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            所有处理器的返回结果列表
        """
        results = []
        with self._lock:
            handlers = self._handlers.copy()

        for handler in handlers:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Hook '{self._name}' handler failed: {e}")

        return results

    def execute_first(self, *args, **kwargs) -> Optional[T]:
        """
        执行第一个钩子处理器并返回结果
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            第一个处理器的返回结果，如果没有处理器则返回None
        """
        with self._lock:
            if not self._handlers:
                return None
            handler = self._handlers[0]

        try:
            return handler(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Hook '{self._name}' first handler failed: {e}")
            return None

    def has_handlers(self) -> bool:
        """是否有注册的处理器"""
        with self._lock:
            return len(self._handlers) > 0


# =============================================================================
# 插件注册表
# =============================================================================

class PluginRegistry:
    """
    插件注册表
    
    管理所有已加载的插件
    """

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._metadata: dict[str, PluginMetadata] = {}
        self._hooks: dict[str, Hook] = {}
        self._lock = threading.RLock()

    def register_plugin(self, plugin_id: str, plugin: Plugin, metadata: PluginMetadata) -> bool:
        """
        注册插件
        
        Args:
            plugin_id: 插件ID
            plugin: 插件实例
            metadata: 插件元数据
        
        Returns:
            是否注册成功
        """
        with self._lock:
            if plugin_id in self._plugins:
                logger.warning(f"Plugin {plugin_id} already registered")
                return False

            self._plugins[plugin_id] = plugin
            self._metadata[plugin_id] = metadata
            logger.info(f"Registered plugin: {plugin_id} ({metadata.name} v{metadata.version})")
            return True

    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        注销插件
        
        Args:
            plugin_id: 插件ID
        
        Returns:
            是否注销成功
        """
        with self._lock:
            if plugin_id not in self._plugins:
                return False

            plugin = self._plugins[plugin_id]
            if plugin.status == PluginStatus.ACTIVE:
                plugin.shutdown()

            del self._plugins[plugin_id]
            del self._metadata[plugin_id]
            logger.info(f"Unregistered plugin: {plugin_id}")
            return True

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """获取插件实例"""
        with self._lock:
            return self._plugins.get(plugin_id)

    def get_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """获取插件元数据"""
        with self._lock:
            return self._metadata.get(plugin_id)

    def get_all_plugins(self) -> dict[str, Plugin]:
        """获取所有插件"""
        with self._lock:
            return self._plugins.copy()

    def get_plugin_ids(self) -> list[str]:
        """获取所有插件ID"""
        with self._lock:
            return list(self._plugins.keys())

    def has_plugin(self, plugin_id: str) -> bool:
        """检查插件是否存在"""
        with self._lock:
            return plugin_id in self._plugins

    def get_or_create_hook(self, hook_name: str) -> Hook:
        """
        获取或创建钩子
        
        Args:
            hook_name: 钩子名称
        
        Returns:
            钩子实例
        """
        with self._lock:
            if hook_name not in self._hooks:
                self._hooks[hook_name] = Hook(hook_name)
            return self._hooks[hook_name]

    def get_hook(self, hook_name: str) -> Optional[Hook]:
        """获取钩子"""
        with self._lock:
            return self._hooks.get(hook_name)

    def get_all_hooks(self) -> dict[str, Hook]:
        """获取所有钩子"""
        with self._lock:
            return self._hooks.copy()


# =============================================================================
# 插件加载器
# =============================================================================

class PluginLoader:
    """
    插件加载器
    
    负责从文件系统加载插件
    """

    def __init__(self, registry: PluginRegistry):
        self._registry = registry
        self._plugin_paths: list[Path] = []

    def add_plugin_path(self, path: str | Path) -> None:
        """
        添加插件搜索路径
        
        Args:
            path: 路径
        """
        path_obj = Path(path)
        if path_obj not in self._plugin_paths:
            self._plugin_paths.append(path_obj)
            logger.debug(f"Added plugin path: {path_obj}")

    def load_plugin_from_file(self, file_path: str | Path, plugin_id: Optional[str] = None) -> Optional[Plugin]:
        """
        从文件加载插件
        
        Args:
            file_path: 插件文件路径
            plugin_id: 插件ID（可选）
        
        Returns:
            插件实例或None
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return None

        try:
            # 动态加载模块
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"Cannot load plugin spec: {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                logger.error(f"No Plugin subclass found in: {file_path}")
                return None

            # 实例化插件
            plugin = plugin_class()
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None

    def load_plugin_from_module(self, module_name: str, plugin_id: Optional[str] = None) -> Optional[Plugin]:
        """
        从模块加载插件
        
        Args:
            module_name: 模块名称
            plugin_id: 插件ID（可选）
        
        Returns:
            插件实例或None
        """
        try:
            module = importlib.import_module(module_name)

            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                logger.error(f"No Plugin subclass found in module: {module_name}")
                return None

            # 实例化插件
            plugin = plugin_class()
            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin from module {module_name}: {e}")
            return None

    def _find_plugin_class(self, module) -> Optional[type[Plugin]]:
        """
        在模块中查找插件类
        
        Args:
            module: 模块对象
        
        Returns:
            插件类或None
        """
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, Plugin) and 
                obj is not Plugin):
                return obj
        return None

    def discover_plugins(self) -> list[tuple[str, Path]]:
        """
        发现所有可用的插件
        
        Returns:
            插件ID和路径的列表
        """
        discovered = []

        for path in self._plugin_paths:
            if not path.exists():
                continue

            # 查找Python文件
            for file_path in path.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                plugin_id = file_path.stem
                discovered.append((plugin_id, file_path))

            # 查找包目录
            for dir_path in path.iterdir():
                if dir_path.is_dir() and (dir_path / "__init__.py").exists():
                    plugin_id = dir_path.name
                    discovered.append((plugin_id, dir_path / "__init__.py"))

        return discovered


# =============================================================================
# 插件管理器
# =============================================================================

class PluginManager:
    """
    插件管理器
    
    统一管理插件的加载、初始化和生命周期
    """

    def __init__(self):
        self._registry = PluginRegistry()
        self._loader = PluginLoader(self._registry)
        self._configs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def registry(self) -> PluginRegistry:
        """获取插件注册表"""
        return self._registry

    def add_plugin_path(self, path: str | Path) -> None:
        """添加插件搜索路径"""
        self._loader.add_plugin_path(path)

    def set_plugin_config(self, plugin_id: str, config: dict[str, Any]) -> None:
        """
        设置插件配置
        
        Args:
            plugin_id: 插件ID
            config: 配置字典
        """
        with self._lock:
            self._configs[plugin_id] = config

    def load_plugin(self, plugin_id: str, source: Optional[str | Path] = None) -> bool:
        """
        加载插件
        
        Args:
            plugin_id: 插件ID
            source: 插件源（文件路径或模块名），None则自动发现
        
        Returns:
            是否加载成功
        """
        with self._lock:
            if self._registry.has_plugin(plugin_id):
                logger.warning(f"Plugin {plugin_id} already loaded")
                return False

            # 加载插件实例
            if source:
                if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
                    plugin = self._loader.load_plugin_from_file(source, plugin_id)
                else:
                    plugin = self._loader.load_plugin_from_module(source, plugin_id)
            else:
                # 自动发现
                discovered = self._loader.discover_plugins()
                plugin = None
                for pid, path in discovered:
                    if pid == plugin_id:
                        plugin = self._loader.load_plugin_from_file(path, plugin_id)
                        break

            if not plugin:
                logger.error(f"Failed to load plugin: {plugin_id}")
                return False

            # 创建元数据（可以从插件获取或从配置文件加载）
            metadata = self._create_metadata(plugin)

            # 创建上下文
            config = self._configs.get(plugin_id, {})
            context = PluginContext(plugin_id=plugin_id, config=config)

            # 注册插件
            if not self._registry.register_plugin(plugin_id, plugin, metadata):
                return False

            # 初始化插件
            if not plugin.initialize(metadata, context):
                logger.error(f"Failed to initialize plugin: {plugin_id}")
                self._registry.unregister_plugin(plugin_id)
                return False

            logger.info(f"Successfully loaded and initialized plugin: {plugin_id}")
            return True

    def unload_plugin(self, plugin_id: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_id: 插件ID
        
        Returns:
            是否卸载成功
        """
        with self._lock:
            plugin = self._registry.get_plugin(plugin_id)
            if not plugin:
                logger.warning(f"Plugin not found: {plugin_id}")
                return False

            # 关闭插件
            if plugin.status == PluginStatus.ACTIVE:
                if not plugin.shutdown():
                    logger.warning(f"Plugin {plugin_id} shutdown failed")

            # 注销插件
            return self._registry.unregister_plugin(plugin_id)

    def reload_plugin(self, plugin_id: str) -> bool:
        """
        重新加载插件
        
        Args:
            plugin_id: 插件ID
        
        Returns:
            是否重新加载成功
        """
        with self._lock:
            # 保存配置
            config = self._configs.get(plugin_id, {})

            # 卸载
            if self._registry.has_plugin(plugin_id):
                self.unload_plugin(plugin_id)

            # 重新加载
            return self.load_plugin(plugin_id)

    def enable_plugin(self, plugin_id: str) -> bool:
        """
        启用插件
        
        Args:
            plugin_id: 插件ID
        
        Returns:
            是否启用成功
        """
        plugin = self._registry.get_plugin(plugin_id)
        if not plugin:
            return False

        if plugin.status == PluginStatus.DISABLED:
            # 重新初始化
            metadata = self._registry.get_metadata(plugin_id)
            config = self._configs.get(plugin_id, {})
            context = PluginContext(plugin_id=plugin_id, config=config)
            return plugin.initialize(metadata, context)

        return True

    def disable_plugin(self, plugin_id: str) -> bool:
        """
        禁用插件
        
        Args:
            plugin_id: 插件ID
        
        Returns:
            是否禁用成功
        """
        plugin = self._registry.get_plugin(plugin_id)
        if not plugin:
            return False

        if plugin.status == PluginStatus.ACTIVE:
            plugin.shutdown()
            plugin._status = PluginStatus.DISABLED
            return True

        return False

    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """获取插件实例"""
        return self._registry.get_plugin(plugin_id)

    def get_plugin_status(self, plugin_id: str) -> Optional[PluginStatus]:
        """获取插件状态"""
        plugin = self._registry.get_plugin(plugin_id)
        return plugin.status if plugin else None

    def get_all_plugins(self) -> dict[str, Plugin]:
        """获取所有插件"""
        return self._registry.get_all_plugins()

    def get_active_plugins(self) -> dict[str, Plugin]:
        """获取所有活跃插件"""
        all_plugins = self._registry.get_all_plugins()
        return {
            pid: plugin 
            for pid, plugin in all_plugins.items() 
            if plugin.status == PluginStatus.ACTIVE
        }

    def register_hook(self, hook_name: str, handler: Callable) -> None:
        """
        注册钩子处理器
        
        Args:
            hook_name: 钩子名称
            handler: 处理函数
        """
        hook = self._registry.get_or_create_hook(hook_name)
        hook.register(handler)

    def unregister_hook(self, hook_name: str, handler: Callable) -> bool:
        """
        注销钩子处理器
        
        Args:
            hook_name: 钩子名称
            handler: 处理函数
        
        Returns:
            是否成功注销
        """
        hook = self._registry.get_hook(hook_name)
        if hook:
            return hook.unregister(handler)
        return False

    def execute_hook(self, hook_name: str, *args, **kwargs) -> list[Any]:
        """
        执行钩子
        
        Args:
            hook_name: 钩子名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            所有处理器的返回结果列表
        """
        hook = self._registry.get_hook(hook_name)
        if hook:
            return hook.execute(*args, **kwargs)
        return []

    def execute_hook_first(self, hook_name: str, *args, **kwargs) -> Optional[Any]:
        """
        执行钩子并返回第一个结果
        
        Args:
            hook_name: 钩子名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            第一个处理器的返回结果
        """
        hook = self._registry.get_hook(hook_name)
        if hook:
            return hook.execute_first(*args, **kwargs)
        return None

    def load_all_plugins(self) -> list[str]:
        """
        加载所有发现的插件
        
        Returns:
            成功加载的插件ID列表
        """
        loaded = []
        discovered = self._loader.discover_plugins()

        for plugin_id, path in discovered:
            if self.load_plugin(plugin_id, path):
                loaded.append(plugin_id)

        return loaded

    def unload_all_plugins(self) -> None:
        """卸载所有插件"""
        plugin_ids = self._registry.get_plugin_ids()
        for plugin_id in plugin_ids:
            self.unload_plugin(plugin_id)

    def get_manager_stats(self) -> dict[str, Any]:
        """
        获取管理器统计
        
        Returns:
            统计信息
        """
        all_plugins = self._registry.get_all_plugins()
        active_count = sum(
            1 for p in all_plugins.values() 
            if p.status == PluginStatus.ACTIVE
        )

        return {
            "total_plugins": len(all_plugins),
            "active_plugins": active_count,
            "plugin_ids": list(all_plugins.keys()),
            "hooks": list(self._registry.get_all_hooks().keys()),
        }

    def _create_metadata(self, plugin: Plugin) -> PluginMetadata:
        """
        创建插件元数据
        
        可以从插件类属性或默认值创建
        """
        # 尝试从插件类获取元数据
        plugin_class = plugin.__class__

        name = getattr(plugin_class, 'PLUGIN_NAME', plugin_class.__name__)
        version = getattr(plugin_class, 'PLUGIN_VERSION', '1.0.0')
        description = getattr(plugin_class, 'PLUGIN_DESCRIPTION', '')
        author = getattr(plugin_class, 'PLUGIN_AUTHOR', '')
        dependencies = getattr(plugin_class, 'PLUGIN_DEPENDENCIES', [])
        hooks = getattr(plugin_class, 'PLUGIN_HOOKS', [])

        return PluginMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            dependencies=dependencies,
            hooks=hooks,
        )
