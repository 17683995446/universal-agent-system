# 阶段26: Agent插件系统 (Agent Plugin System)

## 概述

阶段26实现了Agent插件系统，支持动态扩展和模块化架构。系统提供插件生命周期管理、动态加载、钩子机制等功能，使Agent系统具有高度的可扩展性和灵活性。

## 核心组件

### 1. PluginStatus (插件状态枚举)

定义插件的生命周期状态：

- `UNLOADED`: 未加载
- `LOADING`: 加载中
- `LOADED`: 已加载
- `INITIALIZING`: 初始化中
- `ACTIVE`: 活跃
- `ERROR`: 错误
- `DISABLED`: 已禁用
- `UNLOADING`: 卸载中

```python
from src.agent import PluginStatus

# 检查插件状态
if plugin.status == PluginStatus.ACTIVE:
    # 插件已激活
    pass
```

### 2. PluginMetadata (插件元数据)

描述插件的基本信息：

```python
from src.agent import PluginMetadata

metadata = PluginMetadata(
    name="MyPlugin",
    version="1.0.0",
    description="A sample plugin",
    author="Your Name",
    dependencies=["other_plugin"],
    hooks=["on_message", "on_command"],
)
```

### 3. PluginContext (插件上下文)

提供给插件的运行时环境：

```python
from src.agent import PluginContext

context = PluginContext(
    plugin_id="my_plugin",
    config={"api_key": "secret", "timeout": 30},
)

# 获取配置
api_key = context.get_config("api_key")
timeout = context.get_config("timeout", 30)

# 共享数据
context.set_shared_data("shared_key", "shared_value")
value = context.get_shared_data("shared_key")
```

### 4. Plugin (插件基类)

所有插件必须继承的基类：

```python
from src.agent import Plugin, PluginMetadata, PluginContext

class MyPlugin(Plugin):
    """自定义插件"""
    PLUGIN_NAME = "MyPlugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "My custom plugin"
    PLUGIN_AUTHOR = "Your Name"
    PLUGIN_DEPENDENCIES = []
    PLUGIN_HOOKS = ["on_message"]

    def __init__(self):
        super().__init__()
        self.data = {}

    def on_init(self) -> bool:
        """初始化插件"""
        # 获取配置
        api_key = self.context.get_config("api_key")
        # 初始化逻辑
        return True

    def on_shutdown(self) -> bool:
        """关闭插件"""
        # 清理逻辑
        return True
```

### 5. Hook (钩子)

实现插件间的扩展点：

```python
from src.agent import Hook

# 创建钩子
hook = Hook("on_message")

# 注册处理器
def handle_message(message):
    print(f"Received: {message}")
    return True

hook.register(handle_message)

# 执行钩子
results = hook.execute("Hello World")

# 执行第一个处理器
result = hook.execute_first("Hello World")
```

### 6. PluginRegistry (插件注册表)

管理所有已加载的插件：

```python
from src.agent import PluginRegistry

registry = PluginRegistry()

# 注册插件
registry.register_plugin("my_plugin", plugin, metadata)

# 获取插件
plugin = registry.get_plugin("my_plugin")
metadata = registry.get_metadata("my_plugin")

# 获取所有插件
all_plugins = registry.get_all_plugins()

# 注销插件
registry.unregister_plugin("my_plugin")

# 获取或创建钩子
hook = registry.get_or_create_hook("my_hook")
```

### 7. PluginLoader (插件加载器)

从文件系统加载插件：

```python
from src.agent import PluginLoader, PluginRegistry

registry = PluginRegistry()
loader = PluginLoader(registry)

# 添加插件搜索路径
loader.add_plugin_path("/path/to/plugins")

# 从文件加载
plugin = loader.load_plugin_from_file("/path/to/plugin.py")

# 从模块加载
plugin = loader.load_plugin_from_module("my_package.plugins.my_plugin")

# 自动发现插件
discovered = loader.discover_plugins()
```

### 8. PluginManager (插件管理器)

统一管理插件的完整生命周期：

```python
from src.agent import PluginManager

manager = PluginManager()

# 添加插件路径
manager.add_plugin_path("/path/to/plugins")

# 设置插件配置
manager.set_plugin_config("my_plugin", {"api_key": "secret"})

# 加载插件
manager.load_plugin("my_plugin")

# 从文件加载
manager.load_plugin("my_plugin", "/path/to/plugin.py")

# 从模块加载
manager.load_plugin("my_plugin", "my_package.plugins.my_plugin")

# 自动发现并加载所有插件
loaded = manager.load_all_plugins()

# 卸载插件
manager.unload_plugin("my_plugin")

# 重新加载插件
manager.reload_plugin("my_plugin")

# 启用/禁用插件
manager.enable_plugin("my_plugin")
manager.disable_plugin("my_plugin")

# 获取插件状态
status = manager.get_plugin_status("my_plugin")

# 获取活跃插件
active = manager.get_active_plugins()

# 注册钩子
manager.register_hook("on_message", handle_message)

# 执行钩子
results = manager.execute_hook("on_message", "Hello")

# 获取统计
stats = manager.get_manager_stats()
```

## 使用场景

### 场景1: 创建自定义插件

```python
# my_plugin.py
from src.agent import Plugin, PluginMetadata, PluginContext

class EmailNotificationPlugin(Plugin):
    PLUGIN_NAME = "EmailNotification"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Send email notifications"
    PLUGIN_AUTHOR = "Your Name"

    def on_init(self) -> bool:
        self.smtp_server = self.context.get_config("smtp_server")
        self.smtp_port = self.context.get_config("smtp_port", 587)
        # 初始化SMTP连接
        return True

    def on_shutdown(self) -> bool:
        # 关闭SMTP连接
        return True

    def send_email(self, to, subject, body):
        # 发送邮件逻辑
        pass
```

### 场景2: 使用钩子扩展功能

```python
# 在主应用中定义钩子点
manager = PluginManager()
manager.load_all_plugins()

# 在关键位置执行钩子
results = manager.execute_hook("before_process", data)

# 处理数据
processed = process_data(data)

# 执行后续钩子
manager.execute_hook("after_process", processed)
```

### 场景3: 插件间通信

```python
# Plugin A
class PluginA(Plugin):
    def on_init(self) -> bool:
        # 注册钩子处理器
        self.context.set_shared_data("plugin_a_ready", True)
        return True

# Plugin B
class PluginB(Plugin):
    def on_init(self) -> bool:
        # 检查Plugin A是否就绪
        ready = self.context.get_shared_data("plugin_a_ready", False)
        if ready:
            # 使用Plugin A的功能
            pass
        return True
```

## 插件生命周期

```
UNLOADED -> LOADING -> LOADED -> INITIALIZING -> ACTIVE
                                              |
                                              v
UNLOADED <- UNLOADING <- DISABLED <- ERROR <-+
```

## 钩子执行流程

```python
# 1. 创建钩子
hook = Hook("on_event")

# 2. 注册处理器
hook.register(handler1)
hook.register(handler2)
hook.register(handler3)

# 3. 执行钩子
results = hook.execute(arg1, arg2)
# 返回: [result1, result2, result3]

# 或只执行第一个
result = hook.execute_first(arg1, arg2)
# 返回: result1
```

## 最佳实践

1. **插件命名**: 使用有意义的名称，避免冲突
2. **版本管理**: 使用语义化版本号
3. **依赖声明**: 在 PLUGIN_DEPENDENCIES 中声明依赖
4. **错误处理**: 在 on_init 和 on_shutdown 中正确处理异常
5. **配置验证**: 在初始化时验证配置项
6. **钩子命名**: 使用命名空间避免冲突，如 `plugin_name.event_name`

## 线程安全

- PluginRegistry 使用 threading.RLock 保证线程安全
- Hook 使用 threading.RLock 保证线程安全
- PluginManager 使用 threading.RLock 保证线程安全

## 测试覆盖

- 48个测试用例，覆盖所有核心功能
- 包括状态测试、元数据测试、钩子测试、管理器测试
- 验证生命周期、导出功能

## 总结

阶段26成功实现了Agent插件系统，提供了：
- 完整的插件生命周期管理
- 动态加载和卸载机制
- 灵活的钩子系统
- 插件间通信机制
- 线程安全的实现

为构建可扩展、模块化的Agent系统奠定了坚实基础。
