# 阶段32: Agent配置中心

> 遵循"图难于其易，为大于其细。天下难事，必作于易；天下大事，必作于细。"的原则

## 概述

阶段32实现了完整的Agent配置中心，提供集中式配置管理、版本控制、灰度发布和配置模板等功能，支持多环境、多实例的灵活配置管理。

## 核心组件

### 1. 配置状态与作用域

#### ConfigStatus 枚举
```python
class ConfigStatus(str, Enum):
    DRAFT = "draft"           # 草稿
    PUBLISHED = "published"   # 已发布
    ROLLED_BACK = "rolled_back"  # 已回滚
    DEPRECATED = "deprecated"    # 已废弃
```

#### ConfigScope 枚举
```python
class ConfigScope(str, Enum):
    GLOBAL = "global"         # 全局
    APPLICATION = "application"  # 应用级
    SERVICE = "service"       # 服务级
    INSTANCE = "instance"     # 实例级
```

#### RolloutStrategy 枚举
```python
class RolloutStrategy(str, Enum):
    ALL_AT_ONCE = "all_at_once"      # 全量发布
    PERCENTAGE = "percentage"        # 百分比
    CANARY = "canary"                # 金丝雀
    BLUE_GREEN = "blue_green"        # 蓝绿部署
    A_B_TEST = "a_b_test"            # A/B测试
```

### 2. 核心数据模型

- **ConfigValue**: 配置值，支持加密
- **ConfigVersion**: 配置版本，包含完整版本控制信息
- **ConfigInstance**: 配置实例，关联到具体作用域
- **RolloutPlan**: 灰度发布计划
- **RolloutTarget**: 灰度发布目标
- **ConfigChangeEvent**: 配置变更事件

### 3. 存储后端

#### ConfigStorageBackend (基类)
定义存储后端接口：save, load, delete, list_keys, exists

#### MemoryConfigStorage
内存存储实现，适用于测试和开发环境

#### FileConfigStorage
文件存储实现，适用于单机部署

```python
# 使用内存存储
center = ConfigCenter(MemoryConfigStorage())

# 使用文件存储
center = ConfigCenter(FileConfigStorage("./config_data"))
```

### 4. ConfigCenter - 配置中心核心

#### 配置版本管理
```python
center = ConfigCenter()

# 创建配置
version = center.create_config(
    config_id="app.config",
    data={"timeout": 30, "retries": 3},
    created_by="admin",
    comment="Initial config"
)

# 获取版本
version = center.get_version("app.config", version_id)
latest = center.get_latest_version("app.config")
published = center.get_published_version("app.config")

# 列出所有版本
versions = center.list_versions("app.config")
```

#### 配置发布与回滚
```python
# 发布版本
center.publish_version("app.config", version_id, operator="admin")

# 回滚到指定版本
center.rollback("app.config", to_version_id, operator="admin")
```

#### 实例管理
```python
# 注册配置实例
instance = center.register_instance(
    config_id="app.config",
    scope=ConfigScope.SERVICE,
    target_id="service1",
    metadata={"region": "us-east"}
)

# 应用配置到实例
center.apply_config_to_instance(instance_id, version_id)

# 获取目标的配置
configs = center.get_config_for_target("service1", ConfigScope.SERVICE)
```

#### 灰度发布
```python
# 创建灰度发布计划
target = RolloutTarget(
    target_id="instance1",
    target_type="instance",
    weight=0.1
)

plan = center.create_rollout_plan(
    config_id="app.config",
    version_id=new_version_id,
    strategy=RolloutStrategy.CANARY,
    targets=[target]
)

# 开始发布
center.start_rollout(plan.plan_id)

# 更新进度
center.update_rollout_progress(plan.plan_id, 50.0)

# 完成或取消
center.update_rollout_progress(plan.plan_id, 100.0)  # 完成
center.cancel_rollout(plan.plan_id)  # 取消
```

#### 事件监听
```python
def on_config_change(event: ConfigChangeEvent):
    print(f"Config {event.config_id} changed: {event.change_type}")

center.add_listener(on_config_change)
center.remove_listener(on_config_change)
```

### 5. ConfigTemplate - 配置模板

```python
schema = {
    "type": "object",
    "required": ["name", "port"],
    "properties": {
        "name": {"type": "string"},
        "port": {"type": "integer"},
        "timeout": {"type": "integer", "default": 30}
    }
}

template = ConfigTemplate("app_schema", schema)

# 验证配置
valid, errors = template.validate({"name": "myapp", "port": 8080})

# 应用默认值
config = template.apply_defaults({"name": "myapp", "port": 8080})
# config = {"name": "myapp", "port": 8080, "timeout": 30}
```

## 使用示例

### 基本使用

```python
from src.agent.config_center import (
    ConfigCenter, ConfigScope, RolloutStrategy,
    RolloutTarget, MemoryConfigStorage
)

# 创建配置中心
center = ConfigCenter(MemoryConfigStorage())

# 创建并发布配置
v1 = center.create_config(
    "api.config",
    {"timeout": 30, "retries": 3},
    comment="Initial version"
)
center.publish_version("api.config", v1.version_id)

# 注册服务实例
instance = center.register_instance(
    "api.config",
    ConfigScope.SERVICE,
    "api-service-1"
)

# 应用配置
center.apply_config_to_instance(instance.instance_id, v1.version_id)
```

### 完整配置生命周期

```python
# 1. 创建配置
v1 = center.create_config("app.config", {"v": 1})

# 2. 发布配置
center.publish_version("app.config", v1.version_id)

# 3. 创建新版本
v2 = center.create_config("app.config", {"v": 2})

# 4. 灰度发布
target = RolloutTarget(target_id="service1", target_type="service")
plan = center.create_rollout_plan(
    "app.config", v2.version_id, RolloutStrategy.CANARY, [target]
)
center.start_rollout(plan.plan_id)
center.update_rollout_progress(plan.plan_id, 100.0)

# 5. 回滚（如需要）
center.rollback("app.config", v1.version_id)
```

## 测试覆盖

共46个测试用例，覆盖：
- 配置状态枚举 (1 test)
- 配置作用域枚举 (1 test)
- 灰度发布策略枚举 (1 test)
- 配置值 (2 tests)
- 配置版本 (2 tests)
- 配置实例 (2 tests)
- 灰度发布计划 (2 tests)
- 配置变更事件 (2 tests)
- 内存存储后端 (5 tests)
- 文件存储后端 (4 tests)
- 配置中心核心 (16 tests)
- 配置模板 (4 tests)
- 便捷函数 (3 tests)
- 集成测试 (1 test)

## 文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| `/workspace/src/agent/config_center.py` | 配置中心实现 | 785 |
| `/workspace/tests/test_config_center.py` | 单元测试 | 698 |
| `/workspace/docs/stage32_config_center.md` | 本文档 | - |

## 与前一阶段的集成

配置中心与阶段31的健康检查系统集成：
- 配置变更可触发健康检查
- 灰度发布进度可通过健康端点暴露
- 配置回滚可联动健康监控

## 下一步（阶段33）

下一阶段将实现 **Agent服务发现**，提供服务注册、发现、负载均衡等功能。

## 总结

阶段32成功实现了完整的Agent配置中心，包括：

1. ✅ 定义了配置状态、作用域、灰度策略枚举
2. ✅ 实现了配置版本模型和版本控制
3. ✅ 实现了配置发布和回滚机制
4. ✅ 实现了灰度发布功能（全量、百分比、金丝雀、蓝绿、A/B测试）
5. ✅ 实现了内存和文件存储后端
6. ✅ 实现了配置模板和验证
7. ✅ 编写了46个单元测试
8. ✅ 所有测试通过

系统现在具备了完整的配置管理能力，支持多版本控制、灰度发布、实例管理和配置模板验证。
