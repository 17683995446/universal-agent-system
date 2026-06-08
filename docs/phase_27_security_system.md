# 阶段27: Agent安全与权限系统 (Agent Security & Permission System)

## 概述

阶段27实现了Agent安全与权限系统，保护系统资源和操作。系统提供权限管理、角色定义、安全策略、审计日志等功能，确保Agent系统的安全性和可控性。

## 核心组件

### 1. Permission (权限枚举)

定义系统中的所有权限类型：

- **基础权限**: READ, WRITE, DELETE, EXECUTE
- **Agent权限**: AGENT_CREATE, AGENT_MANAGE, AGENT_STOP, AGENT_QUERY
- **配置权限**: CONFIG_READ, CONFIG_WRITE, CONFIG_RELOAD
- **插件权限**: PLUGIN_LOAD, PLUGIN_UNLOAD, PLUGIN_MANAGE
- **工作流权限**: WORKFLOW_CREATE, WORKFLOW_EXECUTE, WORKFLOW_MANAGE
- **集群权限**: CLUSTER_JOIN, CLUSTER_MANAGE, CLUSTER_QUERY
- **系统权限**: SYSTEM_ADMIN, SYSTEM_MONITOR, SYSTEM_LOG

```python
from src.agent import Permission

# 检查权限
if Permission.READ in user_permissions:
    # 用户有读取权限
    pass
```

### 2. Role (角色枚举)

定义系统角色：

- `ADMIN`: 管理员 - 所有权限
- `OPERATOR`: 操作员 - 执行权限
- `DEVELOPER`: 开发者 - 开发权限
- `VIEWER`: 观察者 - 只读权限
- `GUEST`: 访客 - 最小权限

```python
from src.agent import Role, ROLE_PERMISSIONS

# 获取角色权限
admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
```

### 3. UserIdentity (用户身份)

表示用户或实体的身份信息：

```python
from src.agent import UserIdentity, Role, Permission

# 创建用户
user = UserIdentity(
    user_id="user_001",
    name="张三",
    role=Role.DEVELOPER,
    custom_permissions={Permission.DELETE},  # 自定义权限
)

# 检查权限
if user.has_permission(Permission.READ):
    print("用户有读取权限")

# 添加权限
user.add_permission(Permission.SYSTEM_ADMIN)

# 移除权限
user.remove_permission(Permission.DELETE)
```

### 4. SecurityPolicy (安全策略)

定义特定的安全规则：

```python
from src.agent import SecurityPolicy, SecurityPolicyType, Permission, Role

# 创建拒绝覆盖策略
deny_policy = SecurityPolicy(
    policy_id="deny_delete",
    name="禁止删除",
    policy_type=SecurityPolicyType.DENY_OVERRIDE,
    permissions={Permission.DELETE},
    users=["*"],           # 所有用户
    resources=["critical_*"],  # 关键资源
    priority=10,           # 高优先级
)

# 创建允许覆盖策略
allow_policy = SecurityPolicy(
    policy_id="allow_admin",
    name="管理员特权",
    policy_type=SecurityPolicyType.ALLOW_OVERRIDE,
    permissions={Permission.SYSTEM_ADMIN},
    roles=[Role.ADMIN],
    resources=["*"],
    priority=5,
)
```

### 5. PermissionChecker (权限检查器)

执行权限检查逻辑：

```python
from src.agent import PermissionChecker, UserIdentity, Role, Permission

checker = PermissionChecker()

# 注册用户
user = UserIdentity(user_id="user_001", name="张三", role=Role.VIEWER)
checker.register_user(user)

# 添加策略
checker.add_policy(deny_policy)

# 检查权限
result = checker.check_permission("user_001", Permission.READ, "agent_1")
if result.granted:
    print("权限检查通过")
else:
    print(f"权限检查失败: {result.reason}")
```

### 6. AuditLogger (审计日志器)

记录所有操作的审计日志：

```python
from src.agent import AuditLogger, Permission

logger = AuditLogger(max_entries=10000)

# 记录日志
entry = logger.log(
    user_id="user_001",
    action="read",
    resource="agent_1",
    permission=Permission.READ,
    granted=True,
    ip_address="192.168.1.1",
)

# 获取日志
entries = logger.get_entries(user_id="user_001", limit=100)

# 获取统计
stats = logger.get_stats()
print(f"总记录: {stats['total_entries']}")
print(f"授权数: {stats['granted_count']}")
print(f"拒绝数: {stats['denied_count']}")
```

### 7. SecurityManager (安全管理器)

统一管理权限检查和审计日志：

```python
from src.agent import SecurityManager, UserIdentity, Role, Permission

manager = SecurityManager()

# 注册用户
user = UserIdentity(user_id="user_001", name="张三", role=Role.DEVELOPER)
manager.register_user(user)

# 添加策略
manager.add_policy(deny_policy)

# 检查权限并记录日志
result = manager.check_and_log("user_001", Permission.READ, "agent_1")

# 要求权限（无权限则抛出异常）
try:
    manager.require_permission("user_001", Permission.DELETE, "agent_1")
except PermissionDeniedError as e:
    print(f"权限不足: {e}")

# 获取审计日志
entries = manager.get_audit_entries(user_id="user_001")

# 获取安全统计
stats = manager.get_security_stats()

# 创建默认管理员
admin = manager.create_default_admin()
```

## 安全策略类型

| 类型 | 说明 |
|------|------|
| ALLOW | 允许策略 |
| DENY | 拒绝策略 |
| ALLOW_OVERRIDE | 强制允许（覆盖其他规则） |
| DENY_OVERRIDE | 强制拒绝（覆盖其他规则） |

## 权限检查流程

```
1. 检查用户是否存在
   ↓
2. 应用强制覆盖策略（DENY_OVERRIDE / ALLOW_OVERRIDE）
   ↓ 如果有强制覆盖，直接返回结果
3. 检查用户基础权限（角色权限 + 自定义权限）
   ↓
4. 应用普通安全策略（ALLOW / DENY）
   ↓
5. 执行钩子检查
   ↓
6. 综合判断返回结果
```

## 使用场景

### 场景1: 保护敏感操作

```python
# 创建安全管理器
manager = SecurityManager()

# 创建拒绝策略
deny_policy = SecurityPolicy(
    policy_id="protect_critical",
    name="保护关键资源",
    policy_type=SecurityPolicyType.DENY_OVERRIDE,
    permissions={Permission.DELETE, Permission.WRITE},
    resources=["critical_*"],
    users=["*"],
    priority=1,
)
manager.add_policy(deny_policy)

# 即使管理员也无法删除关键资源
admin = UserIdentity(user_id="admin", name="管理员", role=Role.ADMIN)
manager.register_user(admin)

result = manager.check_permission("admin", Permission.DELETE, "critical_data")
# result.granted = False（策略强制拒绝）
```

### 场景2: 审计追踪

```python
# 记录所有操作
manager.check_and_log("user_001", Permission.READ, "agent_1", "agent_read")

# 查看拒绝记录
denied_entries = manager.get_audit_entries(granted=False, limit=50)

for entry in denied_entries:
    print(f"用户 {entry.user_id} 尝试 {entry.action} 被拒绝")
```

### 场景3: 基于角色的访问控制

```python
# 创建不同角色的用户
viewer = UserIdentity(user_id="viewer_1", name="观察者", role=Role.VIEWER)
developer = UserIdentity(user_id="dev_1", name="开发者", role=Role.DEVELOPER)
admin = UserIdentity(user_id="admin_1", name="管理员", role=Role.ADMIN)

manager.register_user(viewer)
manager.register_user(developer)
manager.register_user(admin)

# 观察者只能读取
assert manager.check_permission("viewer_1", Permission.READ).granted
assert not manager.check_permission("viewer_1", Permission.DELETE).granted

# 开发者可以创建
assert manager.check_permission("dev_1", Permission.AGENT_CREATE).granted

# 管理员有所有权限
assert manager.check_permission("admin_1", Permission.SYSTEM_ADMIN).granted
```

## 最佳实践

1. **最小权限原则**: 只授予必要的权限
2. **策略优先级**: 设置合理的优先级，关键策略优先级更高
3. **审计日志**: 定期检查审计日志，发现异常行为
4. **强制覆盖**: 使用 DENY_OVERRIDE 保护关键资源
5. **角色分离**: 不同职责使用不同角色

## 线程安全

- PermissionChecker 使用 threading.RLock
- AuditLogger 使用 threading.RLock
- SecurityManager 使用 threading.RLock

## 测试覆盖

- 66个测试用例，覆盖所有核心功能
- 包括权限枚举、角色、用户身份、策略、检查器、日志器、管理器测试
- 验证强制覆盖策略、导出功能

## 总结

阶段27成功实现了Agent安全与权限系统，提供了：
- 完整的权限类型定义
- 基于角色的访问控制
- 灵活的安全策略系统
- 完善的审计日志
- 线程安全的实现

为构建安全、可控的Agent系统奠定了坚实基础。