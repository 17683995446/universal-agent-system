# =============================================================================
# Agent安全与权限系统 - 阶段27: Agent Security & Permission System
# =============================================================================
#
# 功能：实现Agent安全与权限系统，保护系统资源和操作
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Callable


# =============================================================================
# 权限类型
# =============================================================================

class Permission(str, Enum):
    """权限类型枚举"""
    # 基础权限
    READ = "read"               # 读取
    WRITE = "write"             # 写入
    DELETE = "delete"           # 删除
    EXECUTE = "execute"         # 执行
    
    # Agent相关权限
    AGENT_CREATE = "agent_create"       # 创建Agent
    AGENT_MANAGE = "agent_manage"       # 管理Agent
    AGENT_STOP = "agent_stop"           # 停止Agent
    AGENT_QUERY = "agent_query"         # 查询Agent
    
    # 配置权限
    CONFIG_READ = "config_read"         # 读取配置
    CONFIG_WRITE = "config_write"       # 写入配置
    CONFIG_RELOAD = "config_reload"     # 重载配置
    
    # 插件权限
    PLUGIN_LOAD = "plugin_load"         # 加载插件
    PLUGIN_UNLOAD = "plugin_unload"     # 卸载插件
    PLUGIN_MANAGE = "plugin_manage"     # 管理插件
    
    # 工作流权限
    WORKFLOW_CREATE = "workflow_create" # 创建工作流
    WORKFLOW_EXECUTE = "workflow_execute" # 执行工作流
    WORKFLOW_MANAGE = "workflow_manage" # 管理工作流
    
    # 集群权限
    CLUSTER_JOIN = "cluster_join"       # 加入集群
    CLUSTER_MANAGE = "cluster_manage"   # 管理集群
    CLUSTER_QUERY = "cluster_query"     # 查询集群
    
    # 系统权限
    SYSTEM_ADMIN = "system_admin"       # 系统管理
    SYSTEM_MONITOR = "system_monitor"   # 系统监控
    SYSTEM_LOG = "system_log"           # 查看日志


# =============================================================================
# 角色类型
# =============================================================================

class Role(str, Enum):
    """角色类型枚举"""
    ADMIN = "admin"             # 管理员 - 所有权限
    OPERATOR = "operator"       # 操作员 - 执行权限
    DEVELOPER = "developer"     # 开发者 - 开发权限
    VIEWER = "viewer"           # 观察者 - 只读权限
    GUEST = "guest"             # 访客 - 最小权限


# =============================================================================
# 预定义角色权限
# =============================================================================

ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # 所有权限
    
    Role.OPERATOR: {
        Permission.READ,
        Permission.EXECUTE,
        Permission.AGENT_QUERY,
        Permission.AGENT_MANAGE,
        Permission.AGENT_STOP,
        Permission.CONFIG_READ,
        Permission.WORKFLOW_EXECUTE,
        Permission.CLUSTER_QUERY,
        Permission.SYSTEM_MONITOR,
    },
    
    Role.DEVELOPER: {
        Permission.READ,
        Permission.WRITE,
        Permission.AGENT_CREATE,
        Permission.AGENT_QUERY,
        Permission.CONFIG_READ,
        Permission.CONFIG_WRITE,
        Permission.PLUGIN_LOAD,
        Permission.PLUGIN_UNLOAD,
        Permission.WORKFLOW_CREATE,
        Permission.WORKFLOW_EXECUTE,
        Permission.SYSTEM_LOG,
    },
    
    Role.VIEWER: {
        Permission.READ,
        Permission.AGENT_QUERY,
        Permission.CONFIG_READ,
        Permission.CLUSTER_QUERY,
        Permission.SYSTEM_MONITOR,
    },
    
    Role.GUEST: {
        Permission.READ,
        Permission.AGENT_QUERY,
    },
}


# =============================================================================
# 用户身份
# =============================================================================

@dataclass
class UserIdentity:
    """
    用户身份
    
    表示一个用户或实体的身份信息
    """
    user_id: str
    name: str
    role: Role = Role.GUEST
    custom_permissions: set[Permission] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.role, str):
            self.role = Role(self.role)
    
    def get_all_permissions(self) -> set[Permission]:
        """获取所有权限（角色权限 + 自定义权限）"""
        base_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return base_permissions.union(self.custom_permissions)
    
    def has_permission(self, permission: Permission) -> bool:
        """检查是否有某个权限"""
        return permission in self.get_all_permissions()
    
    def add_permission(self, permission: Permission) -> None:
        """添加自定义权限"""
        self.custom_permissions.add(permission)
    
    def remove_permission(self, permission: Permission) -> bool:
        """移除自定义权限"""
        if permission in self.custom_permissions:
            self.custom_permissions.remove(permission)
            return True
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role.value,
            "custom_permissions": [p.value for p in self.custom_permissions],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


# =============================================================================
# 权限检查结果
# =============================================================================

@dataclass
class PermissionCheckResult:
    """
    权限检查结果
    """
    granted: bool
    permission: Permission
    user_id: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "granted": self.granted,
            "permission": self.permission.value,
            "user_id": self.user_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# =============================================================================
# 审计日志记录
# =============================================================================

@dataclass
class AuditLogEntry:
    """
    审计日志记录
    
    记录一次操作的审计信息
    """
    entry_id: str
    user_id: str
    action: str
    resource: str
    permission: Optional[Permission] = None
    granted: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "permission": self.permission.value if self.permission else None,
            "granted": self.granted,
            "details": self.details,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
        }


# =============================================================================
# 安全策略
# =============================================================================

class SecurityPolicyType(str, Enum):
    """安全策略类型"""
    ALLOW = "allow"             # 允许
    DENY = "deny"               # 拒绝
    ALLOW_OVERRIDE = "allow_override"  # 允许覆盖
    DENY_OVERRIDE = "deny_override"    # 拒绝覆盖


@dataclass
class SecurityPolicy:
    """
    安全策略
    
    定义特定的安全规则
    """
    policy_id: str
    name: str
    policy_type: SecurityPolicyType = SecurityPolicyType.ALLOW
    permissions: set[Permission] = field(default_factory=set)
    resources: list[str] = field(default_factory=list)  # 资源匹配模式
    users: list[str] = field(default_factory=list)      # 用户匹配模式
    roles: list[Role] = field(default_factory=list)     # 角色匹配
    conditions: dict[str, Any] = field(default_factory=dict)  # 条件
    priority: int = 100           # 优先级（数字越小优先级越高）
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    
    def matches_user(self, user: UserIdentity) -> bool:
        """检查是否匹配用户"""
        # 用户ID匹配
        if user.user_id in self.users:
            return True
        # 角色匹配
        if user.role in self.roles:
            return True
        # 通配符匹配
        if "*" in self.users or "*" in self.roles:
            return True
        return False
    
    def matches_resource(self, resource: str) -> bool:
        """检查是否匹配资源"""
        if "*" in self.resources:
            return True
        for pattern in self.resources:
            if pattern == resource:
                return True
            # 简单的前缀匹配
            if pattern.endswith("*") and resource.startswith(pattern[:-1]):
                return True
        return False
    
    def matches_permission(self, permission: Permission) -> bool:
        """检查是否匹配权限"""
        return permission in self.permissions
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "policy_type": self.policy_type.value,
            "permissions": [p.value for p in self.permissions],
            "resources": self.resources,
            "users": self.users,
            "roles": [r.value for r in self.roles],
            "conditions": self.conditions,
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }


# =============================================================================
# 权限检查器
# =============================================================================

class PermissionChecker:
    """
    权限检查器
    
    执行权限检查逻辑
    """

    def __init__(self):
        self._users: dict[str, UserIdentity] = {}
        self._policies: list[SecurityPolicy] = []
        self._lock = threading.RLock()
        self._check_hooks: list[Callable[[UserIdentity, Permission, str], bool]] = []

    def register_user(self, user: UserIdentity) -> bool:
        """
        注册用户
        
        Args:
            user: 用户身份
        
        Returns:
            是否注册成功
        """
        with self._lock:
            if user.user_id in self._users:
                return False
            self._users[user.user_id] = user
            return True

    def unregister_user(self, user_id: str) -> bool:
        """
        注销用户
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否注销成功
        """
        with self._lock:
            if user_id in self._users:
                del self._users[user_id]
                return True
            return False

    def get_user(self, user_id: str) -> Optional[UserIdentity]:
        """获取用户"""
        with self._lock:
            return self._users.get(user_id)

    def add_policy(self, policy: SecurityPolicy) -> None:
        """
        添加安全策略
        
        Args:
            policy: 安全策略
        """
        with self._lock:
            self._policies.append(policy)
            # 按优先级排序
            self._policies.sort(key=lambda p: p.priority)

    def remove_policy(self, policy_id: str) -> bool:
        """
        移除安全策略
        
        Args:
            policy_id: 策略ID
        
        Returns:
            是否移除成功
        """
        with self._lock:
            for i, policy in enumerate(self._policies):
                if policy.policy_id == policy_id:
                    self._policies.pop(i)
                    return True
            return False

    def register_check_hook(self, hook: Callable[[UserIdentity, Permission, str], bool]) -> None:
        """
        注册权限检查钩子
        
        Args:
            hook: 检查钩子函数
        """
        self._check_hooks.append(hook)

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: str = "*",
    ) -> PermissionCheckResult:
        """
        检查权限
        
        Args:
            user_id: 用户ID
            permission: 权限类型
            resource: 资源标识
        
        Returns:
            权限检查结果
        """
        user = self.get_user(user_id)
        if not user:
            return PermissionCheckResult(
                granted=False,
                permission=permission,
                user_id=user_id,
                reason="用户不存在",
            )

        # 1. 应用安全策略（优先级最高）
        policy_override, policy_granted = self._apply_policies_with_override(user, permission, resource)
        
        if policy_override:
            # 策略强制决定结果
            reason = "安全策略强制" + ("允许" if policy_granted else "拒绝")
            return PermissionCheckResult(
                granted=policy_granted,
                permission=permission,
                user_id=user_id,
                reason=reason if not policy_granted else "",
            )

        # 2. 检查用户基础权限
        if user.has_permission(permission):
            base_granted = True
        else:
            base_granted = False

        # 3. 应用普通安全策略
        policy_result = self._apply_normal_policies(user, permission, resource)

        # 4. 执行钩子检查
        hook_result = True
        for hook in self._check_hooks:
            try:
                if not hook(user, permission, resource):
                    hook_result = False
                    break
            except Exception:
                pass

        # 5. 综合判断
        final_granted = base_granted and policy_result and hook_result

        reason = ""
        if not base_granted:
            reason = "用户无此权限"
        elif not policy_result:
            reason = "安全策略拒绝"
        elif not hook_result:
            reason = "钩子检查拒绝"

        return PermissionCheckResult(
            granted=final_granted,
            permission=permission,
            user_id=user_id,
            reason=reason,
        )

    def _apply_policies_with_override(
        self,
        user: UserIdentity,
        permission: Permission,
        resource: str,
    ) -> tuple[bool, bool]:
        """应用强制覆盖策略，返回 (是否强制覆盖, 结果)"""
        with self._lock:
            for policy in self._policies:
                if not policy.enabled:
                    continue
                if not policy.matches_user(user):
                    continue
                if not policy.matches_permission(permission):
                    continue
                if not policy.matches_resource(resource):
                    continue

                if policy.policy_type == SecurityPolicyType.DENY_OVERRIDE:
                    return (True, False)  # 强制拒绝
                if policy.policy_type == SecurityPolicyType.ALLOW_OVERRIDE:
                    return (True, True)   # 强制允许
        return (False, False)  # 无强制覆盖

    def _apply_normal_policies(
        self,
        user: UserIdentity,
        permission: Permission,
        resource: str,
    ) -> bool:
        """应用普通策略"""
        with self._lock:
            allow_found = False
            deny_found = False

            for policy in self._policies:
                if not policy.enabled:
                    continue
                if policy.policy_type in (SecurityPolicyType.DENY_OVERRIDE, SecurityPolicyType.ALLOW_OVERRIDE):
                    continue  # 跳过强制覆盖策略
                if not policy.matches_user(user):
                    continue
                if not policy.matches_permission(permission):
                    continue
                if not policy.matches_resource(resource):
                    continue

                if policy.policy_type == SecurityPolicyType.DENY:
                    deny_found = True

                if policy.policy_type == SecurityPolicyType.ALLOW:
                    allow_found = True

            # 拒绝优先
            if deny_found:
                return False
            if allow_found:
                return True

            return True  # 无匹配策略，默认允许

    def get_all_users(self) -> dict[str, UserIdentity]:
        """获取所有用户"""
        with self._lock:
            return self._users.copy()

    def get_all_policies(self) -> list[SecurityPolicy]:
        """获取所有策略"""
        with self._lock:
            return self._policies.copy()


# =============================================================================
# 审计日志器
# =============================================================================

class AuditLogger:
    """
    审计日志器
    
    记录所有操作的审计日志
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: list[AuditLogEntry] = []
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._handlers: list[Callable[[AuditLogEntry], None]] = []

    def log(
        self,
        user_id: str,
        action: str,
        resource: str,
        permission: Optional[Permission] = None,
        granted: bool = True,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        记录审计日志
        
        Args:
            user_id: 用户ID
            action: 操作类型
            resource: 资源标识
            permission: 权限类型
            granted: 是否授权
            details: 详细信息
            ip_address: IP地址
            session_id: 会话ID
        
        Returns:
            审计日志记录
        """
        entry = AuditLogEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            action=action,
            resource=resource,
            permission=permission,
            granted=granted,
            details=details or {},
            ip_address=ip_address,
            session_id=session_id,
        )

        with self._lock:
            self._entries.append(entry)
            # 限制最大条数
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

        # 调用处理器
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass

        return entry

    def log_permission_check(
        self,
        result: PermissionCheckResult,
        resource: str,
        action: str = "permission_check",
    ) -> AuditLogEntry:
        """
        记录权限检查日志
        
        Args:
            result: 权限检查结果
            resource: 资源标识
            action: 操作类型
        
        Returns:
            审计日志记录
        """
        return self.log(
            user_id=result.user_id,
            action=action,
            resource=resource,
            permission=result.permission,
            granted=result.granted,
            details={"reason": result.reason},
        )

    def register_handler(self, handler: Callable[[AuditLogEntry], None]) -> None:
        """
        注册日志处理器
        
        Args:
            handler: 处理函数
        """
        self._handlers.append(handler)

    def get_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        granted: Optional[bool] = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        获取日志记录
        
        Args:
            user_id: 过滤用户ID
            action: 过滤操作类型
            granted: 过滤授权状态
            limit: 最大条数
        
        Returns:
            日志记录列表
        """
        with self._lock:
            entries = self._entries.copy()

        # 过滤
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if granted is not None:
            entries = [e for e in entries if e.granted == granted]

        # 限制数量
        return entries[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        with self._lock:
            total = len(self._entries)
            granted = sum(1 for e in self._entries if e.granted)
            denied = total - granted

            action_counts: dict[str, int] = {}
            for e in self._entries:
                action_counts[e.action] = action_counts.get(e.action, 0) + 1

            return {
                "total_entries": total,
                "granted_count": granted,
                "denied_count": denied,
                "action_distribution": action_counts,
                "max_entries": self._max_entries,
            }

    def clear(self) -> int:
        """
        清空日志
        
        Returns:
            清空的条数
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count


# =============================================================================
# 安全管理器
# =============================================================================

class SecurityManager:
    """
    安全管理器
    
    统一管理权限检查和审计日志
    """

    def __init__(self):
        self._checker = PermissionChecker()
        self._audit_logger = AuditLogger()
        self._lock = threading.RLock()

    @property
    def checker(self) -> PermissionChecker:
        """获取权限检查器"""
        return self._checker

    @property
    def audit_logger(self) -> AuditLogger:
        """获取审计日志器"""
        return self._audit_logger

    def register_user(self, user: UserIdentity) -> bool:
        """注册用户"""
        return self._checker.register_user(user)

    def unregister_user(self, user_id: str) -> bool:
        """注销用户"""
        return self._checker.unregister_user(user_id)

    def get_user(self, user_id: str) -> Optional[UserIdentity]:
        """获取用户"""
        return self._checker.get_user(user_id)

    def add_policy(self, policy: SecurityPolicy) -> None:
        """添加安全策略"""
        self._checker.add_policy(policy)

    def remove_policy(self, policy_id: str) -> bool:
        """移除安全策略"""
        return self._checker.remove_policy(policy_id)

    def check_and_log(
        self,
        user_id: str,
        permission: Permission,
        resource: str = "*",
        action: str = "permission_check",
    ) -> PermissionCheckResult:
        """
        检查权限并记录日志
        
        Args:
            user_id: 用户ID
            permission: 权限类型
            resource: 资源标识
            action: 操作类型
        
        Returns:
            权限检查结果
        """
        result = self._checker.check_permission(user_id, permission, resource)
        self._audit_logger.log_permission_check(result, resource, action)
        return result

    def require_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: str = "*",
    ) -> None:
        """
        要求权限（无权限则抛出异常）
        
        Args:
            user_id: 用户ID
            permission: 权限类型
            resource: 资源标识
        
        Raises:
            PermissionDeniedError: 权限不足
        """
        result = self.check_and_log(user_id, permission, resource)
        if not result.granted:
            raise PermissionDeniedError(
                f"用户 {user_id} 无权限 {permission.value} 访问 {resource}: {result.reason}"
            )

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        granted: bool = True,
        details: Optional[dict] = None,
    ) -> AuditLogEntry:
        """记录操作日志"""
        return self._audit_logger.log(
            user_id=user_id,
            action=action,
            resource=resource,
            granted=granted,
            details=details,
        )

    def get_audit_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        granted: Optional[bool] = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """获取审计日志"""
        return self._audit_logger.get_entries(user_id, action, granted, limit)

    def get_security_stats(self) -> dict[str, Any]:
        """
        获取安全统计
        
        Returns:
            统计信息
        """
        return {
            "users_count": len(self._checker.get_all_users()),
            "policies_count": len(self._checker.get_all_policies()),
            "audit_stats": self._audit_logger.get_stats(),
        }

    def create_default_admin(self) -> UserIdentity:
        """
        创建默认管理员
        
        Returns:
            管理员用户身份
        """
        admin = UserIdentity(
            user_id="admin",
            name="Administrator",
            role=Role.ADMIN,
        )
        self.register_user(admin)
        return admin


# =============================================================================
# 异常类
# =============================================================================

class PermissionDeniedError(Exception):
    """权限不足异常"""
    pass


class SecurityError(Exception):
    """安全错误异常"""
    pass