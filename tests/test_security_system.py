# =============================================================================
# 安全系统测试 - 阶段27测试
# =============================================================================

import pytest
from unittest.mock import Mock

from src.agent.security_system import (
    SecurityManager,
    PermissionChecker,
    AuditLogger,
    Permission,
    Role,
    UserIdentity,
    PermissionCheckResult,
    AuditLogEntry,
    SecurityPolicy,
    SecurityPolicyType,
    PermissionDeniedError,
    SecurityError,
    ROLE_PERMISSIONS,
)


class TestPermission:
    """权限枚举测试"""

    def test_enum_values(self):
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.EXECUTE.value == "execute"
        assert Permission.AGENT_CREATE.value == "agent_create"
        assert Permission.SYSTEM_ADMIN.value == "system_admin"


class TestRole:
    """角色枚举测试"""

    def test_enum_values(self):
        assert Role.ADMIN.value == "admin"
        assert Role.OPERATOR.value == "operator"
        assert Role.DEVELOPER.value == "developer"
        assert Role.VIEWER.value == "viewer"
        assert Role.GUEST.value == "guest"


class TestRolePermissions:
    """角色权限映射测试"""

    def test_admin_has_all_permissions(self):
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert len(admin_perms) == len(Permission)

    def test_viewer_has_read_only(self):
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.READ in viewer_perms
        assert Permission.WRITE not in viewer_perms
        assert Permission.DELETE not in viewer_perms

    def test_guest_has_minimal_permissions(self):
        guest_perms = ROLE_PERMISSIONS[Role.GUEST]
        assert Permission.READ in guest_perms
        assert Permission.AGENT_QUERY in guest_perms
        assert Permission.AGENT_CREATE not in guest_perms


class TestUserIdentity:
    """用户身份测试"""

    def test_creation(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.DEVELOPER,
        )
        assert user.user_id == "user_1"
        assert user.name == "Test User"
        assert user.role == Role.DEVELOPER

    def test_get_all_permissions(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.VIEWER,
        )
        perms = user.get_all_permissions()
        assert Permission.READ in perms
        assert Permission.AGENT_QUERY in perms

    def test_custom_permissions(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.VIEWER,
            custom_permissions={Permission.WRITE},
        )
        perms = user.get_all_permissions()
        assert Permission.READ in perms  # 角色权限
        assert Permission.WRITE in perms  # 自定义权限

    def test_has_permission(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.VIEWER,
        )
        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.DELETE) is False

    def test_add_permission(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.VIEWER,
        )
        user.add_permission(Permission.DELETE)
        assert user.has_permission(Permission.DELETE) is True

    def test_remove_permission(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.VIEWER,
            custom_permissions={Permission.DELETE},
        )
        result = user.remove_permission(Permission.DELETE)
        assert result is True
        assert user.has_permission(Permission.DELETE) is False

    def test_to_dict(self):
        user = UserIdentity(
            user_id="user_1",
            name="Test User",
            role=Role.DEVELOPER,
            custom_permissions={Permission.DELETE},
        )
        result = user.to_dict()
        assert result["user_id"] == "user_1"
        assert result["name"] == "Test User"
        assert result["role"] == "developer"
        assert "delete" in result["custom_permissions"]


class TestPermissionCheckResult:
    """权限检查结果测试"""

    def test_creation_granted(self):
        result = PermissionCheckResult(
            granted=True,
            permission=Permission.READ,
            user_id="user_1",
            reason="",
        )
        assert result.granted is True
        assert result.permission == Permission.READ

    def test_creation_denied(self):
        result = PermissionCheckResult(
            granted=False,
            permission=Permission.DELETE,
            user_id="user_1",
            reason="用户无此权限",
        )
        assert result.granted is False
        assert result.reason == "用户无此权限"

    def test_to_dict(self):
        result = PermissionCheckResult(
            granted=True,
            permission=Permission.READ,
            user_id="user_1",
        )
        data = result.to_dict()
        assert data["granted"] is True
        assert data["permission"] == "read"


class TestAuditLogEntry:
    """审计日志记录测试"""

    def test_creation(self):
        entry = AuditLogEntry(
            entry_id="audit_1",
            user_id="user_1",
            action="read",
            resource="agent_1",
            permission=Permission.READ,
            granted=True,
        )
        assert entry.entry_id == "audit_1"
        assert entry.user_id == "user_1"
        assert entry.action == "read"
        assert entry.granted is True

    def test_to_dict(self):
        entry = AuditLogEntry(
            entry_id="audit_1",
            user_id="user_1",
            action="read",
            resource="agent_1",
            permission=Permission.READ,
            granted=True,
            ip_address="127.0.0.1",
        )
        data = entry.to_dict()
        assert data["entry_id"] == "audit_1"
        assert data["ip_address"] == "127.0.0.1"


class TestSecurityPolicy:
    """安全策略测试"""

    def test_creation(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test Policy",
            policy_type=SecurityPolicyType.ALLOW,
            permissions={Permission.READ},
            resources=["agent_*"],
            users=["user_1"],
            priority=100,
        )
        assert policy.policy_id == "policy_1"
        assert policy.policy_type == SecurityPolicyType.ALLOW

    def test_matches_user(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            users=["user_1"],
        )
        user = UserIdentity(user_id="user_1", name="Test", role=Role.GUEST)
        assert policy.matches_user(user) is True

        user2 = UserIdentity(user_id="user_2", name="Test", role=Role.GUEST)
        assert policy.matches_user(user2) is False

    def test_matches_user_by_role(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            roles=[Role.ADMIN],
        )
        user = UserIdentity(user_id="user_1", name="Test", role=Role.ADMIN)
        assert policy.matches_user(user) is True

    def test_matches_user_wildcard(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            users=["*"],
        )
        user = UserIdentity(user_id="any_user", name="Test", role=Role.GUEST)
        assert policy.matches_user(user) is True

    def test_matches_resource(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            resources=["agent_1", "config_*"],
        )
        assert policy.matches_resource("agent_1") is True
        assert policy.matches_resource("config_test") is True
        assert policy.matches_resource("other") is False

    def test_matches_resource_wildcard(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            resources=["*"],
        )
        assert policy.matches_resource("anything") is True

    def test_matches_permission(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            permissions={Permission.READ, Permission.WRITE},
        )
        assert policy.matches_permission(Permission.READ) is True
        assert policy.matches_permission(Permission.DELETE) is False

    def test_to_dict(self):
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test Policy",
            policy_type=SecurityPolicyType.DENY,
            permissions={Permission.READ},
            resources=["*"],
            priority=50,
        )
        data = policy.to_dict()
        assert data["policy_id"] == "policy_1"
        assert data["policy_type"] == "deny"
        assert data["priority"] == 50


class TestPermissionChecker:
    """权限检查器测试"""

    def test_creation(self):
        checker = PermissionChecker()
        assert checker is not None

    def test_register_user(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)

        result = checker.register_user(user)
        assert result is True
        assert checker.get_user("user_1") == user

    def test_register_duplicate_user(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)

        checker.register_user(user)
        result = checker.register_user(user)
        assert result is False

    def test_unregister_user(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)

        checker.register_user(user)
        result = checker.unregister_user("user_1")
        assert result is True
        assert checker.get_user("user_1") is None

    def test_check_permission_granted(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        checker.register_user(user)

        result = checker.check_permission("user_1", Permission.READ)
        assert result.granted is True

    def test_check_permission_denied(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        checker.register_user(user)

        result = checker.check_permission("user_1", Permission.DELETE)
        assert result.granted is False
        assert "用户无此权限" in result.reason

    def test_check_permission_nonexistent_user(self):
        checker = PermissionChecker()

        result = checker.check_permission("nonexistent", Permission.READ)
        assert result.granted is False
        assert "用户不存在" in result.reason

    def test_add_policy(self):
        checker = PermissionChecker()
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
            policy_type=SecurityPolicyType.DENY,
            permissions={Permission.DELETE},
            users=["*"],
            priority=10,
        )

        checker.add_policy(policy)
        policies = checker.get_all_policies()
        assert len(policies) == 1

    def test_remove_policy(self):
        checker = PermissionChecker()
        policy = SecurityPolicy(
            policy_id="policy_1",
            name="Test",
        )

        checker.add_policy(policy)
        result = checker.remove_policy("policy_1")
        assert result is True
        assert len(checker.get_all_policies()) == 0

    def test_policy_deny_override(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.ADMIN)
        checker.register_user(user)

        # 添加拒绝覆盖策略
        policy = SecurityPolicy(
            policy_id="deny_delete",
            name="Deny Delete",
            policy_type=SecurityPolicyType.DENY_OVERRIDE,
            permissions={Permission.DELETE},
            users=["*"],
            resources=["*"],  # 匹配所有资源
            priority=10,
        )
        checker.add_policy(policy)

        # ADMIN有所有权限，但策略强制拒绝
        result = checker.check_permission("user_1", Permission.DELETE)
        assert result.granted is False

    def test_policy_allow_override(self):
        checker = PermissionChecker()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.GUEST)
        checker.register_user(user)

        # 添加允许覆盖策略
        policy = SecurityPolicy(
            policy_id="allow_write",
            name="Allow Write",
            policy_type=SecurityPolicyType.ALLOW_OVERRIDE,
            permissions={Permission.WRITE},
            users=["user_1"],
            resources=["*"],  # 匹配所有资源
            priority=10,
        )
        checker.add_policy(policy)

        # GUEST没有WRITE权限，但策略强制允许
        result = checker.check_permission("user_1", Permission.WRITE)
        assert result.granted is True


class TestAuditLogger:
    """审计日志器测试"""

    def test_creation(self):
        logger = AuditLogger()
        assert logger is not None

    def test_log(self):
        logger = AuditLogger()

        entry = logger.log(
            user_id="user_1",
            action="read",
            resource="agent_1",
            permission=Permission.READ,
            granted=True,
        )

        assert entry.user_id == "user_1"
        assert entry.action == "read"
        assert entry.granted is True

    def test_log_permission_check(self):
        logger = AuditLogger()
        result = PermissionCheckResult(
            granted=True,
            permission=Permission.READ,
            user_id="user_1",
        )

        entry = logger.log_permission_check(result, "agent_1")
        assert entry.permission == Permission.READ
        assert entry.granted is True

    def test_get_entries(self):
        logger = AuditLogger()

        logger.log(user_id="user_1", action="read", resource="r1", granted=True)
        logger.log(user_id="user_2", action="write", resource="r2", granted=False)
        logger.log(user_id="user_1", action="delete", resource="r3", granted=False)

        # 过滤用户
        entries = logger.get_entries(user_id="user_1")
        assert len(entries) == 2

        # 过滤操作
        entries = logger.get_entries(action="write")
        assert len(entries) == 1

        # 过滤授权状态
        entries = logger.get_entries(granted=False)
        assert len(entries) == 2

    def test_get_stats(self):
        logger = AuditLogger()

        logger.log(user_id="user_1", action="read", resource="r1", granted=True)
        logger.log(user_id="user_1", action="write", resource="r2", granted=False)

        stats = logger.get_stats()
        assert stats["total_entries"] == 2
        assert stats["granted_count"] == 1
        assert stats["denied_count"] == 1

    def test_clear(self):
        logger = AuditLogger()

        logger.log(user_id="user_1", action="read", resource="r1")
        count = logger.clear()
        assert count == 1
        assert len(logger.get_entries()) == 0

    def test_max_entries_limit(self):
        logger = AuditLogger(max_entries=5)

        for i in range(10):
            logger.log(user_id=f"user_{i}", action="read", resource=f"r{i}")

        entries = logger.get_entries(limit=100)
        assert len(entries) == 5  # 只保留最后5条


class TestSecurityManager:
    """安全管理器测试"""

    def test_creation(self):
        manager = SecurityManager()
        assert manager.checker is not None
        assert manager.audit_logger is not None

    def test_register_user(self):
        manager = SecurityManager()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)

        result = manager.register_user(user)
        assert result is True
        assert manager.get_user("user_1") == user

    def test_check_and_log(self):
        manager = SecurityManager()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        manager.register_user(user)

        result = manager.check_and_log("user_1", Permission.READ, "agent_1")
        assert result.granted is True

        # 验证日志记录
        entries = manager.get_audit_entries(user_id="user_1")
        assert len(entries) == 1

    def test_require_permission_success(self):
        manager = SecurityManager()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        manager.register_user(user)

        # 不应抛出异常
        manager.require_permission("user_1", Permission.READ)

    def test_require_permission_failure(self):
        manager = SecurityManager()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        manager.register_user(user)

        # 应抛出异常
        with pytest.raises(PermissionDeniedError):
            manager.require_permission("user_1", Permission.DELETE)

    def test_log_action(self):
        manager = SecurityManager()

        entry = manager.log_action(
            user_id="user_1",
            action="custom_action",
            resource="resource_1",
            granted=True,
        )
        assert entry.action == "custom_action"

    def test_get_security_stats(self):
        manager = SecurityManager()
        user = UserIdentity(user_id="user_1", name="Test", role=Role.VIEWER)
        manager.register_user(user)

        stats = manager.get_security_stats()
        assert stats["users_count"] == 1
        assert "audit_stats" in stats

    def test_create_default_admin(self):
        manager = SecurityManager()

        admin = manager.create_default_admin()
        assert admin.user_id == "admin"
        assert admin.role == Role.ADMIN

        # 验证已注册
        assert manager.get_user("admin") == admin


class TestExceptions:
    """异常类测试"""

    def test_permission_denied_error(self):
        error = PermissionDeniedError("无权限")
        assert str(error) == "无权限"

    def test_security_error(self):
        error = SecurityError("安全错误")
        assert str(error) == "安全错误"


class TestExportSecuritySystem:
    """导出测试"""

    def test_security_manager_exported(self):
        from src.agent import SecurityManager
        assert SecurityManager is not None

    def test_permission_checker_exported(self):
        from src.agent import PermissionChecker
        assert PermissionChecker is not None

    def test_audit_logger_exported(self):
        from src.agent import AuditLogger
        assert AuditLogger is not None

    def test_permission_exported(self):
        from src.agent import Permission
        assert Permission.READ is not None

    def test_role_exported(self):
        from src.agent import Role
        assert Role.ADMIN is not None

    def test_user_identity_exported(self):
        from src.agent import UserIdentity
        assert UserIdentity is not None

    def test_permission_check_result_exported(self):
        from src.agent import PermissionCheckResult
        assert PermissionCheckResult is not None

    def test_audit_log_entry_exported(self):
        from src.agent import AuditLogEntry
        assert AuditLogEntry is not None

    def test_security_policy_exported(self):
        from src.agent import SecurityPolicy
        assert SecurityPolicy is not None

    def test_security_policy_type_exported(self):
        from src.agent import SecurityPolicyType
        assert SecurityPolicyType.ALLOW is not None

    def test_permission_denied_error_exported(self):
        from src.agent import PermissionDeniedError
        assert PermissionDeniedError is not None

    def test_security_error_exported(self):
        from src.agent import SecurityError
        assert SecurityError is not None

    def test_role_permissions_exported(self):
        from src.agent import ROLE_PERMISSIONS
        assert ROLE_PERMISSIONS is not None
        assert Role.ADMIN in ROLE_PERMISSIONS


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])