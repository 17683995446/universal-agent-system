"""
无上限上下文Agent系统 - 任务快照管理器

管理任务快照的创建、保存、加载、恢复。
支持自动保存、版本控制、旧快照清理。

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

from typing import Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import OrderedDict

from src.modules.task_manager.task_snapshot import TaskSnapshot, TaskState
from src.modules.task_manager.task_step import TaskStep
from src.core.exceptions import (
    SnapshotNotFound, SnapshotSaveError, SnapshotLoadError,
)
from src.core.constants import (
    SNAPSHOT_AUTO_SAVE_INTERVAL,
    SNAPSHOT_MAX_RETAINED,
)


# =============================================================================
# 持久化接口（抽象）
# =============================================================================

class SnapshotStorage(ABC):
    """快照持久化存储接口"""

    @abstractmethod
    def save(self, task_id: str, snapshot: TaskSnapshot) -> bool:
        """保存快照"""
        ...

    @abstractmethod
    def load(self, task_id: str) -> Optional[TaskSnapshot]:
        """加载快照"""
        ...

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """删除快照"""
        ...

    @abstractmethod
    def list_all(self) -> list[str]:
        """列出所有任务ID"""
        ...

    @abstractmethod
    def exists(self, task_id: str) -> bool:
        """检查快照是否存在"""
        ...


# =============================================================================
# 内存存储实现（用于开发和测试）
# =============================================================================

class InMemorySnapshotStorage(SnapshotStorage):
    """内存快照存储（非持久化，仅用于开发和测试）"""

    def __init__(self):
        self._store: dict[str, TaskSnapshot] = {}

    def save(self, task_id: str, snapshot: TaskSnapshot) -> bool:
        self._store[task_id] = snapshot.copy()
        return True

    def load(self, task_id: str) -> Optional[TaskSnapshot]:
        snapshot = self._store.get(task_id)
        return snapshot.copy() if snapshot else None

    def delete(self, task_id: str) -> bool:
        if task_id in self._store:
            del self._store[task_id]
            return True
        return False

    def list_all(self) -> list[str]:
        return list(self._store.keys())

    def exists(self, task_id: str) -> bool:
        return task_id in self._store


# =============================================================================
# 快照管理器
# =============================================================================

class SnapshotManager:
    """
    任务快照管理器

    功能：
    - 快照CRUD操作
    - 自动保存（按轮次间隔）
    - 版本控制（乐观锁）
    - 旧快照清理
    - 任务恢复
    """

    def __init__(
        self,
        storage: Optional[SnapshotStorage] = None,
        auto_save_interval: int = SNAPSHOT_AUTO_SAVE_INTERVAL,
        max_retained: int = SNAPSHOT_MAX_RETAINED,
    ):
        """
        Args:
            storage: 持久化存储（None则使用内存存储）
            auto_save_interval: 自动保存间隔（轮次）
            max_retained: 最大保留快照数
        """
        self._storage = storage or InMemorySnapshotStorage()
        self._auto_save_interval = auto_save_interval
        self._max_retained = max_retained
        self._active_snapshots: dict[str, TaskSnapshot] = {}
        self._operation_counts: dict[str, int] = {}

    # =========================================================================
    # 核心CRUD
    # =========================================================================

    def create(
        self,
        task_id: str,
        session_id: str,
        goal: str,
        **kwargs
    ) -> TaskSnapshot:
        """
        创建新任务快照

        Args:
            task_id: 任务ID
            session_id: 会话ID
            goal: 任务目标
            **kwargs: 其他TaskSnapshot参数

        Returns:
            创建的快照
        """
        snapshot = TaskSnapshot(
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            **kwargs
        )
        self._active_snapshots[task_id] = snapshot
        self._operation_counts[task_id] = 0
        self._save_to_storage(task_id, snapshot)
        return snapshot

    def get(self, task_id: str, from_storage: bool = False) -> TaskSnapshot:
        """
        获取任务快照

        Args:
            task_id: 任务ID
            from_storage: 是否强制从存储加载

        Returns:
            快照

        Raises:
            SnapshotNotFound: 快照不存在
        """
        if not from_storage and task_id in self._active_snapshots:
            return self._active_snapshots[task_id]

        snapshot = self._storage.load(task_id)
        if snapshot is None:
            raise SnapshotNotFound(task_id)

        self._active_snapshots[task_id] = snapshot
        return snapshot

    def get_optional(self, task_id: str) -> Optional[TaskSnapshot]:
        """获取任务快照（不存在返回None）"""
        try:
            return self.get(task_id)
        except SnapshotNotFound:
            return None

    def update(self, task_id: str, **changes) -> TaskSnapshot:
        """
        更新任务快照

        Args:
            task_id: 任务ID
            **changes: 要更新的字段

        Returns:
            更新后的快照

        Raises:
            SnapshotNotFound: 快照不存在
        """
        snapshot = self.get(task_id)

        for key, value in changes.items():
            if hasattr(snapshot, key):
                setattr(snapshot, key, value)

        snapshot.increment_version()
        self._operation_counts[task_id] = self._operation_counts.get(task_id, 0) + 1

        # 检查自动保存
        if self._should_auto_save(task_id):
            self._save_to_storage(task_id, snapshot)

        return snapshot

    def save(self, task_id: str) -> TaskSnapshot:
        """
        手动保存快照到存储

        Args:
            task_id: 任务ID

        Returns:
            保存的快照
        """
        snapshot = self.get(task_id)
        self._save_to_storage(task_id, snapshot)
        return snapshot

    def delete(self, task_id: str) -> bool:
        """
        删除任务快照

        Args:
            task_id: 任务ID

        Returns:
            是否删除成功
        """
        if task_id in self._active_snapshots:
            del self._active_snapshots[task_id]
        if task_id in self._operation_counts:
            del self._operation_counts[task_id]
        return self._storage.delete(task_id)

    def exists(self, task_id: str) -> bool:
        """检查快照是否存在"""
        return task_id in self._active_snapshots or self._storage.exists(task_id)

    # =========================================================================
    # 自动保存
    # =========================================================================

    def _should_auto_save(self, task_id: str) -> bool:
        """检查是否应该自动保存"""
        count = self._operation_counts.get(task_id, 0)
        return count > 0 and count % self._auto_save_interval == 0

    def _save_to_storage(self, task_id: str, snapshot: TaskSnapshot) -> None:
        """保存到存储"""
        try:
            self._storage.save(task_id, snapshot)
        except Exception as e:
            raise SnapshotSaveError(task_id, str(e))

    # =========================================================================
    # 任务恢复
    # =========================================================================

    def restore(self, task_id: str) -> TaskSnapshot:
        """
        从存储恢复任务快照

        Args:
            task_id: 任务ID

        Returns:
            恢复的快照

        Raises:
            SnapshotNotFound: 快照不存在
            SnapshotLoadError: 加载失败
        """
        try:
            snapshot = self._storage.load(task_id)
            if snapshot is None:
                raise SnapshotNotFound(task_id)

            # 恢复时递增版本
            snapshot.increment_version()
            self._active_snapshots[task_id] = snapshot
            self._operation_counts[task_id] = 0

            return snapshot
        except SnapshotNotFound:
            raise
        except Exception as e:
            raise SnapshotLoadError(task_id, str(e))

    def suspend(self, task_id: str) -> TaskSnapshot:
        """
        暂停任务（保存当前状态）

        Args:
            task_id: 任务ID

        Returns:
            暂停的快照
        """
        snapshot = self.get(task_id)
        snapshot.suspend()
        return self.save(task_id)

    def resume(self, task_id: str) -> TaskSnapshot:
        """
        恢复暂停的任务

        Args:
            task_id: 任务ID

        Returns:
            恢复的快照
        """
        snapshot = self.get(task_id)
        snapshot.resume()
        return self.save(task_id)

    # =========================================================================
    # 旧快照清理
    # =========================================================================

    def cleanup_old_snapshots(self, max_age_hours: Optional[int] = None) -> int:
        """
        清理旧快照

        Args:
            max_age_hours: 最大保留时长（小时），None则使用max_retained

        Returns:
            清理的快照数
        """
        if max_age_hours is not None:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            to_delete = []
            for task_id in self._storage.list_all():
                snapshot = self._storage.load(task_id)
                if snapshot and snapshot.updated_at < cutoff:
                    to_delete.append(task_id)
        else:
            # 按max_retained清理
            all_snapshots = []
            for task_id in self._storage.list_all():
                snapshot = self._storage.load(task_id)
                if snapshot:
                    all_snapshots.append((task_id, snapshot.updated_at))

            # 按更新时间排序，保留最新的
            all_snapshots.sort(key=lambda x: x[1], reverse=True)
            to_delete = [tid for tid, _ in all_snapshots[self._max_retained:]]

        count = 0
        for task_id in to_delete:
            if self.delete(task_id):
                count += 1

        return count

    # =========================================================================
    # 查询
    # =========================================================================

    def list_active(self) -> list[str]:
        """列出所有活跃任务ID"""
        return list(self._active_snapshots.keys())

    def list_all(self) -> list[str]:
        """列出所有存储中的任务ID"""
        return self._storage.list_all()

    def get_stats(self) -> dict[str, Any]:
        """获取管理器统计信息"""
        return {
            "active_count": len(self._active_snapshots),
            "stored_count": len(self._storage.list_all()),
            "auto_save_interval": self._auto_save_interval,
            "max_retained": self._max_retained,
        }

    # =========================================================================
    # 批量操作
    # =========================================================================

    def save_all(self) -> int:
        """
        保存所有活跃快照

        Returns:
            保存的快照数
        """
        count = 0
        for task_id in list(self._active_snapshots.keys()):
            try:
                self.save(task_id)
                count += 1
            except SnapshotSaveError:
                pass
        return count

    def clear_active(self) -> int:
        """
        清空活跃快照缓存（不删除存储）

        Returns:
            清空的快照数
        """
        count = len(self._active_snapshots)
        self._active_snapshots.clear()
        self._operation_counts.clear()
        return count
