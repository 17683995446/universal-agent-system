"""
阶段45: Agent命令总线系统

遵循"图难于其易，为大于其细"的原则，实现命令注册、分发、拦截、超时管理。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class CommandStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class Command:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: CommandStatus = CommandStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "status": self.status.value,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error, "duration": round(self.duration, 4),
        }

@dataclass
class CommandResult:
    success: bool
    result: Any = None
    error: str = ""

@dataclass
class MiddlewareResult:
    """中间件结果: None=继续, CommandResult=短路返回"""
    result: Optional[CommandResult] = None

class CommandBus:
    """命令总线"""
    def __init__(self):
        self._handlers: Dict[str, Callable[[Command], CommandResult]] = {}
        self._middlewares: List[Callable[[Command], MiddlewareResult]] = []
        self._history: List[Command] = []
        self._lock = threading.RLock()
        self._default_timeout = 30.0

    def register(self, command_name: str, handler: Callable[[Command], CommandResult]) -> None:
        with self._lock:
            self._handlers[command_name] = handler

    def unregister(self, command_name: str) -> bool:
        with self._lock:
            return self._handlers.pop(command_name, None) is not None

    def add_middleware(self, middleware: Callable[[Command], MiddlewareResult]) -> None:
        self._middlewares.append(middleware)

    def dispatch(self, command: Command) -> CommandResult:
        with self._lock:
            command.status = CommandStatus.RUNNING
            command.started_at = time.time()
            self._history.append(command)

        # 执行中间件
        for mw in self._middlewares:
            try:
                mr = mw(command)
                if mr.result is not None:
                    command.status = CommandStatus.COMPLETED if mr.result.success else CommandStatus.FAILED
                    command.result = mr.result.result
                    command.error = mr.result.error
                    command.completed_at = time.time()
                    return mr.result
            except Exception as e:
                command.status = CommandStatus.FAILED
                command.error = str(e)
                command.completed_at = time.time()
                return CommandResult(success=False, error=f"中间件错误: {e}")

        # 查找处理器
        handler = self._handlers.get(command.name)
        if not handler:
            command.status = CommandStatus.FAILED
            command.error = f"未知命令: {command.name}"
            command.completed_at = time.time()
            return CommandResult(success=False, error=command.error)

        # 执行处理器
        try:
            result = handler(command)
            command.status = CommandStatus.COMPLETED if result.success else CommandStatus.FAILED
            command.result = result.result
            command.error = result.error
        except Exception as e:
            command.status = CommandStatus.FAILED
            command.error = str(e)
            result = CommandResult(success=False, error=str(e))

        command.completed_at = time.time()
        return result

    def dispatch_simple(self, name: str, payload: Optional[Dict] = None, timeout: float = 30.0) -> CommandResult:
        cmd = Command(name=name, payload=payload or {}, timeout=timeout)
        return self.dispatch(cmd)

    def get_history(self, command_name: Optional[str] = None, limit: int = 50) -> List[Command]:
        with self._lock:
            cmds = list(self._history)
            if command_name:
                cmds = [c for c in cmds if c.name == command_name]
            return cmds[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._history)
            by_status = {}
            for c in self._history:
                by_status[c.status.value] = by_status.get(c.status.value, 0) + 1
            return {
                "total_commands": total,
                "registered_handlers": list(self._handlers.keys()),
                "by_status": by_status,
            }

def create_command_bus() -> CommandBus:
    return CommandBus()
