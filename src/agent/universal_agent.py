"""
阶段100: Agent 通用Agent框架系统

遵循"图难于其易，为大于其细"的原则，实现统一的Agent接口、能力注册、任务路由、生命周期管理。
整合前面99个阶段的所有能力。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time


@dataclass
class Capability:
    """能力描述"""
    name: str
    description: str
    handler: Optional[Callable] = None
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
        }


@dataclass
class AgentTask:
    """Agent任务"""
    task_id: str
    task_type: str
    payload: Any
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    created_at: float = 0.0
    completed_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class CapabilityRegistry:
    """能力注册表"""

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability):
        """注册能力"""
        self._capabilities[capability.name] = capability

    def unregister(self, name: str):
        """注销能力"""
        self._capabilities.pop(name, None)

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        return list(self._capabilities.keys())

    def find_handler(self, task_type: str) -> Optional[Callable]:
        """查找任务处理器"""
        cap = self._capabilities.get(task_type)
        return cap.handler if cap else None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered": len(self._capabilities),
            "capabilities": self.list_capabilities(),
        }


class TaskRouter:
    """任务路由器"""

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self._history: List[AgentTask] = []

    def route(self, task: AgentTask) -> AgentTask:
        """路由任务到对应处理器"""
        handler = self.registry.find_handler(task.task_type)
        if handler is None:
            task.status = "failed"
            task.result = f"No handler for task type: {task.task_type}"
            self._history.append(task)
            return task

        task.status = "running"
        try:
            task.result = handler(task.payload)
            task.status = "completed"
            task.completed_at = time.time()
        except Exception as e:
            task.status = "failed"
            task.result = str(e)

        self._history.append(task)
        return task

    def get_history(self, status: Optional[str] = None) -> List[AgentTask]:
        if status:
            return [t for t in self._history if t.status == status]
        return self._history.copy()

    def get_stats(self) -> Dict[str, Any]:
        completed = len([t for t in self._history if t.status == "completed"])
        failed = len([t for t in self._history if t.status == "failed"])
        return {
            "total": len(self._history),
            "completed": completed,
            "failed": failed,
        }


class LifecycleManager:
    """生命周期管理器"""

    def __init__(self):
        self.state = "initialized"
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self):
        """启动"""
        self.state = "running"
        self._start_time = time.time()

    def pause(self):
        """暂停"""
        if self.state == "running":
            self.state = "paused"

    def resume(self):
        """恢复"""
        if self.state == "paused":
            self.state = "running"

    def stop(self):
        """停止"""
        self.state = "stopped"
        self._stop_time = time.time()

    def uptime(self) -> float:
        """运行时间"""
        if self._start_time is None:
            return 0.0
        end = self._stop_time or time.time()
        return end - self._start_time

    def get_stats(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "uptime": self.uptime(),
        }


class UniversalAgent:
    """通用Agent框架"""

    def __init__(self, agent_id: str = "universal_agent"):
        self.agent_id = agent_id
        self.registry = CapabilityRegistry()
        self.router = TaskRouter(self.registry)
        self.lifecycle = LifecycleManager()
        self._context: Dict[str, Any] = {}
        self._plugins: Dict[str, Any] = {}

    def register_capability(self, name: str, description: str,
                            handler: Callable, priority: int = 0):
        """注册能力"""
        cap = Capability(name, description, handler, priority)
        self.registry.register(cap)

    def execute(self, task_type: str, payload: Any,
                task_id: Optional[str] = None) -> AgentTask:
        """执行任务"""
        if self.lifecycle.state not in ("running", "initialized"):
            task = AgentTask(
                task_id=task_id or f"task_{time.time()}",
                task_type=task_type,
                payload=payload,
                status="failed",
                result="Agent not running",
            )
            return task

        task = AgentTask(
            task_id=task_id or f"task_{time.time()}",
            task_type=task_type,
            payload=payload,
        )
        return self.router.route(task)

    def set_context(self, key: str, value: Any):
        """设置上下文"""
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        """获取上下文"""
        return self._context.get(key)

    def load_plugin(self, name: str, plugin: Any):
        """加载插件"""
        self._plugins[name] = plugin

    def get_plugin(self, name: str) -> Optional[Any]:
        return self._plugins.get(name)

    def start(self):
        """启动Agent"""
        self.lifecycle.start()

    def stop(self):
        """停止Agent"""
        self.lifecycle.stop()

    def get_capabilities(self) -> List[str]:
        return self.registry.list_capabilities()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "lifecycle": self.lifecycle.get_stats(),
            "capabilities": self.registry.get_stats(),
            "tasks": self.router.get_stats(),
            "context_keys": list(self._context.keys()),
            "plugins": list(self._plugins.keys()),
        }


def create_universal_agent(agent_id: str = "agent") -> UniversalAgent:
    return UniversalAgent(agent_id)
