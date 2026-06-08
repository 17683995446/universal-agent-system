"""
阶段95: Agent 持续学习系统

遵循"图难于其易，为大于其细"的原则，实现经验回放、弹性权重巩固(EWC近似)、
任务增量学习、灾难性遗忘缓解。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


@dataclass
class Task:
    """学习任务"""
    task_id: int
    name: str
    data: List[Any] = field(default_factory=list)
    labels: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "data_size": len(self.data),
        }


class ExperienceReplay:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add(self, sample: Dict[str, Any]):
        """添加样本"""
        self.buffer.append(sample)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样"""
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def get_all(self) -> List[Dict[str, Any]]:
        return self.buffer.copy()

    def __len__(self) -> int:
        return len(self.buffer)


class EWCApproximation:
    """弹性权重巩固（简化版）"""

    def __init__(self, importance_factor: float = 1000.0):
        self.importance_factor = importance_factor
        self.params_old: Dict[str, List[float]] = {}
        self.fisher_info: Dict[str, List[float]] = {}

    def compute_fisher(self, params: Dict[str, List[float]],
                       data: List[List[float]],
                       model_fn: Callable[[List[float], Dict[str, List[float]]], Any]) -> Dict[str, List[float]]:
        """计算Fisher信息矩阵（对角近似）"""
        fisher = {}
        for key, values in params.items():
            fisher[key] = [0.0] * len(values)

        for x in data:
            # 计算梯度平方的近似
            pred = model_fn(x, params)
            for key in params:
                for i in range(len(params[key])):
                    # 简化：使用参数值的变化敏感度作为Fisher信息近似
                    perturbed_params = {k: v.copy() for k, v in params.items()}
                    perturbed_params[key][i] += 0.01
                    new_pred = model_fn(x, perturbed_params)
                    grad_sq = (float(new_pred) - float(pred)) ** 2 * 10000
                    fisher[key][i] += grad_sq

        # 平均
        for key in fisher:
            fisher[key] = [v / len(data) if data else 0.0 for v in fisher[key]]

        return fisher

    def register_task(self, params: Dict[str, List[float]],
                      data: List[List[float]],
                      model_fn: Callable):
        """注册任务参数和Fisher信息"""
        self.params_old = {k: v.copy() for k, v in params.items()}
        self.fisher_info = self.compute_fisher(params, data, model_fn)

    def penalty(self, params: Dict[str, List[float]]) -> float:
        """计算EWC惩罚项"""
        if not self.params_old:
            return 0.0
        loss = 0.0
        for key in params:
            if key in self.params_old and key in self.fisher_info:
                for i in range(len(params[key])):
                    diff = params[key][i] - self.params_old[key][i]
                    loss += self.fisher_info[key][i] * diff ** 2
        return loss


class ContinualLearner:
    """持续学习器"""

    def __init__(self, capacity: int = 100, ewc_factor: float = 1000.0):
        self.replay = ExperienceReplay(capacity)
        self.ewc = EWCApproximation(ewc_factor)
        self.tasks: List[Task] = []
        self.current_task_id = 0
        self._history: List[Dict[str, Any]] = []

    def add_task(self, name: str, data: List[Any], labels: List[Any]):
        """添加新任务"""
        task = Task(task_id=self.current_task_id, name=name, data=data, labels=labels)
        self.tasks.append(task)
        self.current_task_id += 1
        self._history.append({"event": "add_task", "task_id": task.task_id})

    def learn(self, params: Dict[str, List[float]],
              model_fn: Callable,
              loss_fn: Callable,
              optimizer_fn: Callable,
              epochs: int = 1):
        """学习当前任务（简化版）"""
        if not self.tasks:
            return

        current_task = self.tasks[-1]
        # 混合当前任务数据和回放数据
        replay_samples = self.replay.sample(min(10, len(self.replay)))

        for epoch in range(epochs):
            for x, y in zip(current_task.data, current_task.labels):
                # 训练一步（简化）
                loss = loss_fn(x, y, params)
                # EWC惩罚
                ewc_loss = self.ewc.penalty(params)
                total_loss = loss + ewc_loss
                # 优化
                params = optimizer_fn(params, total_loss)

            # 添加到回放
            for x, y in zip(current_task.data, current_task.labels):
                self.replay.add({"x": x, "y": y, "task_id": current_task.task_id})

        # 注册EWC
        self.ewc.register_task(params, current_task.data, model_fn)
        self._history.append({"event": "learn", "task_id": current_task.task_id})

    def evaluate_on_task(self, task_id: int,
                         model_fn: Callable[[Any], Any]) -> float:
        """评估在指定任务上的性能"""
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if not task:
            return 0.0
        correct = 0
        for x, y in zip(task.data, task.labels):
            pred = model_fn(x)
            if pred == y:
                correct += 1
        return correct / len(task.data) if task.data else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tasks": len(self.tasks),
            "replay_size": len(self.replay),
            "history_events": len(self._history),
        }


class TaskIncrementalLearning:
    """任务增量学习管理"""

    def __init__(self):
        self.learners: Dict[str, ContinualLearner] = {}

    def create_learner(self, name: str, capacity: int = 100) -> ContinualLearner:
        """创建学习器"""
        learner = ContinualLearner(capacity)
        self.learners[name] = learner
        return learner

    def get_learner(self, name: str) -> Optional[ContinualLearner]:
        return self.learners.get(name)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "learners": len(self.learners),
            "total_tasks": sum(len(l.tasks) for l in self.learners.values()),
        }


def create_continual_learner(capacity: int = 100) -> ContinualLearner:
    return ContinualLearner(capacity)
