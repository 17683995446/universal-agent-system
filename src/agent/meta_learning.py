"""
阶段96: Agent 元学习系统

遵循"图难于其易，为大于其细"的原则，实现MAML近似、任务分布采样、元参数更新、快速适应。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import copy


@dataclass
class MetaTask:
    """元学习任务"""
    task_id: int
    support_x: List[Any]
    support_y: List[Any]
    query_x: List[Any]
    query_y: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "support_size": len(self.support_x),
            "query_size": len(self.query_x),
        }


class MAMLApproximation:
    """MAML近似（模型无关元学习）"""

    def __init__(self, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

    def inner_update(self, params: Dict[str, List[float]],
                     task: MetaTask,
                     loss_fn: Callable[[Any, Any, Dict[str, List[float]]], float],
                     steps: int = 1) -> Dict[str, List[float]]:
        """内循环更新（任务特定适应）"""
        adapted = copy.deepcopy(params)
        for _ in range(steps):
            # 计算support集上的梯度并更新
            for x, y in zip(task.support_x, task.support_y):
                loss = loss_fn(x, y, adapted)
                # 简化梯度下降
                for key in adapted:
                    for i in range(len(adapted[key])):
                        # 数值梯度近似
                        perturbed = copy.deepcopy(adapted)
                        perturbed[key][i] += 0.001
                        new_loss = loss_fn(x, y, perturbed)
                        grad = (new_loss - loss) / 0.001
                        adapted[key][i] -= self.inner_lr * grad
        return adapted

    def meta_update(self, meta_params: Dict[str, List[float]],
                    tasks: List[MetaTask],
                    loss_fn: Callable) -> Dict[str, List[float]]:
        """外循环更新（元参数更新）"""
        # 对每个任务进行内循环适应，然后在query集上评估
        meta_gradients = {k: [0.0] * len(v) for k, v in meta_params.items()}

        for task in tasks:
            adapted = self.inner_update(meta_params, task, loss_fn)
            # 在query集上计算梯度
            for x, y in zip(task.query_x, task.query_y):
                loss = loss_fn(x, y, adapted)
                for key in meta_params:
                    for i in range(len(meta_params[key])):
                        perturbed = copy.deepcopy(adapted)
                        perturbed[key][i] += 0.001
                        new_loss = loss_fn(x, y, perturbed)
                        grad = (new_loss - loss) / 0.001
                        meta_gradients[key][i] += grad

        # 更新元参数
        n_tasks = len(tasks) if tasks else 1
        for key in meta_params:
            for i in range(len(meta_params[key])):
                meta_params[key][i] -= self.meta_lr * (meta_gradients[key][i] / n_tasks)

        return meta_params


class TaskDistribution:
    """任务分布"""

    def __init__(self, task_generator: Optional[Callable[[], MetaTask]] = None):
        self.task_generator = task_generator
        self.tasks: List[MetaTask] = []

    def generate_task(self, task_id: int) -> MetaTask:
        """生成任务"""
        if self.task_generator:
            return self.task_generator()
        # 默认生成简单回归任务
        slope = random.uniform(-2, 2)
        intercept = random.uniform(-1, 1)
        support_x = [[random.uniform(-1, 1)] for _ in range(5)]
        support_y = [slope * x[0] + intercept for x in support_x]
        query_x = [[random.uniform(-1, 1)] for _ in range(5)]
        query_y = [slope * x[0] + intercept for x in query_x]
        return MetaTask(task_id, support_x, support_y, query_x, query_y)

    def sample_tasks(self, n_tasks: int) -> List[MetaTask]:
        """采样任务"""
        return [self.generate_task(i) for i in range(n_tasks)]


class MetaLearner:
    """元学习器"""

    def __init__(self, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.maml = MAMLApproximation(inner_lr, meta_lr)
        self.task_dist = TaskDistribution()
        self.meta_params: Dict[str, List[float]] = {}
        self._history: List[Dict[str, Any]] = []

    def initialize_params(self, param_shapes: Dict[str, int]):
        """初始化元参数"""
        self.meta_params = {
            k: [random.uniform(-0.1, 0.1) for _ in range(v)]
            for k, v in param_shapes.items()
        }

    def meta_train(self, n_iterations: int = 10,
                   tasks_per_iteration: int = 4,
                   loss_fn: Optional[Callable] = None):
        """元训练"""
        if not self.meta_params:
            raise ValueError("Parameters not initialized")

        if loss_fn is None:
            loss_fn = self._default_loss_fn

        for iteration in range(n_iterations):
            tasks = self.task_dist.sample_tasks(tasks_per_iteration)
            self.meta_params = self.maml.meta_update(self.meta_params, tasks, loss_fn)
            self._history.append({
                "iteration": iteration,
                "tasks": tasks_per_iteration,
            })

    def adapt(self, task: MetaTask,
              loss_fn: Optional[Callable] = None,
              steps: int = 5) -> Dict[str, List[float]]:
        """快速适应新任务"""
        if loss_fn is None:
            loss_fn = self._default_loss_fn
        return self.maml.inner_update(self.meta_params, task, loss_fn, steps)

    def evaluate(self, task: MetaTask,
                 adapted_params: Dict[str, List[float]],
                 loss_fn: Optional[Callable] = None) -> float:
        """评估适应后的模型"""
        if loss_fn is None:
            loss_fn = self._default_loss_fn
        total_loss = 0.0
        for x, y in zip(task.query_x, task.query_y):
            total_loss += loss_fn(x, y, adapted_params)
        return total_loss / len(task.query_x) if task.query_x else 0.0

    @staticmethod
    def _default_loss_fn(x: Any, y: Any, params: Dict[str, List[float]]) -> float:
        """默认损失函数：线性回归"""
        pred = params.get("w", [0.0])[0] * x[0] + params.get("b", [0.0])[0]
        return (pred - y) ** 2

    def get_stats(self) -> Dict[str, Any]:
        return {
            "meta_iterations": len(self._history),
            "param_keys": list(self.meta_params.keys()),
        }


def create_meta_learner(inner_lr: float = 0.01, meta_lr: float = 0.001) -> MetaLearner:
    return MetaLearner(inner_lr, meta_lr)
