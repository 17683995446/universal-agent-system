"""
阶段99: Agent 具身智能基础系统

遵循"图难于其易，为大于其细"的原则，实现环境感知、动作空间、状态表示、感知-动作循环。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import random


@dataclass
class Pose:
    """位姿（2D）"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 朝向角度（弧度）

    def distance_to(self, other: "Pose") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}


@dataclass
class Observation:
    """环境观测"""
    pose: Pose
    sensors: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose": self.pose.to_dict(),
            "sensors": self.sensors,
            "timestamp": self.timestamp,
        }


@dataclass
class Action:
    """动作"""
    action_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": self.action_type, "params": self.params}


class ActionSpace:
    """动作空间"""

    def __init__(self):
        self.actions = ["move_forward", "turn_left", "turn_right", "stop", "grab"]

    def sample(self) -> Action:
        """随机采样动作"""
        action_type = random.choice(self.actions)
        if action_type == "move_forward":
            return Action(action_type, {"distance": random.uniform(0.1, 1.0)})
        elif action_type in ("turn_left", "turn_right"):
            return Action(action_type, {"angle": random.uniform(0.1, math.pi / 2)})
        return Action(action_type)

    def get_actions(self) -> List[str]:
        return self.actions.copy()


class Environment:
    """简化环境"""

    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.goals: List[Tuple[float, float]] = []

    def add_obstacle(self, x: float, y: float, radius: float = 0.5):
        self.obstacles.append((x, y, radius))

    def add_goal(self, x: float, y: float):
        self.goals.append((x, y))

    def get_observation(self, pose: Pose) -> Observation:
        """获取当前位姿的观测"""
        sensors = {
            "front_distance": self._raycast_distance(pose, 0.0),
            "left_distance": self._raycast_distance(pose, math.pi / 4),
            "right_distance": self._raycast_distance(pose, -math.pi / 4),
        }
        # 检测是否到达目标
        sensors["goal_detected"] = any(
            math.sqrt((pose.x - gx) ** 2 + (pose.y - gy) ** 2) < 1.0
            for gx, gy in self.goals
        )
        return Observation(pose=pose, sensors=sensors)

    def _raycast_distance(self, pose: Pose, angle_offset: float) -> float:
        """射线投射距离"""
        direction = pose.theta + angle_offset
        max_range = 5.0
        step = 0.1
        for dist in [i * step for i in range(1, int(max_range / step) + 1)]:
            x = pose.x + dist * math.cos(direction)
            y = pose.y + dist * math.sin(direction)
            # 边界检测
            if x < 0 or x > self.width or y < 0 or y > self.height:
                return dist
            # 障碍物检测
            for ox, oy, oradius in self.obstacles:
                if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < oradius:
                    return dist
        return max_range

    def apply_action(self, pose: Pose, action: Action) -> Pose:
        """应用动作，返回新位姿"""
        new_pose = Pose(pose.x, pose.y, pose.theta)
        if action.action_type == "move_forward":
            dist = action.params.get("distance", 0.5)
            new_x = pose.x + dist * math.cos(pose.theta)
            new_y = pose.y + dist * math.sin(pose.theta)
            # 简单碰撞检测
            if not self._is_collision(new_x, new_y):
                new_pose.x = max(0.0, min(self.width, new_x))
                new_pose.y = max(0.0, min(self.height, new_y))
        elif action.action_type == "turn_left":
            new_pose.theta += action.params.get("angle", math.pi / 4)
        elif action.action_type == "turn_right":
            new_pose.theta -= action.params.get("angle", math.pi / 4)
        return new_pose

    def _is_collision(self, x: float, y: float) -> bool:
        for ox, oy, oradius in self.obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < oradius:
                return True
        return False

    def check_goal_reached(self, pose: Pose) -> bool:
        return any(
            math.sqrt((pose.x - gx) ** 2 + (pose.y - gy) ** 2) < 0.5
            for gx, gy in self.goals
        )


class PerceptionActionLoop:
    """感知-动作循环"""

    def __init__(self, environment: Environment, action_space: ActionSpace):
        self.env = environment
        self.action_space = action_space
        self.pose = Pose()
        self._history: List[Tuple[Observation, Action]] = []

    def step(self, policy_fn: Optional[Callable[[Observation], Action]] = None) -> Tuple[Observation, Action, Pose]:
        """执行一步感知-动作循环"""
        # 感知
        obs = self.env.get_observation(self.pose)
        # 决策
        if policy_fn:
            action = policy_fn(obs)
        else:
            action = self.action_space.sample()
        # 执行
        new_pose = self.env.apply_action(self.pose, action)
        self.pose = new_pose
        self._history.append((obs, action))
        return obs, action, new_pose

    def run(self, n_steps: int = 10,
            policy_fn: Optional[Callable[[Observation], Action]] = None) -> List[Tuple[Observation, Action]]:
        """运行多步"""
        for _ in range(n_steps):
            self.step(policy_fn)
        return self._history.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "steps": len(self._history),
            "current_pose": self.pose.to_dict(),
            "goal_reached": self.env.check_goal_reached(self.pose),
        }


class EmbodiedAgent:
    """具身智能体"""

    def __init__(self, environment: Environment):
        self.env = environment
        self.action_space = ActionSpace()
        self.loop = PerceptionActionLoop(environment, self.action_space)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """重置位姿"""
        self.loop.pose = Pose(x, y, theta)
        self.loop._history.clear()

    def navigate_to(self, goal_x: float, goal_y: float, max_steps: int = 100) -> bool:
        """简单导航到目标（贪心策略）"""
        def policy(obs: Observation) -> Action:
            # 简单策略：朝目标方向移动
            dx = goal_x - obs.pose.x
            dy = goal_y - obs.pose.y
            target_theta = math.atan2(dy, dx)
            angle_diff = target_theta - obs.pose.theta
            # 归一化到 [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            if abs(angle_diff) > 0.3:
                if angle_diff > 0:
                    return Action("turn_left", {"angle": 0.3})
                else:
                    return Action("turn_right", {"angle": 0.3})
            return Action("move_forward", {"distance": 0.5})

        for _ in range(max_steps):
            self.loop.step(policy)
            if self.env.check_goal_reached(self.loop.pose):
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return self.loop.get_stats()


def create_embodied_agent(width: float = 10.0, height: float = 10.0) -> EmbodiedAgent:
    env = Environment(width, height)
    return EmbodiedAgent(env)
