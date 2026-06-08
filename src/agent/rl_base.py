"""
阶段84: Agent强化学习基础系统

遵循"图难于其易，为大于其细"的原则，实现环境、智能体、策略、奖励、训练循环。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


@dataclass
class State:
    """状态"""
    features: List[float]
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"features": self.features, "info": self.info}


@dataclass
class Action:
    """动作"""
    value: Any
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "index": self.index}


@dataclass
class Transition:
    """状态转移"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "reward": self.reward,
            "next_state": self.next_state.to_dict(),
            "done": self.done,
        }


class Environment:
    """环境基类"""

    def __init__(self, state_dim: int, action_space: List[Any]):
        self.state_dim = state_dim
        self.action_space = action_space
        self._current_state: Optional[State] = None

    def reset(self) -> State:
        """重置环境"""
        self._current_state = State(features=[0.0] * self.state_dim)
        return self._current_state

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """执行动作，返回 (next_state, reward, done, info)"""
        raise NotImplementedError

    def get_state(self) -> State:
        if self._current_state is None:
            self.reset()
        return self._current_state


class SimpleGridEnv(Environment):
    """简单网格环境"""

    def __init__(self, size: int = 4):
        super().__init__(state_dim=2, action_space=[0, 1, 2, 3])  # 上右下左
        self.size = size
        self.goal = (size - 1, size - 1)
        self._pos = (0, 0)

    def reset(self) -> State:
        self._pos = (0, 0)
        self._current_state = State(features=[float(self._pos[0]), float(self._pos[1])])
        return self._current_state

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        x, y = self._pos
        if action.value == 0:  # 上
            y = max(0, y - 1)
        elif action.value == 1:  # 右
            x = min(self.size - 1, x + 1)
        elif action.value == 2:  # 下
            y = min(self.size - 1, y + 1)
        elif action.value == 3:  # 左
            x = max(0, x - 1)
        self._pos = (x, y)
        self._current_state = State(features=[float(x), float(y)])
        done = self._pos == self.goal
        reward = 1.0 if done else -0.01
        return self._current_state, reward, done, {}


class Policy:
    """策略基类"""

    def select_action(self, state: State, action_space: List[Any]) -> Action:
        raise NotImplementedError

    def update(self, transition: Transition) -> None:
        pass


class RandomPolicy(Policy):
    """随机策略"""

    def select_action(self, state: State, action_space: List[Any]) -> Action:
        idx = random.randint(0, len(action_space) - 1)
        return Action(value=action_space[idx], index=idx)


class EpsilonGreedyPolicy(Policy):
    """ε-贪心策略"""

    def __init__(self, q_table: Dict[Tuple[int, ...], List[float]], epsilon: float = 0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def select_action(self, state: State, action_space: List[Any]) -> Action:
        state_key = tuple(int(f) for f in state.features)
        if state_key not in self.q_table or random.random() < self.epsilon:
            idx = random.randint(0, len(action_space) - 1)
            return Action(value=action_space[idx], index=idx)
        q_values = self.q_table[state_key]
        idx = q_values.index(max(q_values))
        return Action(value=action_space[idx], index=idx)


class QLearningAgent:
    """Q-Learning智能体"""

    def __init__(self, action_space: List[Any], learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}
        self._policy = EpsilonGreedyPolicy(self.q_table, epsilon)
        self.episode_rewards: List[float] = []
        self.transitions: List[Transition] = []

    def select_action(self, state: State) -> Action:
        return self._policy.select_action(state, self.action_space)

    def learn(self, transition: Transition) -> None:
        """从转移中学习"""
        s_key = tuple(int(f) for f in transition.state.features)
        ns_key = tuple(int(f) for f in transition.next_state.features)
        a_idx = transition.action.index

        if s_key not in self.q_table:
            self.q_table[s_key] = [0.0] * len(self.action_space)
        if ns_key not in self.q_table:
            self.q_table[ns_key] = [0.0] * len(self.action_space)

        current_q = self.q_table[s_key][a_idx]
        max_next_q = max(self.q_table[ns_key]) if not transition.done else 0.0
        target = transition.reward + self.gamma * max_next_q
        self.q_table[s_key][a_idx] = current_q + self.lr * (target - current_q)
        self.transitions.append(transition)

    def train_episode(self, env: Environment, max_steps: int = 100) -> float:
        """训练一个回合"""
        state = env.reset()
        total_reward = 0.0
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            transition = Transition(state, action, reward, next_state, done)
            self.learn(transition)
            state = next_state
            if done:
                break
        self.episode_rewards.append(total_reward)
        return total_reward

    def get_stats(self) -> Dict[str, Any]:
        return {
            "q_table_size": len(self.q_table),
            "episodes": len(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "total_transitions": len(self.transitions),
        }


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer: List[Transition] = []
        self._idx = 0

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._idx] = transition
            self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self._buffer) < batch_size:
            return list(self._buffer)
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


class RLTrainer:
    """强化学习训练器"""

    def __init__(self, env: Environment, agent: QLearningAgent):
        self.env = env
        self.agent = agent
        self.replay_buffer = ReplayBuffer()

    def train(self, episodes: int = 100, max_steps: int = 100,
              batch_size: int = 32, use_replay: bool = False) -> Dict[str, Any]:
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                transition = Transition(state, action, reward, next_state, done)
                if use_replay:
                    self.replay_buffer.push(transition)
                    if len(self.replay_buffer) >= batch_size:
                        batch = self.replay_buffer.sample(batch_size)
                        for t in batch:
                            self.agent.learn(t)
                else:
                    self.agent.learn(transition)
                state = next_state
                if done:
                    break
            self.agent.episode_rewards.append(total_reward)
        return self.agent.get_stats()

    def evaluate(self, episodes: int = 10, max_steps: int = 100) -> float:
        """评估策略"""
        total_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            ep_reward = 0.0
            for _ in range(max_steps):
                # 使用贪心策略评估
                old_epsilon = self.agent.epsilon
                self.agent.epsilon = 0.0
                action = self.agent.select_action(state)
                self.agent.epsilon = old_epsilon
                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                state = next_state
                if done:
                    break
            total_rewards.append(ep_reward)
        return sum(total_rewards) / len(total_rewards) if total_rewards else 0.0


def create_q_learning_agent(action_space: List[Any], **kwargs) -> QLearningAgent:
    return QLearningAgent(action_space, **kwargs)


def create_simple_grid_env(size: int = 4) -> SimpleGridEnv:
    return SimpleGridEnv(size)
