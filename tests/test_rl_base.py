"""阶段84: Agent强化学习基础系统 - 单元测试"""
import pytest
from src.agent.rl_base import (
    State, Action, Transition,
    Environment, SimpleGridEnv, RandomPolicy, EpsilonGreedyPolicy,
    QLearningAgent, ReplayBuffer, RLTrainer,
    create_q_learning_agent, create_simple_grid_env,
)


class TestSimpleGridEnv:
    def test_reset(self):
        env = SimpleGridEnv(size=4)
        state = env.reset()
        assert state.features == [0.0, 0.0]

    def test_step_right(self):
        env = SimpleGridEnv(size=4)
        env.reset()
        next_state, reward, done, _ = env.step(Action(value=1, index=1))  # 右
        assert next_state.features == [1.0, 0.0]
        assert not done
        assert reward == -0.01

    def test_step_down(self):
        env = SimpleGridEnv(size=4)
        env.reset()
        next_state, reward, done, _ = env.step(Action(value=2, index=2))  # 下
        assert next_state.features == [0.0, 1.0]

    def test_reach_goal(self):
        env = SimpleGridEnv(size=2)
        env.reset()
        # (0,0) -> (1,0) -> (1,1) goal
        env.step(Action(value=1, index=1))
        next_state, reward, done, _ = env.step(Action(value=2, index=2))
        assert done
        assert reward == 1.0

    def test_boundary(self):
        env = SimpleGridEnv(size=2)
        env.reset()
        env.step(Action(value=0, index=0))  # 上，已在边界
        assert env._pos == (0, 0)


class TestRandomPolicy:
    def test_select_action(self):
        policy = RandomPolicy()
        action = policy.select_action(State(features=[0.0]), [0, 1, 2])
        assert action.value in [0, 1, 2]


class TestEpsilonGreedyPolicy:
    def test_explore(self):
        policy = EpsilonGreedyPolicy({}, epsilon=1.0)  # 总是探索
        action = policy.select_action(State(features=[0.0]), [0, 1])
        assert action.value in [0, 1]

    def test_exploit(self):
        q_table = {(0,): [1.0, 5.0]}
        policy = EpsilonGreedyPolicy(q_table, epsilon=0.0)  # 总是利用
        action = policy.select_action(State(features=[0.0]), [0, 1])
        assert action.value == 1  # Q值最高的动作


class TestQLearningAgent:
    def test_select_action(self):
        agent = QLearningAgent([0, 1, 2, 3])
        action = agent.select_action(State(features=[0.0, 0.0]))
        assert action.value in [0, 1, 2, 3]

    def test_learn(self):
        agent = QLearningAgent([0, 1])
        t = Transition(
            state=State(features=[0.0]),
            action=Action(value=0, index=0),
            reward=1.0,
            next_state=State(features=[1.0]),
            done=False,
        )
        agent.learn(t)
        assert len(agent.q_table) == 2
        assert len(agent.transitions) == 1

    def test_train_episode(self):
        env = SimpleGridEnv(size=2)
        agent = QLearningAgent([0, 1, 2, 3])
        reward = agent.train_episode(env, max_steps=10)
        assert len(agent.episode_rewards) == 1
        assert len(agent.transitions) > 0

    def test_get_stats(self):
        env = SimpleGridEnv(size=2)
        agent = QLearningAgent([0, 1, 2, 3])
        agent.train_episode(env)
        stats = agent.get_stats()
        assert stats["episodes"] == 1
        assert stats["q_table_size"] > 0


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(3):
            t = Transition(
                state=State(features=[float(i)]),
                action=Action(value=i, index=0),
                reward=1.0,
                next_state=State(features=[float(i + 1)]),
                done=False,
            )
            buf.push(t)
        assert len(buf) == 3
        sample = buf.sample(2)
        assert len(sample) == 2

    def test_capacity(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            t = Transition(
                state=State(features=[float(i)]),
                action=Action(value=i, index=0),
                reward=1.0,
                next_state=State(features=[float(i + 1)]),
                done=False,
            )
            buf.push(t)
        assert len(buf) == 3


class TestRLTrainer:
    def test_train(self):
        env = SimpleGridEnv(size=2)
        agent = QLearningAgent([0, 1, 2, 3])
        trainer = RLTrainer(env, agent)
        stats = trainer.train(episodes=5, max_steps=10)
        assert stats["episodes"] == 5

    def test_train_with_replay(self):
        env = SimpleGridEnv(size=2)
        agent = QLearningAgent([0, 1, 2, 3])
        trainer = RLTrainer(env, agent)
        stats = trainer.train(episodes=5, max_steps=10, use_replay=True, batch_size=2)
        assert stats["episodes"] == 5

    def test_evaluate(self):
        env = SimpleGridEnv(size=2)
        agent = QLearningAgent([0, 1, 2, 3])
        trainer = RLTrainer(env, agent)
        trainer.train(episodes=20, max_steps=10)
        avg_reward = trainer.evaluate(episodes=5, max_steps=10)
        assert isinstance(avg_reward, float)


class TestFactories:
    def test_create_q_learning_agent(self):
        agent = create_q_learning_agent([0, 1])
        assert isinstance(agent, QLearningAgent)

    def test_create_simple_grid_env(self):
        env = create_simple_grid_env(5)
        assert isinstance(env, SimpleGridEnv)
        assert env.size == 5
