"""
阶段99测试: 具身智能基础系统
"""
import pytest
import math
from src.agent.embodied_intelligence import (
    Pose, Observation, Action, ActionSpace,
    Environment, PerceptionActionLoop, EmbodiedAgent,
    create_embodied_agent,
)


class TestPose:
    def test_distance_to(self):
        p1 = Pose(x=0, y=0)
        p2 = Pose(x=3, y=4)
        assert p1.distance_to(p2) == 5.0

    def test_to_dict(self):
        p = Pose(x=1.0, y=2.0, theta=math.pi)
        d = p.to_dict()
        assert d["x"] == 1.0
        assert d["y"] == 2.0


class TestObservation:
    def test_to_dict(self):
        obs = Observation(pose=Pose(x=1, y=2), sensors={"d": 1.0})
        d = obs.to_dict()
        assert d["pose"]["x"] == 1
        assert "sensors" in d


class TestAction:
    def test_to_dict(self):
        a = Action("move", {"dist": 1.0})
        d = a.to_dict()
        assert d["action_type"] == "move"
        assert d["params"]["dist"] == 1.0


class TestActionSpace:
    def test_sample(self):
        space = ActionSpace()
        a = space.sample()
        assert a.action_type in space.actions

    def test_get_actions(self):
        space = ActionSpace()
        actions = space.get_actions()
        assert "move_forward" in actions
        assert len(actions) == 5


class TestEnvironment:
    def test_add_obstacle_and_goal(self):
        env = Environment()
        env.add_obstacle(5.0, 5.0)
        env.add_goal(8.0, 8.0)
        assert len(env.obstacles) == 1
        assert len(env.goals) == 1

    def test_get_observation(self):
        env = Environment()
        env.add_goal(1.0, 0.0)
        obs = env.get_observation(Pose(x=0, y=0, theta=0))
        assert "front_distance" in obs.sensors
        assert "goal_detected" in obs.sensors

    def test_apply_action_move(self):
        env = Environment()
        pose = Pose(x=0, y=0, theta=0)
        action = Action("move_forward", {"distance": 1.0})
        new_pose = env.apply_action(pose, action)
        assert new_pose.x == pytest.approx(1.0)
        assert new_pose.y == pytest.approx(0.0)

    def test_apply_action_turn(self):
        env = Environment()
        pose = Pose(x=0, y=0, theta=0)
        action = Action("turn_left", {"angle": math.pi / 2})
        new_pose = env.apply_action(pose, action)
        assert new_pose.theta == pytest.approx(math.pi / 2)

    def test_collision(self):
        env = Environment()
        env.add_obstacle(1.0, 0.0, radius=0.5)
        pose = Pose(x=0, y=0, theta=0)
        action = Action("move_forward", {"distance": 1.0})
        new_pose = env.apply_action(pose, action)
        # 应该被阻挡，位置不变
        assert new_pose.x == pytest.approx(0.0)

    def test_check_goal_reached(self):
        env = Environment()
        env.add_goal(1.0, 1.0)
        assert env.check_goal_reached(Pose(x=1.0, y=1.0)) is True
        assert env.check_goal_reached(Pose(x=0.0, y=0.0)) is False


class TestPerceptionActionLoop:
    def test_step(self):
        env = Environment()
        space = ActionSpace()
        loop = PerceptionActionLoop(env, space)
        obs, action, new_pose = loop.step()
        assert isinstance(obs, Observation)
        assert isinstance(action, Action)
        assert isinstance(new_pose, Pose)

    def test_run(self):
        env = Environment()
        space = ActionSpace()
        loop = PerceptionActionLoop(env, space)
        history = loop.run(n_steps=3)
        assert len(history) == 3

    def test_get_stats(self):
        env = Environment()
        space = ActionSpace()
        loop = PerceptionActionLoop(env, space)
        loop.run(n_steps=2)
        stats = loop.get_stats()
        assert stats["steps"] == 2


class TestEmbodiedAgent:
    def test_reset(self):
        agent = create_embodied_agent()
        agent.reset(x=5.0, y=5.0, theta=math.pi)
        assert agent.loop.pose.x == 5.0
        assert agent.loop.pose.theta == math.pi

    def test_navigate_to(self):
        agent = create_embodied_agent(width=20.0, height=20.0)
        agent.env.add_goal(5.0, 0.0)
        reached = agent.navigate_to(5.0, 0.0, max_steps=50)
        assert isinstance(reached, bool)

    def test_get_stats(self):
        agent = create_embodied_agent()
        agent.loop.run(n_steps=2)
        stats = agent.get_stats()
        assert stats["steps"] == 2


class TestCreateEmbodiedAgent:
    def test_factory(self):
        agent = create_embodied_agent(width=5.0, height=5.0)
        assert isinstance(agent, EmbodiedAgent)
        assert agent.env.width == 5.0
