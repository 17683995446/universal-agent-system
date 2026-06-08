"""
阶段95测试: 持续学习系统
"""
import pytest
from src.agent.continual_learning import (
    Task, ExperienceReplay, EWCApproximation,
    ContinualLearner, TaskIncrementalLearning, create_continual_learner,
)


class TestTask:
    def test_to_dict(self):
        t = Task(task_id=1, name="task_a", data=[1, 2], labels=[0, 1])
        d = t.to_dict()
        assert d["task_id"] == 1
        assert d["name"] == "task_a"
        assert d["data_size"] == 2


class TestExperienceReplay:
    def test_add_and_sample(self):
        er = ExperienceReplay(capacity=3)
        er.add({"x": 1, "y": 0})
        er.add({"x": 2, "y": 1})
        sampled = er.sample(2)
        assert len(sampled) == 2

    def test_capacity(self):
        er = ExperienceReplay(capacity=2)
        er.add({"x": 1})
        er.add({"x": 2})
        er.add({"x": 3})
        assert len(er) == 2
        assert er.get_all()[0] == {"x": 2}

    def test_len(self):
        er = ExperienceReplay()
        assert len(er) == 0
        er.add({"x": 1})
        assert len(er) == 1


class TestEWCApproximation:
    def test_penalty_no_old(self):
        ewc = EWCApproximation()
        assert ewc.penalty({"w": [1.0, 2.0]}) == 0.0

    def test_register_and_penalty(self):
        ewc = EWCApproximation(importance_factor=1.0)
        params = {"w": [1.0, 2.0]}
        model = lambda x, p: sum(p["w"])
        data = [[1.0], [2.0]]
        ewc.register_task(params, data, model)
        # 参数不变时惩罚为0
        assert ewc.penalty(params) == 0.0
        # 参数变化时惩罚大于0
        new_params = {"w": [2.0, 2.0]}
        assert ewc.penalty(new_params) > 0.0


class TestContinualLearner:
    def test_add_task(self):
        learner = ContinualLearner()
        learner.add_task("task1", [1, 2], [0, 1])
        assert len(learner.tasks) == 1
        assert learner.tasks[0].name == "task1"

    def test_learn(self):
        learner = ContinualLearner()
        learner.add_task("task1", [[1.0], [2.0]], [0, 1])
        params = {"w": [0.5]}
        model = lambda x, p: 1 if x[0] * p["w"][0] > 1.0 else 0
        loss_fn = lambda x, y, p: 0.1
        optimizer_fn = lambda p, l: p
        learner.learn(params, model, loss_fn, optimizer_fn, epochs=1)
        assert len(learner.replay) > 0

    def test_evaluate_on_task(self):
        learner = ContinualLearner()
        learner.add_task("task1", [[1.0], [2.0]], [0, 1])
        model = lambda x: x[0]
        acc = learner.evaluate_on_task(0, model)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_on_missing_task(self):
        learner = ContinualLearner()
        assert learner.evaluate_on_task(99, lambda x: x) == 0.0

    def test_get_stats(self):
        learner = ContinualLearner()
        learner.add_task("t", [1], [0])
        stats = learner.get_stats()
        assert stats["tasks"] == 1
        assert stats["replay_size"] == 0


class TestTaskIncrementalLearning:
    def test_create_learner(self):
        til = TaskIncrementalLearning()
        learner = til.create_learner("learner1")
        assert isinstance(learner, ContinualLearner)
        assert "learner1" in til.learners

    def test_get_learner(self):
        til = TaskIncrementalLearning()
        til.create_learner("l1")
        assert til.get_learner("l1") is not None
        assert til.get_learner("l2") is None

    def test_get_stats(self):
        til = TaskIncrementalLearning()
        til.create_learner("l1")
        til.learners["l1"].add_task("t", [1], [0])
        stats = til.get_stats()
        assert stats["learners"] == 1
        assert stats["total_tasks"] == 1


class TestCreateContinualLearner:
    def test_factory(self):
        learner = create_continual_learner(capacity=50)
        assert isinstance(learner, ContinualLearner)
        assert learner.replay.capacity == 50
