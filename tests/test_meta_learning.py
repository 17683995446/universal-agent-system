"""
阶段96测试: 元学习系统
"""
import pytest
from src.agent.meta_learning import (
    MetaTask, MAMLApproximation, TaskDistribution,
    MetaLearner, create_meta_learner,
)


class TestMetaTask:
    def test_to_dict(self):
        t = MetaTask(task_id=1, support_x=[[1.0]], support_y=[2.0],
                     query_x=[[3.0]], query_y=[4.0])
        d = t.to_dict()
        assert d["task_id"] == 1
        assert d["support_size"] == 1
        assert d["query_size"] == 1


class TestMAMLApproximation:
    def test_inner_update(self):
        maml = MAMLApproximation(inner_lr=0.01)
        params = {"w": [1.0], "b": [0.0]}
        task = MetaTask(0, support_x=[[1.0]], support_y=[2.0],
                        query_x=[[2.0]], query_y=[3.0])
        loss_fn = lambda x, y, p: (p["w"][0] * x[0] + p["b"][0] - y) ** 2
        adapted = maml.inner_update(params, task, loss_fn, steps=1)
        assert "w" in adapted
        assert "b" in adapted

    def test_meta_update(self):
        maml = MAMLApproximation(inner_lr=0.01, meta_lr=0.001)
        params = {"w": [1.0], "b": [0.0]}
        tasks = [
            MetaTask(0, support_x=[[1.0]], support_y=[2.0],
                     query_x=[[2.0]], query_y=[3.0]),
            MetaTask(1, support_x=[[1.0]], support_y=[3.0],
                     query_x=[[2.0]], query_y=[4.0]),
        ]
        loss_fn = lambda x, y, p: (p["w"][0] * x[0] + p["b"][0] - y) ** 2
        updated = maml.meta_update(params, tasks, loss_fn)
        assert "w" in updated
        assert "b" in updated


class TestTaskDistribution:
    def test_generate_task(self):
        td = TaskDistribution()
        task = td.generate_task(0)
        assert isinstance(task, MetaTask)
        assert task.task_id == 0
        assert len(task.support_x) == 5
        assert len(task.query_x) == 5

    def test_sample_tasks(self):
        td = TaskDistribution()
        tasks = td.sample_tasks(3)
        assert len(tasks) == 3
        assert all(isinstance(t, MetaTask) for t in tasks)


class TestMetaLearner:
    def test_initialize_params(self):
        ml = MetaLearner()
        ml.initialize_params({"w": 1, "b": 1})
        assert "w" in ml.meta_params
        assert "b" in ml.meta_params
        assert len(ml.meta_params["w"]) == 1

    def test_meta_train(self):
        ml = MetaLearner()
        ml.initialize_params({"w": 1, "b": 1})
        ml.meta_train(n_iterations=2, tasks_per_iteration=2)
        assert len(ml._history) == 2

    def test_meta_train_not_initialized(self):
        ml = MetaLearner()
        with pytest.raises(ValueError):
            ml.meta_train()

    def test_adapt(self):
        ml = MetaLearner()
        ml.initialize_params({"w": 1, "b": 1})
        task = MetaTask(0, support_x=[[1.0]], support_y=[2.0],
                        query_x=[[2.0]], query_y=[3.0])
        adapted = ml.adapt(task, steps=1)
        assert "w" in adapted

    def test_evaluate(self):
        ml = MetaLearner()
        ml.initialize_params({"w": 1, "b": 1})
        task = MetaTask(0, support_x=[[1.0]], support_y=[2.0],
                        query_x=[[1.0]], query_y=[2.0])
        loss = ml.evaluate(task, ml.meta_params)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_get_stats(self):
        ml = MetaLearner()
        ml.initialize_params({"w": 1})
        ml.meta_train(n_iterations=1, tasks_per_iteration=1)
        stats = ml.get_stats()
        assert stats["meta_iterations"] == 1
        assert "w" in stats["param_keys"]


class TestCreateMetaLearner:
    def test_factory(self):
        ml = create_meta_learner(inner_lr=0.02, meta_lr=0.002)
        assert isinstance(ml, MetaLearner)
        assert ml.maml.inner_lr == 0.02
