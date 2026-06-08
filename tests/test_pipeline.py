"""阶段48: Agent管道编排系统 - 单元测试"""
import pytest
from src.agent.pipeline import (
    StepStatus, PipelineMode, PipelineContext, PipelineStep, StepResult,
    PipelineResult, Pipeline, create_pipeline,
)

class TestPipeline:
    def test_sequential(self):
        p = Pipeline("test")
        p.add_handler("step1", lambda ctx: (ctx.set("a", 1), True)[-1])
        p.add_handler("step2", lambda ctx: (ctx.set("b", ctx.get("a", 0) + 1), True)[-1])
        r = p.execute()
        assert r.success
        assert len(r.steps) == 2

    def test_parallel(self):
        p = Pipeline("test", mode=PipelineMode.PARALLEL)
        p.add_handler("s1", lambda ctx: (ctx.set("a", 1), True)[-1])
        p.add_handler("s2", lambda ctx: (ctx.set("b", 2), True)[-1])
        r = p.execute()
        assert r.success

    def test_step_failure(self):
        p = Pipeline("test")
        p.add_handler("fail", lambda ctx: (_ for _ in ()).throw(ValueError("err")))
        r = p.execute()
        assert not r.success
        assert len(r.errors) > 0

    def test_continue_on_error(self):
        p = Pipeline("test")
        p.add_handler("fail", lambda ctx: (_ for _ in ()).throw(ValueError("err")),
                      continue_on_error=True)
        p.add_handler("ok", lambda ctx: True)
        r = p.execute()
        assert not r.success  # fail step failed
        assert len(r.steps) == 2  # but both ran

    def test_condition_skip(self):
        p = Pipeline("test")
        p.add_handler("s1", lambda ctx: (ctx.set("skip", True), True)[-1])
        p.add_step(PipelineStep(name="s2", handler=lambda ctx: True,
                                condition=lambda ctx: not ctx.get("skip", False)))
        r = p.execute()
        assert r.steps[1].status == StepStatus.SKIPPED

    def test_context_passing(self):
        p = Pipeline("test")
        p.add_handler("s1", lambda ctx: (ctx.set("x", 42), True)[-1])
        p.add_handler("s2", lambda ctx: (ctx.set("y", ctx.get("x") * 2), True)[-1])
        ctx = PipelineContext()
        r = p.execute(ctx)
        assert ctx.get("y") == 84

    def test_retry(self):
        attempts = [0]
        def flaky(ctx):
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("not yet")
            return True
        p = Pipeline("test")
        p.add_step(PipelineStep(name="flaky", handler=flaky, retry_count=3, retry_delay=0.01))
        r = p.execute()
        assert r.success
        assert r.steps[0].retries == 2

    def test_stop_on_failure(self):
        p = Pipeline("test")
        p.add_handler("fail", lambda ctx: (_ for _ in ()).throw(ValueError("err")))
        p.add_handler("never", lambda ctx: True)
        r = p.execute()
        assert r.steps[1].status == StepStatus.SKIPPED

    def test_create_pipeline(self):
        p = create_pipeline("test")
        assert p.name == "test"
