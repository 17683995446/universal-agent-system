"""
阶段100测试: 通用Agent框架系统
"""
import pytest
import time
from src.agent.universal_agent import (
    Capability, AgentTask, CapabilityRegistry, TaskRouter,
    LifecycleManager, UniversalAgent, create_universal_agent,
)


class TestCapability:
    def test_to_dict(self):
        c = Capability(name="compute", description="计算能力", priority=1)
        d = c.to_dict()
        assert d["name"] == "compute"
        assert d["priority"] == 1


class TestAgentTask:
    def test_to_dict(self):
        t = AgentTask(task_id="t1", task_type="test", payload="hello")
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["task_type"] == "test"
        assert d["status"] == "pending"
        assert d["created_at"] > 0


class TestCapabilityRegistry:
    def test_register_and_get(self):
        reg = CapabilityRegistry()
        cap = Capability("math", "数学计算", handler=lambda x: x * 2)
        reg.register(cap)
        assert reg.get("math") == cap
        assert reg.list_capabilities() == ["math"]

    def test_unregister(self):
        reg = CapabilityRegistry()
        reg.register(Capability("temp", ""))
        reg.unregister("temp")
        assert reg.get("temp") is None

    def test_find_handler(self):
        reg = CapabilityRegistry()
        reg.register(Capability("double", "", handler=lambda x: x * 2))
        handler = reg.find_handler("double")
        assert handler(5) == 10

    def test_find_handler_missing(self):
        reg = CapabilityRegistry()
        assert reg.find_handler("missing") is None

    def test_get_stats(self):
        reg = CapabilityRegistry()
        reg.register(Capability("a", ""))
        stats = reg.get_stats()
        assert stats["registered"] == 1
        assert "a" in stats["capabilities"]


class TestTaskRouter:
    def test_route_success(self):
        reg = CapabilityRegistry()
        reg.register(Capability("double", "", handler=lambda x: x * 2))
        router = TaskRouter(reg)
        task = AgentTask(task_id="t1", task_type="double", payload=5)
        result = router.route(task)
        assert result.status == "completed"
        assert result.result == 10

    def test_route_no_handler(self):
        reg = CapabilityRegistry()
        router = TaskRouter(reg)
        task = AgentTask(task_id="t1", task_type="unknown", payload=5)
        result = router.route(task)
        assert result.status == "failed"
        assert "No handler" in result.result

    def test_route_exception(self):
        reg = CapabilityRegistry()
        reg.register(Capability("fail", "", handler=lambda x: 1 / 0))
        router = TaskRouter(reg)
        task = AgentTask(task_id="t1", task_type="fail", payload=5)
        result = router.route(task)
        assert result.status == "failed"

    def test_get_history(self):
        reg = CapabilityRegistry()
        reg.register(Capability("ok", "", handler=lambda x: x))
        router = TaskRouter(reg)
        router.route(AgentTask("t1", "ok", 1))
        router.route(AgentTask("t2", "ok", 2))
        assert len(router.get_history()) == 2
        assert len(router.get_history("completed")) == 2

    def test_get_stats(self):
        reg = CapabilityRegistry()
        reg.register(Capability("ok", "", handler=lambda x: x))
        router = TaskRouter(reg)
        router.route(AgentTask("t1", "ok", 1))
        stats = router.get_stats()
        assert stats["total"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 0


class TestLifecycleManager:
    def test_start_stop(self):
        lm = LifecycleManager()
        assert lm.state == "initialized"
        lm.start()
        assert lm.state == "running"
        lm.stop()
        assert lm.state == "stopped"

    def test_pause_resume(self):
        lm = LifecycleManager()
        lm.start()
        lm.pause()
        assert lm.state == "paused"
        lm.resume()
        assert lm.state == "running"

    def test_uptime(self):
        lm = LifecycleManager()
        assert lm.uptime() == 0.0
        lm.start()
        time.sleep(0.01)
        assert lm.uptime() > 0.0
        lm.stop()
        uptime = lm.uptime()
        time.sleep(0.01)
        assert lm.uptime() == uptime  # 停止后不再增长

    def test_get_stats(self):
        lm = LifecycleManager()
        stats = lm.get_stats()
        assert stats["state"] == "initialized"


class TestUniversalAgent:
    def test_register_and_execute(self):
        agent = UniversalAgent()
        agent.register_capability("double", "", lambda x: x * 2)
        result = agent.execute("double", 5)
        assert result.status == "completed"
        assert result.result == 10

    def test_execute_no_handler(self):
        agent = UniversalAgent()
        result = agent.execute("unknown", 5)
        assert result.status == "failed"

    def test_context(self):
        agent = UniversalAgent()
        agent.set_context("key", "value")
        assert agent.get_context("key") == "value"
        assert agent.get_context("missing") is None

    def test_plugin(self):
        agent = UniversalAgent()
        agent.load_plugin("math", {"add": lambda a, b: a + b})
        assert agent.get_plugin("math") is not None
        assert agent.get_plugin("missing") is None

    def test_start_stop(self):
        agent = UniversalAgent()
        agent.start()
        assert agent.lifecycle.state == "running"
        agent.stop()
        assert agent.lifecycle.state == "stopped"

    def test_execute_when_stopped(self):
        agent = UniversalAgent()
        agent.start()
        agent.stop()
        result = agent.execute("any", 1)
        assert result.status == "failed"
        assert "not running" in result.result

    def test_get_capabilities(self):
        agent = UniversalAgent()
        agent.register_capability("a", "", lambda x: x)
        agent.register_capability("b", "", lambda x: x)
        caps = agent.get_capabilities()
        assert "a" in caps
        assert "b" in caps

    def test_get_stats(self):
        agent = UniversalAgent(agent_id="test_agent")
        agent.register_capability("echo", "", lambda x: x)
        agent.execute("echo", "hello")
        stats = agent.get_stats()
        assert stats["agent_id"] == "test_agent"
        assert stats["capabilities"]["registered"] == 1
        assert stats["tasks"]["total"] == 1


class TestCreateUniversalAgent:
    def test_factory(self):
        agent = create_universal_agent("my_agent")
        assert isinstance(agent, UniversalAgent)
        assert agent.agent_id == "my_agent"
