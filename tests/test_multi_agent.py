"""
阶段98测试: 多智能体协作系统
"""
import pytest
from src.agent.multi_agent import (
    Message, Agent, MessageBus, NegotiationProtocol,
    TaskAllocator, MultiAgentSystem, create_multi_agent_system,
)


class TestMessage:
    def test_to_dict(self):
        m = Message(sender="a", receiver="b", content="hello", msg_type="inform")
        d = m.to_dict()
        assert d["sender"] == "a"
        assert d["receiver"] == "b"
        assert d["msg_type"] == "inform"


class TestAgent:
    def test_receive_and_read(self):
        a = Agent(agent_id="agent1", capabilities=["compute"])
        a.receive(Message("a2", "agent1", "hi"))
        assert len(a.inbox) == 1
        msgs = a.read_messages()
        assert len(msgs) == 1
        assert len(a.inbox) == 0

    def test_has_capability(self):
        a = Agent(agent_id="a1", capabilities=["read", "write"])
        assert a.has_capability("read") is True
        assert a.has_capability("delete") is False

    def test_to_dict(self):
        a = Agent(agent_id="a1")
        d = a.to_dict()
        assert d["agent_id"] == "a1"
        assert d["inbox_size"] == 0


class TestMessageBus:
    def test_register_and_send(self):
        bus = MessageBus()
        a1 = Agent(agent_id="a1")
        a2 = Agent(agent_id="a2")
        bus.register(a1)
        bus.register(a2)
        bus.send(Message("a1", "a2", "hello"))
        assert len(a2.inbox) == 1
        assert bus.get_stats()["total_messages"] == 1

    def test_broadcast(self):
        bus = MessageBus()
        a1 = Agent(agent_id="a1")
        a2 = Agent(agent_id="a2")
        a3 = Agent(agent_id="a3")
        bus.register(a1)
        bus.register(a2)
        bus.register(a3)
        bus.broadcast("a1", "alert")
        assert len(a2.inbox) == 1
        assert len(a3.inbox) == 1
        assert len(a1.inbox) == 0


class TestNegotiationProtocol:
    def test_call_for_proposal(self):
        bus = MessageBus()
        bus.register(Agent("manager"))
        bus.register(Agent("worker1"))
        bus.register(Agent("worker2"))
        neg = NegotiationProtocol(bus)
        msgs = neg.call_for_proposal("manager", "task1", ["worker1", "worker2"])
        assert len(msgs) == 2

    def test_propose_and_accept(self):
        bus = MessageBus()
        bus.register(Agent("m"))
        bus.register(Agent("w"))
        neg = NegotiationProtocol(bus)
        neg.propose("w", "m", "task1", 10.0)
        neg.accept_proposal("m", "w", "task1")
        assert len(bus.get_messages()) == 2


class TestTaskAllocator:
    def test_allocate(self):
        bus = MessageBus()
        bus.register(Agent("a1", capabilities=["compute"]))
        bus.register(Agent("a2", capabilities=["compute"]))
        allocator = TaskAllocator(bus)
        chosen = allocator.allocate("task1", "compute")
        assert chosen in ["a1", "a2"]
        assert allocator.get_allocation("task1") == chosen

    def test_allocate_no_candidate(self):
        bus = MessageBus()
        bus.register(Agent("a1", capabilities=["read"]))
        allocator = TaskAllocator(bus)
        assert allocator.allocate("task1", "compute") is None


class TestMultiAgentSystem:
    def test_add_agent(self):
        sys = MultiAgentSystem()
        sys.add_agent("agent1", ["compute"])
        assert sys.bus.get_stats()["registered_agents"] == 1

    def test_send_message(self):
        sys = MultiAgentSystem()
        sys.add_agent("a1")
        sys.add_agent("a2")
        sys.send_message("a1", "a2", "hello")
        assert sys.bus.get_stats()["total_messages"] == 1

    def test_assign_task(self):
        sys = MultiAgentSystem()
        sys.add_agent("worker", ["compute"])
        chosen = sys.assign_task("task1", "compute")
        assert chosen == "worker"

    def test_submit_and_get_result(self):
        sys = MultiAgentSystem()
        sys.submit_result("worker", "task1", "done")
        result = sys.get_result("task1")
        assert result["agent_id"] == "worker"
        assert result["result"] == "done"

    def test_get_stats(self):
        sys = MultiAgentSystem()
        sys.add_agent("a1")
        stats = sys.get_stats()
        assert stats["agents"]["registered_agents"] == 1


class TestCreateMultiAgentSystem:
    def test_factory(self):
        sys = create_multi_agent_system()
        assert isinstance(sys, MultiAgentSystem)
