"""
阶段98: Agent 多智能体协作系统

遵循"图难于其易，为大于其细"的原则，实现智能体注册、消息传递、协商机制、协作任务分配。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random


@dataclass
class Message:
    """智能体间消息"""
    sender: str
    receiver: str
    content: Any
    msg_type: str = "inform"  # inform, request, propose, accept, reject

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "msg_type": self.msg_type,
            "content": self.content,
        }


@dataclass
class Agent:
    """智能体"""
    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    inbox: List[Message] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    def receive(self, msg: Message):
        """接收消息"""
        self.inbox.append(msg)

    def read_messages(self) -> List[Message]:
        """读取所有消息并清空收件箱"""
        msgs = self.inbox.copy()
        self.inbox.clear()
        return msgs

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "inbox_size": len(self.inbox),
        }


class MessageBus:
    """消息总线"""

    def __init__(self):
        self._messages: List[Message] = []
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent):
        """注册智能体"""
        self._agents[agent.agent_id] = agent

    def send(self, msg: Message):
        """发送消息"""
        self._messages.append(msg)
        if msg.receiver in self._agents:
            self._agents[msg.receiver].receive(msg)

    def broadcast(self, sender: str, content: Any, msg_type: str = "inform"):
        """广播消息"""
        for agent_id in self._agents:
            if agent_id != sender:
                self.send(Message(sender, agent_id, content, msg_type))

    def get_messages(self) -> List[Message]:
        return self._messages.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_agents": len(self._agents),
            "total_messages": len(self._messages),
        }


class NegotiationProtocol:
    """协商协议（合同网协议简化版）"""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus

    def call_for_proposal(self, manager: str, task: str,
                          candidates: List[str]) -> List[Message]:
        """发起招标"""
        for candidate in candidates:
            self.bus.send(Message(
                sender=manager,
                receiver=candidate,
                content={"task": task, "action": "cfp"},
                msg_type="request",
            ))
        return [m for m in self.bus.get_messages() if m.sender == manager and m.msg_type == "request"]

    def propose(self, agent_id: str, manager: str, task: str, bid: float):
        """提交投标"""
        self.bus.send(Message(
            sender=agent_id,
            receiver=manager,
            content={"task": task, "bid": bid},
            msg_type="propose",
        ))

    def accept_proposal(self, manager: str, agent_id: str, task: str):
        """接受投标"""
        self.bus.send(Message(
            sender=manager,
            receiver=agent_id,
            content={"task": task, "result": "accepted"},
            msg_type="accept",
        ))

    def reject_proposal(self, manager: str, agent_id: str, task: str):
        """拒绝投标"""
        self.bus.send(Message(
            sender=manager,
            receiver=agent_id,
            content={"task": task, "result": "rejected"},
            msg_type="reject",
        ))


class TaskAllocator:
    """任务分配器"""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.negotiation = NegotiationProtocol(message_bus)
        self._allocations: Dict[str, str] = {}  # task -> agent_id

    def allocate(self, task: str, required_capability: str) -> Optional[str]:
        """分配任务给有能力的智能体"""
        candidates = [
            aid for aid, agent in self.bus._agents.items()
            if agent.has_capability(required_capability)
        ]
        if not candidates:
            return None

        # 简化的合同网：向所有候选者发CFP，选择第一个响应的（简化）
        self.negotiation.call_for_proposal("allocator", task, candidates)

        # 简化：随机选择一个候选者
        chosen = random.choice(candidates)
        self.negotiation.accept_proposal("allocator", chosen, task)
        self._allocations[task] = chosen
        return chosen

    def get_allocation(self, task: str) -> Optional[str]:
        return self._allocations.get(task)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "allocated_tasks": len(self._allocations),
        }


class MultiAgentSystem:
    """多智能体系统"""

    def __init__(self):
        self.bus = MessageBus()
        self.negotiation = NegotiationProtocol(self.bus)
        self.allocator = TaskAllocator(self.bus)
        self._task_results: Dict[str, Any] = {}

    def add_agent(self, agent_id: str, capabilities: List[str] = None):
        """添加智能体"""
        agent = Agent(agent_id=agent_id, capabilities=capabilities or [])
        self.bus.register(agent)

    def send_message(self, sender: str, receiver: str, content: Any, msg_type: str = "inform"):
        """发送消息"""
        self.bus.send(Message(sender, receiver, content, msg_type))

    def broadcast(self, sender: str, content: Any):
        """广播"""
        self.bus.broadcast(sender, content)

    def assign_task(self, task: str, required_capability: str) -> Optional[str]:
        """分配任务"""
        return self.allocator.allocate(task, required_capability)

    def submit_result(self, agent_id: str, task: str, result: Any):
        """提交任务结果"""
        self._task_results[task] = {
            "agent_id": agent_id,
            "result": result,
        }

    def get_result(self, task: str) -> Optional[Dict[str, Any]]:
        return self._task_results.get(task)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agents": self.bus.get_stats(),
            "allocations": self.allocator.get_stats(),
            "completed_tasks": len(self._task_results),
        }


def create_multi_agent_system() -> MultiAgentSystem:
    return MultiAgentSystem()
