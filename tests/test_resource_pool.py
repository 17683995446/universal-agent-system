"""阶段44: Agent资源池管理系统 - 单元测试"""
import pytest
from src.agent.resource_pool import (
    ResourceType, PoolStatus, ResourceUnit, Allocation, ResourceQuota,
    ResourcePool, ResourceManager, create_resource_manager,
)

class TestResourcePool:
    def test_allocate(self):
        p = ResourcePool("cpu", ResourceType.CPU, 4.0)
        a = p.allocate(2.0, "agent1")
        assert a is not None
        assert a.amount == 2.0
        assert p.available == 2.0

    def test_allocate_exhaust(self):
        p = ResourcePool("cpu", ResourceType.CPU, 2.0)
        p.allocate(2.0, "a1")
        assert p.allocate(1.0, "a2") is None
        assert p.status == PoolStatus.EXHAUSTED

    def test_release(self):
        p = ResourcePool("cpu", ResourceType.CPU, 4.0)
        a = p.allocate(2.0, "a1")
        assert p.release(a.id)
        assert p.available == 4.0

    def test_utilization(self):
        p = ResourcePool("mem", ResourceType.MEMORY, 1000)
        p.allocate(250, "a1")
        assert abs(p.utilization - 0.25) < 0.01

    def test_with_unit(self):
        p = ResourcePool("gpu", ResourceType.GPU, 4.0)
        u = ResourceUnit(id="gpu1", resource_type=ResourceType.GPU, capacity=2.0)
        p.add_unit(u)
        a = p.allocate(1.0, "a1", labels={"gpu_id": "gpu1"})
        assert a is not None

    def test_closed_pool(self):
        p = ResourcePool("cpu", ResourceType.CPU, 4.0)
        p.close()
        assert p.allocate(1.0, "a1") is None

    def test_get_allocations(self):
        p = ResourcePool("cpu", ResourceType.CPU, 4.0)
        p.allocate(1.0, "a1")
        p.allocate(1.0, "a2")
        allocs = p.get_allocations("a1")
        assert len(allocs) == 1

class TestResourceManager:
    def test_create_pool(self):
        m = ResourceManager()
        m.create_pool("cpu", ResourceType.CPU, 4.0)
        assert "cpu" in m.list_pools()

    def test_allocate_and_release(self):
        m = ResourceManager()
        m.create_pool("cpu", ResourceType.CPU, 4.0)
        a = m.allocate("cpu", 2.0, "agent1")
        assert a is not None
        assert m.release("cpu", a.id)

    def test_quota_enforcement(self):
        m = ResourceManager()
        m.create_pool("cpu", ResourceType.CPU, 10.0)
        m.set_quota("agent1", ResourceQuota(owner="agent1", max_cpu=2.0))
        m.allocate("cpu", 1.0, "agent1")
        assert m.allocate("cpu", 2.0, "agent1") is None

    def test_nonexistent_pool(self):
        m = ResourceManager()
        assert m.allocate("nope", 1.0, "a") is None

    def test_create_manager(self):
        m = create_resource_manager()
        assert isinstance(m, ResourceManager)
