"""
阶段33测试: Agent服务发现系统
"""

import time
import threading
import pytest
from datetime import datetime, timedelta

from src.agent.service_discovery import (
    ServiceStatus,
    LoadBalancePolicy,
    ServiceEndpoint,
    ServiceInstance,
    ServiceRegistration,
    ServiceChangeEvent,
    ServiceRegistry,
    ServiceDiscovery,
    create_service_registry,
    create_service_discovery,
    create_registration,
    _lb_round_robin,
    _lb_random,
    _lb_weighted,
    _lb_consistent_hash,
    _lb_least_connections,
)


# ============================================================
# 模型测试
# ============================================================

class TestServiceStatus:
    def test_values(self):
        assert ServiceStatus.UP.value == "up"
        assert ServiceStatus.DOWN.value == "down"
        assert ServiceStatus.STARTING.value == "starting"
        assert ServiceStatus.DRAINING.value == "draining"
        assert ServiceStatus.OUT_OF_SERVICE.value == "out_of_service"


class TestLoadBalancePolicy:
    def test_values(self):
        assert LoadBalancePolicy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancePolicy.RANDOM.value == "random"
        assert LoadBalancePolicy.WEIGHTED.value == "weighted"
        assert LoadBalancePolicy.CONSISTENT_HASH.value == "consistent_hash"
        assert LoadBalancePolicy.LEAST_CONNECTIONS.value == "least_connections"


class TestServiceEndpoint:
    def test_address(self):
        ep = ServiceEndpoint(host="10.0.0.1", port=8080)
        assert ep.address == "10.0.0.1:8080"

    def test_url(self):
        ep = ServiceEndpoint(host="10.0.0.1", port=8080, protocol="https", path="/api")
        assert ep.url == "https://10.0.0.1:8080/api"

    def test_to_dict(self):
        ep = ServiceEndpoint(host="10.0.0.1", port=8080, tls=True)
        d = ep.to_dict()
        assert d["host"] == "10.0.0.1"
        assert d["tls"] is True


class TestServiceInstance:
    def test_creation(self):
        ep = ServiceEndpoint("10.0.0.1", 8080)
        inst = ServiceInstance(
            instance_id="i1", service_name="svc", endpoint=ep,
            zone="a", region="us-east", tags={"v": "1"},
        )
        assert inst.status == ServiceStatus.UP
        assert inst.zone == "a"
        assert inst.tags["v"] == "1"

    def test_to_dict(self):
        ep = ServiceEndpoint("10.0.0.1", 8080)
        inst = ServiceInstance(instance_id="i1", service_name="svc", endpoint=ep)
        d = inst.to_dict()
        assert d["instance_id"] == "i1"
        assert d["status"] == "up"
        assert "endpoint" in d


class TestServiceRegistration:
    def test_creation(self):
        reg = create_registration("svc", "10.0.0.1", 8080, zone="a")
        assert reg.service_name == "svc"
        assert reg.endpoint.host == "10.0.0.1"
        assert reg.zone == "a"


class TestServiceChangeEvent:
    def test_creation(self):
        evt = ServiceChangeEvent(
            event_id="e1", service_name="svc",
            instance_id="i1", change_type="register",
        )
        assert evt.change_type == "register"
        d = evt.to_dict()
        assert d["service_name"] == "svc"


# ============================================================
# ServiceRegistry 测试
# ============================================================

class TestServiceRegistry:
    def setup_method(self):
        self.registry = ServiceRegistry(heartbeat_ttl_seconds=5.0)

    def _make_reg(self, name="svc", host="10.0.0.1", port=8080, iid="i1", **kw):
        return create_registration(name, host, port, instance_id=iid, **kw)

    # --- 注册/注销 ---

    def test_register(self):
        reg = self._make_reg()
        inst = self.registry.register(reg)
        assert inst.instance_id == "i1"
        assert inst.status == ServiceStatus.UP

    def test_register_multiple(self):
        self.registry.register(self._make_reg(iid="i1"))
        self.registry.register(self._make_reg(iid="i2", host="10.0.0.2"))
        self.registry.register(self._make_reg(iid="i3", host="10.0.0.3"))
        assert len(self.registry.get_all_instances("svc")) == 3

    def test_deregister(self):
        self.registry.register(self._make_reg())
        assert self.registry.deregister("svc", "i1") is True
        assert self.registry.get_instance("svc", "i1") is None

    def test_deregister_nonexistent(self):
        assert self.registry.deregister("svc", "nope") is False

    # --- 心跳 ---

    def test_heartbeat(self):
        self.registry.register(self._make_reg())
        assert self.registry.heartbeat("svc", "i1") is True

    def test_heartbeat_nonexistent(self):
        assert self.registry.heartbeat("svc", "nope") is False

    def test_heartbeat_revives_down(self):
        self.registry.register(self._make_reg())
        self.registry.set_status("svc", "i1", ServiceStatus.DOWN)
        self.registry.heartbeat("svc", "i1")
        assert self.registry.get_instance("svc", "i1").status == ServiceStatus.UP

    # --- 状态 ---

    def test_set_status(self):
        self.registry.register(self._make_reg())
        assert self.registry.set_status("svc", "i1", ServiceStatus.DRAINING) is True
        assert self.registry.get_instance("svc", "i1").status == ServiceStatus.DRAINING

    # --- 查询 ---

    def test_get_all_instances(self):
        self.registry.register(self._make_reg(iid="i1"))
        self.registry.register(self._make_reg(iid="i2", host="10.0.0.2"))
        assert len(self.registry.get_all_instances("svc")) == 2

    def test_get_healthy_instances(self):
        self.registry.register(self._make_reg(iid="i1"))
        self.registry.register(self._make_reg(iid="i2", host="10.0.0.2"))
        self.registry.set_status("svc", "i2", ServiceStatus.DOWN)
        healthy = self.registry.get_healthy_instances("svc")
        assert len(healthy) == 1
        assert healthy[0].instance_id == "i1"

    def test_get_all_service_names(self):
        self.registry.register(self._make_reg(name="svc-a", iid="i1"))
        self.registry.register(self._make_reg(name="svc-b", iid="i2"))
        names = self.registry.get_all_service_names()
        assert set(names) == {"svc-a", "svc-b"}

    # --- 过期清理 ---

    def test_evict_expired(self):
        self.registry.register(self._make_reg(iid="i1"))
        self.registry.register(self._make_reg(iid="i2", host="10.0.0.2"))

        # 模拟i1心跳过期
        inst = self.registry.get_instance("svc", "i1")
        inst.last_heartbeat = datetime.now() - timedelta(seconds=10)

        evicted = self.registry.evict_expired()
        assert evicted == 1
        assert self.registry.get_instance("svc", "i1").status == ServiceStatus.DOWN
        assert self.registry.get_instance("svc", "i2").status == ServiceStatus.UP

    # --- 监听 ---

    def test_listener(self):
        events = []
        self.registry.add_listener(lambda e: events.append(e))
        self.registry.register(self._make_reg())
        assert len(events) == 1
        assert events[0].change_type == "register"

    def test_remove_listener(self):
        events = []
        fn = lambda e: events.append(e)
        self.registry.add_listener(fn)
        self.registry.register(self._make_reg())
        assert len(events) == 1
        self.registry.remove_listener(fn)
        self.registry.register(self._make_reg(iid="i2", host="10.0.0.2"))
        assert len(events) == 1  # 不再收到


# ============================================================
# ServiceDiscovery 测试
# ============================================================

class TestServiceDiscovery:
    def setup_method(self):
        self.registry = ServiceRegistry()
        self.discovery = ServiceDiscovery(self.registry)

    def _reg(self, name="svc", host="10.0.0.1", port=8080, iid="i1", **kw):
        return create_registration(name, host, port, instance_id=iid, **kw)

    def test_discover_all(self):
        self.registry.register(self._reg(iid="i1"))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2"))
        result = self.discovery.discover("svc", healthy_only=False)
        assert len(result) == 2

    def test_discover_healthy_only(self):
        self.registry.register(self._reg(iid="i1"))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2"))
        self.registry.set_status("svc", "i2", ServiceStatus.DOWN)
        result = self.discovery.discover("svc", healthy_only=True)
        assert len(result) == 1
        assert result[0].instance_id == "i1"

    def test_discover_by_zone(self):
        self.registry.register(self._reg(iid="i1", zone="a"))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2", zone="b"))
        result = self.discovery.discover("svc", zone="a")
        assert len(result) == 1
        assert result[0].zone == "a"

    def test_discover_by_region(self):
        self.registry.register(self._reg(iid="i1", region="us-east"))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2", region="eu-west"))
        result = self.discovery.discover("svc", region="us-east")
        assert len(result) == 1

    def test_discover_by_tags(self):
        self.registry.register(self._reg(iid="i1", tags={"v": "1"}))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2", tags={"v": "2"}))
        result = self.discovery.discover("svc", tags={"v": "2"})
        assert len(result) == 1
        assert result[0].instance_id == "i2"

    def test_discover_empty(self):
        result = self.discovery.discover("nonexistent")
        assert result == []

    def test_discover_one(self):
        self.registry.register(self._reg(iid="i1"))
        self.registry.register(self._reg(iid="i2", host="10.0.0.2"))
        inst = self.discovery.discover_one("svc")
        assert inst is not None
        assert inst.service_name == "svc"

    def test_discover_one_empty(self):
        assert self.discovery.discover_one("nonexistent") is None


# ============================================================
# 负载均衡测试
# ============================================================

class TestLoadBalance:
    def _make_instances(self, n=3):
        return [
            ServiceInstance(
                instance_id=f"i{j}", service_name="svc",
                endpoint=ServiceEndpoint(f"10.0.0.{j}", 8080),
                weight=float(j),
                active_connections=j,
            )
            for j in range(1, n + 1)
        ]

    def test_round_robin_distribution(self):
        insts = self._make_instances(3)
        # 重置计数器
        from src.agent.service_discovery import _round_robin_counters, _rr_lock
        with _rr_lock:
            _round_robin_counters.pop("svc", None)

        picked = [_lb_round_robin(insts) for _ in range(6)]
        ids = [p.instance_id for p in picked]
        assert ids == ["i1", "i2", "i3", "i1", "i2", "i3"]

    def test_random(self):
        insts = self._make_instances(3)
        picked = _lb_random(insts)
        assert picked.instance_id in ["i1", "i2", "i3"]

    def test_weighted(self):
        insts = self._make_instances(3)
        picked = _lb_weighted(insts)
        assert picked.instance_id in ["i1", "i2", "i3"]

    def test_consistent_hash_same_key(self):
        insts = self._make_instances(5)
        a = _lb_consistent_hash(insts, "user-123")
        b = _lb_consistent_hash(insts, "user-123")
        assert a.instance_id == b.instance_id

    def test_consistent_hash_different_key(self):
        insts = self._make_instances(5)
        a = _lb_consistent_hash(insts, "key-a")
        b = _lb_consistent_hash(insts, "key-b")
        # 不一定不同，但概率极低
        assert isinstance(a, ServiceInstance)

    def test_least_connections(self):
        insts = self._make_instances(3)
        picked = _lb_least_connections(insts)
        assert picked.instance_id == "i1"  # active_connections=1 最小


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    def test_create_service_registry(self):
        r = create_service_registry(60.0)
        assert isinstance(r, ServiceRegistry)

    def test_create_service_discovery(self):
        r = create_service_registry()
        d = create_service_discovery(r)
        assert isinstance(d, ServiceDiscovery)

    def test_create_registration(self):
        reg = create_registration("svc", "10.0.0.1", 8080, zone="a", tags={"v": "1"})
        assert reg.service_name == "svc"
        assert reg.zone == "a"
        assert reg.tags["v"] == "1"


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    def test_full_lifecycle(self):
        registry = create_service_registry()
        discovery = create_service_discovery(registry)

        # 注册3个实例
        for i in range(3):
            registry.register(create_registration(
                "api", f"10.0.0.{i+1}", 8080,
                instance_id=f"inst-{i}", zone="a", weight=float(i + 1),
            ))

        # 发现
        found = discovery.discover("api", zone="a")
        assert len(found) == 3

        # 负载均衡选择
        one = discovery.discover_one("api", zone="a", policy=LoadBalancePolicy.LEAST_CONNECTIONS)
        assert one is not None

        # 注销一个
        registry.deregister("api", "inst-1")
        found = discovery.discover("api")
        assert len(found) == 2

        # 标记下线
        registry.set_status("api", "inst-2", ServiceStatus.DOWN)
        healthy = discovery.discover("api", healthy_only=True)
        assert len(healthy) == 1
        assert healthy[0].instance_id == "inst-0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
