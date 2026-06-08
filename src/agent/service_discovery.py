"""
阶段33: Agent服务发现系统

遵循"图难于其易，为大于其细"的原则，实现服务注册、发现、负载均衡。
"""

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================
# 33.1 服务注册模型
# ============================================================

class ServiceStatus(str, Enum):
    """服务实例状态"""
    UP = "up"               # 正常运行
    DOWN = "down"           # 已停止
    STARTING = "starting"   # 启动中
    DRAINING = "draining"   # 流量排空中
    OUT_OF_SERVICE = "out_of_service"  # 停服维护


class LoadBalancePolicy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"       # 轮询
    RANDOM = "random"                 # 随机
    WEIGHTED = "weighted"             # 加权
    CONSISTENT_HASH = "consistent_hash"  # 一致性哈希
    LEAST_CONNECTIONS = "least_connections"  # 最少连接


@dataclass
class ServiceEndpoint:
    """服务端点"""
    host: str
    port: int
    protocol: str = "http"
    path: str = ""
    tls: bool = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def url(self) -> str:
        base = f"{self.protocol}://{self.host}:{self.port}"
        return f"{base}{self.path}" if self.path else base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "path": self.path,
            "tls": self.tls,
            "address": self.address,
            "url": self.url,
        }


@dataclass
class ServiceInstance:
    """服务实例"""
    instance_id: str
    service_name: str
    endpoint: ServiceEndpoint
    status: ServiceStatus = ServiceStatus.UP
    weight: float = 1.0
    zone: str = ""
    region: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_connections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "service_name": self.service_name,
            "endpoint": self.endpoint.to_dict(),
            "status": self.status.value,
            "weight": self.weight,
            "zone": self.zone,
            "region": self.region,
            "tags": self.tags,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "active_connections": self.active_connections,
        }


@dataclass
class ServiceRegistration:
    """服务注册信息"""
    instance_id: str
    service_name: str
    endpoint: ServiceEndpoint
    weight: float = 1.0
    zone: str = ""
    region: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: float = 30.0  # 心跳TTL，0表示永不过期


@dataclass
class ServiceChangeEvent:
    """服务变更事件"""
    event_id: str
    service_name: str
    instance_id: str
    change_type: str  # register, deregister, heartbeat, status_change
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "service_name": self.service_name,
            "instance_id": self.instance_id,
            "change_type": self.change_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


# ============================================================
# 33.2 服务注册中心
# ============================================================

class ServiceRegistry:
    """服务注册中心 - 管理服务实例的注册、注销和心跳"""

    def __init__(self, heartbeat_ttl_seconds: float = 30.0):
        self._heartbeat_ttl = heartbeat_ttl_seconds
        self._services: Dict[str, Dict[str, ServiceInstance]] = {}  # name -> {instance_id -> instance}
        self._lock = threading.RLock()
        self._listeners: List[Callable[[ServiceChangeEvent], None]] = []
        self._running = False
        self._evict_thread: Optional[threading.Thread] = None

    # --- 注册与注销 ---

    def register(self, registration: ServiceRegistration) -> ServiceInstance:
        """注册服务实例"""
        with self._lock:
            instance = ServiceInstance(
                instance_id=registration.instance_id,
                service_name=registration.service_name,
                endpoint=registration.endpoint,
                status=ServiceStatus.UP,
                weight=registration.weight,
                zone=registration.zone,
                region=registration.region,
                tags=registration.tags,
                metadata=registration.metadata,
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
            )

            if registration.service_name not in self._services:
                self._services[registration.service_name] = {}
            self._services[registration.service_name][instance.instance_id] = instance

        self._notify(ServiceChangeEvent(
            event_id=self._gen_id(),
            service_name=registration.service_name,
            instance_id=registration.instance_id,
            change_type="register",
            details={"endpoint": instance.endpoint.address, "zone": instance.zone},
        ))
        return instance

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """注销服务实例"""
        with self._lock:
            bucket = self._services.get(service_name)
            if not bucket or instance_id not in bucket:
                return False
            del bucket[instance_id]
            if not bucket:
                del self._services[service_name]

        self._notify(ServiceChangeEvent(
            event_id=self._gen_id(),
            service_name=service_name,
            instance_id=instance_id,
            change_type="deregister",
        ))
        return True

    # --- 心跳 ---

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """续约心跳"""
        with self._lock:
            instance = self._get_instance(service_name, instance_id)
            if not instance:
                return False
            instance.last_heartbeat = datetime.now()
            if instance.status == ServiceStatus.DOWN:
                instance.status = ServiceStatus.UP
        return True

    def set_status(
        self, service_name: str, instance_id: str, status: ServiceStatus
    ) -> bool:
        """设置实例状态"""
        with self._lock:
            instance = self._get_instance(service_name, instance_id)
            if not instance:
                return False
            old = instance.status
            instance.status = status

        self._notify(ServiceChangeEvent(
            event_id=self._gen_id(),
            service_name=service_name,
            instance_id=instance_id,
            change_type="status_change",
            details={"old_status": old.value, "new_status": status.value},
        ))
        return True

    # --- 查询 ---

    def get_instance(
        self, service_name: str, instance_id: str
    ) -> Optional[ServiceInstance]:
        """获取单个实例"""
        with self._lock:
            return self._get_instance(service_name, instance_id)

    def get_all_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取服务的所有实例"""
        with self._lock:
            bucket = self._services.get(service_name, {})
            return list(bucket.values())

    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """获取健康实例"""
        with self._lock:
            bucket = self._services.get(service_name, {})
            return [
                inst for inst in bucket.values()
                if inst.status == ServiceStatus.UP
            ]

    def get_all_service_names(self) -> List[str]:
        """获取所有服务名"""
        with self._lock:
            return list(self._services.keys())

    # --- 过期清理 ---

    def evict_expired(self) -> int:
        """清理过期实例，返回清理数量"""
        now = datetime.now()
        evicted = 0
        with self._lock:
            for name in list(self._services.keys()):
                bucket = self._services[name]
                for iid in list(bucket.keys()):
                    inst = bucket[iid]
                    if self._heartbeat_ttl > 0:
                        elapsed = (now - inst.last_heartbeat).total_seconds()
                        if elapsed > self._heartbeat_ttl:
                            inst.status = ServiceStatus.DOWN
                            evicted += 1
        return evicted

    def start_eviction(self, interval_seconds: float = 10.0) -> None:
        """启动后台过期清理"""
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                self.evict_expired()
                time.sleep(interval_seconds)

        self._evict_thread = threading.Thread(target=_loop, daemon=True)
        self._evict_thread.start()

    def stop_eviction(self) -> None:
        """停止后台过期清理"""
        self._running = False
        if self._evict_thread:
            self._evict_thread.join(timeout=5.0)
            self._evict_thread = None

    # --- 监听 ---

    def add_listener(
        self, listener: Callable[[ServiceChangeEvent], None]
    ) -> None:
        self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[ServiceChangeEvent], None]
    ) -> bool:
        if listener in self._listeners:
            self._listeners.remove(listener)
            return True
        return False

    # --- 内部 ---

    def _get_instance(
        self, service_name: str, instance_id: str
    ) -> Optional[ServiceInstance]:
        bucket = self._services.get(service_name)
        if bucket:
            return bucket.get(instance_id)
        return None

    def _notify(self, event: ServiceChangeEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4())


# ============================================================
# 33.3 服务发现
# ============================================================

class ServiceDiscovery:
    """服务发现 - 按条件查找可用服务"""

    def __init__(self, registry: ServiceRegistry):
        self._registry = registry
        self._lock = threading.RLock()

    def discover(
        self,
        service_name: str,
        healthy_only: bool = True,
        zone: str = "",
        region: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> List[ServiceInstance]:
        """发现服务实例"""
        instances = self._registry.get_all_instances(service_name)

        if healthy_only:
            instances = [i for i in instances if i.status == ServiceStatus.UP]

        if zone:
            instances = [i for i in instances if i.zone == zone]

        if region:
            instances = [i for i in instances if i.region == region]

        if tags:
            instances = [
                i for i in instances
                if all(i.tags.get(k) == v for k, v in tags.items())
            ]

        return instances

    def discover_one(
        self,
        service_name: str,
        healthy_only: bool = True,
        zone: str = "",
        region: str = "",
        tags: Optional[Dict[str, str]] = None,
        policy: LoadBalancePolicy = LoadBalancePolicy.ROUND_ROBIN,
        key: str = "",
    ) -> Optional[ServiceInstance]:
        """发现一个服务实例（通过负载均衡选择）"""
        instances = self.discover(
            service_name, healthy_only, zone, region, tags
        )
        if not instances:
            return None
        return _load_balance(instances, policy, key)


# ============================================================
# 33.4 负载均衡策略
# ============================================================

# 轮询计数器（per service_name）
_round_robin_counters: Dict[str, int] = {}
_rr_lock = threading.Lock()


def _load_balance(
    instances: List[ServiceInstance],
    policy: LoadBalancePolicy,
    key: str = "",
) -> ServiceInstance:
    """根据策略选择一个实例"""
    if len(instances) == 1:
        return instances[0]

    if policy == LoadBalancePolicy.ROUND_ROBIN:
        return _lb_round_robin(instances)
    elif policy == LoadBalancePolicy.RANDOM:
        return _lb_random(instances)
    elif policy == LoadBalancePolicy.WEIGHTED:
        return _lb_weighted(instances)
    elif policy == LoadBalancePolicy.CONSISTENT_HASH:
        return _lb_consistent_hash(instances, key)
    elif policy == LoadBalancePolicy.LEAST_CONNECTIONS:
        return _lb_least_connections(instances)
    else:
        return instances[0]


def _lb_round_robin(instances: List[ServiceInstance]) -> ServiceInstance:
    """轮询"""
    name = instances[0].service_name
    with _rr_lock:
        idx = _round_robin_counters.get(name, 0)
        _round_robin_counters[name] = (idx + 1) % len(instances)
    return instances[idx]


def _lb_random(instances: List[ServiceInstance]) -> ServiceInstance:
    """随机"""
    import random
    return random.choice(instances)


def _lb_weighted(instances: List[ServiceInstance]) -> ServiceInstance:
    """加权随机"""
    import random
    weights = [max(0.01, i.weight) for i in instances]
    return random.choices(instances, weights=weights, k=1)[0]


def _lb_consistent_hash(
    instances: List[ServiceInstance], key: str
) -> ServiceInstance:
    """一致性哈希"""
    if not key:
        key = str(time.time())
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return instances[hash_val % len(instances)]


def _lb_least_connections(instances: List[ServiceInstance]) -> ServiceInstance:
    """最少连接"""
    return min(instances, key=lambda i: i.active_connections)


# ============================================================
# 便捷函数
# ============================================================

def create_service_registry(
    heartbeat_ttl_seconds: float = 30.0,
) -> ServiceRegistry:
    return ServiceRegistry(heartbeat_ttl_seconds)


def create_service_discovery(
    registry: ServiceRegistry,
) -> ServiceDiscovery:
    return ServiceDiscovery(registry)


def create_registration(
    service_name: str,
    host: str,
    port: int,
    instance_id: str = "",
    weight: float = 1.0,
    zone: str = "",
    region: str = "",
    tags: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ServiceRegistration:
    return ServiceRegistration(
        instance_id=instance_id or str(uuid.uuid4()),
        service_name=service_name,
        endpoint=ServiceEndpoint(host=host, port=port),
        weight=weight,
        zone=zone,
        region=region,
        tags=tags or {},
        metadata=metadata or {},
    )
