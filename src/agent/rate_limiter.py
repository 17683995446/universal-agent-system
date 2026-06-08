"""
阶段34: Agent限流与熔断系统

遵循"图难于其易，为大于其细"的原则，实现请求限流、熔断降级、舱壁隔离。
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


# ============================================================
# 34.1 模型定义
# ============================================================

class RateLimitAlgorithm(str, Enum):
    """限流算法"""
    TOKEN_BUCKET = "token_bucket"       # 令牌桶
    SLIDING_WINDOW = "sliding_window"   # 滑动窗口
    FIXED_WINDOW = "fixed_window"       # 固定窗口
    LEAKY_BUCKET = "leaky_bucket"       # 漏桶


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭（正常通行）
    OPEN = "open"           # 打开（拒绝请求）
    HALF_OPEN = "half_open" # 半开（试探通行）


class BulkheadState(str, Enum):
    """隔离舱状态"""
    HEALTHY = "healthy"     # 健康
    REJECTED = "rejected"   # 拒绝（已满）
    DEGRADED = "degraded"   # 降级


@dataclass
class RateLimitResult:
    """限流结果"""
    allowed: bool
    remaining: int = 0
    limit: int = 0
    retry_after_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "limit": self.limit,
            "retry_after_seconds": self.retry_after_seconds,
            "metadata": self.metadata,
        }


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败阈值（触发打开）
    success_threshold: int = 3          # 成功阈值（触发关闭）
    timeout_seconds: float = 30.0       # 打开→半开等待时间
    half_open_max_calls: int = 5        # 半开状态最大试探请求数
    slow_call_duration_seconds: float = 5.0  # 慢调用阈值
    slow_call_rate_threshold: float = 0.5    # 慢调用比例阈值
    failure_rate_threshold: float = 0.5      # 失败率阈值


@dataclass
class CircuitBreakerStats:
    """熔断器统计"""
    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    success_calls: int = 0
    failure_calls: int = 0
    slow_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    half_open_successes: int = 0       # 半开期间成功数
    half_open_failures: int = 0        # 半开期间失败数

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failure_calls / self.total_calls

    @property
    def slow_call_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "success_calls": self.success_calls,
            "failure_calls": self.failure_calls,
            "slow_calls": self.slow_calls,
            "failure_rate": round(self.failure_rate, 4),
            "slow_call_rate": round(self.slow_call_rate, 4),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
        }


@dataclass
class BulkheadConfig:
    """隔离舱配置"""
    max_concurrent_calls: int = 10     # 最大并发数
    max_pending_calls: int = 20        # 最大等待队列
    timeout_seconds: float = 5.0       # 等待超时


@dataclass
class BulkheadStats:
    """隔离舱统计"""
    active_calls: int = 0
    pending_calls: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_timed_out: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_calls": self.active_calls,
            "pending_calls": self.pending_calls,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_timed_out": self.total_timed_out,
        }


# ============================================================
# 34.2 限流器
# ============================================================

class TokenBucketLimiter:
    """令牌桶限流器"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 桶容量（最大令牌数）
            refill_rate: 每秒补充令牌数
        """
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, permits: int = 1) -> RateLimitResult:
        """获取令牌"""
        with self._lock:
            self._refill()
            if self._tokens >= permits:
                self._tokens -= permits
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self._tokens),
                    limit=self._capacity,
                )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._capacity,
                retry_after_seconds=(permits - self._tokens) / self._refill_rate if self._refill_rate > 0 else float('inf'),
            )

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


class SlidingWindowLimiter:
    """滑动窗口限流器"""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self, permits: int = 1) -> RateLimitResult:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_seconds

            # 清理过期记录
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            remaining = max(0, self._max_requests - len(self._timestamps))
            if remaining >= permits:
                for _ in range(permits):
                    self._timestamps.append(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining - permits,
                    limit=self._max_requests,
                )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._max_requests,
                retry_after_seconds=self._window_seconds,
            )


class FixedWindowLimiter:
    """固定窗口限流器"""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._count = 0
        self._window_start = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, permits: int = 1) -> RateLimitResult:
        with self._lock:
            now = time.monotonic()
            if now - self._window_start >= self._window_seconds:
                self._count = 0
                self._window_start = now

            remaining = max(0, self._max_requests - self._count)
            if remaining >= permits:
                self._count += permits
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining - permits,
                    limit=self._max_requests,
                )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._max_requests,
                retry_after_seconds=self._window_seconds - (now - self._window_start),
            )


class LeakyBucketLimiter:
    """漏桶限流器"""

    def __init__(self, capacity: int, leak_rate: float):
        """
        Args:
            capacity: 桶容量
            leak_rate: 每秒漏出速率
        """
        self._capacity = capacity
        self._leak_rate = leak_rate
        self._water = 0.0
        self._last_leak = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, permits: int = 1) -> RateLimitResult:
        with self._lock:
            self._leak()
            if self._water + permits <= self._capacity:
                self._water += permits
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self._capacity - self._water),
                    limit=self._capacity,
                )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._capacity,
                retry_after_seconds=(self._capacity - self._water) / self._leak_rate if self._leak_rate > 0 else float('inf'),
            )

    def _leak(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_leak
        self._water = max(0.0, self._water - elapsed * self._leak_rate)
        self._last_leak = now


class RateLimiter:
    """限流器统一入口"""

    def __init__(self, algorithm: RateLimitAlgorithm, **kwargs):
        if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self._limiter = TokenBucketLimiter(
                kwargs.get("capacity", 100), kwargs.get("refill_rate", 10.0)
            )
        elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self._limiter = SlidingWindowLimiter(
                kwargs.get("max_requests", 100), kwargs.get("window_seconds", 1.0)
            )
        elif algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self._limiter = FixedWindowLimiter(
                kwargs.get("max_requests", 100), kwargs.get("window_seconds", 1.0)
            )
        elif algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            self._limiter = LeakyBucketLimiter(
                kwargs.get("capacity", 100), kwargs.get("leak_rate", 10.0)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self._algorithm = algorithm

    def acquire(self, permits: int = 1) -> RateLimitResult:
        return self._limiter.acquire(permits)

    @property
    def algorithm(self) -> RateLimitAlgorithm:
        return self._algorithm


# ============================================================
# 34.3 熔断器
# ============================================================

class CircuitBreaker:
    """熔断器 - 三态状态机"""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._stats = CircuitBreakerStats()
        self._stats.last_state_change = datetime.now()
        self._lock = threading.RLock()
        self._opened_at: float = 0.0  # monotonic time when opened
        self._half_open_calls: int = 0
        self._listeners: List[Callable[[str, CircuitState, CircuitState], None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._stats.state

    @property
    def stats(self) -> CircuitBreakerStats:
        return CircuitBreakerStats(
            state=self._stats.state,
            total_calls=self._stats.total_calls,
            success_calls=self._stats.success_calls,
            failure_calls=self._stats.failure_calls,
            slow_calls=self._stats.slow_calls,
            last_failure_time=self._stats.last_failure_time,
            last_state_change=self._stats.last_state_change,
        )

    def allow_request(self) -> bool:
        """判断是否允许请求通过"""
        with self._lock:
            self._check_state_transition()

            if self._stats.state == CircuitState.CLOSED:
                return True

            if self._stats.state == CircuitState.OPEN:
                return False

            # HALF_OPEN: 限制试探请求数
            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self, duration_seconds: float = 0.0) -> None:
        """记录成功调用"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.success_calls += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.half_open_successes += 1
                if self._stats.half_open_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, duration_seconds: float = 0.0) -> None:
        """记录失败调用"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.failure_calls += 1
            self._stats.last_failure_time = datetime.now()

            if duration_seconds > self._config.slow_call_duration_seconds:
                self._stats.slow_calls += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.half_open_failures += 1
                self._transition_to(CircuitState.OPEN)
            elif self._stats.state == CircuitState.CLOSED:
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)

    def record_slow_call(self) -> None:
        """记录慢调用"""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.slow_calls += 1
            self._stats.last_failure_time = datetime.now()

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.half_open_failures += 1
                self._transition_to(CircuitState.OPEN)
            elif self._stats.state == CircuitState.CLOSED:
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """重置熔断器"""
        with self._lock:
            old = self._stats.state
            self._stats = CircuitBreakerStats()
            self._stats.last_state_change = datetime.now()
            self._half_open_calls = 0
            if old != CircuitState.CLOSED:
                self._notify_listeners(old, CircuitState.CLOSED)

    def add_state_listener(
        self, listener: Callable[[str, CircuitState, CircuitState], None]
    ) -> None:
        self._listeners.append(listener)

    # --- 内部 ---

    def _should_open(self) -> bool:
        cfg = self._config
        if self._stats.total_calls < cfg.failure_threshold:
            return False
        if self._stats.failure_rate >= cfg.failure_rate_threshold:
            return True
        if self._stats.slow_call_rate >= cfg.slow_call_rate_threshold:
            return True
        return False

    def _check_state_transition(self) -> None:
        """检查是否应从OPEN转为HALF_OPEN"""
        if self._stats.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        old = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = datetime.now()

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._stats.half_open_successes = 0
            self._stats.half_open_failures = 0
        elif new_state == CircuitState.CLOSED:
            # 保留total_calls但重置失败计数
            self._stats.failure_calls = 0
            self._stats.slow_calls = 0
            self._stats.half_open_successes = 0
            self._stats.half_open_failures = 0

        self._notify_listeners(old, new_state)

    def _notify_listeners(
        self, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        for listener in self._listeners:
            try:
                listener(self._name, old_state, new_state)
            except Exception:
                pass


# ============================================================
# 34.4 服务隔离舱
# ============================================================

class Bulkhead:
    """舱壁隔离 - 限制并发和资源"""

    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self._name = name
        self._config = config or BulkheadConfig()
        self._stats = BulkheadStats()
        self._active = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> BulkheadStats:
        with self._lock:
            return BulkheadStats(
                active_calls=self._active,
                pending_calls=self._pending,
                total_accepted=self._stats.total_accepted,
                total_rejected=self._stats.total_rejected,
                total_timed_out=self._stats.total_timed_out,
            )

    def try_acquire(self) -> bool:
        """尝试获取许可（非阻塞）"""
        with self._lock:
            if self._active < self._config.max_concurrent_calls:
                self._active += 1
                self._stats.total_accepted += 1
                return True
            self._stats.total_rejected += 1
            return False

    def acquire(self, timeout_seconds: Optional[float] = None) -> bool:
        """获取许可（可阻塞等待）"""
        timeout = timeout_seconds if timeout_seconds is not None else self._config.timeout_seconds
        deadline = time.monotonic() + timeout

        with self._condition:
            while True:
                if self._active < self._config.max_concurrent_calls:
                    self._active += 1
                    self._stats.total_accepted += 1
                    return True

                if self._pending < self._config.max_pending_calls:
                    self._pending += 1
                    self._stats.total_accepted += 1
                    # 等待变为活跃
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._pending -= 1
                        self._stats.total_rejected += 1
                        return False
                    self._condition.wait(timeout=remaining)
                    continue

                # 直接拒绝
                self._stats.total_rejected += 1
                return False

    def release(self) -> None:
        """释放许可"""
        with self._condition:
            if self._active > 0:
                self._active -= 1
            elif self._pending > 0:
                self._pending -= 1
            self._condition.notify()

    def get_state(self) -> BulkheadState:
        """获取隔离舱状态"""
        with self._lock:
            if self._active >= self._config.max_concurrent_calls and \
               self._pending >= self._config.max_pending_calls:
                return BulkheadState.REJECTED
            if self._active >= self._config.max_concurrent_calls:
                return BulkheadState.DEGRADED
            return BulkheadState.HEALTHY


# ============================================================
# 便捷函数
# ============================================================

def create_rate_limiter(
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    **kwargs,
) -> RateLimiter:
    return RateLimiter(algorithm, **kwargs)


def create_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    return CircuitBreaker(name, config)


def create_bulkhead(
    name: str, config: Optional[BulkheadConfig] = None,
) -> Bulkhead:
    return Bulkhead(name, config)
