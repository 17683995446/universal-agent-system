"""
阶段34测试: Agent限流与熔断系统
"""

import threading
import time
import pytest

from src.agent.rate_limiter import (
    RateLimitAlgorithm,
    CircuitState,
    BulkheadState,
    RateLimitResult,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    BulkheadConfig,
    BulkheadStats,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    FixedWindowLimiter,
    LeakyBucketLimiter,
    RateLimiter,
    CircuitBreaker,
    Bulkhead,
    create_rate_limiter,
    create_circuit_breaker,
    create_bulkhead,
)


# ============================================================
# 模型测试
# ============================================================

class TestEnums:
    def test_rate_limit_algorithm(self):
        assert RateLimitAlgorithm.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitAlgorithm.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitAlgorithm.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitAlgorithm.LEAKY_BUCKET.value == "leaky_bucket"

    def test_circuit_state(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_bulkhead_state(self):
        assert BulkheadState.HEALTHY.value == "healthy"
        assert BulkheadState.REJECTED.value == "rejected"
        assert BulkheadState.DEGRADED.value == "degraded"


class TestRateLimitResult:
    def test_allowed(self):
        r = RateLimitResult(allowed=True, remaining=5, limit=10)
        assert r.allowed is True
        d = r.to_dict()
        assert d["allowed"] is True
        assert d["remaining"] == 5

    def test_rejected(self):
        r = RateLimitResult(allowed=False, retry_after_seconds=2.0)
        assert r.allowed is False
        assert r.retry_after_seconds == 2.0


class TestCircuitBreakerStats:
    def test_failure_rate(self):
        s = CircuitBreakerStats(total_calls=10, failure_calls=3)
        assert abs(s.failure_rate - 0.3) < 0.001

    def test_failure_rate_zero(self):
        s = CircuitBreakerStats()
        assert s.failure_rate == 0.0

    def test_slow_call_rate(self):
        s = CircuitBreakerStats(total_calls=10, slow_calls=2)
        assert abs(s.slow_call_rate - 0.2) < 0.001


class TestBulkheadStats:
    def test_to_dict(self):
        s = BulkheadStats(active_calls=3, total_accepted=10, total_rejected=1)
        d = s.to_dict()
        assert d["active_calls"] == 3
        assert d["total_accepted"] == 10


# ============================================================
# 令牌桶测试
# ============================================================

class TestTokenBucketLimiter:
    def test_acquire_within_capacity(self):
        limiter = TokenBucketLimiter(capacity=5, refill_rate=1.0)
        r = limiter.acquire()
        assert r.allowed is True
        assert r.remaining >= 0

    def test_acquire_exhausted(self):
        limiter = TokenBucketLimiter(capacity=2, refill_rate=0.0)
        limiter.acquire()
        limiter.acquire()
        r = limiter.acquire()
        assert r.allowed is False

    def test_refill(self):
        limiter = TokenBucketLimiter(capacity=1, refill_rate=100.0)
        limiter.acquire()
        time.sleep(0.05)
        r = limiter.acquire()
        assert r.allowed is True


# ============================================================
# 滑动窗口测试
# ============================================================

class TestSlidingWindowLimiter:
    def test_within_limit(self):
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            assert limiter.acquire().allowed is True

    def test_exceed_limit(self):
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=1.0)
        for _ in range(3):
            limiter.acquire()
        r = limiter.acquire()
        assert r.allowed is False

    def test_window_slide(self):
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=0.1)
        limiter.acquire()
        r = limiter.acquire()
        assert r.allowed is False
        time.sleep(0.15)
        r = limiter.acquire()
        assert r.allowed is True


# ============================================================
# 固定窗口测试
# ============================================================

class TestFixedWindowLimiter:
    def test_within_limit(self):
        limiter = FixedWindowLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            assert limiter.acquire().allowed is True

    def test_exceed_limit(self):
        limiter = FixedWindowLimiter(max_requests=2, window_seconds=1.0)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire().allowed is False

    def test_window_reset(self):
        limiter = FixedWindowLimiter(max_requests=1, window_seconds=0.1)
        limiter.acquire()
        assert limiter.acquire().allowed is False
        time.sleep(0.15)
        assert limiter.acquire().allowed is True


# ============================================================
# 漏桶测试
# ============================================================

class TestLeakyBucketLimiter:
    def test_within_capacity(self):
        limiter = LeakyBucketLimiter(capacity=5, leak_rate=10.0)
        for _ in range(5):
            assert limiter.acquire().allowed is True

    def test_exhausted(self):
        limiter = LeakyBucketLimiter(capacity=2, leak_rate=0.0)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire().allowed is False

    def test_leak(self):
        limiter = LeakyBucketLimiter(capacity=1, leak_rate=100.0)
        limiter.acquire()
        time.sleep(0.05)
        assert limiter.acquire().allowed is True


# ============================================================
# RateLimiter 统一入口测试
# ============================================================

class TestRateLimiter:
    def test_token_bucket(self):
        rl = create_rate_limiter(RateLimitAlgorithm.TOKEN_BUCKET, capacity=5, refill_rate=1.0)
        assert rl.algorithm == RateLimitAlgorithm.TOKEN_BUCKET
        assert rl.acquire().allowed is True

    def test_sliding_window(self):
        rl = create_rate_limiter(RateLimitAlgorithm.SLIDING_WINDOW, max_requests=5, window_seconds=1.0)
        assert rl.acquire().allowed is True

    def test_fixed_window(self):
        rl = create_rate_limiter(RateLimitAlgorithm.FIXED_WINDOW, max_requests=5, window_seconds=1.0)
        assert rl.acquire().allowed is True

    def test_leaky_bucket(self):
        rl = create_rate_limiter(RateLimitAlgorithm.LEAKY_BUCKET, capacity=5, leak_rate=1.0)
        assert rl.acquire().allowed is True

    def test_unknown_algorithm(self):
        with pytest.raises(ValueError):
            RateLimiter("unknown_algo")


# ============================================================
# 熔断器测试
# ============================================================

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = create_circuit_breaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_when_closed(self):
        cb = create_circuit_breaker("test")
        assert cb.allow_request() is True

    def test_open_after_failures(self):
        config = CircuitBreakerConfig(failure_threshold=3, failure_rate_threshold=0.5)
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reject_when_open(self):
        config = CircuitBreakerConfig(failure_threshold=1, failure_rate_threshold=0.5)
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.1,
            failure_rate_threshold=0.5,
        )
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_close_after_success_in_half_open(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2,
            timeout_seconds=0.05, failure_rate_threshold=0.5,
        )
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopen_on_failure_in_half_open(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.05,
            failure_rate_threshold=0.5,
        )
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stats(self):
        cb = create_circuit_breaker("test")
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        stats = cb.stats
        assert stats.total_calls == 3
        assert stats.success_calls == 2
        assert stats.failure_calls == 1

    def test_reset(self):
        config = CircuitBreakerConfig(failure_threshold=1, failure_rate_threshold=0.5)
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_state_listener(self):
        events = []
        cb = create_circuit_breaker("test")
        cb.add_state_listener(lambda name, old, new: events.append((old, new)))
        config = CircuitBreakerConfig(failure_threshold=1, failure_rate_threshold=0.5)
        cb2 = CircuitBreaker("test2", config)
        cb2.add_state_listener(lambda name, old, new: events.append((old, new)))
        cb2.record_failure()
        assert len(events) == 1
        assert events[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_slow_call_triggers_open(self):
        config = CircuitBreakerConfig(
            failure_threshold=3, slow_call_rate_threshold=0.5,
        )
        cb = CircuitBreaker("test", config)
        cb.record_success()
        cb.record_slow_call()
        cb.record_slow_call()
        assert cb.state == CircuitState.OPEN

    def test_half_open_max_calls(self):
        config = CircuitBreakerConfig(
            failure_threshold=1, timeout_seconds=0.05,
            half_open_max_calls=1, failure_rate_threshold=0.5,
        )
        cb = CircuitBreaker("test", config)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True   # 第1个试探
        assert cb.allow_request() is False  # 超过最大试探数


# ============================================================
# 隔离舱测试
# ============================================================

class TestBulkhead:
    def test_try_acquire_within_limit(self):
        bh = create_bulkhead("test", BulkheadConfig(max_concurrent_calls=3))
        assert bh.try_acquire() is True

    def test_try_acquire_exhausted(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=1, max_pending_calls=0))
        bh.try_acquire()
        assert bh.try_acquire() is False

    def test_release(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=1))
        bh.try_acquire()
        assert bh.try_acquire() is False
        bh.release()
        assert bh.try_acquire() is True

    def test_stats(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=2))
        bh.try_acquire()
        bh.try_acquire()
        bh.try_acquire()  # rejected
        stats = bh.stats
        assert stats.active_calls == 2
        assert stats.total_accepted == 2
        assert stats.total_rejected == 1

    def test_state_healthy(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=5))
        assert bh.get_state() == BulkheadState.HEALTHY

    def test_state_rejected(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=1, max_pending_calls=0))
        bh.try_acquire()
        assert bh.get_state() == BulkheadState.REJECTED

    def test_concurrent_access(self):
        bh = Bulkhead("test", BulkheadConfig(max_concurrent_calls=5))
        lock = threading.Lock()
        results = []

        def worker():
            if bh.try_acquire():
                time.sleep(0.05)  # 保持占用足够长
                bh.release()
                with lock:
                    results.append(True)
            else:
                with lock:
                    results.append(False)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 至少有一些被接受，至少有一些被拒绝
        accepted = sum(results)
        rejected = len(results) - accepted
        assert accepted >= 1
        assert rejected >= 1


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    def test_create_rate_limiter(self):
        rl = create_rate_limiter(RateLimitAlgorithm.TOKEN_BUCKET, capacity=10)
        assert isinstance(rl, RateLimiter)

    def test_create_circuit_breaker(self):
        cb = create_circuit_breaker("test")
        assert isinstance(cb, CircuitBreaker)

    def test_create_bulkhead(self):
        bh = create_bulkhead("test")
        assert isinstance(bh, Bulkhead)


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    def test_rate_limit_with_circuit_breaker(self):
        """限流+熔断联合使用"""
        rl = create_rate_limiter(RateLimitAlgorithm.TOKEN_BUCKET, capacity=5, refill_rate=1.0)
        cb = create_circuit_breaker("api", CircuitBreakerConfig(failure_threshold=3, failure_rate_threshold=0.5))

        # 模拟请求处理
        for i in range(10):
            if not rl.acquire().allowed:
                continue
            if not cb.allow_request():
                continue
            if i < 7:
                cb.record_success()
            else:
                cb.record_failure()

        stats = cb.stats
        assert stats.total_calls > 0

    def test_bulkhead_with_circuit_breaker(self):
        """隔离舱+熔断联合使用"""
        bh = create_bulkhead("api", BulkheadConfig(max_concurrent_calls=3))
        cb = create_circuit_breaker("api", CircuitBreakerConfig(failure_threshold=2, failure_rate_threshold=0.5))

        results = []
        for i in range(5):
            if not bh.try_acquire():
                results.append("rejected")
                continue
            if not cb.allow_request():
                bh.release()
                results.append("circuit_open")
                continue
            bh.release()
            if i < 2:
                cb.record_success()
                results.append("success")
            else:
                cb.record_failure()
                results.append("failure")

        assert "success" in results
        assert "failure" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
