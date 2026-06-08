# =============================================================================
# API网关测试 - 阶段29测试
# =============================================================================

import time
import pytest
from unittest.mock import Mock

from src.agent.api_gateway import (
    ApiGateway,
    Route,
    ApiRequest,
    ApiResponse,
    HttpMethod,
    Middleware,
    MiddlewareType,
    RateLimiter,
)


class TestHttpMethod:
    """HTTP方法枚举测试"""

    def test_enum_values(self):
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.PUT.value == "PUT"
        assert HttpMethod.DELETE.value == "DELETE"
        assert HttpMethod.PATCH.value == "PATCH"
        assert HttpMethod.HEAD.value == "HEAD"
        assert HttpMethod.OPTIONS.value == "OPTIONS"


class TestApiRequest:
    """API请求测试"""

    def test_creation(self):
        req = ApiRequest(
            method=HttpMethod.GET,
            path="/api/agents",
            headers={"Authorization": "Bearer token"},
            query_params={"page": "1"},
            body=None,
            client_id="client_1",
        )
        assert req.method == HttpMethod.GET
        assert req.path == "/api/agents"
        assert req.request_id  # 自动生成

    def test_creation_with_string_method(self):
        req = ApiRequest(method="POST", path="/api/data")
        assert req.method == HttpMethod.POST

    def test_get_header(self):
        req = ApiRequest(headers={"Content-Type": "application/json"})
        assert req.get_header("Content-Type") == "application/json"
        assert req.get_header("Missing", "default") == "default"

    def test_get_query(self):
        req = ApiRequest(query_params={"page": "2", "size": "10"})
        assert req.get_query("page") == "2"
        assert req.get_query("missing", "0") == "0"

    def test_to_dict(self):
        req = ApiRequest(method=HttpMethod.GET, path="/test")
        data = req.to_dict()
        assert data["method"] == "GET"
        assert data["path"] == "/test"
        assert "request_id" in data


class TestApiResponse:
    """API响应测试"""

    def test_ok(self):
        resp = ApiResponse.ok({"data": "test"}, "req_1")
        assert resp.status_code == 200
        assert resp.body == {"data": "test"}
        assert resp.request_id == "req_1"

    def test_created(self):
        resp = ApiResponse.created({"id": 1})
        assert resp.status_code == 201

    def test_bad_request(self):
        resp = ApiResponse.bad_request("Invalid input")
        assert resp.status_code == 400
        assert resp.error == "Invalid input"

    def test_unauthorized(self):
        resp = ApiResponse.unauthorized()
        assert resp.status_code == 401

    def test_forbidden(self):
        resp = ApiResponse.forbidden()
        assert resp.status_code == 403

    def test_not_found(self):
        resp = ApiResponse.not_found("Route missing")
        assert resp.status_code == 404
        assert resp.error == "Route missing"

    def test_rate_limited(self):
        resp = ApiResponse.rate_limited()
        assert resp.status_code == 429

    def test_server_error(self):
        resp = ApiResponse.server_error("DB error")
        assert resp.status_code == 500

    def test_to_dict(self):
        resp = ApiResponse.ok("data")
        data = resp.to_dict()
        assert data["status_code"] == 200
        assert data["body"] == "data"


class TestRoute:
    """路由测试"""

    def test_creation(self):
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="list_agents",
            method=HttpMethod.GET,
            path_pattern="/api/agents",
            handler=handler,
            name="List Agents",
        )
        assert route.route_id == "list_agents"
        assert route.method == HttpMethod.GET
        assert route.name == "List Agents"

    def test_match_exact(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents",
            handler=handler,
        )
        result = route.match(HttpMethod.GET, "/api/agents")
        assert result is not None
        assert result == {}

    def test_match_with_params(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents/{agent_id}",
            handler=handler,
        )
        result = route.match(HttpMethod.GET, "/api/agents/123")
        assert result is not None
        assert result == {"agent_id": "123"}

    def test_match_multiple_params(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents/{agent_id}/tasks/{task_id}",
            handler=handler,
        )
        result = route.match(HttpMethod.GET, "/api/agents/abc/tasks/xyz")
        assert result == {"agent_id": "abc", "task_id": "xyz"}

    def test_no_match_wrong_method(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents",
            handler=handler,
        )
        result = route.match(HttpMethod.POST, "/api/agents")
        assert result is None

    def test_no_match_wrong_path(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents",
            handler=handler,
        )
        result = route.match(HttpMethod.GET, "/api/tasks")
        assert result is None

    def test_disabled_route(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents",
            handler=handler,
            enabled=False,
        )
        result = route.match(HttpMethod.GET, "/api/agents")
        assert result is None

    def test_to_dict(self):
        handler = Mock()
        route = Route(
            route_id="r1",
            method=HttpMethod.POST,
            path_pattern="/api/agents",
            handler=handler,
            rate_limit=100,
        )
        data = route.to_dict()
        assert data["route_id"] == "r1"
        assert data["method"] == "POST"
        assert data["rate_limit"] == 100


class TestMiddleware:
    """中间件测试"""

    def test_process_before(self):
        handler = Mock(return_value=None)
        mw = Middleware(
            middleware_id="auth",
            name="Auth",
            handler=handler,
        )
        req = ApiRequest()
        result = mw.process_before(req)
        assert result is None
        handler.assert_called_once_with(req)

    def test_process_before_returns_response(self):
        handler = Mock(return_value=ApiResponse.unauthorized())
        mw = Middleware(
            middleware_id="auth",
            name="Auth",
            handler=handler,
        )
        req = ApiRequest()
        result = mw.process_before(req)
        assert result is not None
        assert result.status_code == 401

    def test_process_after(self):
        after_handler = Mock(side_effect=lambda r: r)
        mw = Middleware(
            middleware_id="log",
            name="Log",
            handler=Mock(),
            after_handler=after_handler,
        )
        resp = ApiResponse.ok()
        result = mw.process_after(resp)
        assert result.status_code == 200
        after_handler.assert_called_once_with(resp)

    def test_disabled_middleware(self):
        handler = Mock(return_value=ApiResponse.unauthorized())
        mw = Middleware(
            middleware_id="auth",
            name="Auth",
            handler=handler,
            enabled=False,
        )
        req = ApiRequest()
        result = mw.process_before(req)
        assert result is None  # 禁用时跳过


class TestRateLimiter:
    """限流器测试"""

    def test_allow_within_limit(self):
        limiter = RateLimiter()
        allowed, count = limiter.check("key1", limit=5)
        assert allowed is True
        assert count == 1

    def test_deny_over_limit(self):
        limiter = RateLimiter()
        for _ in range(3):
            limiter.check("key1", limit=3)

        allowed, count = limiter.check("key1", limit=3)
        assert allowed is False
        assert count == 3

    def test_different_keys_independent(self):
        limiter = RateLimiter()
        for _ in range(5):
            limiter.check("key1", limit=3)

        allowed, _ = limiter.check("key2", limit=3)
        assert allowed is True

    def test_get_usage(self):
        limiter = RateLimiter()
        limiter.check("key1", limit=10)
        limiter.check("key1", limit=10)
        assert limiter.get_usage("key1") == 2

    def test_reset_key(self):
        limiter = RateLimiter()
        limiter.check("key1", limit=3)
        limiter.check("key1", limit=3)
        limiter.reset("key1")
        assert limiter.get_usage("key1") == 0

    def test_reset_all(self):
        limiter = RateLimiter()
        limiter.check("key1", limit=3)
        limiter.check("key2", limit=3)
        limiter.reset()
        assert limiter.get_usage("key1") == 0
        assert limiter.get_usage("key2") == 0

    def test_get_stats(self):
        limiter = RateLimiter()
        limiter.check("key1", limit=10)
        limiter.check("key1", limit=10)
        limiter.check("key2", limit=10)

        stats = limiter.get_stats()
        assert stats["key1"] == 2
        assert stats["key2"] == 1


class TestApiGateway:
    """API网关测试"""

    def test_creation(self):
        gw = ApiGateway("test_gw")
        assert gw.gateway_id == "test_gw"

    def test_add_and_get_route(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        found = gw.get_route("r1")
        assert found is not None
        assert found.route_id == "r1"

    def test_remove_route(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)
        result = gw.remove_route("r1")
        assert result is True
        assert gw.get_route("r1") is None

    def test_handle_request_success(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok({"result": "success"}))
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)

        assert resp.status_code == 200
        assert resp.body == {"result": "success"}
        assert resp.request_id == req.request_id
        assert resp.duration_ms > 0
        handler.assert_called_once()

    def test_handle_request_with_params(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/agents/{agent_id}",
            handler=handler,
        )
        gw.add_route(route)

        req = ApiRequest(method=HttpMethod.GET, path="/api/agents/123")
        resp = gw.handle_request(req)

        assert resp.status_code == 200
        # 验证路径参数注入
        call_args = handler.call_args[0][0]
        assert call_args.metadata["path_params"] == {"agent_id": "123"}

    def test_handle_request_not_found(self):
        gw = ApiGateway()
        req = ApiRequest(method=HttpMethod.GET, path="/api/nonexistent")
        resp = gw.handle_request(req)

        assert resp.status_code == 404

    def test_handle_request_custom_not_found(self):
        gw = ApiGateway()
        gw.set_not_found_handler(
            lambda req: ApiResponse.not_found("Custom 404", req.request_id)
        )
        req = ApiRequest(method=HttpMethod.GET, path="/missing")
        resp = gw.handle_request(req)

        assert resp.status_code == 404
        assert resp.error == "Custom 404"

    def test_handle_request_error(self):
        gw = ApiGateway()
        handler = Mock(side_effect=RuntimeError("DB Error"))
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)

        assert resp.status_code == 500
        assert "DB Error" in resp.error

    def test_handle_request_custom_error_handler(self):
        gw = ApiGateway()
        handler = Mock(side_effect=ValueError("Bad input"))
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)
        gw.set_error_handler(
            lambda req, err: ApiResponse.bad_request(str(err), req.request_id)
        )

        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)

        assert resp.status_code == 400

    def test_rate_limiting(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
            rate_limit=2,
        )
        gw.add_route(route)

        req = ApiRequest(method=HttpMethod.GET, path="/api/test", client_id="c1")

        resp1 = gw.handle_request(req)
        assert resp1.status_code == 200

        resp2 = gw.handle_request(req)
        assert resp2.status_code == 200

        resp3 = gw.handle_request(req)
        assert resp3.status_code == 429

    def test_middleware_before_blocks(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        mw = Middleware(
            middleware_id="blocker",
            name="Blocker",
            handler=lambda req: ApiResponse.forbidden("Blocked", req.request_id),
            priority=1,
        )
        gw.add_middleware(mw)

        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)

        assert resp.status_code == 403
        handler.assert_not_called()  # 路由处理器未被调用

    def test_middleware_after_modifies(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok({"data": "test"}))
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        def add_header(resp):
            resp.headers["X-Custom"] = "value"
            return resp

        mw = Middleware(
            middleware_id="header_adder",
            name="Header Adder",
            handler=Mock(return_value=None),
            after_handler=add_header,
        )
        gw.add_middleware(mw)

        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)

        assert resp.status_code == 200
        assert resp.headers["X-Custom"] == "value"

    def test_enable_disable_route(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        route = Route(
            route_id="r1",
            method=HttpMethod.GET,
            path_pattern="/api/test",
            handler=handler,
        )
        gw.add_route(route)

        gw.disable_route("r1")
        req = ApiRequest(method=HttpMethod.GET, path="/api/test")
        resp = gw.handle_request(req)
        assert resp.status_code == 404

        gw.enable_route("r1")
        resp = gw.handle_request(req)
        assert resp.status_code == 200

    def test_get_all_routes(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        gw.add_route(Route("r1", HttpMethod.GET, "/a", handler))
        gw.add_route(Route("r2", HttpMethod.POST, "/b", handler))

        routes = gw.get_all_routes()
        assert len(routes) == 2

    def test_remove_middleware(self):
        gw = ApiGateway()
        mw = Middleware("mw1", "Test", Mock(return_value=None))
        gw.add_middleware(mw)
        result = gw.remove_middleware("mw1")
        assert result is True

    def test_get_gateway_stats(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        gw.add_route(Route("r1", HttpMethod.GET, "/test", handler))

        req = ApiRequest(method=HttpMethod.GET, path="/test")
        gw.handle_request(req)

        stats = gw.get_gateway_stats()
        assert stats["routes_count"] == 1
        assert stats["total_requests"] == 1
        assert 200 in stats["status_distribution"]

    def test_get_request_log(self):
        gw = ApiGateway()
        handler = Mock(return_value=ApiResponse.ok())
        gw.add_route(Route("r1", HttpMethod.GET, "/test", handler))

        gw.handle_request(ApiRequest(method=HttpMethod.GET, path="/test"))
        gw.handle_request(ApiRequest(method=HttpMethod.GET, path="/missing"))

        logs = gw.get_request_log()
        assert len(logs) == 2

        logs_404 = gw.get_request_log(status_code=404)
        assert len(logs_404) == 1


class TestExportApiGateway:
    """导出测试"""

    def test_api_gateway_exported(self):
        from src.agent import ApiGateway
        assert ApiGateway is not None

    def test_route_exported(self):
        from src.agent import Route
        assert Route is not None

    def test_api_request_exported(self):
        from src.agent import ApiRequest
        assert ApiRequest is not None

    def test_api_response_exported(self):
        from src.agent import ApiResponse
        assert ApiResponse is not None

    def test_http_method_exported(self):
        from src.agent import HttpMethod
        assert HttpMethod.GET is not None

    def test_middleware_exported(self):
        from src.agent import Middleware
        assert Middleware is not None

    def test_middleware_type_exported(self):
        from src.agent import MiddlewareType
        assert MiddlewareType.BEFORE is not None

    def test_rate_limiter_exported(self):
        from src.agent import RateLimiter
        assert RateLimiter is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])