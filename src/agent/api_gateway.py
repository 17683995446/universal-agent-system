# =============================================================================
# Agent API网关 - 阶段29: Agent API Gateway
# =============================================================================
#
# 功能：实现Agent API网关，统一管理请求路由、限流、中间件
# 原则：图难于其易，为大于其细
#
# =============================================================================

from __future__ import annotations

import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


# =============================================================================
# HTTP方法
# =============================================================================

class HttpMethod(str, Enum):
    """HTTP方法枚举"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


# =============================================================================
# API请求
# =============================================================================

@dataclass
class ApiRequest:
    """
    API请求
    
    表示一个进入网关的请求
    """
    request_id: str = ""
    method: HttpMethod = HttpMethod.GET
    path: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)
    body: Any = None
    client_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req_{uuid.uuid4().hex[:8]}"
        if isinstance(self.method, str):
            self.method = HttpMethod(self.method.upper())

    def get_header(self, name: str, default: str = "") -> str:
        """获取请求头"""
        return self.headers.get(name, default)

    def get_query(self, name: str, default: str = "") -> str:
        """获取查询参数"""
        return self.query_params.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "body": str(self.body) if self.body else None,
            "client_id": self.client_id,
            "timestamp": self.timestamp,
        }


# =============================================================================
# API响应
# =============================================================================

@dataclass
class ApiResponse:
    """
    API响应
    
    表示从网关返回的响应
    """
    status_code: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    request_id: str = ""
    duration_ms: float = 0.0
    error: Optional[str] = None

    @staticmethod
    def ok(body: Any = None, request_id: str = "") -> ApiResponse:
        """创建成功响应"""
        return ApiResponse(
            status_code=200,
            body=body,
            request_id=request_id,
        )

    @staticmethod
    def created(body: Any = None, request_id: str = "") -> ApiResponse:
        """创建资源响应"""
        return ApiResponse(
            status_code=201,
            body=body,
            request_id=request_id,
        )

    @staticmethod
    def bad_request(message: str = "Bad Request", request_id: str = "") -> ApiResponse:
        """创建400响应"""
        return ApiResponse(
            status_code=400,
            error=message,
            request_id=request_id,
        )

    @staticmethod
    def unauthorized(message: str = "Unauthorized", request_id: str = "") -> ApiResponse:
        """创建401响应"""
        return ApiResponse(
            status_code=401,
            error=message,
            request_id=request_id,
        )

    @staticmethod
    def forbidden(message: str = "Forbidden", request_id: str = "") -> ApiResponse:
        """创建403响应"""
        return ApiResponse(
            status_code=403,
            error=message,
            request_id=request_id,
        )

    @staticmethod
    def not_found(message: str = "Not Found", request_id: str = "") -> ApiResponse:
        """创建404响应"""
        return ApiResponse(
            status_code=404,
            error=message,
            request_id=request_id,
        )

    @staticmethod
    def rate_limited(message: str = "Rate Limited", request_id: str = "") -> ApiResponse:
        """创建429响应"""
        return ApiResponse(
            status_code=429,
            error=message,
            request_id=request_id,
        )

    @staticmethod
    def server_error(message: str = "Internal Server Error", request_id: str = "") -> ApiResponse:
        """创建500响应"""
        return ApiResponse(
            status_code=500,
            error=message,
            request_id=request_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "status_code": self.status_code,
            "body": self.body,
            "headers": self.headers,
            "request_id": self.request_id,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# =============================================================================
# 路由定义
# =============================================================================

@dataclass
class Route:
    """
    路由定义
    
    定义一个API路由规则
    """
    route_id: str
    method: HttpMethod
    path_pattern: str          # 路径模式，支持 {param} 参数
    handler: Callable[[ApiRequest], ApiResponse]
    name: str = ""
    description: str = ""
    rate_limit: Optional[int] = None  # 每分钟请求限制，None表示不限
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.route_id
        if isinstance(self.method, str):
            self.method = HttpMethod(self.method.upper())
        # 编译路径模式为正则
        self._compiled_pattern = self._compile_pattern(self.path_pattern)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """将路径模式编译为正则表达式"""
        # 将 {param} 转换为命名捕获组
        regex = re.sub(r'\{(\w+)\}', r'(?P<\1>[^/]+)', pattern)
        return re.compile(f"^{regex}$")

    def match(self, method: HttpMethod, path: str) -> Optional[dict[str, str]]:
        """
        匹配请求
        
        Args:
            method: HTTP方法
            path: 请求路径
        
        Returns:
            匹配成功返回参数字典，否则返回None
        """
        if not self.enabled:
            return None
        if self.method != method:
            return None
        m = self._compiled_pattern.match(path)
        if m:
            return m.groupdict()
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "route_id": self.route_id,
            "method": self.method.value,
            "path_pattern": self.path_pattern,
            "name": self.name,
            "description": self.description,
            "rate_limit": self.rate_limit,
            "enabled": self.enabled,
        }


# =============================================================================
# 中间件
# =============================================================================

class MiddlewareType(str, Enum):
    """中间件类型"""
    BEFORE = "before"     # 前置中间件
    AFTER = "after"       # 后置中间件
    AROUND = "around"     # 环绕中间件


@dataclass
class Middleware:
    """
    中间件
    
    在请求处理前后执行的自定义逻辑
    """
    middleware_id: str
    name: str
    handler: Callable[[ApiRequest], Optional[ApiResponse]]  # 前置
    after_handler: Optional[Callable[[ApiResponse], ApiResponse]] = None  # 后置
    middleware_type: MiddlewareType = MiddlewareType.BEFORE
    priority: int = 100  # 数字越小优先级越高
    enabled: bool = True

    def process_before(self, request: ApiRequest) -> Optional[ApiResponse]:
        """前置处理"""
        if not self.enabled:
            return None
        try:
            return self.handler(request)
        except Exception:
            return None

    def process_after(self, response: ApiResponse) -> ApiResponse:
        """后置处理"""
        if not self.enabled or not self.after_handler:
            return response
        try:
            return self.after_handler(response)
        except Exception:
            return response


# =============================================================================
# 限流器
# =============================================================================

class RateLimiter:
    """
    限流器
    
    基于滑动窗口的请求限流
    """

    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def check(self, key: str, limit: int, window_seconds: float = 60.0) -> tuple[bool, int]:
        """
        检查是否允许请求
        
        Args:
            key: 限流键（如 client_id 或 route_id）
            limit: 窗口内最大请求数
            window_seconds: 窗口时间（秒）
        
        Returns:
            (是否允许, 当前窗口请求数)
        """
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            # 清理过期记录
            self._windows[key] = [
                t for t in self._windows[key] if t > cutoff
            ]

            current_count = len(self._windows[key])
            if current_count >= limit:
                return (False, current_count)

            self._windows[key].append(now)
            return (True, current_count + 1)

    def get_usage(self, key: str, window_seconds: float = 60.0) -> int:
        """获取当前窗口使用量"""
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            self._windows[key] = [
                t for t in self._windows[key] if t > cutoff
            ]
            return len(self._windows[key])

    def reset(self, key: Optional[str] = None) -> None:
        """重置限流计数"""
        with self._lock:
            if key:
                self._windows.pop(key, None)
            else:
                self._windows.clear()

    def get_stats(self) -> dict[str, int]:
        """获取限流统计"""
        with self._lock:
            return {k: len(v) for k, v in self._windows.items()}


# =============================================================================
# API网关
# =============================================================================

class ApiGateway:
    """
    API网关
    
    统一管理请求路由、限流、中间件
    
    Attributes:
        routes: 路由列表
        middlewares: 中间件列表
        rate_limiter: 限流器
    """

    def __init__(self, gateway_id: str = "default"):
        self._gateway_id = gateway_id
        self._routes: list[Route] = []
        self._middlewares: list[Middleware] = []
        self._rate_limiter = RateLimiter()
        self._lock = threading.RLock()
        self._request_log: list[dict[str, Any]] = []
        self._max_log_size = 1000
        self._not_found_handler: Optional[Callable[[ApiRequest], ApiResponse]] = None
        self._error_handler: Optional[Callable[[ApiRequest, Exception], ApiResponse]] = None

    @property
    def gateway_id(self) -> str:
        return self._gateway_id

    # =====================================================================
    # 路由管理
    # =====================================================================

    def add_route(self, route: Route) -> None:
        """添加路由"""
        with self._lock:
            self._routes.append(route)

    def remove_route(self, route_id: str) -> bool:
        """移除路由"""
        with self._lock:
            for i, route in enumerate(self._routes):
                if route.route_id == route_id:
                    self._routes.pop(i)
                    return True
            return False

    def get_route(self, route_id: str) -> Optional[Route]:
        """获取路由"""
        with self._lock:
            for route in self._routes:
                if route.route_id == route_id:
                    return route
            return None

    def get_all_routes(self) -> list[Route]:
        """获取所有路由"""
        with self._lock:
            return self._routes.copy()

    def enable_route(self, route_id: str) -> bool:
        """启用路由"""
        route = self.get_route(route_id)
        if route:
            route.enabled = True
            return True
        return False

    def disable_route(self, route_id: str) -> bool:
        """禁用路由"""
        route = self.get_route(route_id)
        if route:
            route.enabled = False
            return True
        return False

    # =====================================================================
    # 中间件管理
    # =====================================================================

    def add_middleware(self, middleware: Middleware) -> None:
        """添加中间件"""
        with self._lock:
            self._middlewares.append(middleware)
            self._middlewares.sort(key=lambda m: m.priority)

    def remove_middleware(self, middleware_id: str) -> bool:
        """移除中间件"""
        with self._lock:
            for i, mw in enumerate(self._middlewares):
                if mw.middleware_id == middleware_id:
                    self._middlewares.pop(i)
                    return True
            return False

    # =====================================================================
    # 请求处理
    # =====================================================================

    def handle_request(self, request: ApiRequest) -> ApiResponse:
        """
        处理请求
        
        Args:
            request: API请求
        
        Returns:
            API响应
        """
        start_time = time.time()

        # 1. 执行前置中间件
        for mw in self._middlewares:
            result = mw.process_before(request)
            if result is not None:
                return self._finalize(result, request, start_time)

        # 2. 匹配路由
        route, params = self._match_route(request.method, request.path)

        if route is None:
            response = self._handle_not_found(request)
            return self._finalize_with_after(response, request, start_time)

        # 3. 限流检查
        if route.rate_limit is not None:
            client_key = request.client_id or request.request_id
            rl_key = f"{route.route_id}:{client_key}"
            allowed, count = self._rate_limiter.check(rl_key, route.rate_limit)
            if not allowed:
                response = ApiResponse.rate_limited(
                    f"Rate limit exceeded: {route.rate_limit}/min",
                    request.request_id,
                )
                return self._finalize_with_after(response, request, start_time)

        # 4. 将路径参数注入请求元数据
        request.metadata["path_params"] = params

        # 5. 执行路由处理器
        try:
            response = route.handler(request)
        except Exception as e:
            response = self._handle_error(request, e)

        return self._finalize_with_after(response, request, start_time)

    def _match_route(
        self, method: HttpMethod, path: str,
    ) -> tuple[Optional[Route], dict[str, str]]:
        """匹配路由"""
        with self._lock:
            for route in self._routes:
                params = route.match(method, path)
                if params is not None:
                    return route, params
        return None, {}

    def _handle_not_found(self, request: ApiRequest) -> ApiResponse:
        """处理404"""
        if self._not_found_handler:
            try:
                return self._not_found_handler(request)
            except Exception:
                pass
        return ApiResponse.not_found(
            f"No route for {request.method.value} {request.path}",
            request.request_id,
        )

    def _handle_error(self, request: ApiRequest, error: Exception) -> ApiResponse:
        """处理错误"""
        if self._error_handler:
            try:
                return self._error_handler(request, error)
            except Exception:
                pass
        return ApiResponse.server_error(str(error), request.request_id)

    def _finalize(
        self, response: ApiResponse, request: ApiRequest, start_time: float,
    ) -> ApiResponse:
        """完成响应"""
        response.request_id = request.request_id
        response.duration_ms = (time.time() - start_time) * 1000
        self._log_request(request, response)
        return response

    def _finalize_with_after(
        self, response: ApiResponse, request: ApiRequest, start_time: float,
    ) -> ApiResponse:
        """完成响应（含后置中间件）"""
        response = self._finalize(response, request, start_time)
        for mw in reversed(self._middlewares):
            response = mw.process_after(response)
        return response

    def _log_request(self, request: ApiRequest, response: ApiResponse) -> None:
        """记录请求日志"""
        with self._lock:
            self._request_log.append({
                "request_id": request.request_id,
                "method": request.method.value,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": response.duration_ms,
                "client_id": request.client_id,
                "timestamp": time.time(),
            })
            if len(self._request_log) > self._max_log_size:
                self._request_log = self._request_log[-self._max_log_size:]

    # =====================================================================
    # 自定义处理器
    # =====================================================================

    def set_not_found_handler(self, handler: Callable[[ApiRequest], ApiResponse]) -> None:
        """设置404处理器"""
        self._not_found_handler = handler

    def set_error_handler(self, handler: Callable[[ApiRequest, Exception], ApiResponse]) -> None:
        """设置错误处理器"""
        self._error_handler = handler

    # =====================================================================
    # 统计
    # =====================================================================

    def get_gateway_stats(self) -> dict[str, Any]:
        """获取网关统计"""
        with self._lock:
            status_counts: dict[int, int] = defaultdict(int)
            for log in self._request_log:
                status_counts[log["status_code"]] += 1

            return {
                "gateway_id": self._gateway_id,
                "routes_count": len(self._routes),
                "middlewares_count": len(self._middlewares),
                "total_requests": len(self._request_log),
                "status_distribution": dict(status_counts),
                "rate_limiter_stats": self._rate_limiter.get_stats(),
            }

    def get_request_log(
        self,
        limit: int = 100,
        status_code: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """获取请求日志"""
        with self._lock:
            logs = self._request_log.copy()
        if status_code:
            logs = [l for l in logs if l["status_code"] == status_code]
        return logs[-limit:]
