# 阶段29: Agent API网关 (Agent API Gateway)

## 概述

阶段29实现了Agent API网关，统一管理请求路由、限流、中间件。系统提供灵活的路由匹配、滑动窗口限流、前置/后置中间件、请求日志等功能。

## 核心组件

### 1. HttpMethod (HTTP方法枚举)

支持 GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS。

### 2. ApiRequest / ApiResponse

请求和响应数据类：

```python
from src.agent import ApiRequest, ApiResponse, HttpMethod

# 创建请求
req = ApiRequest(
    method=HttpMethod.GET,
    path="/api/agents",
    headers={"Authorization": "Bearer token"},
    query_params={"page": "1"},
    client_id="client_1",
)

# 创建响应
resp = ApiResponse.ok({"agents": []}, req.request_id)
resp = ApiResponse.not_found("Route missing")
resp = ApiResponse.rate_limited("Too many requests")
```

### 3. Route (路由)

支持路径参数匹配：

```python
from src.agent import Route, HttpMethod, ApiResponse

def list_agents(req):
    return ApiResponse.ok({"agents": []})

route = Route(
    route_id="list_agents",
    method=HttpMethod.GET,
    path_pattern="/api/agents",
    handler=list_agents,
    rate_limit=100,  # 每分钟100次
)
```

### 4. Middleware (中间件)

前置和后置处理：

```python
from src.agent import Middleware

# 认证中间件
def auth_check(req):
    if not req.get_header("Authorization"):
        return ApiResponse.unauthorized("Missing token", req.request_id)
    return None  # 通过

auth_mw = Middleware(
    middleware_id="auth",
    name="Authentication",
    handler=auth_check,
    priority=1,  # 高优先级
)

# 日志中间件（后置）
def add_log_header(resp):
    resp.headers["X-Logged"] = "true"
    return resp

log_mw = Middleware(
    middleware_id="logger",
    name="Logger",
    handler=lambda req: None,
    after_handler=add_log_header,
)
```

### 5. RateLimiter (限流器)

基于滑动窗口的限流：

```python
from src.agent import RateLimiter

limiter = RateLimiter()

# 检查限流
allowed, count = limiter.check("client_1", limit=60, window_seconds=60)
# allowed=True, count=1

# 获取使用量
usage = limiter.get_usage("client_1")

# 重置
limiter.reset("client_1")
```

### 6. ApiGateway (API网关)

统一管理入口：

```python
from src.agent import ApiGateway, Route, ApiRequest, HttpMethod, ApiResponse

gw = ApiGateway("agent_api")

# 添加路由
gw.add_route(Route("list", HttpMethod.GET, "/api/agents", list_handler))
gw.add_route(Route("get", HttpMethod.GET, "/api/agents/{id}", get_handler))
gw.add_route(Route("create", HttpMethod.POST, "/api/agents", create_handler, rate_limit=30))

# 添加中间件
gw.add_middleware(auth_middleware)

# 处理请求
req = ApiRequest(method=HttpMethod.GET, path="/api/agents/123")
resp = gw.handle_request(req)

# 自定义处理器
gw.set_not_found_handler(lambda req: ApiResponse.not_found("Custom 404"))
gw.set_error_handler(lambda req, err: ApiResponse.server_error(str(err)))

# 统计
stats = gw.get_gateway_stats()
logs = gw.get_request_log(status_code=404)
```

## 请求处理流程

```
请求 → 前置中间件 → 路由匹配 → 限流检查 → 处理器 → 后置中间件 → 响应
```

## 测试覆盖

- 59个测试用例全部通过
- 覆盖路由匹配、参数提取、限流、中间件、错误处理、日志等

## 总结

阶段29成功实现了Agent API网关，提供了：
- 灵活的路由匹配（支持路径参数）
- 滑动窗口限流
- 前置/后置中间件
- 请求日志和统计
- 自定义错误处理
- 线程安全的实现