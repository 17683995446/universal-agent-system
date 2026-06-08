"""
阶段73: Agent机器学习推理服务系统

遵循"图难于其易，为大于其细"的原则，实现模型加载、推理请求、批处理、超时控制。
"""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class InferenceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class InferenceRequest:
    request_id: str
    model_id: str
    inputs: Any
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    status: InferenceStatus = InferenceStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class InferenceResult:
    request_id: str
    outputs: Any
    latency_ms: float
    status: InferenceStatus
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "status": self.status.value,
            "error": self.error,
        }


class ModelLoader:
    """模型加载器"""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable[[str], Any]] = {}

    def register_loader(self, framework: str, loader: Callable[[str], Any]) -> None:
        self._loaders[framework] = loader

    def load(self, model_id: str, framework: str = "custom", source: str = "") -> bool:
        loader = self._loaders.get(framework)
        if loader:
            try:
                self._models[model_id] = loader(source)
                return True
            except Exception:
                return False
        # 默认加载一个占位模型
        self._models[model_id] = {"model_id": model_id, "framework": framework}
        return True

    def get(self, model_id: str) -> Optional[Any]:
        return self._models.get(model_id)

    def unload(self, model_id: str) -> bool:
        return self._models.pop(model_id, None) is not None

    def list_loaded(self) -> List[str]:
        return list(self._models.keys())


class InferenceEngine:
    """推理引擎"""

    def __init__(self, max_workers: int = 4):
        self._loader = ModelLoader()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._predictors: Dict[str, Callable[[Any, Any], Any]] = {}

    @property
    def loader(self) -> ModelLoader:
        return self._loader

    def register_predictor(self, model_id: str, predictor: Callable[[Any, Any], Any]) -> None:
        self._predictors[model_id] = predictor

    def predict(self, model_id: str, inputs: Any) -> Any:
        model = self._loader.get(model_id)
        if model is None:
            raise RuntimeError(f"Model {model_id} not loaded")
        predictor = self._predictors.get(model_id)
        if predictor:
            return predictor(model, inputs)
        # 默认预测器: 返回输入的包装
        return {"model_id": model_id, "inputs": inputs}

    def infer(self, request: InferenceRequest) -> InferenceResult:
        start = time.time()
        request.status = InferenceStatus.RUNNING
        try:
            outputs = self.predict(request.model_id, request.inputs)
            latency = (time.time() - start) * 1000
            request.status = InferenceStatus.SUCCESS
            request.result = outputs
            request.latency_ms = latency
            return InferenceResult(
                request_id=request.request_id,
                outputs=outputs,
                latency_ms=latency,
                status=InferenceStatus.SUCCESS,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            request.status = InferenceStatus.FAILED
            request.error = str(e)
            request.latency_ms = latency
            return InferenceResult(
                request_id=request.request_id,
                outputs=None,
                latency_ms=latency,
                status=InferenceStatus.FAILED,
                error=str(e),
            )

    def infer_async(self, request: InferenceRequest) -> Any:
        future = self._executor.submit(self.infer, request)
        try:
            return future.result(timeout=request.timeout)
        except FutureTimeoutError:
            request.status = InferenceStatus.TIMEOUT
            request.error = "Inference timeout"
            return InferenceResult(
                request_id=request.request_id,
                outputs=None,
                latency_ms=request.timeout * 1000,
                status=InferenceStatus.TIMEOUT,
                error="Inference timeout",
            )

    def batch_infer(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        return [self.infer(req) for req in requests]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class InferenceService:
    """推理服务"""

    def __init__(self, max_workers: int = 4):
        self._engine = InferenceEngine(max_workers=max_workers)
        self._history: List[InferenceRequest] = []

    @property
    def engine(self) -> InferenceEngine:
        return self._engine

    def load_model(self, model_id: str, framework: str = "custom", source: str = "") -> bool:
        return self._engine.loader.load(model_id, framework, source)

    def unload_model(self, model_id: str) -> bool:
        return self._engine.loader.unload(model_id)

    def predict(self, model_id: str, inputs: Any) -> Any:
        return self._engine.predict(model_id, inputs)

    def request(self, model_id: str, inputs: Any, timeout: float = 30.0) -> InferenceResult:
        req = InferenceRequest(
            request_id=str(uuid.uuid4())[:8],
            model_id=model_id,
            inputs=inputs,
            timeout=timeout,
        )
        result = self._engine.infer_async(req)
        self._history.append(req)
        return result

    def batch_request(self, model_id: str, inputs_list: List[Any]) -> List[InferenceResult]:
        requests = [
            InferenceRequest(
                request_id=str(uuid.uuid4())[:8],
                model_id=model_id,
                inputs=inputs,
            )
            for inputs in inputs_list
        ]
        results = self._engine.batch_infer(requests)
        self._history.extend(requests)
        return results

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._history)
        success = sum(1 for r in self._history if r.status == InferenceStatus.SUCCESS)
        failed = sum(1 for r in self._history if r.status == InferenceStatus.FAILED)
        timeouts = sum(1 for r in self._history if r.status == InferenceStatus.TIMEOUT)
        latencies = [r.latency_ms for r in self._history if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "total_requests": total,
            "success": success,
            "failed": failed,
            "timeouts": timeouts,
            "avg_latency_ms": round(avg_latency, 2),
            "loaded_models": len(self._engine.loader.list_loaded()),
        }

    def shutdown(self) -> None:
        self._engine.shutdown()


def create_inference_service(max_workers: int = 4) -> InferenceService:
    return InferenceService(max_workers=max_workers)
