"""阶段73: Agent机器学习推理服务系统 - 单元测试"""
import pytest
import time
from src.agent.inference_service import (
    InferenceStatus, InferenceRequest, InferenceResult,
    ModelLoader, InferenceEngine, InferenceService, create_inference_service,
)


class TestModelLoader:
    def test_load_and_get(self):
        loader = ModelLoader()
        assert loader.load("m1", "custom")
        assert loader.get("m1") is not None

    def test_unload(self):
        loader = ModelLoader()
        loader.load("m1")
        assert loader.unload("m1")
        assert loader.get("m1") is None

    def test_list_loaded(self):
        loader = ModelLoader()
        loader.load("m1")
        loader.load("m2")
        assert set(loader.list_loaded()) == {"m1", "m2"}

    def test_register_loader(self):
        loader = ModelLoader()
        loader.register_loader("dummy", lambda s: {"source": s})
        assert loader.load("m1", "dummy", "path")
        assert loader.get("m1") == {"source": "path"}


class TestInferenceEngine:
    def test_predict_not_loaded(self):
        engine = InferenceEngine()
        with pytest.raises(RuntimeError):
            engine.predict("m1", [1, 2])

    def test_predict_default(self):
        engine = InferenceEngine()
        engine.loader.load("m1")
        result = engine.predict("m1", [1, 2])
        assert result["model_id"] == "m1"

    def test_predict_custom(self):
        engine = InferenceEngine()
        engine.loader.load("m1")
        engine.register_predictor("m1", lambda model, inputs: sum(inputs))
        assert engine.predict("m1", [1, 2, 3]) == 6

    def test_infer_success(self):
        engine = InferenceEngine()
        engine.loader.load("m1")
        req = InferenceRequest(request_id="r1", model_id="m1", inputs=[1])
        result = engine.infer(req)
        assert result.status == InferenceStatus.SUCCESS
        assert result.latency_ms >= 0

    def test_infer_failed(self):
        engine = InferenceEngine()
        req = InferenceRequest(request_id="r1", model_id="m1", inputs=[1])
        result = engine.infer(req)
        assert result.status == InferenceStatus.FAILED

    def test_infer_async(self):
        engine = InferenceEngine()
        engine.loader.load("m1")
        req = InferenceRequest(request_id="r1", model_id="m1", inputs=[1], timeout=5.0)
        result = engine.infer_async(req)
        assert result.status == InferenceStatus.SUCCESS

    def test_batch_infer(self):
        engine = InferenceEngine()
        engine.loader.load("m1")
        reqs = [
            InferenceRequest(request_id="r1", model_id="m1", inputs=[1]),
            InferenceRequest(request_id="r2", model_id="m1", inputs=[2]),
        ]
        results = engine.batch_infer(reqs)
        assert len(results) == 2
        assert all(r.status == InferenceStatus.SUCCESS for r in results)

    def test_infer_timeout(self):
        engine = InferenceEngine(max_workers=1)
        engine.loader.load("m1")
        # 注册一个慢预测器
        engine.register_predictor("m1", lambda model, inputs: time.sleep(2))
        req = InferenceRequest(request_id="r1", model_id="m1", inputs=[], timeout=0.1)
        result = engine.infer_async(req)
        assert result.status == InferenceStatus.TIMEOUT


class TestInferenceService:
    def test_load_and_predict(self):
        svc = InferenceService()
        svc.load_model("m1")
        result = svc.predict("m1", [1, 2])
        assert result["inputs"] == [1, 2]
        svc.shutdown()

    def test_request(self):
        svc = InferenceService()
        svc.load_model("m1")
        result = svc.request("m1", [1, 2])
        assert result.status == InferenceStatus.SUCCESS
        svc.shutdown()

    def test_batch_request(self):
        svc = InferenceService()
        svc.load_model("m1")
        results = svc.batch_request("m1", [[1], [2], [3]])
        assert len(results) == 3
        assert all(r.status == InferenceStatus.SUCCESS for r in results)
        svc.shutdown()

    def test_stats(self):
        svc = InferenceService()
        svc.load_model("m1")
        svc.request("m1", [1])
        svc.request("m1", [2])
        stats = svc.get_stats()
        assert stats["total_requests"] == 2
        assert stats["success"] == 2
        assert stats["loaded_models"] == 1
        svc.shutdown()

    def test_create_inference_service(self):
        svc = create_inference_service()
        assert isinstance(svc, InferenceService)
        svc.shutdown()
