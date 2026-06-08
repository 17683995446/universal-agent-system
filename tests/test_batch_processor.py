"""阶段50: Agent批处理系统 - 单元测试"""
import pytest
from src.agent.batch_processor import (
    BatchStatus, BatchItem, BatchResult, BatchProcessor, create_batch_processor,
)

class TestBatchProcessor:
    def test_process_sequential(self):
        bp = BatchProcessor()
        results = []
        r = bp.process_sequential([1, 2, 3], lambda x: results.append(x * 2))
        assert r.success and r.completed == 3
        assert results == [2, 4, 6]

    def test_process_parallel(self):
        bp = BatchProcessor(batch_size=2, max_workers=2)
        results = []
        r = bp.process([1, 2, 3, 4], lambda x: results.append(x * 10))
        assert r.success and r.completed == 4

    def test_handle_errors(self):
        bp = BatchProcessor()
        r = bp.process_sequential([1, "bad", 3], lambda x: x * 2 if isinstance(x, int) else (_ for _ in ()).throw(ValueError("err")))
        assert not r.success
        assert r.completed == 2
        assert r.failed == 1

    def test_empty_batch(self):
        bp = BatchProcessor()
        r = bp.process_sequential([], lambda x: None)
        assert r.success and r.total == 0

    def test_progress_callback(self):
        progress = []
        bp = BatchProcessor(batch_size=2, max_workers=2)
        bp.process([1, 2, 3, 4], lambda x: None,
                   progress_callback=lambda done, total: progress.append((done, total)))
        assert len(progress) > 0

    def test_create_batch_processor(self):
        bp = create_batch_processor(50, 2)
        assert isinstance(bp, BatchProcessor)
