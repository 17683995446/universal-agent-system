"""
阶段50: Agent批处理系统

遵循"图难于其易，为大于其细"的原则，实现批量任务定义、分片、执行、进度追踪。
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchItem:
    id: str
    data: Any = None
    status: BatchStatus = BatchStatus.PENDING
    result: Any = None
    error: str = ""

@dataclass
class BatchResult:
    success: bool
    total: int = 0
    completed: int = 0
    failed: int = 0
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)

class BatchProcessor:
    """批处理器"""
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._lock = threading.RLock()

    def process(self, items: List[Any],
                handler: Callable[[Any], Any],
                progress_callback: Optional[Callable[[int, int], None]] = None) -> BatchResult:
        start = time.time()
        total = len(items)
        completed = 0
        failed = 0
        errors = []

        batches = self._split(items)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {}
            for batch in batches:
                f = executor.submit(self._process_batch, batch, handler)
                futures[f] = len(batch)

            for future in as_completed(futures):
                try:
                    results = future.result()
                    completed += results[0]
                    failed += results[1]
                    errors.extend(results[2])
                except Exception as e:
                    failed += futures[future]
                    errors.append(str(e))

                if progress_callback:
                    progress_callback(completed + failed, total)

        duration = time.time() - start
        return BatchResult(
            success=failed == 0,
            total=total, completed=completed, failed=failed,
            duration=duration, errors=errors,
        )

    def process_sequential(self, items: List[Any],
                           handler: Callable[[Any], Any]) -> BatchResult:
        start = time.time()
        completed = 0
        failed = 0
        errors = []

        for item in items:
            try:
                handler(item)
                completed += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))

        return BatchResult(
            success=failed == 0,
            total=len(items), completed=completed, failed=failed,
            duration=time.time() - start, errors=errors,
        )

    def _split(self, items: List[Any]) -> List[List[Any]]:
        return [items[i:i+self._batch_size] for i in range(0, len(items), self._batch_size)]

    def _process_batch(self, batch: List[Any],
                       handler: Callable[[Any], Any]) -> tuple:
        completed = 0
        failed = 0
        errors = []
        for item in batch:
            try:
                handler(item)
                completed += 1
            except Exception as e:
                failed += 1
                errors.append(str(e))
        return (completed, failed, errors)

def create_batch_processor(batch_size: int = 100, max_workers: int = 4) -> BatchProcessor:
    return BatchProcessor(batch_size, max_workers)
