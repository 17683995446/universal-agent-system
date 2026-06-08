"""
阶段48: Agent管道编排系统

遵循"图难于其易，为大于其细"的原则，实现管道定义、步骤编排、并行执行、错误处理。
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PipelineMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

@dataclass
class PipelineContext:
    """管道上下文，在步骤间传递数据"""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

@dataclass
class PipelineStep:
    name: str
    handler: Callable[[PipelineContext], bool]
    timeout: float = 30.0
    retry_count: int = 0
    retry_delay: float = 1.0
    continue_on_error: bool = False
    condition: Optional[Callable[[PipelineContext], bool]] = None

@dataclass
class StepResult:
    name: str
    status: StepStatus = StepStatus.PENDING
    duration: float = 0.0
    error: str = ""
    retries: int = 0

@dataclass
class PipelineResult:
    success: bool
    steps: List[StepResult] = field(default_factory=list)
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)

class Pipeline:
    """管道"""
    def __init__(self, name: str, mode: PipelineMode = PipelineMode.SEQUENTIAL):
        self.name = name
        self.mode = mode
        self._steps: List[PipelineStep] = []
        self._lock = threading.RLock()

    def add_step(self, step: PipelineStep) -> "Pipeline":
        self._steps.append(step)
        return self

    def add_handler(self, name: str, handler: Callable[[PipelineContext], bool],
                     timeout: float = 30.0, continue_on_error: bool = False) -> "Pipeline":
        step = PipelineStep(name=name, handler=handler, timeout=timeout,
                            continue_on_error=continue_on_error)
        self._steps.append(step)
        return self

    def execute(self, context: Optional[PipelineContext] = None,
                max_workers: int = 4) -> PipelineResult:
        ctx = context or PipelineContext()
        results: List[StepResult] = []
        start = time.time()

        if self.mode == PipelineMode.SEQUENTIAL:
            results = self._execute_sequential(ctx)
        else:
            results = self._execute_parallel(ctx, max_workers)

        duration = time.time() - start
        success = all(r.status == StepStatus.COMPLETED for r in results)
        errors = [r.error for r in results if r.error]

        return PipelineResult(
            success=success, steps=results,
            duration=duration, errors=errors,
        )

    def _execute_sequential(self, ctx: PipelineContext) -> List[StepResult]:
        results = []
        for step in self._steps:
            if step.condition and not step.condition(ctx):
                results.append(StepResult(name=step.name, status=StepStatus.SKIPPED))
                continue
            result = self._run_step(step, ctx)
            results.append(result)
            if result.status == StepStatus.FAILED and not step.continue_on_error:
                # 跳过后续步骤
                for remaining in self._steps[self._steps.index(step)+1:]:
                    results.append(StepResult(name=remaining.name, status=StepStatus.SKIPPED))
                break
        return results

    def _execute_parallel(self, ctx: PipelineContext, max_workers: int) -> List[StepResult]:
        results = []
        runnable = [s for s in self._steps if not s.condition or s.condition(ctx)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._run_step, s, ctx): s for s in runnable}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _run_step(self, step: PipelineStep, ctx: PipelineContext) -> StepResult:
        retries = 0
        for attempt in range(step.retry_count + 1):
            start = time.time()
            try:
                success = step.handler(ctx)
                duration = time.time() - start
                if success:
                    return StepResult(name=step.name, status=StepStatus.COMPLETED,
                                      duration=duration, retries=retries)
                else:
                    return StepResult(name=step.name, status=StepStatus.FAILED,
                                      duration=duration, error="处理器返回False",
                                      retries=retries)
            except Exception as e:
                retries += 1
                if retries <= step.retry_count:
                    time.sleep(step.retry_delay)
                    continue
                duration = time.time() - start
                return StepResult(name=step.name, status=StepStatus.FAILED,
                                  duration=duration, error=str(e), retries=retries)
        return StepResult(name=step.name, status=StepStatus.FAILED, error="重试耗尽")

def create_pipeline(name: str, mode: PipelineMode = PipelineMode.SEQUENTIAL) -> Pipeline:
    return Pipeline(name, mode)
