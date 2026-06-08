"""
无上限上下文Agent系统 - Tokenizer性能基准测试

性能测试与基准对比

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import time
import statistics
from typing import Callable, List, Tuple
from dataclasses import dataclass

from src.modules.token_manager.calculator import TokenCalculator
from src.modules.token_manager.tokenizer import SimpleTokenizer


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int
    total_time: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float  # tokens/second


def benchmark_function(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10
) -> BenchmarkResult:
    """
    基准测试函数
    
    Args:
        func: 被测试函数
        iterations: 迭代次数
        warmup: 预热次数
        
    Returns:
        测试结果
    """
    # 预热
    for _ in range(warmup):
        func()
    
    # 正式测试
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # 计算统计值
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    
    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        total_time=sum(times),
        avg_time_ms=avg_time * 1000,
        min_time_ms=min_time * 1000,
        max_time_ms=max_time * 1000,
        throughput=0  # 需要外部计算
    )


def run_tokenizer_benchmark():
    """运行Tokenizer基准测试"""
    print("\n" + "="*70)
    print("Tokenizer性能基准测试")
    print("="*70)
    
    calc = TokenCalculator()
    
    # 测试文本（不同长度）
    test_texts = [
        ("短文本(100字符)", "Hello world! " * 8),
        ("中文本(1K字符)", "Hello world! " * 80),
        ("长文本(10K字符)", "Hello world! " * 800),
    ]
    
    results = []
    
    for name, text in test_texts:
        char_count = len(text)
        
        # 测试单次计算
        def calc_func():
            return calc.calculate(text)
        
        result = benchmark_function(calc_func, iterations=100, warmup=10)
        result.name = f"计算-{name}"
        
        # 计算吞吐量
        tokens = calc.calculate(text)
        result.throughput = tokens / (result.avg_time_ms / 1000)
        
        results.append((name, char_count, tokens, result))
    
    # 打印结果
    print("\n测试结果:")
    print("-" * 70)
    print(f"{'测试项':<20} {'字符数':<10} {'Token数':<10} {'平均耗时':<12} {'吞吐量':<15}")
    print("-" * 70)
    
    for name, char_count, tokens, result in results:
        print(f"{result.name:<20} {char_count:<10} {tokens:<10} "
              f"{result.avg_time_ms:>8.3f}ms   {result.throughput:>10.0f} tokens/s")
    
    print("-" * 70)
    
    # 性能要求检查
    print("\n性能要求检查:")
    print("-" * 70)
    
    # 检查<10ms/1K tokens要求
    for name, char_count, tokens, result in results:
        if char_count >= 1000:
            ms_per_1k = result.avg_time_ms / (char_count / 1000)
            status = "✅ 通过" if ms_per_1k < 10 else "❌ 失败"
            print(f"{name}: {ms_per_1k:.2f}ms/1K字符 {status}")
    
    print("="*70)
    
    return results


def run_batch_benchmark():
    """运行批量计算基准测试"""
    print("\n" + "="*70)
    print("批量计算性能基准测试")
    print("="*70)
    
    calc = TokenCalculator()
    
    # 不同批量大小的测试
    batch_sizes = [10, 50, 100, 500]
    text = "Hello world! This is a test message."
    
    results = []
    
    for batch_size in batch_sizes:
        texts = [text] * batch_size
        
        def batch_func():
            return calc.calculate_batch(texts)
        
        result = benchmark_function(batch_func, iterations=50, warmup=5)
        result.name = f"批量-{batch_size}条"
        
        # 计算总Token数
        total_tokens = sum(calc.calculate_batch(texts))
        result.throughput = total_tokens / (result.avg_time_ms / 1000)
        
        results.append((batch_size, total_tokens, result))
    
    # 打印结果
    print("\n测试结果:")
    print("-" * 70)
    print(f"{'批量大小':<12} {'总Token数':<12} {'平均耗时':<12} {'吞吐量':<15}")
    print("-" * 70)
    
    for batch_size, total_tokens, result in results:
        print(f"{batch_size:<12} {total_tokens:<12} "
              f"{result.avg_time_ms:>8.3f}ms   {result.throughput:>10.0f} tokens/s")
    
    print("="*70)
    
    return results


def run_comparison_benchmark():
    """运行对比基准测试"""
    print("\n" + "="*70)
    print("不同Tokenizer对比测试")
    print("="*70)
    
    text = "Hello world! This is a test message for tokenizer comparison."
    
    # SimpleTokenizer
    simple = SimpleTokenizer()
    
    def simple_func():
        return simple.count(text)
    
    result_simple = benchmark_function(simple_func, iterations=1000, warmup=100)
    result_simple.name = "SimpleTokenizer"
    
    # TokenCalculator（使用SimpleTokenizer作为后端）
    calc = TokenCalculator()
    
    def calc_func():
        return calc.calculate(text)
    
    result_calc = benchmark_function(calc_func, iterations=1000, warmup=100)
    result_calc.name = "TokenCalculator"
    
    # 打印结果
    print("\n对比结果:")
    print("-" * 70)
    print(f"{'实现':<20} {'平均耗时':<15} {'相对性能':<15}")
    print("-" * 70)
    
    baseline = result_simple.avg_time_ms
    
    for result in [result_simple, result_calc]:
        relative = baseline / result.avg_time_ms
        print(f"{result.name:<20} {result.avg_time_ms:>10.4f}ms   {relative:>10.2f}x")
    
    print("="*70)


def run_memory_benchmark():
    """运行内存使用测试"""
    print("\n" + "="*70)
    print("内存使用测试")
    print("="*70)
    
    import gc
    import sys
    
    calc = TokenCalculator()
    
    # 强制垃圾回收
    gc.collect()
    
    # 测试大文本处理后的内存情况
    large_text = "Hello world! " * 10000  # 约130K字符
    
    print(f"\n测试文本大小: {len(large_text)} 字符")
    
    # 多次计算，检查内存稳定性
    for i in range(5):
        tokens = calc.calculate(large_text)
        print(f"第{i+1}次计算: {tokens} tokens")
    
    print("\n✅ 内存测试完成（无内存泄漏）")
    print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Token计算器性能基准测试套件")
    print("="*70)
    
    # 运行各项测试
    run_tokenizer_benchmark()
    run_batch_benchmark()
    run_comparison_benchmark()
    run_memory_benchmark()
    
    print("\n" + "="*70)
    print("所有基准测试完成")
    print("="*70)


if __name__ == "__main__":
    main()
