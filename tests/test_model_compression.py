"""
阶段90测试: 模型压缩与加速系统
"""
import pytest
from src.agent.model_compression import (
    CompressedModel, WeightPruner, Quantizer,
    KnowledgeDistiller, LowRankDecomposer, ModelCompressor,
    create_model_compressor,
)


class TestCompressedModel:
    def test_compression_ratio(self):
        m = CompressedModel(original_size=100, compressed_size=60, method="pruning")
        assert m.compression_ratio() == 0.4

    def test_compression_ratio_zero(self):
        m = CompressedModel(original_size=0, compressed_size=0, method="pruning")
        assert m.compression_ratio() == 0.0

    def test_to_dict(self):
        m = CompressedModel(original_size=100, compressed_size=75, method="quantization", metrics={"acc": 0.9})
        d = m.to_dict()
        assert d["original_size"] == 100
        assert d["compressed_size"] == 75
        assert d["method"] == "quantization"
        assert "compression_ratio" in d


class TestWeightPruner:
    def test_prune_basic(self):
        pruner = WeightPruner(sparsity=0.5)
        weights = {"w1": [0.1, -0.9, 0.3, -0.2, 0.8]}
        pruned = pruner.prune(weights)
        assert len(pruned["w1"]) == 5
        # 大约一半的权重被置零
        zeros = sum(1 for v in pruned["w1"] if v == 0.0)
        assert zeros >= 1

    def test_prune_empty(self):
        pruner = WeightPruner(sparsity=0.5)
        pruned = pruner.prune({"w1": []})
        assert pruned["w1"] == []

    def test_get_sparsity(self):
        pruner = WeightPruner(sparsity=0.5)
        weights = {"w1": [1.0, 0.0, 2.0, 0.0]}
        assert pruner.get_sparsity(weights) == 0.5

    def test_get_sparsity_empty(self):
        pruner = WeightPruner(sparsity=0.5)
        assert pruner.get_sparsity({}) == 0.0


class TestQuantizer:
    def test_quantize_basic(self):
        q = Quantizer(bits=8)
        weights = {"w1": [0.0, 0.5, 1.0]}
        quantized = q.quantize(weights)
        assert all(isinstance(v, int) for v in quantized["w1"])
        assert len(quantized["w1"]) == 3

    def test_quantize_constant(self):
        q = Quantizer(bits=8)
        weights = {"w1": [2.0, 2.0, 2.0]}
        quantized = q.quantize(weights)
        assert quantized["w1"] == [0, 0, 0]

    def test_dequantize(self):
        q = Quantizer(bits=8)
        weights = {"w1": [0.0, 0.5, 1.0]}
        quantized = q.quantize(weights)
        ranges = q.get_ranges(weights)
        dequantized = q.dequantize(quantized, ranges)
        assert len(dequantized["w1"]) == 3
        # 反量化后应该接近原值
        assert abs(dequantized["w1"][0] - 0.0) < 0.1
        assert abs(dequantized["w1"][-1] - 1.0) < 0.1

    def test_get_ranges(self):
        q = Quantizer(bits=8)
        weights = {"w1": [-1.0, 0.0, 1.0]}
        ranges = q.get_ranges(weights)
        assert ranges["w1"] == (-1.0, 1.0)


class TestKnowledgeDistiller:
    def test_distill_logits(self):
        d = KnowledgeDistiller(temperature=2.0)
        teacher = [2.0, 1.0, 0.1]
        student = [1.5, 1.2, 0.3]
        loss = d.distill_logits(teacher, student)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_distill_identical(self):
        d = KnowledgeDistiller(temperature=1.0)
        logits = [1.0, 2.0, 3.0]
        loss = d.distill_logits(logits, logits)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_softmax(self):
        d = KnowledgeDistiller()
        probs = d._softmax([1.0, 1.0, 1.0])
        assert sum(probs) == pytest.approx(1.0)
        assert len(probs) == 3


class TestLowRankDecomposer:
    def test_decompose_matrix(self):
        decomp = LowRankDecomposer(rank=2)
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        U, V = decomp.decompose_matrix(matrix)
        assert len(U) == 2
        assert len(U[0]) == 2
        assert len(V) == 2
        assert len(V[0]) == 3

    def test_decompose_empty(self):
        decomp = LowRankDecomposer(rank=2)
        U, V = decomp.decompose_matrix([])
        assert U == []
        assert V == []

    def test_reconstruct(self):
        decomp = LowRankDecomposer(rank=2)
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        U, V = decomp.decompose_matrix(matrix)
        recon = decomp.reconstruct(U, V)
        assert len(recon) == 3
        assert len(recon[0]) == 2


class TestModelCompressor:
    def test_compress_prune(self):
        mc = ModelCompressor()
        weights = {"layer1": [0.1, -0.5, 0.3, -0.8, 0.2]}
        pruned = mc.compress_prune(weights, sparsity=0.4)
        assert len(pruned["layer1"]) == 5
        assert len(mc._compression_history) == 1
        assert mc._compression_history[0].method == "pruning"

    def test_compress_quantize(self):
        mc = ModelCompressor()
        weights = {"layer1": [0.0, 0.25, 0.5, 0.75, 1.0]}
        quantized, ranges = mc.compress_quantize(weights, bits=8)
        assert all(isinstance(v, int) for v in quantized["layer1"])
        assert "layer1" in ranges
        assert len(mc._compression_history) == 1

    def test_compress_low_rank(self):
        mc = ModelCompressor()
        matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        U, V = mc.compress_low_rank(matrix, rank=2)
        assert len(U) == 2
        assert len(V) == 2
        assert mc._compression_history[0].method == "low_rank"

    def test_get_stats_empty(self):
        mc = ModelCompressor()
        stats = mc.get_stats()
        assert stats["total_compressions"] == 0

    def test_get_stats_multiple(self):
        mc = ModelCompressor()
        mc.compress_prune({"w": [1.0, 2.0]})
        mc.compress_quantize({"w": [1.0, 2.0]})
        stats = mc.get_stats()
        assert stats["total_compressions"] == 2
        assert "pruning" in stats["methods_used"]
        assert "quantization" in stats["methods_used"]


class TestCreateModelCompressor:
    def test_factory(self):
        mc = create_model_compressor()
        assert isinstance(mc, ModelCompressor)
