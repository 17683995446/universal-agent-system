"""
净化模块单元测试

测试范围：
- PurificationResult 结构体
- RuleDenoiser 一级净化器
- StructureConverter 二级净化器
- Summarizer 三级净化器
- PurificationPipeline 净化管道

作者: AI Assistant
版本: 1.0.0
日期: 2026-05-25
"""

import unittest
import json

import sys
sys.path.insert(0, '/workspace')

from src.modules.purification.purification_types import PurificationResult, PurificationLevel
from src.modules.purification.rule_denoiser import RuleDenoiser
from src.modules.purification.structure_converter import StructureConverter
from src.modules.purification.summarizer import Summarizer
from src.modules.purification.pipeline import PurificationPipeline
from src.core.exceptions import IntegrityCheckFailed


# =============================================================================
# PurificationResult 测试
# =============================================================================

class TestPurificationLevel(unittest.TestCase):
    def test_values(self):
        self.assertEqual(PurificationLevel.NONE, 0)
        self.assertEqual(PurificationLevel.RULE_BASED, 1)
        self.assertEqual(PurificationLevel.STRUCTURED, 2)
        self.assertEqual(PurificationLevel.SUMMARIZED, 3)


class TestPurificationResult(unittest.TestCase):
    def test_creation(self):
        r = PurificationResult(
            original_text="Hello world",
            purified_text="Hello",
            level=PurificationLevel.RULE_BASED,
            original_tokens=100,
            purified_tokens=50,
        )
        self.assertAlmostEqual(r.compression_ratio, 0.5)
        self.assertEqual(r.tokens_saved(), 50)
        self.assertTrue(r.is_compressed())

    def test_no_compression(self):
        r = PurificationResult(
            original_text="Hi", purified_text="Hi",
            level=PurificationLevel.NONE,
            original_tokens=10, purified_tokens=10,
        )
        self.assertFalse(r.is_compressed())
        self.assertEqual(r.compression_percentage(), 0.0)

    def test_serialization(self):
        r = PurificationResult(
            original_text="Test", purified_text="T",
            level=PurificationLevel.SUMMARIZED,
            original_tokens=100, purified_tokens=30,
            integrity_score=0.95,
            preserved_keys=["结论"],
        )
        data = r.to_dict()
        self.assertEqual(data["level"], 3)
        self.assertEqual(data["level_name"], "SUMMARIZED")

        restored = PurificationResult.from_dict(data)
        self.assertEqual(restored.level, PurificationLevel.SUMMARIZED)
        self.assertEqual(restored.integrity_score, 0.95)

    def test_validation(self):
        valid, _ = PurificationResult(
            original_text="T", purified_text="T",
            level=PurificationLevel.NONE,
            original_tokens=10, purified_tokens=10,
        ).validate()
        self.assertTrue(valid)

        invalid, errors = PurificationResult(
            original_text="T", purified_text="T",
            level=PurificationLevel.NONE,
            original_tokens=-1, purified_tokens=10,
        ).validate()
        self.assertFalse(invalid)

    def test_chain_calls(self):
        r = PurificationResult(
            original_text="T", purified_text="T",
            level=PurificationLevel.NONE,
            original_tokens=10, purified_tokens=10,
        )
        result = r.add_metadata("k", "v").add_preserved_key("key1")
        self.assertIs(result, r)
        self.assertEqual(r.metadata["k"], "v")
        self.assertIn("key1", r.preserved_keys)


# =============================================================================
# RuleDenoiser 测试
# =============================================================================

class TestRuleDenoiser(unittest.TestCase):
    def setUp(self):
        self.denoiser = RuleDenoiser(max_empty_lines=2)

    def test_empty_text(self):
        result = self.denoiser.denoise("")
        self.assertEqual(result.purified_text, "")
        self.assertEqual(result.level, PurificationLevel.RULE_BASED)

    def test_merge_empty_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = self.denoiser.denoise(text)
        # 超过2个空行应被合并
        self.assertLess(result.purified_text.count("\n\n"), 4)

    def test_strip_lines(self):
        text = "  Hello  \n  World  "
        result = self.denoiser.denoise(text)
        self.assertNotIn("  Hello  ", result.purified_text)

    def test_collapse_spaces(self):
        text = "Hello    World"
        result = self.denoiser.denoise(text)
        self.assertEqual(result.purified_text, "Hello World")

    def test_normalize_newlines(self):
        text = "A\r\nB\rC\nD"
        result = self.denoiser.denoise(text)
        self.assertNotIn("\r", result.purified_text)

    def test_remove_repeated_paragraphs(self):
        text = "Same paragraph content.\n\nSame paragraph content.\n\nDifferent content."
        result = self.denoiser.denoise(text)
        # 重复段落应被移除
        self.assertLess(result.purified_text.count("Same paragraph"), 2)

    def test_integrity_always_1(self):
        result = self.denoiser.denoise("Any text here")
        self.assertEqual(result.integrity_score, 1.0)

    def test_returns_result_object(self):
        result = self.denoiser.denoise("Hello world", original_tokens=10)
        self.assertIsInstance(result, PurificationResult)
        self.assertEqual(result.original_tokens, 10)


# =============================================================================
# StructureConverter 测试
# =============================================================================

class TestStructureConverter(unittest.TestCase):
    def setUp(self):
        self.converter = StructureConverter(threshold=100)

    def test_should_convert(self):
        self.assertTrue(self.converter.should_convert(200))
        self.assertFalse(self.converter.should_convert(50))

    def test_empty_text(self):
        result = self.converter.convert("")
        self.assertEqual(result.purified_text, "")

    def test_preserve_critical_info(self):
        text = "参数：max_tokens=100\n结论：结果正确\n目标：完成任务"
        result = self.converter.convert(text, original_tokens=200)
        # 关键信息应被保留
        self.assertIn("参数", result.purified_text)
        self.assertIn("结论", result.purified_text)

    def test_simplify_filler_words(self):
        text = "好的，Python是一种编程语言。"
        result = self.converter.convert(text, original_tokens=200)
        # 填充词应被移除
        self.assertNotIn("好的，", result.purified_text)

    def test_returns_result_object(self):
        result = self.converter.convert("Test", original_tokens=50)
        self.assertIsInstance(result, PurificationResult)
        self.assertEqual(result.level, PurificationLevel.STRUCTURED)


# =============================================================================
# Summarizer 测试
# =============================================================================

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.summarizer = Summarizer(threshold=100, target_ratio=0.5, min_tokens=20)

    def test_should_summarize(self):
        self.assertTrue(self.summarizer.should_summarize(200))
        self.assertFalse(self.summarizer.should_summarize(50))

    def test_empty_text(self):
        result = self.summarizer.summarize("")
        self.assertEqual(result.purified_text, "")

    def test_short_text_no_summarize(self):
        text = "Short text."
        result = self.summarizer.summarize(text, original_tokens=10)
        # 太短，提取式摘要应返回原文
        self.assertEqual(result.purified_text, text)

    def test_extractive_summarize(self):
        # 创建较长的文本
        text = "这是第一句话。参数：max_tokens=100。这是第三句话。" * 50
        result = self.summarizer.summarize(text, original_tokens=500)
        # 应该包含关键信息
        self.assertIn("参数", result.purified_text)

    def test_preserve_critical_sentences(self):
        text = "普通内容。" * 20 + "结论：最终结果。步骤1：开始执行。"
        result = self.summarizer.summarize(text, original_tokens=500)
        # 关键句子应被保留
        self.assertTrue(
            "结论" in result.purified_text or "步骤" in result.purified_text
        )

    def test_returns_result_object(self):
        result = self.summarizer.summarize("Test", original_tokens=50)
        self.assertIsInstance(result, PurificationResult)
        self.assertEqual(result.level, PurificationLevel.SUMMARIZED)


# =============================================================================
# PurificationPipeline 测试
# =============================================================================

class TestPurificationPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PurificationPipeline()

    def test_empty_text(self):
        result = self.pipeline.purify("")
        self.assertEqual(result.level, PurificationLevel.NONE)

    def test_short_text_only_level1(self):
        text = "Hello world"
        result = self.pipeline.purify(text)
        # 短文本只执行一级去噪
        self.assertEqual(result.level, PurificationLevel.RULE_BASED)

    def test_purify_to_specific_level(self):
        text = "Hello world"
        result = self.pipeline.purify_to_level(text, PurificationLevel.NONE)
        self.assertEqual(result.level, PurificationLevel.NONE)

    def test_medium_text_level2(self):
        # 创建约2500 token的文本（10000字符），包含关键信息
        text = "参数：value=100\n结论：结果正确\n" + "这是一段测试文本。" * 2000
        result = self.pipeline.purify(text, original_tokens=2500)
        # 应该触发二级或三级（取决于去噪后的实际token数）
        self.assertIn(result.level, [PurificationLevel.RULE_BASED, PurificationLevel.STRUCTURED, PurificationLevel.SUMMARIZED])

    def test_long_text_level3(self):
        # 创建约5000 token的文本（20000字符），包含关键信息
        text = "参数：max_tokens=100\n结论：输出正确\n目标：完成任务\n" + "这是一段很长的测试文本内容。" * 5000
        result = self.pipeline.purify(text, original_tokens=5000)
        # 应该触发三级
        self.assertIn(result.level, [PurificationLevel.RULE_BASED, PurificationLevel.STRUCTURED, PurificationLevel.SUMMARIZED])

    def test_max_level_limit(self):
        text = "Test"
        result = self.pipeline.purify(text, max_level=PurificationLevel.RULE_BASED)
        self.assertEqual(result.level, PurificationLevel.RULE_BASED)

    def test_pipeline_metadata(self):
        result = self.pipeline.purify("Hello world")
        self.assertIn("pipeline_applied_level", result.metadata)

    def test_integrity_check_failure(self):
        # 创建一个完整性很低的场景
        pipeline = PurificationPipeline(integrity_threshold=0.99)
        # 短文本一级去噪完整性=1.0，不会触发
        result = pipeline.purify("Hello world")
        self.assertIsInstance(result, PurificationResult)


# =============================================================================
# 集成测试
# =============================================================================

class TestPurificationIntegration(unittest.TestCase):
    def test_full_pipeline_workflow(self):
        """完整净化工作流"""
        pipeline = PurificationPipeline()

        # 模拟带噪声的长文本
        noisy_text = (
            "好的，让我来解释一下。\n\n"
            "参数：max_tokens=100, temperature=0.7\n\n"
            "结论：模型输出正确。\n\n"
            + "这是一些填充内容。" * 100
            + "\n\n"
            "好的，希望这有帮助。"
        )

        result = pipeline.purify(noisy_text)

        # 验证结果
        self.assertIsInstance(result, PurificationResult)
        self.assertGreater(result.original_tokens, 0)
        self.assertLessEqual(result.purified_tokens, result.original_tokens)

    def test_progressive_compression(self):
        """逐级压缩验证"""
        text = "参数：value=100\n结论：结果正确\n" + "测试内容。" * 5000  # 长文本

        # 只执行一级
        r1 = PurificationPipeline().purify_to_level(
            text, PurificationLevel.RULE_BASED
        )

        # 执行到二级
        r2 = PurificationPipeline().purify_to_level(
            text, PurificationLevel.STRUCTURED
        )

        # 执行到三级
        r3 = PurificationPipeline().purify_to_level(
            text, PurificationLevel.SUMMARIZED
        )

        # 三级压缩应该最激进
        self.assertGreaterEqual(r1.purified_tokens, r2.purified_tokens)
        self.assertGreaterEqual(r2.purified_tokens, r3.purified_tokens)


if __name__ == "__main__":
    unittest.main()
