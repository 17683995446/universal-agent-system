"""阶段89: Agent数据合成与增强系统 - 单元测试"""
import pytest
from src.agent.data_synthesis import (
    SyntheticSample,
    DataAugmenter, NumericPerturbation, SMOTESampler,
    TextAugmenter, DataSynthesisEngine, create_data_synthesis_engine,
)


class TestDataAugmenter:
    def test_augment_no_transforms(self):
        a = DataAugmenter()
        result = a.augment([1, 2, 3], n=2)
        assert len(result) == 2
        assert result[0] == [1, 2, 3]

    def test_augment_with_transform(self):
        a = DataAugmenter()
        a.register_transform(lambda x: [v * 2 for v in x])
        result = a.augment([1, 2, 3], n=1)
        assert result[0] == [2, 4, 6]


class TestNumericPerturbation:
    def test_perturb(self):
        p = NumericPerturbation(noise_scale=0.1)
        features = [1.0, 2.0, 3.0]
        result = p.perturb(features)
        assert len(result) == 3
        assert result != features  # 应该被扰动

    def test_perturb_batch(self):
        p = NumericPerturbation(noise_scale=0.1)
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = p.perturb_batch(data, n=2)
        assert len(result) == 4  # 2 samples * 2


class TestSMOTESampler:
    def test_sample(self):
        s = SMOTESampler(k_neighbors=2)
        samples = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        synthetic = s.sample(samples, n_synthetic=5)
        assert len(synthetic) == 5
        assert len(synthetic[0]) == 2

    def test_sample_insufficient(self):
        s = SMOTESampler()
        assert s.sample([[1.0]], n_synthetic=5) == []
        assert s.sample([], n_synthetic=5) == []

    def test_find_neighbors(self):
        s = SMOTESampler(k_neighbors=2)
        samples = [[0.0], [1.0], [10.0]]
        neighbors = s._find_neighbors([0.0], samples)
        assert len(neighbors) == 2
        assert [1.0] in neighbors


class TestTextAugmenter:
    def test_register_and_replace(self):
        t = TextAugmenter()
        t.register_synonyms("good", ["great", "excellent"])
        result = t.synonym_replace("this is good")
        assert result in ["this is great", "this is excellent"]

    def test_random_delete(self):
        t = TextAugmenter()
        text = "a b c d e"
        result = t.random_delete(text, p=0.5)
        # 应该删除了一些词
        assert len(result.split()) <= 5

    def test_random_swap(self):
        t = TextAugmenter()
        text = "a b c"
        result = t.random_swap(text, n=1)
        assert set(result.split()) == {"a", "b", "c"}

    def test_no_synonyms(self):
        t = TextAugmenter()
        result = t.synonym_replace("hello world")
        assert result == "hello world"


class TestDataSynthesisEngine:
    def test_synthesize_numeric_perturbation(self):
        e = DataSynthesisEngine()
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = e.synthesize_numeric(data, n=3, method="perturbation")
        assert len(result) == 6  # 2 * 3

    def test_synthesize_numeric_smote(self):
        e = DataSynthesisEngine()
        data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        result = e.synthesize_numeric(data, n=5, method="smote")
        assert len(result) == 5

    def test_synthesize_text(self):
        e = DataSynthesisEngine()
        e.text_augmenter.register_synonyms("hello", ["hi", "hey"])
        texts = ["hello world"]
        result = e.synthesize_text(texts, n_per_sample=2)
        assert len(result) > 0

    def test_augment(self):
        e = DataSynthesisEngine()
        augmenter = DataAugmenter()
        augmenter.register_transform(lambda x: x * 2)
        e.register_augmenter("number", augmenter)
        result = e.augment("number", 5, n=1)
        assert result == [10]

    def test_balance_dataset(self):
        e = DataSynthesisEngine()
        data = [[1.0], [1.1], [2.0], [2.1], [2.2]]
        labels = ["A", "A", "B", "B", "B"]
        new_data, new_labels = e.balance_dataset(data, labels)
        # A类有2个，B类有3个，应该生成1个A类样本
        assert len(new_data) == 6
        assert new_labels.count("A") == 3
        assert new_labels.count("B") == 3

    def test_get_stats(self):
        e = DataSynthesisEngine()
        e.synthesize_numeric([[1.0]], n=2, method="perturbation")
        stats = e.get_stats()
        assert stats["total_synthetic"] == 2
        assert "perturbation" in stats["methods_used"]

    def test_create_data_synthesis_engine(self):
        e = create_data_synthesis_engine()
        assert isinstance(e, DataSynthesisEngine)
