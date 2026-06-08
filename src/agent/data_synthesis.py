"""
阶段89: Agent数据合成与增强系统

遵循"图难于其易，为大于其细"的原则，实现数据增强、合成采样、特征扰动、SMOTE。
"""
import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SyntheticSample:
    data: Any
    source: str = "synthetic"
    augmentation_method: str = ""
    label: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "method": self.augmentation_method,
            "label": self.label,
        }


class DataAugmenter:
    """数据增强器"""

    def __init__(self):
        self._transforms: List[Callable[[Any], Any]] = []

    def register_transform(self, transform: Callable[[Any], Any]) -> None:
        self._transforms.append(transform)

    def augment(self, data: Any, n: int = 1) -> List[Any]:
        """对数据应用随机变换生成增强样本"""
        results = []
        for _ in range(n):
            augmented = copy.deepcopy(data)
            if self._transforms:
                transform = random.choice(self._transforms)
                augmented = transform(augmented)
            results.append(augmented)
        return results


class NumericPerturbation:
    """数值特征扰动"""

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def perturb(self, features: List[float]) -> List[float]:
        return [v + random.gauss(0, self.noise_scale * abs(v) if v != 0 else self.noise_scale)
                for v in features]

    def perturb_batch(self, data: List[List[float]], n: int = 1) -> List[List[float]]:
        results = []
        for sample in data:
            for _ in range(n):
                results.append(self.perturb(sample))
        return results


class SMOTESampler:
    """SMOTE合成采样器（简化版）"""

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors

    def sample(self, minority_samples: List[List[float]],
               n_synthetic: int) -> List[List[float]]:
        """生成合成样本"""
        if not minority_samples or len(minority_samples) < 2:
            return []

        synthetic = []
        for _ in range(n_synthetic):
            # 随机选择一个样本
            sample = random.choice(minority_samples)
            # 找到最近邻
            neighbors = self._find_neighbors(sample, minority_samples)
            if not neighbors:
                continue
            # 随机选择一个邻居
            neighbor = random.choice(neighbors)
            # 线性插值
            alpha = random.random()
            new_sample = [a + alpha * (b - a) for a, b in zip(sample, neighbor)]
            synthetic.append(new_sample)
        return synthetic

    def _find_neighbors(self, sample: List[float],
                        candidates: List[List[float]]) -> List[List[float]]:
        """找到k个最近邻"""
        distances = []
        for c in candidates:
            if c == sample:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(sample, c)) ** 0.5
            distances.append((dist, c))
        distances.sort(key=lambda x: x[0])
        return [c for _, c in distances[:self.k]]


class TextAugmenter:
    """文本增强器"""

    def __init__(self):
        self._synonyms: Dict[str, List[str]] = {}

    def register_synonyms(self, word: str, synonyms: List[str]) -> None:
        self._synonyms[word] = synonyms

    def synonym_replace(self, text: str, n: int = 1) -> str:
        """同义词替换"""
        words = text.split()
        if not words:
            return text
        for _ in range(n):
            # 找可以替换的词
            replaceable = [(i, w) for i, w in enumerate(words) if w in self._synonyms]
            if not replaceable:
                break
            idx, word = random.choice(replaceable)
            words[idx] = random.choice(self._synonyms[word])
        return " ".join(words)

    def random_delete(self, text: str, p: float = 0.1) -> str:
        """随机删除词"""
        words = text.split()
        if len(words) <= 1:
            return text
        kept = [w for w in words if random.random() > p]
        return " ".join(kept) if kept else words[0]

    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换词"""
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)


class DataSynthesisEngine:
    """数据合成引擎"""

    def __init__(self):
        self.numeric_perturbation = NumericPerturbation()
        self.smote = SMOTESampler()
        self.text_augmenter = TextAugmenter()
        self._augmenters: Dict[str, DataAugmenter] = {}
        self._history: List[SyntheticSample] = []

    def register_augmenter(self, data_type: str, augmenter: DataAugmenter) -> None:
        self._augmenters[data_type] = augmenter

    def synthesize_numeric(self, data: List[List[float]], n: int = 10,
                           method: str = "perturbation") -> List[List[float]]:
        """合成数值数据"""
        if method == "perturbation":
            result = self.numeric_perturbation.perturb_batch(data, n=n)
            for r in result:
                self._history.append(SyntheticSample(data=r, augmentation_method="perturbation"))
            return result
        elif method == "smote":
            result = self.smote.sample(data, n_synthetic=n)
            for r in result:
                self._history.append(SyntheticSample(data=r, augmentation_method="smote"))
            return result
        return []

    def synthesize_text(self, texts: List[str], n_per_sample: int = 2,
                        methods: Optional[List[str]] = None) -> List[str]:
        """合成文本数据"""
        methods = methods or ["synonym", "delete", "swap"]
        result = []
        for text in texts:
            for _ in range(n_per_sample):
                method = random.choice(methods)
                if method == "synonym":
                    new_text = self.text_augmenter.synonym_replace(text)
                elif method == "delete":
                    new_text = self.text_augmenter.random_delete(text)
                elif method == "swap":
                    new_text = self.text_augmenter.random_swap(text)
                else:
                    new_text = text
                if new_text != text:
                    result.append(new_text)
                    self._history.append(SyntheticSample(data=new_text, augmentation_method=method))
        return result

    def augment(self, data_type: str, data: Any, n: int = 1) -> List[Any]:
        """使用注册的增强器增强数据"""
        augmenter = self._augmenters.get(data_type)
        if not augmenter:
            return []
        return augmenter.augment(data, n)

    def balance_dataset(self, data: List[List[float]],
                        labels: List[Any],
                        target_ratio: Optional[Dict[Any, float]] = None) -> Tuple[List[List[float]], List[Any]]:
        """平衡数据集（对少数类过采样）"""
        # 统计各类别
        class_counts: Dict[Any, int] = {}
        class_samples: Dict[Any, List[List[float]]] = {}
        for sample, label in zip(data, labels):
            class_counts[label] = class_counts.get(label, 0) + 1
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)

        max_count = max(class_counts.values())
        new_data = []
        new_labels = []

        for label, samples in class_samples.items():
            new_data.extend(samples)
            new_labels.extend([label] * len(samples))
            # 如果需要更多样本
            needed = max_count - len(samples)
            if needed > 0:
                synthetic = self.smote.sample(samples, n_synthetic=needed)
                new_data.extend(synthetic)
                new_labels.extend([label] * len(synthetic))

        return new_data, new_labels

    def get_stats(self) -> Dict[str, Any]:
        methods = {}
        for s in self._history:
            methods[s.augmentation_method] = methods.get(s.augmentation_method, 0) + 1
        return {
            "total_synthetic": len(self._history),
            "methods_used": methods,
            "registered_augmenters": len(self._augmenters),
        }


def create_data_synthesis_engine() -> DataSynthesisEngine:
    return DataSynthesisEngine()
