"""
阶段83: Agent多模态数据处理系统

遵循"图难于其易，为大于其细"的原则，实现文本、图像、音频、视频等多模态数据的统一处理。
"""
import base64
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class ModalityData:
    modality: ModalityType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class MultimodalSample:
    sample_id: str
    modalities: Dict[ModalityType, ModalityData] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "modalities": [m.value for m in self.modalities.keys()],
            "labels": self.labels,
        }


class TextProcessor:
    """文本处理器"""

    def __init__(self):
        self._processors: List[Callable[[str], Dict[str, Any]]] = []

    def register(self, processor: Callable[[str], Dict[str, Any]]) -> None:
        self._processors.append(processor)

    def process(self, text: str) -> Dict[str, Any]:
        result = {"text": text, "length": len(text), "word_count": len(text.split())}
        for processor in self._processors:
            result.update(processor(text))
        return result


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        self._processors: List[Callable[[bytes], Dict[str, Any]]] = []

    def register(self, processor: Callable[[bytes], Dict[str, Any]]) -> None:
        self._processors.append(processor)

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        result = {
            "size_bytes": len(image_bytes),
            "checksum": hashlib.md5(image_bytes).hexdigest()[:8],
        }
        for processor in self._processors:
            result.update(processor(image_bytes))
        return result

    def encode_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def decode_base64(self, encoded: str) -> bytes:
        return base64.b64decode(encoded)


class AudioProcessor:
    """音频处理器"""

    def __init__(self):
        self._processors: List[Callable[[bytes], Dict[str, Any]]] = []

    def register(self, processor: Callable[[bytes], Dict[str, Any]]) -> None:
        self._processors.append(processor)

    def process(self, audio_bytes: bytes) -> Dict[str, Any]:
        result = {
            "size_bytes": len(audio_bytes),
            "duration_estimate_sec": len(audio_bytes) / 16000,  # 假设16kHz采样
        }
        for processor in self._processors:
            result.update(processor(audio_bytes))
        return result


class VideoProcessor:
    """视频处理器"""

    def __init__(self):
        self._processors: List[Callable[[bytes], Dict[str, Any]]] = []

    def register(self, processor: Callable[[bytes], Dict[str, Any]]) -> None:
        self._processors.append(processor)

    def process(self, video_bytes: bytes) -> Dict[str, Any]:
        result = {
            "size_bytes": len(video_bytes),
            "checksum": hashlib.md5(video_bytes).hexdigest()[:8],
        }
        for processor in self._processors:
            result.update(processor(video_bytes))
        return result


class MultimodalFusion:
    """多模态融合器"""

    def __init__(self):
        self._fusion_strategies: Dict[str, Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = {}

    def register_strategy(self, name: str, strategy: Callable[[List[Dict[str, Any]]], Dict[str, Any]]) -> None:
        self._fusion_strategies[name] = strategy

    def fuse(self, features: List[Dict[str, Any]], strategy: str = "concat") -> Dict[str, Any]:
        if strategy == "concat":
            return self._concat_fusion(features)
        elif strategy == "sum":
            return self._sum_fusion(features)
        elif strategy in self._fusion_strategies:
            return self._fusion_strategies[strategy](features)
        return self._concat_fusion(features)

    def _concat_fusion(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {"fusion_type": "concat", "modalities": len(features)}
        for i, feat in enumerate(features):
            for k, v in feat.items():
                result[f"mod_{i}_{k}"] = v
        return result

    def _sum_fusion(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {"fusion_type": "sum", "modalities": len(features)}
        numeric_values = {}
        for feat in features:
            for k, v in feat.items():
                if isinstance(v, (int, float)):
                    numeric_values[k] = numeric_values.get(k, 0) + v
        result.update(numeric_values)
        return result


class MultimodalProcessor:
    """多模态处理器"""

    def __init__(self):
        self._text_processor = TextProcessor()
        self._image_processor = ImageProcessor()
        self._audio_processor = AudioProcessor()
        self._video_processor = VideoProcessor()
        self._fusion = MultimodalFusion()
        self._samples: Dict[str, MultimodalSample] = {}

    def create_sample(self, labels: Optional[Dict[str, Any]] = None) -> str:
        sample_id = str(uuid.uuid4())[:8]
        sample = MultimodalSample(sample_id=sample_id, labels=labels or {})
        self._samples[sample_id] = sample
        return sample_id

    def add_text(self, sample_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        sample = self._samples.get(sample_id)
        if not sample:
            return False
        sample.modalities[ModalityType.TEXT] = ModalityData(
            modality=ModalityType.TEXT,
            content=text,
            metadata=metadata or {},
        )
        return True

    def add_image(self, sample_id: str, image_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> bool:
        sample = self._samples.get(sample_id)
        if not sample:
            return False
        sample.modalities[ModalityType.IMAGE] = ModalityData(
            modality=ModalityType.IMAGE,
            content=image_bytes,
            metadata=metadata or {},
        )
        return True

    def add_audio(self, sample_id: str, audio_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> bool:
        sample = self._samples.get(sample_id)
        if not sample:
            return False
        sample.modalities[ModalityType.AUDIO] = ModalityData(
            modality=ModalityType.AUDIO,
            content=audio_bytes,
            metadata=metadata or {},
        )
        return True

    def add_video(self, sample_id: str, video_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> bool:
        sample = self._samples.get(sample_id)
        if not sample:
            return False
        sample.modalities[ModalityType.VIDEO] = ModalityData(
            modality=ModalityType.VIDEO,
            content=video_bytes,
            metadata=metadata or {},
        )
        return True

    def process_sample(self, sample_id: str, fusion_strategy: str = "concat") -> Optional[Dict[str, Any]]:
        sample = self._samples.get(sample_id)
        if not sample:
            return None
        features = []
        for modality, data in sample.modalities.items():
            if modality == ModalityType.TEXT:
                feat = self._text_processor.process(data.content)
            elif modality == ModalityType.IMAGE:
                feat = self._image_processor.process(data.content)
            elif modality == ModalityType.AUDIO:
                feat = self._audio_processor.process(data.content)
            elif modality == ModalityType.VIDEO:
                feat = self._video_processor.process(data.content)
            else:
                continue
            feat["modality"] = modality.value
            features.append(feat)

        if not features:
            return None

        fused = self._fusion.fuse(features, strategy=fusion_strategy)
        fused["sample_id"] = sample_id
        fused["labels"] = sample.labels
        return fused

    def get_sample(self, sample_id: str) -> Optional[MultimodalSample]:
        return self._samples.get(sample_id)

    def list_samples(self) -> List[str]:
        return list(self._samples.keys())

    def get_stats(self) -> Dict[str, Any]:
        modality_counts = {}
        for sample in self._samples.values():
            for mod in sample.modalities:
                modality_counts[mod.value] = modality_counts.get(mod.value, 0) + 1
        return {
            "total_samples": len(self._samples),
            "modality_distribution": modality_counts,
        }


def create_multimodal_processor() -> MultimodalProcessor:
    return MultimodalProcessor()
