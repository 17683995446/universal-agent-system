"""阶段83: Agent多模态数据处理系统 - 单元测试"""
import pytest
from src.agent.multimodal_processor import (
    ModalityType, ModalityData, MultimodalSample,
    TextProcessor, ImageProcessor, AudioProcessor, VideoProcessor,
    MultimodalFusion, MultimodalProcessor, create_multimodal_processor,
)


class TestTextProcessor:
    def test_process(self):
        p = TextProcessor()
        result = p.process("hello world")
        assert result["length"] == 11
        assert result["word_count"] == 2

    def test_register(self):
        p = TextProcessor()
        p.register(lambda t: {"upper": t.upper()})
        result = p.process("hi")
        assert result["upper"] == "HI"


class TestImageProcessor:
    def test_process(self):
        p = ImageProcessor()
        data = b"fake_image_data"
        result = p.process(data)
        assert result["size_bytes"] == len(data)
        assert "checksum" in result

    def test_base64(self):
        p = ImageProcessor()
        data = b"test"
        encoded = p.encode_base64(data)
        decoded = p.decode_base64(encoded)
        assert decoded == data


class TestAudioProcessor:
    def test_process(self):
        p = AudioProcessor()
        data = b"x" * 32000  # 2秒@16kHz
        result = p.process(data)
        assert result["size_bytes"] == 32000
        assert result["duration_estimate_sec"] == 2.0


class TestVideoProcessor:
    def test_process(self):
        p = VideoProcessor()
        data = b"fake_video"
        result = p.process(data)
        assert result["size_bytes"] == len(data)


class TestMultimodalFusion:
    def test_concat(self):
        f = MultimodalFusion()
        features = [{"a": 1}, {"b": 2}]
        result = f.fuse(features, "concat")
        assert result["fusion_type"] == "concat"
        assert result["modalities"] == 2
        assert result["mod_0_a"] == 1
        assert result["mod_1_b"] == 2

    def test_sum(self):
        f = MultimodalFusion()
        features = [{"x": 10}, {"x": 20}]
        result = f.fuse(features, "sum")
        assert result["fusion_type"] == "sum"
        assert result["x"] == 30


class TestMultimodalProcessor:
    def test_create_sample(self):
        p = MultimodalProcessor()
        sid = p.create_sample(labels={"class": "cat"})
        assert sid in p.list_samples()
        sample = p.get_sample(sid)
        assert sample.labels["class"] == "cat"

    def test_add_text(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        assert p.add_text(sid, "hello")
        assert not p.add_text("bad_id", "hello")

    def test_add_image(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        assert p.add_image(sid, b"img_data")

    def test_add_audio(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        assert p.add_audio(sid, b"audio_data")

    def test_add_video(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        assert p.add_video(sid, b"video_data")

    def test_process_sample_text(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        p.add_text(sid, "hello world")
        result = p.process_sample(sid)
        assert result is not None
        assert result["sample_id"] == sid
        assert "mod_0_length" in result

    def test_process_sample_multi(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        p.add_text(sid, "hello")
        p.add_image(sid, b"img")
        result = p.process_sample(sid, "concat")
        assert result["modalities"] == 2

    def test_process_sample_sum(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        p.add_text(sid, "hello world")
        p.add_image(sid, b"img")
        result = p.process_sample(sid, "sum")
        assert result["fusion_type"] == "sum"

    def test_process_sample_not_found(self):
        p = MultimodalProcessor()
        assert p.process_sample("bad_id") is None

    def test_process_sample_empty(self):
        p = MultimodalProcessor()
        sid = p.create_sample()
        assert p.process_sample(sid) is None

    def test_get_stats(self):
        p = MultimodalProcessor()
        sid1 = p.create_sample()
        p.add_text(sid1, "hello")
        sid2 = p.create_sample()
        p.add_image(sid2, b"img")
        stats = p.get_stats()
        assert stats["total_samples"] == 2
        assert stats["modality_distribution"]["text"] == 1
        assert stats["modality_distribution"]["image"] == 1

    def test_create_multimodal_processor(self):
        p = create_multimodal_processor()
        assert isinstance(p, MultimodalProcessor)
