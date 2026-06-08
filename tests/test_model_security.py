"""
阶段94测试: 模型安全与对抗防御系统
"""
import pytest
from src.agent.model_security import (
    AdversarialSample, AdversarialDetector, InputPurifier,
    RobustnessEvaluator, DefenseStrategy, ModelSecuritySystem,
    create_model_security_system,
)


class TestAdversarialSample:
    def test_perturbation_norm(self):
        s = AdversarialSample(
            original=[1.0, 2.0],
            perturbed=[1.1, 2.2],
            perturbation=[0.1, 0.2],
            label=0,
            predicted_label=1,
        )
        assert s.perturbation_norm("l2") == pytest.approx(0.2236, abs=0.01)
        assert s.perturbation_norm("linf") == pytest.approx(0.2)
        assert s.perturbation_norm("l1") == pytest.approx(0.3)

    def test_to_dict(self):
        s = AdversarialSample(
            original=[1.0], perturbed=[1.5], perturbation=[0.5],
            label=0, predicted_label=1, attack_method="fgsm",
        )
        d = s.to_dict()
        assert d["label"] == 0
        assert d["predicted_label"] == 1
        assert d["attack_method"] == "fgsm"


class TestAdversarialDetector:
    def test_detect_statistical_clean(self):
        det = AdversarialDetector(threshold=0.1)
        assert det.detect_statistical([0.1, 0.2, 0.3]) is False

    def test_detect_statistical_anomaly(self):
        det = AdversarialDetector(threshold=0.01)
        assert det.detect_statistical([0.0, 100.0, 0.0]) is True

    def test_detect_consistency(self):
        det = AdversarialDetector()
        # 稳定模型（远离边界）
        stable_model = lambda v: 1 if sum(v) > 2.0 else 0
        assert det.detect_consistency(stable_model, [1.5, 1.5], n_perturbations=10) is False

    def test_detect_clean(self):
        det = AdversarialDetector()
        is_adv, reason = det.detect([0.1, 0.2, 0.3])
        assert is_adv is False
        assert reason == "clean"


class TestInputPurifier:
    def test_clip(self):
        purifier = InputPurifier(clip_min=0.0, clip_max=1.0)
        result = purifier.clip([-0.5, 0.5, 1.5])
        assert result == [0.0, 0.5, 1.0]

    def test_smooth(self):
        purifier = InputPurifier()
        result = purifier.smooth([1.0, 2.0, 3.0], window=3)
        assert len(result) == 3
        assert result[1] == pytest.approx(2.0)

    def test_quantize(self):
        purifier = InputPurifier(clip_min=0.0, clip_max=1.0)
        result = purifier.quantize([0.123, 0.456, 0.789], levels=10)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_purify(self):
        purifier = InputPurifier(clip_min=0.0, clip_max=1.0)
        result = purifier.purify([-0.1, 0.5, 1.1], methods=["clip"])
        assert result == [0.0, 0.5, 1.0]


class TestRobustnessEvaluator:
    def test_evaluate_under_attack(self):
        evalr = RobustnessEvaluator()
        model = lambda v: 1 if v[0] > 0.5 else 0
        test_data = [[0.6], [0.4]]
        labels = [1, 0]
        attack = lambda x: [x[0] + 0.1]
        result = evalr.evaluate_under_attack(model, test_data, labels, attack)
        assert "accuracy" in result
        assert result["total"] == 2

    def test_random_attack(self):
        evalr = RobustnessEvaluator()
        adv = evalr.random_attack([0.5, 0.5], epsilon=0.1)
        assert len(adv) == 2
        assert all(abs(adv[i] - 0.5) <= 0.1 for i in range(2))

    def test_fgsm_attack(self):
        evalr = RobustnessEvaluator()
        model = lambda v: 1 if v[0] > 0.5 else 0
        adv = evalr.fgsm_attack(model, [0.4], epsilon=0.1)
        assert len(adv) == 1

    def test_get_stats(self):
        evalr = RobustnessEvaluator()
        assert evalr.get_stats()["evaluations"] == 0


class TestDefenseStrategy:
    def test_defend_clean(self):
        defense = DefenseStrategy()
        model = lambda v: 1
        result, was_attacked = defense.defend([0.1, 0.2], model)
        assert was_attacked is False
        assert result == [0.1, 0.2]

    def test_evaluate_robustness(self):
        defense = DefenseStrategy()
        model = lambda v: 1 if v[0] > 0.5 else 0
        test_data = [[0.6], [0.4]]
        labels = [1, 0]
        result = defense.evaluate_robustness(model, test_data, labels)
        assert "original_accuracy" in result
        assert "adversarial_accuracy" in result
        assert "robustness_drop" in result

    def test_get_stats(self):
        defense = DefenseStrategy()
        model = lambda v: 1
        defense.defend([0.1], model)
        stats = defense.get_stats()
        assert stats["total_requests"] == 1


class TestModelSecuritySystem:
    def test_predict_secure_clean(self):
        sys = ModelSecuritySystem()
        # 使用一个远离边界的稳定模型，避免 consistency 检测误报
        model = lambda v: 1 if v[0] > 10.0 else 0
        pred, was_attacked = sys.predict_secure([20.0], model)
        assert pred == 1
        assert was_attacked is False

    def test_evaluate(self):
        sys = ModelSecuritySystem()
        model = lambda v: 1 if v[0] > 0.5 else 0
        result = sys.evaluate(model, [[0.6], [0.4]], [1, 0])
        assert "original_accuracy" in result

    def test_get_alerts(self):
        sys = ModelSecuritySystem()
        assert sys.get_alerts() == []

    def test_get_stats(self):
        sys = ModelSecuritySystem()
        stats = sys.get_stats()
        assert "alerts" in stats
        assert "defense" in stats


class TestCreateModelSecuritySystem:
    def test_factory(self):
        sys = create_model_security_system()
        assert isinstance(sys, ModelSecuritySystem)
