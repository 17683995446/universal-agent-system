"""阶段80: Agent智能决策系统 - 单元测试"""
import pytest
from src.agent.intelligent_decision import (
    DecisionContext, DecisionResult,
    Rule, RuleEngine, DecisionTreeNode, DecisionTree,
    MultiObjectiveOptimizer, DecisionEvaluator,
    IntelligentDecisionSystem, create_intelligent_decision_system,
)


class TestDecisionContext:
    def test_get_set(self):
        ctx = DecisionContext()
        ctx.set("temperature", 30)
        assert ctx.get("temperature") == 30
        assert ctx.get("missing") is None


class TestRule:
    def test_evaluate_match(self):
        rule = Rule(
            name="hot_weather",
            condition=lambda ctx: ctx.get("temperature", 0) > 25,
            action="turn_on_ac",
            priority=1,
        )
        ctx = DecisionContext(facts={"temperature": 30})
        result = rule.evaluate(ctx)
        assert result is not None
        assert result.action == "turn_on_ac"

    def test_evaluate_no_match(self):
        rule = Rule(
            name="hot_weather",
            condition=lambda ctx: ctx.get("temperature", 0) > 25,
            action="turn_on_ac",
        )
        ctx = DecisionContext(facts={"temperature": 20})
        result = rule.evaluate(ctx)
        assert result is None


class TestRuleEngine:
    def test_add_and_evaluate(self):
        engine = RuleEngine()
        engine.add_rule(Rule("hot", lambda ctx: ctx.get("temp", 0) > 30, "ac_on", priority=2))
        engine.add_rule(Rule("warm", lambda ctx: ctx.get("temp", 0) > 20, "fan_on", priority=1))
        ctx = DecisionContext(facts={"temp": 35})
        result = engine.evaluate(ctx)
        assert result is not None
        assert result.action == "ac_on"  # 高优先级先匹配

    def test_evaluate_all(self):
        engine = RuleEngine()
        engine.add_rule(Rule("warm", lambda ctx: ctx.get("temp", 0) > 20, "fan_on"))
        engine.add_rule(Rule("hot", lambda ctx: ctx.get("temp", 0) > 30, "ac_on"))
        ctx = DecisionContext(facts={"temp": 35})
        results = engine.evaluate_all(ctx)
        assert len(results) == 2

    def test_no_match(self):
        engine = RuleEngine()
        ctx = DecisionContext(facts={"temp": 10})
        result = engine.evaluate(ctx)
        assert result is None


class TestDecisionTree:
    def test_predict(self):
        root = DecisionTreeNode(
            feature="temperature",
            threshold=25.0,
            left=DecisionTreeNode(label="comfortable", is_leaf=True),
            right=DecisionTreeNode(label="hot", is_leaf=True),
        )
        tree = DecisionTree()
        tree.set_root(root)
        assert tree.predict({"temperature": 20.0}) == "comfortable"
        assert tree.predict({"temperature": 30.0}) == "hot"

    def test_build_simple_tree(self):
        tree = DecisionTree()
        tree.build_simple_tree([
            ("temperature", 25.0, "comfortable", "hot"),
            ("humidity", 60.0, "dry", "humid"),
        ])
        assert tree.predict({"temperature": 20.0, "humidity": 50.0}) == "comfortable"
        assert tree.predict({"temperature": 30.0, "humidity": 50.0}) == "dry"

    def test_empty_tree(self):
        tree = DecisionTree()
        assert tree.predict({"temperature": 20.0}) == "unknown"


class TestMultiObjectiveOptimizer:
    def test_evaluate(self):
        opt = MultiObjectiveOptimizer()
        opt.add_objective("cost", -0.5, lambda x: -x["cost"])  # 成本越低越好
        opt.add_objective("quality", 0.5, lambda x: x["quality"])  # 质量越高越好
        option = {"cost": 100, "quality": 0.9}
        total, scores = opt.evaluate(option)
        # total = -0.5 * 100 + 0.5 * 0.9 = -50 + 0.45 = -49.55
        assert "cost" in scores
        assert "quality" in scores
        assert scores["cost"] == -100
        assert scores["quality"] == 0.9

    def test_optimize(self):
        opt = MultiObjectiveOptimizer()
        opt.add_objective("score", 1.0, lambda x: x)
        options = [0.5, 0.8, 0.3]
        results = opt.optimize(options)
        assert len(results) == 3
        assert results[0][0] == 0.8  # 最高分


class TestDecisionEvaluator:
    def test_record_and_accuracy(self):
        eval = DecisionEvaluator()
        eval.record(DecisionResult(action="yes", confidence=0.9), "yes")
        eval.record(DecisionResult(action="no", confidence=0.8), "yes")
        assert eval.get_accuracy() == 0.5
        stats = eval.get_stats()
        assert stats["total"] == 2
        assert stats["accuracy"] == 0.5


class TestIntelligentDecisionSystem:
    def test_decide_rule(self):
        sys = IntelligentDecisionSystem()
        sys.rule_engine.add_rule(Rule("hot", lambda ctx: ctx.get("temp", 0) > 30, "ac_on"))
        ctx = DecisionContext(facts={"temp": 35})
        result = sys.decide(ctx)
        assert result.action == "ac_on"

    def test_decide_tree(self):
        sys = IntelligentDecisionSystem()
        tree = DecisionTree()
        tree.build_simple_tree([("temp", 25.0, "comfortable", "hot")])
        sys.set_decision_tree(tree)
        ctx = DecisionContext(facts={"temp": 30.0})
        result = sys.decide(ctx)
        assert result.action == "hot"

    def test_decide_optimizer(self):
        sys = IntelligentDecisionSystem()
        sys.optimizer.add_objective("value", 1.0, lambda x: x)
        ctx = DecisionContext()
        result = sys.decide(ctx, options=[0.3, 0.7, 0.5])
        assert result.action == "0.7"

    def test_decide_default(self):
        sys = IntelligentDecisionSystem()
        ctx = DecisionContext()
        result = sys.decide(ctx)
        assert result.action == "default"

    def test_evaluate_decision(self):
        sys = IntelligentDecisionSystem()
        result = DecisionResult(action="yes", confidence=0.9)
        sys.evaluate_decision(result, "yes")
        stats = sys.get_evaluation_stats()
        assert stats["total"] == 1
        assert stats["accuracy"] == 1.0

    def test_get_stats(self):
        sys = IntelligentDecisionSystem()
        sys.rule_engine.add_rule(Rule("test", lambda ctx: True, "action"))
        tree = DecisionTree()
        tree.build_simple_tree([("x", 0.5, "a", "b")])
        sys.set_decision_tree(tree)
        sys.optimizer.add_objective("score", 1.0, lambda x: x)
        stats = sys.get_stats()
        assert stats["rules_count"] == 1
        assert stats["has_decision_tree"] is True
        assert stats["objectives_count"] == 1

    def test_create_intelligent_decision_system(self):
        sys = create_intelligent_decision_system()
        assert isinstance(sys, IntelligentDecisionSystem)
