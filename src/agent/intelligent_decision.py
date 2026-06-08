"""
阶段80: Agent智能决策系统

遵循"图难于其易，为大于其细"的原则，实现决策树、规则引擎、多目标优化、决策评估。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class DecisionContext:
    """决策上下文"""
    facts: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value


@dataclass
class DecisionResult:
    """决策结果"""
    action: str
    confidence: float
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


class Rule:
    """规则"""

    def __init__(self, name: str, condition: Callable[[DecisionContext], bool],
                 action: str, priority: int = 0, confidence: float = 1.0):
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.confidence = confidence

    def evaluate(self, context: DecisionContext) -> Optional[DecisionResult]:
        if self.condition(context):
            return DecisionResult(
                action=self.action,
                confidence=self.confidence,
                reasoning=f"Rule '{self.name}' matched",
            )
        return None


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self._rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)

    def evaluate(self, context: DecisionContext) -> Optional[DecisionResult]:
        for rule in self._rules:
            result = rule.evaluate(context)
            if result:
                return result
        return None

    def evaluate_all(self, context: DecisionContext) -> List[DecisionResult]:
        results = []
        for rule in self._rules:
            result = rule.evaluate(context)
            if result:
                results.append(result)
        return results


class DecisionTreeNode:
    """决策树节点"""

    def __init__(self, feature: str = "", threshold: float = 0.0,
                 left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None,
                 label: str = "", is_leaf: bool = False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf

    def predict(self, features: Dict[str, float]) -> str:
        if self.is_leaf:
            return self.label
        value = features.get(self.feature, 0.0)
        if value <= self.threshold:
            if self.left:
                return self.left.predict(features)
            return self.label
        else:
            if self.right:
                return self.right.predict(features)
            return self.label


class DecisionTree:
    """决策树"""

    def __init__(self):
        self._root: Optional[DecisionTreeNode] = None

    def set_root(self, node: DecisionTreeNode) -> None:
        self._root = node

    def predict(self, features: Dict[str, float]) -> str:
        if not self._root:
            return "unknown"
        return self._root.predict(features)

    def build_simple_tree(self, rules: List[Tuple[str, float, str, str]]) -> None:
        """构建简单决策树: (feature, threshold, true_label, false_label)"""
        if not rules:
            return
        # 递归构建
        def build(idx: int) -> DecisionTreeNode:
            if idx >= len(rules):
                return DecisionTreeNode(label="unknown", is_leaf=True)
            feature, threshold, true_label, false_label = rules[idx]
            node = DecisionTreeNode(
                feature=feature,
                threshold=threshold,
                left=DecisionTreeNode(label=true_label, is_leaf=True),
                right=build(idx + 1) if idx + 1 < len(rules) else DecisionTreeNode(label=false_label, is_leaf=True),
            )
            return node
        self._root = build(0)


class MultiObjectiveOptimizer:
    """多目标优化器（简化版加权评分）"""

    def __init__(self):
        self._objectives: List[Tuple[str, float, Callable[[Any], float]]] = []

    def add_objective(self, name: str, weight: float,
                      scorer: Callable[[Any], float]) -> None:
        self._objectives.append((name, weight, scorer))

    def evaluate(self, option: Any) -> Tuple[float, Dict[str, float]]:
        """返回 (总分, 各目标得分)"""
        total = 0.0
        scores = {}
        for name, weight, scorer in self._objectives:
            score = scorer(option)
            scores[name] = score
            total += weight * score
        return total, scores

    def optimize(self, options: List[Any]) -> List[Tuple[Any, float, Dict[str, float]]]:
        """对所有选项排序"""
        results = []
        for option in options:
            total, scores = self.evaluate(option)
            results.append((option, total, scores))
        return sorted(results, key=lambda x: -x[1])


class DecisionEvaluator:
    """决策评估器"""

    def __init__(self):
        self._history: List[Tuple[DecisionResult, Any]] = []

    def record(self, result: DecisionResult, outcome: Any) -> None:
        self._history.append((result, outcome))

    def get_accuracy(self) -> float:
        if not self._history:
            return 0.0
        correct = sum(1 for r, o in self._history if r.action == str(o))
        return correct / len(self._history)

    def get_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {"total": 0, "accuracy": 0.0}
        return {
            "total": len(self._history),
            "accuracy": round(self.get_accuracy(), 4),
        }


class IntelligentDecisionSystem:
    """智能决策系统"""

    def __init__(self):
        self._rule_engine = RuleEngine()
        self._decision_tree: Optional[DecisionTree] = None
        self._optimizer = MultiObjectiveOptimizer()
        self._evaluator = DecisionEvaluator()

    @property
    def rule_engine(self) -> RuleEngine:
        return self._rule_engine

    @property
    def optimizer(self) -> MultiObjectiveOptimizer:
        return self._optimizer

    def set_decision_tree(self, tree: DecisionTree) -> None:
        self._decision_tree = tree

    def decide(self, context: DecisionContext,
               options: Optional[List[Any]] = None) -> DecisionResult:
        """综合决策"""
        # 1. 先尝试规则引擎
        result = self._rule_engine.evaluate(context)
        if result:
            return result

        # 2. 尝试决策树
        if self._decision_tree:
            features = {k: float(v) for k, v in context.facts.items() if isinstance(v, (int, float))}
            label = self._decision_tree.predict(features)
            return DecisionResult(
                action=label,
                confidence=0.8,
                reasoning="Decision tree prediction",
            )

        # 3. 多目标优化
        if options:
            best = self._optimizer.optimize(options)
            if best:
                option, score, _ = best[0]
                return DecisionResult(
                    action=str(option),
                    confidence=min(1.0, max(0.0, score)),
                    reasoning="Multi-objective optimization",
                )

        # 4. 默认决策
        return DecisionResult(
            action="default",
            confidence=0.0,
            reasoning="No matching rule or model",
        )

    def evaluate_decision(self, result: DecisionResult, outcome: Any) -> None:
        self._evaluator.record(result, outcome)

    def get_evaluation_stats(self) -> Dict[str, Any]:
        return self._evaluator.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "rules_count": len(self._rule_engine._rules),
            "has_decision_tree": self._decision_tree is not None,
            "objectives_count": len(self._optimizer._objectives),
            "evaluation": self._evaluator.get_stats(),
        }


def create_intelligent_decision_system() -> IntelligentDecisionSystem:
    return IntelligentDecisionSystem()
