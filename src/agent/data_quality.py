"""
阶段70: Agent数据质量检查系统

遵循"图难于其易，为大于其细"的原则，实现完整性、准确性、一致性、及时性检查。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

class QualityDimension(str, Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"

class QualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class QualityRule:
    name: str
    dimension: QualityDimension
    check_fn: Callable[[Dict[str, Any]], bool]
    weight: float = 1.0
    description: str = ""

@dataclass
class QualityResult:
    rule_name: str
    passed: bool
    dimension: QualityDimension
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityReport:
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    score: float = 0.0
    level: QualityLevel = QualityLevel.GOOD
    results: List[QualityResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "score": round(self.score, 2),
            "level": self.level.value,
        }

class DataQualityChecker:
    """数据质量检查器"""
    def __init__(self):
        self._rules: List[QualityRule] = []

    def add_rule(self, rule: QualityRule) -> None:
        self._rules.append(rule)

    def check(self, data: Dict[str, Any]) -> QualityReport:
        report = QualityReport()
        total_weight = sum(r.weight for r in self._rules)
        weighted_score = 0.0

        for rule in self._rules:
            report.total_checks += 1
            try:
                passed = rule.check_fn(data)
                result = QualityResult(
                    rule_name=rule.name,
                    passed=passed,
                    dimension=rule.dimension,
                    message="通过" if passed else "未通过",
                )
                if passed:
                    report.passed_checks += 1
                    weighted_score += rule.weight
                else:
                    report.failed_checks += 1
            except Exception as e:
                result = QualityResult(
                    rule_name=rule.name,
                    passed=False,
                    dimension=rule.dimension,
                    message=f"检查异常: {e}",
                )
                report.failed_checks += 1
            report.results.append(result)

        report.score = (weighted_score / total_weight * 100) if total_weight > 0 else 0
        report.level = self._calculate_level(report.score)
        return report

    def check_batch(self, data_list: List[Dict[str, Any]]) -> List[QualityReport]:
        return [self.check(data) for data in data_list]

    def _calculate_level(self, score: float) -> QualityLevel:
        if score >= 95:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        return QualityLevel.CRITICAL

    def get_stats(self) -> Dict[str, Any]:
        by_dimension: Dict[str, int] = {}
        for rule in self._rules:
            dim = rule.dimension.value
            by_dimension[dim] = by_dimension.get(dim, 0) + 1
        return {
            "total_rules": len(self._rules),
            "by_dimension": by_dimension,
        }

class QualityRuleBuilder:
    """质量规则构建器"""
    @staticmethod
    def not_null(field: str, weight: float = 1.0) -> QualityRule:
        return QualityRule(
            name=f"{field}_not_null",
            dimension=QualityDimension.COMPLETENESS,
            check_fn=lambda d: d.get(field) is not None,
            weight=weight,
            description=f"{field} 不能为空",
        )

    @staticmethod
    def in_range(field: str, min_val: Any, max_val: Any, weight: float = 1.0) -> QualityRule:
        return QualityRule(
            name=f"{field}_in_range",
            dimension=QualityDimension.VALIDITY,
            check_fn=lambda d: min_val <= d.get(field) <= max_val if d.get(field) is not None else True,
            weight=weight,
            description=f"{field} 必须在 [{min_val}, {max_val}] 范围内",
        )

    @staticmethod
    def matches_pattern(field: str, pattern: str, weight: float = 1.0) -> QualityRule:
        import re
        compiled = re.compile(pattern)
        return QualityRule(
            name=f"{field}_matches_pattern",
            dimension=QualityDimension.VALIDITY,
            check_fn=lambda d: bool(compiled.match(str(d.get(field, "")))) if d.get(field) is not None else True,
            weight=weight,
            description=f"{field} 必须匹配模式 {pattern}",
        )

    @staticmethod
    def unique(fields: List[str], weight: float = 1.0) -> QualityRule:
        seen = set()
        def check_fn(d):
            key = tuple(d.get(f) for f in fields)
            if key in seen:
                return False
            seen.add(key)
            return True
        return QualityRule(
            name=f"{'_'.join(fields)}_unique",
            dimension=QualityDimension.CONSISTENCY,
            check_fn=check_fn,
            weight=weight,
            description=f"{'/'.join(fields)} 必须唯一",
        )

def create_data_quality_checker() -> DataQualityChecker:
    return DataQualityChecker()
