"""阶段70: Agent数据质量检查系统 - 单元测试"""
import pytest
from src.agent.data_quality import (
    QualityDimension, QualityLevel, QualityRule, QualityResult, QualityReport,
    DataQualityChecker, QualityRuleBuilder, create_data_quality_checker,
)

class TestDataQualityChecker:
    def test_check_pass(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name"))
        report = checker.check({"name": "Alice"})
        assert report.passed_checks == 1
        assert report.score == 100.0
        assert report.level == QualityLevel.EXCELLENT

    def test_check_fail(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name"))
        report = checker.check({"name": None})
        assert report.failed_checks == 1
        assert report.score == 0.0
        assert report.level == QualityLevel.CRITICAL

    def test_in_range(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.in_range("age", 0, 120))
        assert checker.check({"age": 30}).passed_checks == 1
        assert checker.check({"age": 200}).failed_checks == 1

    def test_matches_pattern(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.matches_pattern("email", r"^[\w\.-]+@[\w\.-]+\.\w+$"))
        assert checker.check({"email": "test@example.com"}).passed_checks == 1
        assert checker.check({"email": "invalid"}).failed_checks == 1

    def test_weighted_score(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name", weight=2.0))
        checker.add_rule(QualityRuleBuilder.not_null("age", weight=1.0))
        report = checker.check({"name": "Alice", "age": None})
        assert abs(report.score - 66.67) < 0.01  # 2/3 * 100

    def test_check_batch(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name"))
        reports = checker.check_batch([{"name": "A"}, {"name": None}])
        assert len(reports) == 2
        assert reports[0].passed_checks == 1
        assert reports[1].failed_checks == 1

    def test_report_to_dict(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name"))
        report = checker.check({"name": "Alice"})
        d = report.to_dict()
        assert d["score"] == 100.0
        assert d["level"] == "excellent"

    def test_stats(self):
        checker = DataQualityChecker()
        checker.add_rule(QualityRuleBuilder.not_null("name"))
        checker.add_rule(QualityRuleBuilder.in_range("age", 0, 120))
        stats = checker.get_stats()
        assert stats["total_rules"] == 2

    def test_create_data_quality_checker(self):
        checker = create_data_quality_checker()
        assert isinstance(checker, DataQualityChecker)
