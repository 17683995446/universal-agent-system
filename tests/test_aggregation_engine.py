"""阶段65: Agent数据聚合引擎 - 单元测试"""
import pytest
from src.agent.aggregation_engine import AggregationEngine, create_aggregation_engine

DATA = [
    {"id": 1, "city": "Beijing", "age": 30, "salary": 10000},
    {"id": 2, "city": "Shanghai", "age": 25, "salary": 8000},
    {"id": 3, "city": "Beijing", "age": 35, "salary": 12000},
    {"id": 4, "city": "Shanghai", "age": 28, "salary": 9000},
    {"id": 5, "city": "Beijing", "age": 40, "salary": 15000},
]

class TestAggregationEngine:
    def test_group_by(self):
        engine = AggregationEngine()
        result = engine.group_by(DATA, "city")
        assert len(result["Beijing"]) == 3
        assert len(result["Shanghai"]) == 2

    def test_aggregate_sum(self):
        engine = AggregationEngine()
        result = engine.aggregate(DATA, "city", "salary", "sum")
        assert result["Beijing"] == 37000
        assert result["Shanghai"] == 17000

    def test_aggregate_avg(self):
        engine = AggregationEngine()
        result = engine.aggregate(DATA, "city", "age", "avg")
        assert result["Beijing"] == 35.0

    def test_multi_aggregate(self):
        engine = AggregationEngine()
        result = engine.multi_aggregate(DATA, ["city"], {"salary": "sum", "age": "avg"})
        assert result[("Beijing",)]["salary"] == 37000

    def test_top_n(self):
        engine = AggregationEngine()
        result = engine.top_n(DATA, "salary", n=3)
        assert len(result) == 3
        assert result[0]["salary"] == 15000

    def test_histogram(self):
        engine = AggregationEngine()
        result = engine.histogram(DATA, "age", bins=3)
        assert len(result) == 3
        assert sum(result.values()) == 5

    def test_percentile(self):
        engine = AggregationEngine()
        result = engine.percentile(DATA, "salary", p=0.5)
        assert result == 10000

    def test_distinct(self):
        engine = AggregationEngine()
        result = engine.distinct(DATA, "city")
        assert set(result) == {"Beijing", "Shanghai"}

    def test_create_aggregation_engine(self):
        engine = create_aggregation_engine()
        assert isinstance(engine, AggregationEngine)
