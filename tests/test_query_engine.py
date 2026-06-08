"""阶段64: Agent数据查询引擎 - 单元测试"""
import pytest
from src.agent.query_engine import (
    QueryOperator, QueryCondition, QueryPlan, QueryEngine, create_query_engine,
)

DATA = [
    {"id": 1, "name": "Alice", "age": 30, "city": "Beijing"},
    {"id": 2, "name": "Bob", "age": 25, "city": "Shanghai"},
    {"id": 3, "name": "Charlie", "age": 35, "city": "Beijing"},
    {"id": 4, "name": "David", "age": 28, "city": "Guangzhou"},
]

class TestQueryEngine:
    def test_filter_eq(self):
        engine = QueryEngine()
        plan = QueryPlan().where("city", QueryOperator.EQ, "Beijing")
        result = engine.execute(DATA, plan)
        assert len(result) == 2
        assert all(r["city"] == "Beijing" for r in result)

    def test_filter_gt(self):
        engine = QueryEngine()
        plan = QueryPlan().where("age", QueryOperator.GT, 28)
        result = engine.execute(DATA, plan)
        assert len(result) == 2

    def test_filter_in(self):
        engine = QueryEngine()
        plan = QueryPlan().where("city", QueryOperator.IN, ["Beijing", "Shanghai"])
        result = engine.execute(DATA, plan)
        assert len(result) == 3

    def test_filter_contains(self):
        engine = QueryEngine()
        plan = QueryPlan().where("name", QueryOperator.CONTAINS, "li")
        result = engine.execute(DATA, plan)
        assert len(result) == 2  # Alice, Charlie

    def test_sort_asc(self):
        engine = QueryEngine()
        plan = QueryPlan().order_by("age")
        result = engine.execute(DATA, plan)
        assert result[0]["age"] == 25

    def test_sort_desc(self):
        engine = QueryEngine()
        plan = QueryPlan().order_by("age", desc=True)
        result = engine.execute(DATA, plan)
        assert result[0]["age"] == 35

    def test_paginate(self):
        engine = QueryEngine()
        plan = QueryPlan().paginate(limit=2, offset=1)
        result = engine.execute(DATA, plan)
        assert len(result) == 2
        assert result[0]["id"] == 2

    def test_select(self):
        engine = QueryEngine()
        plan = QueryPlan().select(["name", "age"])
        result = engine.execute(DATA, plan)
        assert set(result[0].keys()) == {"name", "age"}

    def test_count(self):
        engine = QueryEngine()
        plan = QueryPlan().where("city", QueryOperator.EQ, "Beijing")
        assert engine.count(DATA, plan) == 2

    def test_exists(self):
        engine = QueryEngine()
        plan = QueryPlan().where("name", QueryOperator.EQ, "Alice")
        assert engine.exists(DATA, plan)

    def test_aggregate_sum(self):
        engine = QueryEngine()
        plan = QueryPlan()
        assert engine.aggregate(DATA, plan, "age", "sum") == 118

    def test_aggregate_avg(self):
        engine = QueryEngine()
        plan = QueryPlan()
        assert engine.aggregate(DATA, plan, "age", "avg") == 29.5

    def test_create_query_engine(self):
        engine = create_query_engine()
        assert isinstance(engine, QueryEngine)
