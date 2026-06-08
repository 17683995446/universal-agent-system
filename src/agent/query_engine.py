"""
阶段64: Agent数据查询引擎

遵循"图难于其易，为大于其细"的原则，实现查询解析、过滤、排序、分页、投影。
"""
import operator
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

class QueryOperator(str):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"

@dataclass
class QueryCondition:
    field: str
    op: str
    value: Any

@dataclass
class QueryPlan:
    conditions: List[QueryCondition] = field(default_factory=list)
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: Optional[int] = None
    offset: int = 0
    select_fields: Optional[List[str]] = None

    def where(self, field: str, op: str, value: Any) -> "QueryPlan":
        self.conditions.append(QueryCondition(field, op, value))
        return self

    def order_by(self, field: str, desc: bool = False) -> "QueryPlan":
        self.sort_by = field
        self.sort_desc = desc
        return self

    def paginate(self, limit: int, offset: int = 0) -> "QueryPlan":
        self.limit = limit
        self.offset = offset
        return self

    def select(self, fields: List[str]) -> "QueryPlan":
        self.select_fields = fields
        return self

class QueryEngine:
    """查询引擎"""
    def __init__(self):
        self._operators: Dict[str, Callable[[Any, Any], bool]] = {
            QueryOperator.EQ: operator.eq,
            QueryOperator.NE: operator.ne,
            QueryOperator.GT: operator.gt,
            QueryOperator.GTE: operator.ge,
            QueryOperator.LT: operator.lt,
            QueryOperator.LTE: operator.le,
            QueryOperator.IN: lambda a, b: a in b,
            QueryOperator.CONTAINS: lambda a, b: b in a if isinstance(a, (str, list)) else False,
            QueryOperator.STARTSWITH: lambda a, b: str(a).startswith(str(b)),
            QueryOperator.ENDSWITH: lambda a, b: str(a).endswith(str(b)),
        }

    def execute(self, data: List[Dict[str, Any]], plan: QueryPlan) -> List[Dict[str, Any]]:
        # Filter
        result = [item for item in data if self._matches(item, plan.conditions)]
        # Sort
        if plan.sort_by:
            result.sort(key=lambda x: x.get(plan.sort_by), reverse=plan.sort_desc)
        # Paginate
        total = len(result)
        result = result[plan.offset:plan.offset + plan.limit] if plan.limit else result[plan.offset:]
        # Project
        if plan.select_fields:
            result = [{k: v for k, v in item.items() if k in plan.select_fields} for item in result]
        return result

    def _matches(self, item: Dict[str, Any], conditions: List[QueryCondition]) -> bool:
        for cond in conditions:
            field_value = item.get(cond.field)
            op_fn = self._operators.get(cond.op)
            if not op_fn:
                return False
            if not op_fn(field_value, cond.value):
                return False
        return True

    def count(self, data: List[Dict[str, Any]], plan: QueryPlan) -> int:
        return len([item for item in data if self._matches(item, plan.conditions)])

    def exists(self, data: List[Dict[str, Any]], plan: QueryPlan) -> bool:
        for item in data:
            if self._matches(item, plan.conditions):
                return True
        return False

    def aggregate(self, data: List[Dict[str, Any]], plan: QueryPlan,
                  agg_field: str, agg_op: str) -> Any:
        filtered = [item for item in data if self._matches(item, plan.conditions)]
        values = [item.get(agg_field) for item in filtered if item.get(agg_field) is not None]
        if not values:
            return None
        if agg_op == "sum":
            return sum(values)
        elif agg_op == "avg":
            return sum(values) / len(values)
        elif agg_op == "min":
            return min(values)
        elif agg_op == "max":
            return max(values)
        elif agg_op == "count":
            return len(values)
        return None

def create_query_engine() -> QueryEngine:
    return QueryEngine()
