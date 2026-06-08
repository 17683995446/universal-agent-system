"""
阶段65: Agent数据聚合引擎

遵循"图难于其易，为大于其细"的原则，实现分组聚合、多维分析、TopN、直方图。
"""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

class AggregationEngine:
    """聚合引擎"""

    def group_by(self, data: List[Dict[str, Any]], field: str) -> Dict[Any, List[Dict[str, Any]]]:
        """按字段分组"""
        result: Dict[Any, List[Dict]] = defaultdict(list)
        for item in data:
            key = item.get(field)
            result[key].append(item)
        return dict(result)

    def aggregate(self, data: List[Dict[str, Any]], group_field: str,
                  agg_field: str, agg_func: str) -> Dict[Any, Any]:
        """分组聚合"""
        groups = self.group_by(data, group_field)
        result = {}
        for key, items in groups.items():
            values = [item.get(agg_field) for item in items if item.get(agg_field) is not None]
            result[key] = self._apply_agg(values, agg_func)
        return result

    def multi_aggregate(self, data: List[Dict[str, Any]], group_fields: List[str],
                        aggregations: Dict[str, str]) -> Dict[Tuple, Dict[str, Any]]:
        """多字段分组 + 多聚合"""
        result: Dict[Tuple, List[Dict]] = defaultdict(list)
        for item in data:
            key = tuple(item.get(f) for f in group_fields)
            result[key].append(item)

        output = {}
        for key, items in result.items():
            agg_result = {}
            for field, func in aggregations.items():
                values = [item.get(field) for item in items if item.get(field) is not None]
                agg_result[field] = self._apply_agg(values, func)
            output[key] = agg_result
        return output

    def top_n(self, data: List[Dict[str, Any]], field: str, n: int = 10,
              desc: bool = True) -> List[Dict[str, Any]]:
        """TopN排序"""
        sorted_data = sorted(data, key=lambda x: x.get(field) or 0, reverse=desc)
        return sorted_data[:n]

    def histogram(self, data: List[Dict[str, Any]], field: str,
                  bins: int = 10) -> Dict[str, int]:
        """直方图"""
        values = [item.get(field) for item in data if isinstance(item.get(field), (int, float))]
        if not values:
            return {}
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {str(min_val): len(values)}
        bin_size = (max_val - min_val) / bins
        result: Dict[str, int] = defaultdict(int)
        for v in values:
            bin_idx = min(int((v - min_val) / bin_size), bins - 1)
            bin_start = min_val + bin_idx * bin_size
            bin_end = min_val + (bin_idx + 1) * bin_size
            key = f"{bin_start:.2f}-{bin_end:.2f}"
            result[key] += 1
        return dict(result)

    def percentile(self, data: List[Dict[str, Any]], field: str, p: float = 0.5) -> Optional[float]:
        """百分位数"""
        values = sorted([item.get(field) for item in data
                        if isinstance(item.get(field), (int, float))])
        if not values:
            return None
        idx = int(len(values) * p)
        return values[min(idx, len(values) - 1)]

    def distinct(self, data: List[Dict[str, Any]], field: str) -> List[Any]:
        """去重值列表"""
        return list({item.get(field) for item in data if field in item})

    def _apply_agg(self, values: List[Any], func: str) -> Any:
        if not values:
            return None
        if func == "sum":
            return sum(values)
        elif func == "avg":
            return sum(values) / len(values)
        elif func == "min":
            return min(values)
        elif func == "max":
            return max(values)
        elif func == "count":
            return len(values)
        elif func == "first":
            return values[0]
        elif func == "last":
            return values[-1]
        return None

def create_aggregation_engine() -> AggregationEngine:
    return AggregationEngine()
