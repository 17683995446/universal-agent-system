"""
阶段69: Agent数据血缘追踪系统

遵循"图难于其易，为大于其细"的原则，实现数据来源追踪、转换链路、影响分析、血缘图谱。
"""
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

class LineageType(str, Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    SINK = "sink"

@dataclass
class LineageNode:
    node_id: str
    name: str
    node_type: LineageType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LineageEdge:
    from_node: str
    to_node: str
    transform_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataLineage:
    """数据血缘"""
    def __init__(self):
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._upstream: Dict[str, Set[str]] = {}  # node -> upstream nodes
        self._downstream: Dict[str, Set[str]] = {}  # node -> downstream nodes

    def add_node(self, node: LineageNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._upstream:
            self._upstream[node.node_id] = set()
        if node.node_id not in self._downstream:
            self._downstream[node.node_id] = set()

    def add_edge(self, edge: LineageEdge) -> None:
        self._edges.append(edge)
        if edge.from_node in self._downstream:
            self._downstream[edge.from_node].add(edge.to_node)
        if edge.to_node in self._upstream:
            self._upstream[edge.to_node].add(edge.from_node)

    def get_upstream(self, node_id: str, depth: int = -1) -> List[LineageNode]:
        """获取上游节点"""
        result = []
        visited = set()
        self._traverse(node_id, self._upstream, result, visited, depth)
        return result

    def get_downstream(self, node_id: str, depth: int = -1) -> List[LineageNode]:
        """获取下游节点"""
        result = []
        visited = set()
        self._traverse(node_id, self._downstream, result, visited, depth)
        return result

    def _traverse(self, node_id: str, graph: Dict[str, Set[str]],
                  result: List[LineageNode], visited: Set[str], depth: int) -> None:
        if depth == 0 or node_id in visited:
            return
        visited.add(node_id)
        for neighbor in graph.get(node_id, set()):
            if neighbor not in visited:
                node = self._nodes.get(neighbor)
                if node:
                    result.append(node)
                self._traverse(neighbor, graph, result, visited, depth - 1)

    def get_lineage_path(self, from_node: str, to_node: str) -> List[str]:
        """获取两个节点之间的路径"""
        visited = set()
        path = []
        if self._dfs(from_node, to_node, visited, path):
            return path
        return []

    def _dfs(self, current: str, target: str, visited: Set[str], path: List[str]) -> bool:
        if current == target:
            path.append(current)
            return True
        visited.add(current)
        path.append(current)
        for neighbor in self._downstream.get(current, set()):
            if neighbor not in visited:
                if self._dfs(neighbor, target, visited, path):
                    return True
        path.pop()
        return False

    def get_impact(self, node_id: str) -> Dict[str, Any]:
        """获取节点变更影响分析"""
        downstream = self.get_downstream(node_id)
        return {
            "node": node_id,
            "direct_downstream": len(self._downstream.get(node_id, set())),
            "total_downstream": len(downstream),
            "affected_nodes": [n.name for n in downstream],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "sources": sum(1 for n in self._nodes.values() if n.node_type == LineageType.SOURCE),
            "transforms": sum(1 for n in self._nodes.values() if n.node_type == LineageType.TRANSFORM),
            "sinks": sum(1 for n in self._nodes.values() if n.node_type == LineageType.SINK),
        }

class LineageTracker:
    """血缘追踪器"""
    def __init__(self):
        self._lineage = DataLineage()

    def track_source(self, name: str, metadata: Optional[Dict] = None) -> str:
        node_id = str(uuid.uuid4())[:8]
        self._lineage.add_node(LineageNode(node_id, name, LineageType.SOURCE, metadata or {}))
        return node_id

    def track_transform(self, name: str, from_nodes: List[str],
                        transform_type: str = "", metadata: Optional[Dict] = None) -> str:
        node_id = str(uuid.uuid4())[:8]
        self._lineage.add_node(LineageNode(node_id, name, LineageType.TRANSFORM, metadata or {}))
        for from_node in from_nodes:
            self._lineage.add_edge(LineageEdge(from_node, node_id, transform_type))
        return node_id

    def track_sink(self, name: str, from_nodes: List[str], metadata: Optional[Dict] = None) -> str:
        node_id = str(uuid.uuid4())[:8]
        self._lineage.add_node(LineageNode(node_id, name, LineageType.SINK, metadata or {}))
        for from_node in from_nodes:
            self._lineage.add_edge(LineageEdge(from_node, node_id))
        return node_id

    def get_lineage(self) -> DataLineage:
        return self._lineage

def create_lineage_tracker() -> LineageTracker:
    return LineageTracker()
