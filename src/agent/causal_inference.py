"""
阶段86: Agent因果推断系统

遵循"图难于其易，为大于其细"的原则，实现因果图、干预推断、反事实推理、后门准则。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class CausalVariable:
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    domain: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parents": self.parents,
            "children": self.children,
        }


@dataclass
class CausalEdge:
    source: str
    target: str
    strength: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
        }


@dataclass
class Intervention:
    variable: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "value": self.value,
        }


@dataclass
class CounterfactualQuery:
    target: str
    intervention: Intervention
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "intervention": self.intervention.to_dict(),
            "evidence": self.evidence,
        }


class CausalGraph:
    """因果图"""

    def __init__(self):
        self._variables: Dict[str, CausalVariable] = {}
        self._edges: List[CausalEdge] = []

    def add_variable(self, name: str, domain: Optional[List[Any]] = None) -> CausalVariable:
        var = CausalVariable(name=name, domain=domain or [])
        self._variables[name] = var
        return var

    def add_edge(self, source: str, target: str, strength: float = 1.0) -> bool:
        if source not in self._variables or target not in self._variables:
            return False
        self._edges.append(CausalEdge(source=source, target=target, strength=strength))
        self._variables[source].children.append(target)
        self._variables[target].parents.append(source)
        return True

    def get_parents(self, variable: str) -> List[str]:
        var = self._variables.get(variable)
        return var.parents if var else []

    def get_children(self, variable: str) -> List[str]:
        var = self._variables.get(variable)
        return var.children if var else []

    def get_ancestors(self, variable: str) -> Set[str]:
        """获取祖先节点"""
        ancestors = set()
        queue = [variable]
        visited = {variable}
        while queue:
            current = queue.pop(0)
            for parent in self.get_parents(current):
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.append(parent)
        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """获取后代节点"""
        descendants = set()
        queue = [variable]
        visited = {variable}
        while queue:
            current = queue.pop(0)
            for child in self.get_children(current):
                if child not in visited:
                    visited.add(child)
                    descendants.add(child)
                    queue.append(child)
        return descendants

    def is_d_separated(self, x: str, y: str, conditioning: Set[str]) -> bool:
        """检查d-分离（简化版）"""
        # 如果y是x的后代且conditioning不包含中间节点，则不分离
        if y in self.get_descendants(x) and y not in conditioning:
            return False
        # 如果conditioning包含共同原因，则阻断路径
        common_ancestors = self.get_ancestors(x) & self.get_ancestors(y)
        if common_ancestors and common_ancestors <= conditioning:
            return True
        return False

    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """查找后门路径（以箭头指向treatment开始的路径）"""
        paths = []
        # 简化：查找所有从outcome到treatment的路径中，以箭头指向treatment结束的路径
        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == treatment:
                # 检查是否以箭头指向treatment
                if len(path) >= 2:
                    paths.append(path + [treatment])
                return
            for parent in self.get_parents(current):
                if parent not in visited:
                    visited.add(parent)
                    dfs(parent, path + [current], visited)
                    visited.remove(parent)

        visited = {outcome}
        dfs(outcome, [], visited)
        return paths

    def get_backdoor_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """获取后门调整集合"""
        # 简化版：返回所有同时影响treatment和outcome的变量（共同原因）
        ancestors_t = self.get_ancestors(treatment)
        ancestors_o = self.get_ancestors(outcome)
        common_causes = ancestors_t & ancestors_o
        # 排除outcome的后代
        descendants_o = self.get_descendants(outcome)
        return common_causes - descendants_o - {treatment, outcome}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self._variables.values()],
            "edges": [e.to_dict() for e in self._edges],
        }


class StructuralCausalModel:
    """结构因果模型"""

    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self._functions: Dict[str, Callable] = {}
        self._noise: Dict[str, Any] = {}

    def set_function(self, variable: str, func: Callable) -> None:
        self._functions[variable] = func

    def intervene(self, intervention: Intervention,
                  evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行干预 do(X=x)"""
        result = evidence.copy() if evidence else {}
        result[intervention.variable] = intervention.value

        # 按照拓扑顺序计算各变量值
        order = self._topological_sort()
        for var in order:
            if var == intervention.variable:
                continue  # 干预变量的值已固定
            if var in result:
                continue
            func = self._functions.get(var)
            if func:
                # 获取父变量的值
                parent_values = {p: result.get(p, 0) for p in self.graph.get_parents(var)}
                result[var] = func(parent_values)
            else:
                result[var] = 0
        return result

    def counterfactual(self, query: CounterfactualQuery) -> Any:
        """反事实推理"""
        # 步骤1：用证据拟合噪声项（简化版）
        # 步骤2：用拟合的噪声和干预计算反事实结果
        factual = self.intervene(Intervention(query.target, 0), query.evidence)
        counterfactual_result = self.intervene(query.intervention, query.evidence)
        return counterfactual_result.get(query.target, 0)

    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree = {name: 0 for name in self.graph._variables}
        for edge in self.graph._edges:
            in_degree[edge.target] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in self.graph.get_children(current):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order


class CausalInferenceEngine:
    """因果推断引擎"""

    def __init__(self):
        self.graph = CausalGraph()
        self.scm: Optional[StructuralCausalModel] = None

    def build_graph(self, variables: List[str], edges: List[Tuple[str, str]]) -> CausalGraph:
        for var in variables:
            self.graph.add_variable(var)
        for source, target in edges:
            self.graph.add_edge(source, target)
        self.scm = StructuralCausalModel(self.graph)
        return self.graph

    def estimate_ate(self, treatment: str, outcome: str,
                     data: List[Dict[str, Any]]) -> float:
        """估计平均处理效应 (ATE)"""
        # 简化版：计算处理组和对照组的平均差异
        treated = [d[outcome] for d in data if d.get(treatment) == 1]
        control = [d[outcome] for d in data if d.get(treatment) == 0]
        if not treated or not control:
            return 0.0
        return sum(treated) / len(treated) - sum(control) / len(control)

    def estimate_cate(self, treatment: str, outcome: str,
                      conditioning: str, data: List[Dict[str, Any]]) -> Dict[Any, float]:
        """估计条件平均处理效应 (CATE)"""
        cate = {}
        values = set(d.get(conditioning) for d in data)
        for val in values:
            subset = [d for d in data if d.get(conditioning) == val]
            cate[val] = self.estimate_ate(treatment, outcome, subset)
        return cate

    def identify_backdoor(self, treatment: str, outcome: str) -> Set[str]:
        """识别后门路径并返回调整变量集合"""
        return self.graph.get_backdoor_adjustment_set(treatment, outcome)

    def do_calculus(self, treatment: str, value: Any,
                    evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行 do-演算"""
        if not self.scm:
            return {}
        return self.scm.intervene(Intervention(treatment, value), evidence)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "variable_count": len(self.graph._variables),
            "edge_count": len(self.graph._edges),
            "has_scm": self.scm is not None,
        }


def create_causal_inference_engine() -> CausalInferenceEngine:
    return CausalInferenceEngine()
