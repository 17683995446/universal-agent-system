"""
阶段97: Agent 神经架构搜索系统 (NAS)

遵循"图难于其易，为大于其细"的原则，实现搜索空间定义、随机搜索、进化搜索、架构评估。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import copy


@dataclass
class Architecture:
    """神经网络架构"""
    name: str
    layers: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layers": len(self.layers),
            "score": round(self.score, 4),
        }

    def copy(self) -> "Architecture":
        return Architecture(
            name=self.name,
            layers=copy.deepcopy(self.layers),
            score=self.score,
        )


class SearchSpace:
    """搜索空间"""

    def __init__(self):
        self.operations = ["linear", "conv", "pool", "relu", "tanh"]
        self.layer_sizes = [16, 32, 64, 128]
        self.max_layers = 5

    def sample_random(self, name: str = "arch") -> Architecture:
        """随机采样架构"""
        n_layers = random.randint(1, self.max_layers)
        layers = []
        for i in range(n_layers):
            op = random.choice(self.operations)
            size = random.choice(self.layer_sizes)
            layers.append({
                "id": i,
                "op": op,
                "size": size,
            })
        return Architecture(name=f"{name}_{random.randint(0, 10000)}", layers=layers)

    def sample_neighbors(self, arch: Architecture, n: int = 3) -> List[Architecture]:
        """采样邻居架构"""
        neighbors = []
        for _ in range(n):
            neighbor = arch.copy()
            if neighbor.layers:
                # 随机修改一层
                idx = random.randint(0, len(neighbor.layers) - 1)
                neighbor.layers[idx]["op"] = random.choice(self.operations)
                neighbor.layers[idx]["size"] = random.choice(self.layer_sizes)
            neighbor.name = f"{arch.name}_mut_{random.randint(0, 1000)}"
            neighbors.append(neighbor)
        return neighbors

    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """交叉操作"""
        child = Architecture(name=f"cross_{random.randint(0, 10000)}")
        min_len = min(len(parent1.layers), len(parent2.layers))
        if min_len > 0:
            split = random.randint(1, min_len)
            child.layers = copy.deepcopy(parent1.layers[:split]) + copy.deepcopy(parent2.layers[split:])
        return child


class ArchitectureEvaluator:
    """架构评估器"""

    def __init__(self):
        self._cache: Dict[str, float] = {}

    def evaluate(self, arch: Architecture,
                 evaluator_fn: Optional[Callable[[Architecture], float]] = None) -> float:
        """评估架构"""
        cache_key = str(arch.to_dict())
        if cache_key in self._cache:
            return self._cache[cache_key]

        if evaluator_fn:
            score = evaluator_fn(arch)
        else:
            score = self._default_evaluate(arch)

        arch.score = score
        self._cache[cache_key] = score
        return score

    def _default_evaluate(self, arch: Architecture) -> float:
        """默认评估：基于层数和操作类型的启发式"""
        score = 0.0
        for layer in arch.layers:
            if layer["op"] == "linear":
                score += 1.0
            elif layer["op"] == "conv":
                score += 1.5
            elif layer["op"] == "relu":
                score += 0.5
            score += layer["size"] / 128.0
        # 惩罚过深的网络
        score -= len(arch.layers) * 0.2
        return max(0.0, score)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "evaluated": len(self._cache),
        }


class RandomSearchNAS:
    """随机搜索NAS"""

    def __init__(self, search_space: SearchSpace, evaluator: ArchitectureEvaluator):
        self.search_space = search_space
        self.evaluator = evaluator
        self.best_arch: Optional[Architecture] = None
        self._history: List[Architecture] = []

    def search(self, n_samples: int = 20,
               evaluator_fn: Optional[Callable] = None) -> Architecture:
        """随机搜索"""
        best_score = float("-inf")
        for _ in range(n_samples):
            arch = self.search_space.sample_random()
            score = self.evaluator.evaluate(arch, evaluator_fn)
            self._history.append(arch)
            if score > best_score:
                best_score = score
                self.best_arch = arch
        return self.best_arch if self.best_arch else Architecture(name="empty")


class EvolutionaryNAS:
    """进化搜索NAS"""

    def __init__(self, search_space: SearchSpace, evaluator: ArchitectureEvaluator,
                 population_size: int = 10):
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = population_size
        self.population: List[Architecture] = []

    def initialize(self):
        """初始化种群"""
        self.population = [self.search_space.sample_random() for _ in range(self.population_size)]
        for arch in self.population:
            self.evaluator.evaluate(arch)

    def evolve(self, n_generations: int = 5,
               evaluator_fn: Optional[Callable] = None) -> Architecture:
        """进化"""
        if not self.population:
            self.initialize()

        for _ in range(n_generations):
            # 评估
            for arch in self.population:
                self.evaluator.evaluate(arch, evaluator_fn)

            # 选择前50%
            sorted_pop = sorted(self.population, key=lambda a: a.score, reverse=True)
            survivors = sorted_pop[:self.population_size // 2]

            # 生成后代
            offspring = []
            while len(offspring) < self.population_size - len(survivors):
                if len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self.search_space.crossover(p1, p2)
                else:
                    child = survivors[0].copy()
                # 变异
                mutated = self.search_space.sample_neighbors(child, n=1)[0]
                offspring.append(mutated)

            self.population = survivors + offspring

        best = max(self.population, key=lambda a: a.score)
        return best


class NASEngine:
    """NAS引擎"""

    def __init__(self):
        self.search_space = SearchSpace()
        self.evaluator = ArchitectureEvaluator()
        self.random_search = RandomSearchNAS(self.search_space, self.evaluator)
        self.evolutionary = EvolutionaryNAS(self.search_space, self.evaluator)

    def search_random(self, n_samples: int = 20,
                      evaluator_fn: Optional[Callable] = None) -> Architecture:
        """随机搜索"""
        return self.random_search.search(n_samples, evaluator_fn)

    def search_evolutionary(self, n_generations: int = 5,
                            evaluator_fn: Optional[Callable] = None) -> Architecture:
        """进化搜索"""
        return self.evolutionary.evolve(n_generations, evaluator_fn)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "random_search_history": len(self.random_search._history),
            "evaluator": self.evaluator.get_stats(),
        }


def create_nas_engine() -> NASEngine:
    return NASEngine()
