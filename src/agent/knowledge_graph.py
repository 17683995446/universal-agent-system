"""
阶段77: Agent知识图谱构建系统

遵循"图难于其易，为大于其细"的原则，实现实体抽取、关系构建、图存储、路径查询。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class KGEntity:
    entity_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
        }


@dataclass
class KGRelation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
        }


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


class KnowledgeGraph:
    """知识图谱"""

    def __init__(self):
        self._entities: Dict[str, KGEntity] = {}
        self._relations: Dict[str, KGRelation] = {}
        # 邻接表: entity_id -> [(target_id, relation_type, relation_id)]
        self._adj: Dict[str, List[Tuple[str, str, str]]] = {}
        # 反向邻接表
        self._rev_adj: Dict[str, List[Tuple[str, str, str]]] = {}

    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   properties: Optional[Dict[str, Any]] = None) -> KGEntity:
        entity = KGEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        self._entities[entity_id] = entity
        if entity_id not in self._adj:
            self._adj[entity_id] = []
        if entity_id not in self._rev_adj:
            self._rev_adj[entity_id] = []
        return entity

    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[KGEntity]:
        for entity in self._entities.values():
            if entity.name == name:
                return entity
        return None

    def add_relation(self, relation_id: str, source_id: str, target_id: str,
                     relation_type: str, properties: Optional[Dict[str, Any]] = None) -> Optional[KGRelation]:
        if source_id not in self._entities or target_id not in self._entities:
            return None
        relation = KGRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
        )
        self._relations[relation_id] = relation
        self._adj[source_id].append((target_id, relation_type, relation_id))
        self._rev_adj[target_id].append((source_id, relation_type, relation_id))
        return relation

    def get_relation(self, relation_id: str) -> Optional[KGRelation]:
        return self._relations.get(relation_id)

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[KGEntity]:
        if entity_id not in self._adj:
            return []
        neighbors = []
        for target_id, rel_type, _ in self._adj[entity_id]:
            if relation_type is None or rel_type == relation_type:
                target = self._entities.get(target_id)
                if target:
                    neighbors.append(target)
        return neighbors

    def get_relations_between(self, source_id: str, target_id: str) -> List[KGRelation]:
        results = []
        for tid, _, rid in self._adj.get(source_id, []):
            if tid == target_id:
                rel = self._relations.get(rid)
                if rel:
                    results.append(rel)
        return results

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> Optional[List[Triple]]:
        """BFS查找最短路径"""
        if start_id not in self._entities or end_id not in self._entities:
            return None
        if start_id == end_id:
            return []

        visited = {start_id}
        queue = deque([(start_id, [])])

        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for target_id, rel_type, _ in self._adj.get(current, []):
                triple = Triple(
                    subject=self._entities[current].name,
                    predicate=rel_type,
                    object=self._entities[target_id].name,
                )
                new_path = path + [triple]
                if target_id == end_id:
                    return new_path
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, new_path))
        return None

    def query_by_type(self, entity_type: str) -> List[KGEntity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def query_triples(self, subject: Optional[str] = None,
                      predicate: Optional[str] = None,
                      obj: Optional[str] = None) -> List[Triple]:
        results = []
        for relation in self._relations.values():
            source = self._entities.get(relation.source_id)
            target = self._entities.get(relation.target_id)
            if not source or not target:
                continue
            if subject and source.name != subject:
                continue
            if predicate and relation.relation_type != predicate:
                continue
            if obj and target.name != obj:
                continue
            results.append(Triple(
                subject=source.name,
                predicate=relation.relation_type,
                object=target.name,
            ))
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "entity_types": list(set(e.entity_type for e in self._entities.values())),
            "relation_types": list(set(r.relation_type for r in self._relations.values())),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations.values()],
            "stats": self.get_stats(),
        }


class EntityExtractor:
    """实体抽取器（简化版）"""

    def __init__(self):
        self._type_patterns: Dict[str, List[str]] = {}

    def register_pattern(self, entity_type: str, keywords: List[str]) -> None:
        self._type_patterns[entity_type] = keywords

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """返回 [(entity_name, entity_type), ...]"""
        results = []
        for entity_type, keywords in self._type_patterns.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    results.append((keyword, entity_type))
        return results


class RelationExtractor:
    """关系抽取器（简化版）"""

    def __init__(self):
        self._relation_patterns: Dict[str, List[Tuple[str, str]]] = {}

    def register_pattern(self, relation_type: str, pairs: List[Tuple[str, str]]) -> None:
        self._relation_patterns[relation_type] = pairs

    def extract(self, text: str, entities: List[Tuple[str, str]]) -> List[Triple]:
        """基于共现的简单关系抽取"""
        results = []
        entity_names = [e[0] for e in entities]
        for relation_type, pairs in self._relation_patterns.items():
            for subj_keyword, obj_keyword in pairs:
                if subj_keyword in text and obj_keyword in text:
                    # 查找匹配的实体
                    for subj in entity_names:
                        if subj_keyword in subj or subj in subj_keyword:
                            for obj in entity_names:
                                if obj != subj and (obj_keyword in obj or obj in obj_keyword):
                                    results.append(Triple(subj, relation_type, obj))
        return results


class GraphBuilder:
    """图谱构建器"""

    def __init__(self):
        self._kg = KnowledgeGraph()
        self._entity_extractor = EntityExtractor()
        self._relation_extractor = RelationExtractor()
        self._entity_counter = 0
        self._relation_counter = 0

    @property
    def kg(self) -> KnowledgeGraph:
        return self._kg

    def register_entity_type(self, entity_type: str, keywords: List[str]) -> None:
        self._entity_extractor.register_pattern(entity_type, keywords)

    def register_relation_type(self, relation_type: str, pairs: List[Tuple[str, str]]) -> None:
        self._relation_extractor.register_pattern(relation_type, pairs)

    def build_from_text(self, text: str) -> KnowledgeGraph:
        entities = self._entity_extractor.extract(text)
        # 去重添加实体
        seen = set()
        entity_map: Dict[str, str] = {}
        for name, entity_type in entities:
            key = (name, entity_type)
            if key not in seen:
                seen.add(key)
                self._entity_counter += 1
                entity_id = f"e{self._entity_counter}"
                self._kg.add_entity(entity_id, name, entity_type)
                entity_map[name] = entity_id

        # 抽取关系
        triples = self._relation_extractor.extract(text, entities)
        for triple in triples:
            source_id = entity_map.get(triple.subject)
            target_id = entity_map.get(triple.object)
            if source_id and target_id:
                self._relation_counter += 1
                relation_id = f"r{self._relation_counter}"
                self._kg.add_relation(relation_id, source_id, target_id, triple.predicate)

        return self._kg

    def add_fact(self, subject: str, predicate: str, obj: str,
                 subj_type: str = "thing", obj_type: str = "thing") -> bool:
        source = self._kg.get_entity_by_name(subject)
        if not source:
            self._entity_counter += 1
            source = self._kg.add_entity(f"e{self._entity_counter}", subject, subj_type)
        target = self._kg.get_entity_by_name(obj)
        if not target:
            self._entity_counter += 1
            target = self._kg.add_entity(f"e{self._entity_counter}", obj, obj_type)
        self._relation_counter += 1
        relation_id = f"r{self._relation_counter}"
        self._kg.add_relation(relation_id, source.entity_id, target.entity_id, predicate)
        return True


def create_knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph()


def create_graph_builder() -> GraphBuilder:
    return GraphBuilder()
