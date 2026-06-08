"""阶段77: Agent知识图谱构建系统 - 单元测试"""
import pytest
from src.agent.knowledge_graph import (
    KGEntity, KGRelation, Triple,
    KnowledgeGraph, EntityExtractor, RelationExtractor, GraphBuilder,
    create_knowledge_graph, create_graph_builder,
)


class TestKnowledgeGraph:
    def test_add_and_get_entity(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("e1", "Alice", "person")
        assert kg.get_entity("e1") == e
        assert e.name == "Alice"

    def test_get_entity_by_name(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        assert kg.get_entity_by_name("Alice") is not None
        assert kg.get_entity_by_name("Bob") is None

    def test_add_relation(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        r = kg.add_relation("r1", "e1", "e2", "knows")
        assert r is not None
        assert r.relation_type == "knows"

    def test_add_relation_missing_entity(self):
        kg = KnowledgeGraph()
        r = kg.add_relation("r1", "e1", "e2", "knows")
        assert r is None

    def test_get_neighbors(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_entity("e3", "Charlie", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        kg.add_relation("r2", "e1", "e3", "likes")
        neighbors = kg.get_neighbors("e1")
        assert len(neighbors) == 2
        neighbors_filtered = kg.get_neighbors("e1", "knows")
        assert len(neighbors_filtered) == 1
        assert neighbors_filtered[0].name == "Bob"

    def test_get_relations_between(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        rels = kg.get_relations_between("e1", "e2")
        assert len(rels) == 1

    def test_find_path_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        path = kg.find_path("e1", "e2")
        assert len(path) == 1
        assert path[0].predicate == "knows"

    def test_find_path_indirect(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_entity("e3", "Charlie", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        kg.add_relation("r2", "e2", "e3", "knows")
        path = kg.find_path("e1", "e3")
        assert len(path) == 2

    def test_find_path_no_path(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        path = kg.find_path("e1", "e2")
        assert path is None

    def test_query_by_type(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Apple", "company")
        persons = kg.query_by_type("person")
        assert len(persons) == 1
        assert persons[0].name == "Alice"

    def test_query_triples(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        triples = kg.query_triples(subject="Alice")
        assert len(triples) == 1
        assert triples[0].predicate == "knows"

    def test_get_stats(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Alice", "person")
        kg.add_entity("e2", "Bob", "person")
        kg.add_relation("r1", "e1", "e2", "knows")
        stats = kg.get_stats()
        assert stats["entity_count"] == 2
        assert stats["relation_count"] == 1


class TestEntityExtractor:
    def test_extract(self):
        extractor = EntityExtractor()
        extractor.register_pattern("person", ["Alice", "Bob"])
        extractor.register_pattern("company", ["Apple"])
        results = extractor.extract("Alice works at Apple.")
        assert ("Alice", "person") in results
        assert ("Apple", "company") in results


class TestRelationExtractor:
    def test_extract(self):
        rel_ext = RelationExtractor()
        rel_ext.register_pattern("works_at", [("Alice", "Apple")])
        entities = [("Alice", "person"), ("Apple", "company")]
        triples = rel_ext.extract("Alice works at Apple.", entities)
        assert len(triples) >= 1
        assert triples[0].predicate == "works_at"


class TestGraphBuilder:
    def test_build_from_text(self):
        builder = GraphBuilder()
        builder.register_entity_type("person", ["Alice", "Bob"])
        builder.register_relation_type("knows", [("Alice", "Bob")])
        kg = builder.build_from_text("Alice knows Bob.")
        assert kg.get_stats()["entity_count"] == 2
        assert kg.get_stats()["relation_count"] == 1

    def test_add_fact(self):
        builder = GraphBuilder()
        assert builder.add_fact("Alice", "knows", "Bob")
        stats = builder.kg.get_stats()
        assert stats["entity_count"] == 2
        assert stats["relation_count"] == 1

    def test_add_fact_existing_entity(self):
        builder = GraphBuilder()
        builder.add_fact("Alice", "knows", "Bob")
        builder.add_fact("Alice", "likes", "Charlie")
        stats = builder.kg.get_stats()
        assert stats["entity_count"] == 3
        assert stats["relation_count"] == 2


class TestFactories:
    def test_create_knowledge_graph(self):
        kg = create_knowledge_graph()
        assert isinstance(kg, KnowledgeGraph)

    def test_create_graph_builder(self):
        builder = create_graph_builder()
        assert isinstance(builder, GraphBuilder)
