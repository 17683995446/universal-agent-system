"""
阶段78: Agent语义搜索系统

遵循"图难于其易，为大于其细"的原则，实现文档索引、向量相似度、语义匹配、排序。
"""
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Document:
    doc_id: str
    content: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
        }


@dataclass
class SearchResult:
    doc_id: str
    score: float
    doc: Optional[Document] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "doc": self.doc.to_dict() if self.doc else None,
        }


class Tokenizer:
    """简单分词器"""

    def tokenize(self, text: str) -> List[str]:
        # 简单按非字母数字字符分割
        import re
        return [t.lower() for t in re.findall(r'[a-zA-Z0-9\u4e00-\u9fff]+', text)]


class TFIDFIndex:
    """TF-IDF倒排索引"""

    def __init__(self):
        self._docs: Dict[str, Document] = {}
        self._index: Dict[str, Dict[str, float]] = {}  # term -> {doc_id: tf}
        self._idf: Dict[str, float] = {}
        self._doc_norms: Dict[str, float] = {}
        self._tokenizer = Tokenizer()
        self._doc_count = 0

    def add_document(self, doc: Document) -> None:
        if doc.doc_id in self._docs:
            self._remove_document(doc.doc_id)
        self._docs[doc.doc_id] = doc
        tokens = self._tokenizer.tokenize(doc.content)
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

        # 计算TF
        max_freq = max(token_freq.values()) if token_freq else 1
        for token, freq in token_freq.items():
            tf = 0.5 + 0.5 * freq / max_freq
            if token not in self._index:
                self._index[token] = {}
            self._index[token][doc.doc_id] = tf

        self._doc_count += 1
        self._update_idf()
        self._update_norms()

    def _remove_document(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        for term, docs in self._index.items():
            docs.pop(doc_id, None)
        self._docs.pop(doc_id, None)
        self._doc_count -= 1

    def _update_idf(self) -> None:
        for term, docs in self._index.items():
            df = len(docs)
            self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1

    def _update_norms(self) -> None:
        self._doc_norms = {}
        for term, docs in self._index.items():
            idf = self._idf.get(term, 1.0)
            for doc_id, tf in docs.items():
                weight = tf * idf
                self._doc_norms[doc_id] = self._doc_norms.get(doc_id, 0.0) + weight ** 2
        for doc_id in self._doc_norms:
            self._doc_norms[doc_id] = math.sqrt(self._doc_norms[doc_id])

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []

        scores: Dict[str, float] = {}
        query_weights: Dict[str, float] = {}
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        max_freq = max(token_freq.values())

        for token, freq in token_freq.items():
            if token not in self._index:
                continue
            tf = 0.5 + 0.5 * freq / max_freq
            idf = self._idf.get(token, 1.0)
            query_weights[token] = tf * idf
            for doc_id, doc_tf in self._index[token].items():
                doc_idf = self._idf.get(token, 1.0)
                scores[doc_id] = scores.get(doc_id, 0.0) + query_weights[token] * doc_tf * doc_idf

        # 归一化
        query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))
        results = []
        for doc_id, score in scores.items():
            doc_norm = self._doc_norms.get(doc_id, 1.0)
            if query_norm > 0 and doc_norm > 0:
                score = score / (query_norm * doc_norm)
            results.append(SearchResult(doc_id=doc_id, score=score, doc=self._docs.get(doc_id)))

        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "doc_count": len(self._docs),
            "term_count": len(self._index),
        }


class VectorIndex:
    """向量索引（简化版，使用词袋向量）"""

    def __init__(self, dim: int = 100):
        self._dim = dim
        self._docs: Dict[str, Document] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._tokenizer = Tokenizer()

    def _hash_vector(self, text: str) -> List[float]:
        """使用哈希将文本映射到固定维度向量"""
        tokens = self._tokenizer.tokenize(text)
        vec = [0.0] * self._dim
        for token in tokens:
            # 简单哈希
            h = hash(token) % self._dim
            vec[h] += 1.0
        # 归一化
        norm = math.sqrt(sum(v ** 2 for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def add_document(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc
        self._vectors[doc.doc_id] = self._hash_vector(doc.content)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        query_vec = self._hash_vector(query)
        results = []
        for doc_id, doc_vec in self._vectors.items():
            score = sum(a * b for a, b in zip(query_vec, doc_vec))
            results.append(SearchResult(doc_id=doc_id, score=score, doc=self._docs.get(doc_id)))
        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "doc_count": len(self._docs),
            "dim": self._dim,
        }


class SemanticSearchEngine:
    """语义搜索引擎"""

    def __init__(self):
        self._tfidf = TFIDFIndex()
        self._vector = VectorIndex()
        self._docs: Dict[str, Document] = {}

    def index(self, content: str, title: str = "",
              metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            doc_id=doc_id,
            content=content,
            title=title,
            metadata=metadata or {},
        )
        self._docs[doc_id] = doc
        self._tfidf.add_document(doc)
        self._vector.add_document(doc)
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    def search(self, query: str, top_k: int = 10, method: str = "hybrid") -> List[SearchResult]:
        if method == "tfidf":
            return self._tfidf.search(query, top_k)
        elif method == "vector":
            return self._vector.search(query, top_k)
        elif method == "hybrid":
            tfidf_results = self._tfidf.search(query, top_k)
            vector_results = self._vector.search(query, top_k)
            return self._merge_results(tfidf_results, vector_results, top_k)
        return []

    def _merge_results(self, results1: List[SearchResult], results2: List[SearchResult],
                       top_k: int) -> List[SearchResult]:
        merged: Dict[str, float] = {}
        for r in results1:
            merged[r.doc_id] = merged.get(r.doc_id, 0.0) + r.score * 0.6
        for r in results2:
            merged[r.doc_id] = merged.get(r.doc_id, 0.0) + r.score * 0.4

        final = []
        for doc_id, score in sorted(merged.items(), key=lambda x: -x[1]):
            final.append(SearchResult(doc_id=doc_id, score=score, doc=self._docs.get(doc_id)))
        return final[:top_k]

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        self._docs.pop(doc_id)
        # 重新构建索引（简化处理）
        self._tfidf = TFIDFIndex()
        self._vector = VectorIndex()
        for doc in self._docs.values():
            self._tfidf.add_document(doc)
            self._vector.add_document(doc)
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_docs": len(self._docs),
            "tfidf": self._tfidf.get_stats(),
            "vector": self._vector.get_stats(),
        }


def create_semantic_search_engine() -> SemanticSearchEngine:
    return SemanticSearchEngine()
