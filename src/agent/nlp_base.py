"""
阶段76: Agent自然语言处理基础系统

遵循"图难于其易，为大于其细"的原则，实现分词、词性标注、命名实体识别、文本分类。
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


@dataclass
class Token:
    text: str
    start: int
    end: int
    pos: str = ""  # 词性
    lemma: str = ""  # 词干

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "pos": self.pos,
        }


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    tokens: List[Token] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "tokens": [t.to_dict() for t in self.tokens],
        }


class Tokenizer:
    """分词器"""

    # 简单标点符号
    PUNCTUATION = set(".,!?;:\"'()[]{}【】")

    def __init__(self):
        # 英文单词、数字、中文字符
        self._pattern = re.compile(r"[a-zA-Z]+|\d+|[^\s\w]|[\u4e00-\u9fff]")

    def tokenize(self, text: str) -> List[Token]:
        tokens = []
        for match in self._pattern.finditer(text):
            token = Token(
                text=match.group(),
                start=match.start(),
                end=match.end(),
            )
            tokens.append(token)
        return tokens

    def tokenize_words(self, text: str) -> List[str]:
        return [t.text for t in self.tokenize(text)]


class SentenceSplitter:
    """句子分割器"""

    def __init__(self):
        # 中英文句子结束符
        self._pattern = re.compile(r'[。！？.!?]+\s*')

    def split(self, text: str) -> List[Sentence]:
        sentences = []
        start = 0
        for match in self._pattern.finditer(text):
            end = match.end()
            sent_text = text[start:end].strip()
            if sent_text:
                sentences.append(Sentence(
                    text=sent_text,
                    start=start,
                    end=end,
                ))
            start = end
        # 最后一段
        if start < len(text):
            sent_text = text[start:].strip()
            if sent_text:
                sentences.append(Sentence(
                    text=sent_text,
                    start=start,
                    end=len(text),
                ))
        return sentences


class POSTagger:
    """词性标注器（简化版）"""

    # 常见词性规则
    POS_RULES = [
        (r'^\d+$', 'NUM'),  # 数字
        (r'^[A-Z][a-z]+$', 'NNP'),  # 专有名词
        (r'^[a-z]+ing$', 'VBG'),  # 动名词
        (r'^[a-z]+ed$', 'VBD'),  # 过去式
        (r'^[a-z]+ly$', 'RB'),  # 副词
        (r'^[a-z]+tion$', 'NN'),  # 名词
        (r'^[a-z]+ment$', 'NN'),  # 名词
        (r'^[a-z]+ness$', 'NN'),  # 名词
        (r'^[a-z]+ful$', 'JJ'),  # 形容词
        (r'^[a-z]+less$', 'JJ'),  # 形容词
    ]

    def __init__(self):
        self._patterns = [(re.compile(p), pos) for p, pos in self.POS_RULES]

    def tag(self, token: Token) -> str:
        text = token.text.lower()
        for pattern, pos in self._patterns:
            if pattern.match(text):
                return pos
        # 默认规则
        if token.text in Tokenizer.PUNCTUATION:
            return 'PUNCT'
        return 'UNK'

    def tag_tokens(self, tokens: List[Token]) -> List[Token]:
        for token in tokens:
            token.pos = self.tag(token)
        return tokens


class NERRecognizer:
    """命名实体识别器（基于规则简化版）"""

    # 常见实体模式
    ENTITY_PATTERNS = [
        (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'PERSON'),  # 人名
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', 'DATE'),
        (r'\b\d{4}-\d{2}-\d{2}\b', 'DATE'),
        (r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*,?\s+(?:Inc|Corp|Ltd|LLC|Company)\b', 'ORG'),
        (r'\b(?:北京|上海|广州|深圳|纽约|伦敦|东京|巴黎)\b', 'GPE'),
    ]

    def __init__(self):
        self._patterns = [(re.compile(p, re.IGNORECASE), label) for p, label in self.ENTITY_PATTERNS]

    def recognize(self, text: str) -> List[Entity]:
        entities = []
        for pattern, label in self._patterns:
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                )
                entities.append(entity)
        return entities


class TextClassifier:
    """文本分类器（基于关键词简化版）"""

    # 简单关键词分类规则
    CATEGORY_KEYWORDS = {
        'sports': ['sport', 'game', 'team', 'player', 'match', '足球', '篮球', '比赛'],
        'technology': ['tech', 'software', 'computer', 'ai', 'data', '软件', '技术', '人工智能'],
        'business': ['business', 'company', 'market', 'stock', '公司', '市场', '股票'],
        'politics': ['politic', 'government', 'election', '政治', '政府', '选举'],
        'entertainment': ['movie', 'music', 'celebrity', '电影', '音乐', '明星'],
    }

    def __init__(self):
        self._keywords = self.CATEGORY_KEYWORDS

    def classify(self, text: str) -> List[Tuple[str, float]]:
        text_lower = text.lower()
        scores = {}
        for category, keywords in self._keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[category] = score
        # 归一化
        total = sum(scores.values())
        if total == 0:
            return [('unknown', 1.0)]
        return sorted([(c, s/total) for c, s in scores.items()], key=lambda x: -x[1])

    def predict(self, text: str) -> str:
        results = self.classify(text)
        return results[0][0] if results else 'unknown'


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        self._tokenizer = Tokenizer()

    def normalize(self, text: str) -> str:
        # 统一空白字符
        text = re.sub(r'\s+', ' ', text)
        # 统一引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text.strip()

    def remove_punctuation(self, text: str) -> str:
        return ''.join(c for c in text if c not in Tokenizer.PUNCTUATION)

    def to_lower(self, text: str) -> str:
        return text.lower()

    def preprocess(self, text: str, lower: bool = True, remove_punct: bool = False) -> str:
        text = self.normalize(text)
        if lower:
            text = self.to_lower(text)
        if remove_punct:
            text = self.remove_punctuation(text)
        return text


class NLPipeline:
    """NLP处理流水线"""

    def __init__(self):
        self._tokenizer = Tokenizer()
        self._sent_splitter = SentenceSplitter()
        self._pos_tagger = POSTagger()
        self._ner = NERRecognizer()
        self._classifier = TextClassifier()
        self._preprocessor = TextPreprocessor()

    def process(self, text: str) -> Dict[str, Any]:
        # 预处理
        clean_text = self._preprocessor.preprocess(text, lower=False)

        # 句子分割
        sentences = self._sent_splitter.split(clean_text)

        # 分词和词性标注
        all_tokens = []
        for sent in sentences:
            tokens = self._tokenizer.tokenize(sent.text)
            tokens = self._pos_tagger.tag_tokens(tokens)
            sent.tokens = tokens
            all_tokens.extend(tokens)

        # 命名实体识别
        entities = self._ner.recognize(clean_text)

        # 文本分类
        categories = self._classifier.classify(clean_text)

        return {
            "text": clean_text,
            "sentences": [s.to_dict() for s in sentences],
            "tokens": [t.to_dict() for t in all_tokens],
            "entities": [e.to_dict() for e in entities],
            "categories": categories,
            "token_count": len(all_tokens),
            "sentence_count": len(sentences),
        }

    def extract_keywords(self, text: str, top_k: int = 5) -> List[Tuple[str, int]]:
        tokens = self._tokenizer.tokenize_words(text)
        # 简单的词频统计
        freq = {}
        for token in tokens:
            token = token.lower()
            if len(token) > 2 and token not in Tokenizer.PUNCTUATION:
                freq[token] = freq.get(token, 0) + 1
        return sorted(freq.items(), key=lambda x: -x[1])[:top_k]


def create_nlp_pipeline() -> NLPipeline:
    return NLPipeline()
