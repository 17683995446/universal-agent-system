"""阶段76: Agent自然语言处理基础系统 - 单元测试"""
import pytest
from src.agent.nlp_base import (
    Token, Entity, Sentence,
    Tokenizer, SentenceSplitter, POSTagger, NERRecognizer,
    TextClassifier, TextPreprocessor, NLPipeline, create_nlp_pipeline,
)


class TestTokenizer:
    def test_tokenize_english(self):
        t = Tokenizer()
        tokens = t.tokenize("Hello world!")
        assert len(tokens) == 3
        assert tokens[0].text == "Hello"
        assert tokens[1].text == "world"
        assert tokens[2].text == "!"

    def test_tokenize_chinese(self):
        t = Tokenizer()
        tokens = t.tokenize("你好世界")
        assert len(tokens) == 4
        assert tokens[0].text == "你"

    def test_tokenize_numbers(self):
        t = Tokenizer()
        tokens = t.tokenize("The price is $100.")
        assert any(t.text == "100" for t in tokens)

    def test_tokenize_words(self):
        t = Tokenizer()
        words = t.tokenize_words("Hello world")
        assert words == ["Hello", "world"]


class TestSentenceSplitter:
    def test_split_english(self):
        s = SentenceSplitter()
        sentences = s.split("Hello world. How are you? I am fine.")
        assert len(sentences) == 3
        assert "Hello world." in sentences[0].text

    def test_split_chinese(self):
        s = SentenceSplitter()
        sentences = s.split("你好。你好吗？我很好。")
        assert len(sentences) == 3

    def test_split_single(self):
        s = SentenceSplitter()
        sentences = s.split("Just one sentence")
        assert len(sentences) == 1


class TestPOSTagger:
    def test_tag_number(self):
        tagger = POSTagger()
        token = Token(text="123", start=0, end=3)
        assert tagger.tag(token) == "NUM"

    def test_tag_punctuation(self):
        tagger = POSTagger()
        token = Token(text=".", start=0, end=1)
        assert tagger.tag(token) == "PUNCT"

    def test_tag_verb(self):
        tagger = POSTagger()
        token = Token(text="running", start=0, end=7)
        assert tagger.tag(token) == "VBG"

    def test_tag_unknown(self):
        tagger = POSTagger()
        token = Token(text="xyz", start=0, end=3)
        assert tagger.tag(token) == "UNK"

    def test_tag_tokens(self):
        tagger = POSTagger()
        tokens = [Token(text="123", start=0, end=3), Token(text=".", start=3, end=4)]
        tagged = tagger.tag_tokens(tokens)
        assert tagged[0].pos == "NUM"
        assert tagged[1].pos == "PUNCT"


class TestNERRecognizer:
    def test_recognize_person(self):
        ner = NERRecognizer()
        entities = ner.recognize("John Smith is here.")
        assert any(e.label == "PERSON" for e in entities)

    def test_recognize_date(self):
        ner = NERRecognizer()
        entities = ner.recognize("The date is 2024-01-15.")
        assert any(e.label == "DATE" for e in entities)

    def test_recognize_gpe(self):
        ner = NERRecognizer()
        entities = ner.recognize("I live in 北京.")
        assert any(e.label == "GPE" and e.text == "北京" for e in entities)


class TestTextClassifier:
    def test_classify_sports(self):
        c = TextClassifier()
        result = c.predict("The team won the match.")
        assert result == "sports"

    def test_classify_technology(self):
        c = TextClassifier()
        result = c.predict("AI and software technology.")
        assert result == "technology"

    def test_classify_unknown(self):
        c = TextClassifier()
        result = c.predict("xyz abc def")
        assert result == "unknown"

    def test_classify_scores(self):
        c = TextClassifier()
        scores = c.classify("The company stock market business.")
        assert scores[0][0] == "business"
        assert scores[0][1] > 0


class TestTextPreprocessor:
    def test_normalize(self):
        p = TextPreprocessor()
        result = p.normalize("  Hello   world  ")
        assert result == "Hello world"

    def test_remove_punctuation(self):
        p = TextPreprocessor()
        result = p.remove_punctuation("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_preprocess(self):
        p = TextPreprocessor()
        result = p.preprocess("  Hello, WORLD!  ", lower=True, remove_punct=True)
        assert result == "hello world"


class TestNLPipeline:
    def test_process(self):
        pipe = NLPipeline()
        result = pipe.process("Hello world. How are you?")
        assert "tokens" in result
        assert "sentences" in result
        assert "entities" in result
        assert "categories" in result
        assert result["sentence_count"] == 2
        assert result["token_count"] > 0

    def test_extract_keywords(self):
        pipe = NLPipeline()
        keywords = pipe.extract_keywords("hello hello world test test test", top_k=2)
        assert len(keywords) == 2
        assert keywords[0][0] == "test"
        assert keywords[0][1] == 3

    def test_create_nlp_pipeline(self):
        pipe = create_nlp_pipeline()
        assert isinstance(pipe, NLPipeline)
