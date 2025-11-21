
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Match



_class_registry = {}
def register_class(registeer_as: str):
    def decorator(cls):
        _class_registry[registeer_as] = cls
        return cls
    return decorator


class Tokenizer:
    def tokenize(self, text: str):
        raise NotImplementedError("Tokenizer must implement the tokenize method.")
    
class Filter:
    def apply(self, tokens: List[str]):
        raise NotImplementedError("Filter must implement the apply method.")

class Analyzer:
    def __init__(
        self,
        name: str,
        tokenizer: Tokenizer,
        filters: Optional[List[Filter]] = None,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.filters = filters if filters is not None else []
    
    def __call__(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        for filter in self.filters:
            tokens = filter.apply(tokens)
        return tokens
    

@register_class("JiebaTokenizer")
class JiebaTokenizer(Tokenizer):
    def tokenize(self, text: str):
        import jieba
        return jieba.lcut(text)

@register_class("StopwordFilter")
class StopwordFilter(Filter):
    def __init__(self, language: str, stopwords_list: Optional[List[str]] = None):
        self.language = language
        if stopwords_list is None:
            stopwords_list = []
        self.stopwords = set(self._load_default_stopwords(language) + stopwords_list)

    def _load_default_stopwords(self, language: str) -> List[str]:
        import nltk
        from nltk.corpus import stopwords
        try:
            stopwords.words(language)
        except LookupError:
            nltk.download("stopwords")
        return stopwords.words(language)

    def apply(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords]

@register_class("PunctuationFilter")
class PunctuationFilter(Filter):
    def __init__(self, extras: str = ""):
        self.punctuation = set(string.punctuation + extras)

    def apply(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.punctuation]
    

def build_default_analyzer(language: str) -> Analyzer:
    default_config_path = Path(__file__).parent / "lang.yaml"
    return build_default_analyzer_from_yaml(default_config_path, language)

def build_default_analyzer_from_yaml(yaml_path: Path, language: str) -> Analyzer:
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    lang_config = config.get(language)
    if lang_config is None:
        raise ValueError(f"No configuration found for language: {language}")

    tokenizer = _class_registry[lang_config["tokenizer"]["class"]](**lang_config["tokenizer"].get("params", {}))
    filters = []
    if "filters" in lang_config:
        for filter_config in lang_config["filters"]:
            filter_class = _class_registry[filter_config["class"]]
            filter_params = filter_config.get("params", {})
            filters.append(filter_class(**filter_params))
    analyzer = Analyzer(name=language, tokenizer=tokenizer, filters=filters)
    return analyzer
    





