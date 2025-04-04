import pytest
from bpe import Tokenizer
import os

manual_text =  open('manual.txt', 'r', encoding="utf-8").read()


@pytest.fixture
def sample_data():
    return [
        "这是一个测试句子，检查编码解码一致性。",
        "pku 2025 spring llm",
        "原来你也玩鸣潮🤓",
        "!@#$%^&*()_+-=[]{};':\",./<>?",
        "",
        manual_text
    ]


def test_encode_decode_consistency(sample_data):
    for text in sample_data:
        tokenizer = Tokenizer()
        tokenizer.train(text, min(256 + len(text), 1024), verbose=False)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Decoded text '{decoded}' does not match original '{text}'"


def test_save_load(sample_data):
    for text in sample_data[:-1]:
        tokenizer = Tokenizer()
        tokenizer.train(text, min(256 + len(text), 1024), verbose=False)
        ids1 = tokenizer.encode(text)
        assert tokenizer.decode(ids1) == text
        model_file, vocab_file = tokenizer.save('test_tokenizer_tmp')

        tokenizer = Tokenizer()
        tokenizer.load(model_file)
        ids2 = tokenizer.encode(text)
        assert ids2 == ids1
        assert tokenizer.decode(ids2) == text
        os.remove(model_file)
        os.remove(vocab_file)


    
