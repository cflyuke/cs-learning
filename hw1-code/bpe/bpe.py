from collections import Counter
import regex as re
import unicodedata


def compute_bigram_statistics(
        token_ids: list[int], counter: Counter | None = None
) -> Counter[tuple[int, int], int]:
    """
    Compute adjacent token (Tuple[int, int]) statistics

    Args:
        token_ids : sequence of token ids

    returns:
        A counter with frequencies of bigrams
    """
    bigram_counter = Counter() if counter is None else counter
    for left, right in zip(token_ids, token_ids[1:]):
        bigram_counter[(left, right)] += 1
    return bigram_counter


def replace_bigram(
        token_ids: list[int], bigram: tuple[int, int], bigram_id: int
) -> list[int]:
    """
    Replace all copies of a bigram with "bigram_id"

    Args:
        token_ids: List of token ids
        bigram: The bigram to replace
        bigram_id: New id for the bigram

    returns:
        list[int]: New token_ids
    """
    idx = 0
    new_token_ids = []
    while idx < len(token_ids):
        if token_ids[idx:idx + 2] == list(bigram):
            new_token_ids.append(bigram_id)
            idx += 2
        else:
            new_token_ids.append(token_ids[idx])
            idx += 1
    return new_token_ids


def replace_control_character(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    s = replace_control_character(s)
    return s


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            if len(ids) < 2:
                break
            stats = compute_bigram_statistics(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = replace_bigram(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            states = compute_bigram_statistics(ids)
            pair = min(states, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = replace_bigram(ids, pair, idx)
        return ids


    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return text

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (left, right), idx in self.merges.items():
            vocab[idx] = vocab[left] + vocab[right]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        # write the model: to be used in load()
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("bpe-tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for (idx1, idx2), idx in self.merges.items():
                f.write(f"{idx1} {idx2} {idx}\n")

        # write the vocab: for human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = render_token(self.vocab[idx0])
                        s1 = render_token(self.vocab[idx1])
                        f.write(f"[{s0}] [{s1}] -> [{s}] {idx}\n")
                    else:
                        f.write(f"[{s}] {idx}\n")
        return model_file, vocab_file

    def load(self, model_file: str):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "bpe-tokenizer v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2, idx = map(int, line.split())
                merges[(idx1, idx2)] = idx
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
