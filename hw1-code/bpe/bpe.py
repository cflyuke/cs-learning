class BPETokenizer:
    def __init__(self):
        pass
    def merge(self, spans: list[list[int]]) -> list[list[int]]:
        pass
    def encode(self, text: str) -> list[int]:
        pass
    def decode(self, token_ids: list[int]) -> str:
        pass
    def from_config(cls, config_file: str):
        pass
    def from_data(cls, train_data: str, n_merges: int):
        pass
    def save(self, path: str):
        pass