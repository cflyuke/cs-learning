
import math
from multiprocessing import Pool
from typing import List, Optional, Dict
from collections import defaultdict, Counter

from rag.utils import run_in_executor
from rag.llm.bm25.analyzer import Analyzer, build_default_analyzer

from scipy.sparse import csr_array, vstack
import numpy as np

class BM25Embedding:
    def __init__(
        self,
        analyzer: Optional[Analyzer] = None,
        corpus: Optional[List] = None,
        k1: float = 1.5,
        b : float = 0.75,
        epsilon : float = 0.25,
        num_workers : int = 1,
    ):
        if analyzer is None:
            analyzer = build_default_analyzer("zh")
        self.analyzer = analyzer

        self.corpus_size = 0
        self.avgdl = 0
        self.idf = defaultdict(list)

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.num_workers = num_workers
        
        if analyzer and corpus is not None:
            self.fit(corpus)
    
    def _tokenize_corpus(self, corpus: List[str]):
        if self.num_workers == 1:
            return [self.analyzer(text) for text in corpus]
        pool = Pool(self.num_workers)
        return pool.map(self.analyzer, corpus)

    def _compute_statistics(self, tokenized_corpus: List[List[str]]):
        term_document_frequencies = defaultdict(int)
        for tokens in tokenized_corpus:
            unique_terms = set(tokens)
            for term in unique_terms:
                term_document_frequencies[term] += 1
        self.avgdl = sum(len(tokens) for tokens in tokenized_corpus) / len(tokenized_corpus) if tokenized_corpus else 0
        self.corpus_size = len(tokenized_corpus)
        return term_document_frequencies

    def _calc_idf(self, term_document_frequencies: Dict[str, int]):
        idf_sum = 0
        negative_idfs = []
        for term, df in term_document_frequencies.items():
            idf_score = math.log(self.corpus_size - df + 0.5) -  math.log(df + 0.5)
            self.idf[term] = [idf_score, 0]
            idf_sum += idf_score
            if idf_score < 0:
                negative_idfs.append(term)
        average_idf = idf_sum / len(self.idf) if self.idf else 0
        eps = self.epsilon * average_idf
        for term in negative_idfs:
            self.idf[term][0] = eps
        for index, term in enumerate(self.idf):
            self.idf[term][1] = index

    def _rebuild(self, corpus: List[str]):
        self._clear()
        tokenized_corpus = self._tokenize_corpus(corpus)
        term_document_frequencies = self._compute_statistics(tokenized_corpus)
        self._calc_idf(term_document_frequencies)
    
    def _clear(self):
        self.corpus_size = 0
        self.avgdl = 0
        self.idf = defaultdict(list)

    def fit(self, corpus: List[str]):
        self._rebuild(corpus)

    def _encode_query(self, query: str) -> csr_array:
        terms = self.analyzer(query)
        indices, values = [], []
        for term in terms:
            if term in self.idf:
                indices.append(self.idf[term][1])
                values.append(self.idf[term][0])
        return csr_array((values, ([0] * len(indices), indices)), shape=(1, len(self.idf))).astype(np.float32)

    def _encode_document(self, doc: str) -> csr_array:
        terms = self.analyzer(doc)
        term_frequencies = Counter(terms)
        doc_len = len(terms)
        indices, values = [], []
        for term, freq in term_frequencies.items():
            if term in self.idf:
                value = (
                    freq
                    * (self.k1 + 1)
                    / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                )
                values.append(value)
                indices.append(self.idf[term][1])
        return csr_array((values, ([0] * len(indices), indices)), shape=(1, len(self.idf))).astype(np.float32)
    
    def encode_queries(self, queries: List[str]):
        sparse_embeddings = [self._encode_query(query) for query in queries]
        return vstack(sparse_embeddings).tocsr()
    
    def encode_documents(self, documents: List[str]):
        sparse_embeddings = [self._encode_document(doc) for doc in documents]
        return vstack(sparse_embeddings).tocsr()
    
    async def aencode_queries(self, queries: List[str]):
        return await run_in_executor(None, self.encode_queries, queries)
    
    @property
    def dim(self):
        return len(self.idf)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.analyzer = obj.analyzer
        self.corpus_size = obj.corpus_size
        self.avgdl = obj.avgdl
        self.idf = obj.idf
        self.k1 = obj.k1
        self.b = obj.b
        self.epsilon = obj.epsilon
        self.num_workers = obj.num_workers