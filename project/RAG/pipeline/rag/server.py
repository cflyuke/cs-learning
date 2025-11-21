from typing import Any, Dict, Optional, AsyncIterable, List, Union, cast
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
import asyncio
import json

from rag.schema import Document
from rag.vectorstore import Milvus
from rag.llm import BM25Embedding, OpenAIEmbedding, OpenAIRerank, OpenAIChat
from rag.utils import (
    SYSTEMPROMPT,
    get_handler,
    format_user_prompt,
    format_qwen3_embedding_input,
    format_qwen3_reranker_input,
    get_documents,
    extract_references
)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from aiohttp import ClientSession, ClientTimeout
from pydantic import BaseModel

@dataclass
class ServerConfig:
    timeout: float = 30.0
    milvus_uri: str = "./data/index/milvus.db"
    collection_name: str = "rag"

    ## Embedding and Index Configurations (make sure to keep them in sync with preprocess/embedding/vectorstore.py)
    bm25_index_path: str = "./data/index/bm25.pkl"
    bm25_milvus_field: str = "BM25"

    bgem3_embedding_model: str = "models/BAAI/bge-m3"
    bgem3_embedding_url = "http://localhost:8080/v1/embeddings"
    bgem3_embedding_api_key: Optional[str] = None
    bgem3_milvus_field : str = "BgeM3"

    qwen3_embedding_model: str = "models/Qwen/Qwen3-Embedding-0.6B"
    qwen3_embedding_url = "http://localhost:8081/v1/embeddings"
    qwen3_embedding_api_key: Optional[str] = None
    qwen3_milvus_field: str = "Qwen3"

    rerank_model: str = "models/Qwen/Qwen3-Reranker-4B"
    rerank_model_url = "http://localhost:8082/v1/rerank"
    rerank_api_key: Optional[str] = None
    
    chat_model: str = "models/Qwen/Qwen3-4B"
    chat_model_url = "http://localhost:8083/v1/chat/completions"
    chat_api_key: Optional[str] = None
    

    ## Other server configurations
    rrf_top_n: int = 8
    rerank_top_n: int = 4


class RagRequest(BaseModel):
    query: str
    streaming: bool = True


class RAGService:
    def __init__(self, cfg: ServerConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def start_up(self):
        self.logger.info("RAG Service is starting up...")

        self.milvus = Milvus(uri=self.cfg.milvus_uri, collection_name=self.cfg.collection_name)
        self.bm25 = BM25Embedding()
        self.bm25.load(self.cfg.bm25_index_path)

        self.bgem3 = OpenAIEmbedding(
            model=self.cfg.bgem3_embedding_model,
            base_url=self.cfg.bgem3_embedding_url,
            api_key=self.cfg.bgem3_embedding_api_key,
            session=ClientSession(timeout=ClientTimeout(total=self.cfg.timeout)),
            timeout=self.cfg.timeout,
        )

        self.qwen3_embedding = OpenAIEmbedding(
            model=self.cfg.qwen3_embedding_model,
            base_url=self.cfg.qwen3_embedding_url,
            api_key=self.cfg.qwen3_embedding_api_key,
            session=ClientSession(timeout=ClientTimeout(total=self.cfg.timeout)),
            timeout=self.cfg.timeout,
        )

        self.reranker = OpenAIRerank(
            model=self.cfg.rerank_model,
            base_url=self.cfg.rerank_model_url,
            api_key=self.cfg.rerank_api_key,
            session=ClientSession(timeout=ClientTimeout(total=self.cfg.timeout)),
            timeout=self.cfg.timeout,
        )

        self.chat = OpenAIChat(
            model=self.cfg.chat_model,
            base_url=self.cfg.chat_model_url,
            api_key=self.cfg.chat_api_key,
            session=ClientSession(timeout=ClientTimeout(total=self.cfg.timeout)),
            timeout=self.cfg.timeout,
        )

        self.logger.info("RAG Service started.")

    async def shut_down(self):
        self.logger.info("RAG Service is shutting down...")
        await self.bgem3.session.close()
        await self.qwen3_embedding.session.close()
        await self.reranker.session.close()
        await self.chat.session.close()
        self.logger.info("RAG Service shut down.")

    async def retrieve(self, query: str) -> List[Document]:
        # concurrent encode with three encoders
        task1 = asyncio.create_task(self.bm25.aencode_queries([query]))
        task2 = asyncio.create_task(self.bgem3.aencode([query]))
        task3 = asyncio.create_task(self.qwen3_embedding.aencode([format_qwen3_embedding_input(query)]))

        results = await asyncio.gather(task1, task2, task3)

        vectors: Dict[str, Any] = {}
        vectors[self.cfg.bm25_milvus_field] = results[0]
        vectors[self.cfg.bgem3_milvus_field] = results[1]
        vectors[self.cfg.qwen3_milvus_field] = results[2]

        search_results: List[Dict[str, Any]] = await self.milvus.asearch(vector_dict=vectors, limit=self.cfg.rrf_top_n)
        unique_ids = [res["unique_id"] for res in search_results]
        documents = get_documents(unique_ids)
        return documents
    
    async def rerank(self, documents: List[Document], query: str) -> List[Document]:
        docs = [doc.page_content for doc in documents]
        rerank_inputs = format_qwen3_reranker_input(query, docs)
        scores = await self.reranker.arerank(rerank_inputs[0], rerank_inputs[1])
        top_indices = scores.argsort()[-self.cfg.rerank_top_n:][::-1]
        rerank_documents = [documents[idx] for idx in top_indices]
        return rerank_documents

    async def chat_with_rag(self, documents: List[Document], request: RagRequest) -> Union[AsyncIterable[bytes], Dict[str, Any]]:
        query = request.query
        streaming = request.streaming
        docs = [doc.page_content for doc in documents]
        user_prompt = format_user_prompt(docs, query)
        messages = [
            {"role": "system", "content": SYSTEMPROMPT},
            {"role": "user", "content": user_prompt},
        ]
        answer = await self.chat.achat(messages, stream=streaming)

        # streaming path
        if streaming:
            reasoning_content = ""
            content = ""
            async def gen() -> AsyncIterable[bytes]:
                nonlocal reasoning_content, content
                async for chunk in answer: # bytes from upstream SSE
                    try:
                        chunk_str = chunk.decode("utf-8")
                    except Exception:
                        continue
                    if not chunk_str.startswith("data: "):
                        continue
                    data_str = chunk_str[len("data: "):].strip()
                    if data_str == "[DONE]":
                        final_payload = {
                            "message": {"content": content, "reasoning_content": reasoning_content},
                            "refs": [documents[idx - 1].__dict__ for idx in extract_references(content)]
                        }
                        yield ("data: " + json.dumps(final_payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    delta_reasoning = delta.get("reasoning_content", "")
                    delta_text = delta.get("content", "")
                    if delta_reasoning:
                        reasoning_content += delta_reasoning
                    if delta_text:
                        content += delta_text
                    delta_payload = {"delta": {"reasoning_content": delta_reasoning, "content": delta_text}}
                    yield ("data: " + json.dumps(delta_payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                yield b"data: [DONE]\n\n"
            return gen()

        # non-streaming path
        if isinstance(answer, dict):
            message = answer.get("choices", [{}])[0].get("message", {})
            reasoning_content = message.get("reasoning_content", "")
            content = message.get("content", "")
            return {
                "message": {"content": content, "reasoning_content": reasoning_content},
                "refs": [documents[idx - 1].__dict__ for idx in extract_references(content)]
            }
        self.logger.error("Unexpected answer format received from chat model.")
        return {"error": "Unexpected answer format received from chat model."}
    
    async def pipeline(self, request: RagRequest) -> Union[AsyncIterable[bytes], Dict[str, Any]]:
        query = request.query
        
        # retrieve documents
        documents = await self.retrieve(query)

        # rerank documents
        rerank_documents = await self.rerank(documents, query)

        # generate answer
        return await self.chat_with_rag(rerank_documents, request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("RAGServer")
    logger.addHandler(get_handler())
    logger.setLevel(logging.DEBUG)
    cfg = ServerConfig()
    service = RAGService(cfg, logger)
    service.start_up()
    app.state.service = service
    yield
    await service.shut_down()

app = FastAPI(lifespan=lifespan)


@app.get("/v1/config")
async def get_config() -> Dict[str, Any]:
    return app.state.service.cfg.__dict__

@app.post("/v1/rag")
async def rag_endpoint(request: RagRequest):
    service: RAGService = app.state.service
    response = await service.pipeline(request)
    if request.streaming:
        stream = cast(AsyncIterable[bytes], response)
        return StreamingResponse(stream, media_type="text/event-stream")
    else:
        payload = cast(Dict[str, Any], response)
        return JSONResponse(content=payload, media_type="application/json")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)