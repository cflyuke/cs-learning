from typing import List
from functools import partial
import asyncio
import contextvars
import logging

from rag.schema import Document


def get_handler():
    import colorlog
    handler = colorlog.StreamHandler()
    format = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(levelname)s%(reset)s %(message)s %(asctime)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white"
        }
    )
    handler.setFormatter(format)
    handler.setLevel(logging.DEBUG)
    return handler

def get_documents(unique_ids: List[str]) -> List[Document]:
    from rag.database import MongoDB
    collection = MongoDB.get_collection("docs", "split_docs")
    documents = []
    for unique_id in unique_ids:
        doc = collection.find_one({"unique_id": unique_id})
        if doc is None:
            continue
        documents.append(Document(page_content=doc["page_content"], metadata=doc["metadata"]))
    return documents


async def run_in_executor(executor, func, *args, **kwargs):
    ctx = contextvars.copy_context()
    loop = asyncio.get_event_loop()
    func_call = partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(executor, func_call)


def format_qwen3_embedding_input(query: str):
    return f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}'

def format_qwen3_reranker_input(query, docs: List[str]):

    def _format_query(query: str) -> str:
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return prefix + "<Instruct>: {instruction}\n<Query>: {query}\n".format(instruction=instruction, query=query)

    def _format_doc(doc: str) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return "<Document>: " + doc + suffix
    
    return _format_query(query), [_format_doc(doc) for doc in docs]
    

SYSTEMPROMPT = "你是特斯拉电动汽车Model 3车型的用户手册问答系统。"
LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
请根据信息中的内容回答问题：
{query}

### 要求
1. 答案需要精准，语句通顺，并严格按照以下格式输出。
答案内容
【引用编号1, 引用编号2, ...】
2. 如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""
def format_user_prompt(docs: List[str], query: str) -> str:
    context = "\n".join([f"【{idx + 1}】" + doc for idx, doc in enumerate(docs)])
    return LLM_CHAT_PROMPT.format(context=context, query=query)

def extract_references(answer: str) -> List[int]:
    import re
    pattern = r"【([\d, ]+)】"
    match = re.search(pattern, answer)
    if not match:
        return []
    ref_str = match.group(1)
    ref_indices = [int(idx.strip()) for idx in ref_str.split(",") if idx.strip().isdigit()]
    return ref_indices