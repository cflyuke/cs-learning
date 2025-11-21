import json
from typing import Optional, List

import aiohttp
import numpy as np

class OpenAIRerank:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        session: aiohttp.ClientSession,
        timeout: float
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.session = session
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def arerank(self, query: str, documents: List[str], **kargs):
        if not documents:
            raise ValueError("Input texts list is empty.")
        payload = {"model": self.model, "query": query, "documents": documents, "top_n": len(documents), **kargs}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with self.session.post(self.base_url, json=payload, timeout=timeout, headers=self.headers) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Score request failed ({response.status}): {body}")
            data = json.loads(body)
        rank = np.zeros(len(documents), dtype=float)
        try:
            for item in data["results"]:
                rank[item["index"]] = item["relevance_score"]
        except Exception as e:
            raise RuntimeError(f"Failed to parse rerank response: {e}")
        return rank