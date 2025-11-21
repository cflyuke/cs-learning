import json
from typing import List, Optional

import aiohttp
import numpy as np


class OpenAIEmbedding:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        session: aiohttp.ClientSession,
        timeout: float,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.session = session
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def aencode(self, texts: List[str], normalize: bool = True, **kargs):
        if not texts:
            raise ValueError("Input texts list is empty.")
        payload = {"model": self.model, "input": texts, "normalize": normalize, **kargs}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with self.session.post(self.base_url, json=payload, timeout=timeout, headers=self.headers) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Embedding request failed ({response.status}): {body}")
            data = json.loads(body)
        embeddings = []
        for item in data.get("data", []):
            vector = item.get("embedding")
            if vector is None:
                continue
            embeddings.append(vector)
        return np.array(embeddings).astype(np.float32)