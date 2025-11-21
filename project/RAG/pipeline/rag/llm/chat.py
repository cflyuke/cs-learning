import json
import aiohttp
from typing import Optional, List, Dict, Any


class OpenAIChat:
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
        self.headers =  {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def achat(self, messages: List[Dict[str, str]], stream: bool = True, **kargs):
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        payload.update(kargs)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        if stream:
            async def _stream():
                async with self.session.post(self.base_url, json=payload, headers=self.headers, timeout=timeout) as response:
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(f"LLM request failed ({response.status}): {body}")
                    async for chunk in response.content:
                        yield chunk
            return _stream()

        async with self.session.post(self.base_url, json=payload, headers=self.headers, timeout=timeout) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"LLM request failed ({response.status}): {body}")
            return json.loads(body)