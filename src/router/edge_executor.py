"""Edge executor — runs light tasks locally via the edge LLM."""

from __future__ import annotations

import httpx

from .config import EdgeLLMSettings


EDGE_SYSTEM_PROMPT = (
    "You are a helpful local assistant. Answer concisely. If the request is "
    "beyond your capability, reply with a brief acknowledgement so the router "
    "can escalate it on the next turn."
)


class EdgeExecutor:
    def __init__(self, settings: EdgeLLMSettings) -> None:
        self.settings = settings

    async def answer(self, prompt: str) -> str:
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": EDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_s) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]
