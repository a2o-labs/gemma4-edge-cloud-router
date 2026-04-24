"""Cloud forwarder — ships a CompactTask to the litellm endpoint.

Wire format: we embed the CompactTask JSON as the user message content and add
a system preamble that tells the cloud model how to interpret the schema. This
keeps us OpenAI-compatible so any litellm-proxied backend works unchanged.
"""

from __future__ import annotations

import httpx

from .config import CloudLLMSettings
from .schema import CompactTask


CLOUD_SYSTEM_PROMPT = (
    "You are receiving a CompactTask v1.0 envelope in the next user message. "
    "Fields: schema_version, task_id, task_type, instruction (may contain "
    "@symbol_N placeholders — treat as opaque identifiers), symbols "
    "(typed references), context_refs (opaque ids you may request via "
    "callback), privacy (tells you whether redaction was applied). Answer the "
    "instruction. Do not attempt to de-anonymise symbols; echo them through "
    "unchanged so the edge can resolve them."
)


class CloudForwarder:
    def __init__(self, settings: CloudLLMSettings) -> None:
        self.settings = settings

    async def forward(self, task: CompactTask) -> str:
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": CLOUD_SYSTEM_PROMPT},
                {"role": "user", "content": task.model_dump_json()},
            ],
            "temperature": 0.3,
            "metadata": {"task_id": task.task_id, "schema_version": task.schema_version},
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_s) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]
