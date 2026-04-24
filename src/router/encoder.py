"""CompactTask encoder + session-scoped mask_map.

Symbols replace PII-like spans in the user prompt before the task is forwarded
to the cloud. The mapping lives per session_id with a TTL so memory stays
bounded even without an explicit cleanup call.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from .schema import CompactTask, Privacy, SymbolPacket

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\- ]{7,}\d")


@dataclass
class _MaskEntry:
    symbols: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class MaskMap:
    """Session-scoped mask store. TTL enforced lazily on read/write."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, _MaskEntry] = {}
        self._lock = Lock()

    def _gc(self, now: float) -> None:
        expired = [
            sid for sid, e in self._store.items() if now - e.created_at > self.ttl_seconds
        ]
        for sid in expired:
            self._store.pop(sid, None)

    def get_or_create(self, session_id: str) -> _MaskEntry:
        with self._lock:
            self._gc(time.time())
            entry = self._store.get(session_id)
            if entry is None:
                entry = _MaskEntry()
                self._store[session_id] = entry
            return entry

    def symbols_for(self, session_id: str) -> dict[str, str]:
        return dict(self.get_or_create(session_id).symbols)

    def unmask(self, session_id: str, text: str) -> str:
        mapping = self.symbols_for(session_id)
        for sym, original in mapping.items():
            text = text.replace(sym, original)
        return text


class Encoder:
    def __init__(self, mask_map: MaskMap) -> None:
        self.mask_map = mask_map

    def encode(
        self,
        prompt: str,
        *,
        session_id: str,
        task_type: str = "other",
        context_refs: list[str] | None = None,
    ) -> CompactTask:
        entry = self.mask_map.get_or_create(session_id)
        masked, packets = self._mask(prompt, entry)
        applied = bool(packets)
        return CompactTask(
            task_id=uuid.uuid4().hex,
            task_type=task_type,  # type: ignore[arg-type]
            instruction=masked,
            symbols=packets,
            context_refs=list(context_refs or []),
            privacy=Privacy(
                mask_applied=applied,
                redaction_level="medium" if applied else "none",
            ),
        )

    @staticmethod
    def _next_symbol(entry: _MaskEntry, prefix: str) -> str:
        idx = sum(1 for s in entry.symbols if s.startswith(f"@{prefix}_"))
        return f"@{prefix}_{idx}"

    def _mask(
        self, prompt: str, entry: _MaskEntry
    ) -> tuple[str, list[SymbolPacket]]:
        packets: list[SymbolPacket] = []
        masked = prompt

        for match in _EMAIL_RE.findall(prompt):
            sym = self._next_symbol(entry, "email")
            entry.symbols[sym] = match
            masked = masked.replace(match, sym)
            packets.append(SymbolPacket(symbol=sym, kind="pii", hint="email-like"))

        for match in _PHONE_RE.findall(prompt):
            sym = self._next_symbol(entry, "phone")
            entry.symbols[sym] = match
            masked = masked.replace(match, sym)
            packets.append(SymbolPacket(symbol=sym, kind="pii", hint="phone-like"))

        return masked, packets
