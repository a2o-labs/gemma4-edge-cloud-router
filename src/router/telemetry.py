"""Telemetry stub — tracks latency, split ratio, cost.

OTel wiring is deferred to V1.1; for now we expose an in-process counter that
/metrics can render as plain text. Swap `record()` for a real OTel meter when
the collector lands.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Counters:
    total: int = 0
    light: int = 0
    heavy: int = 0
    v15: int = 0
    classifier_fallback: int = 0
    latency_samples_ms: list[float] = field(default_factory=list)


class Telemetry:
    def __init__(self) -> None:
        self._counters = Counters()
        self._lock = threading.Lock()

    def record(self, *, path: str, latency_ms: float, classifier_fell_back: bool) -> None:
        with self._lock:
            self._counters.total += 1
            if path == "light":
                self._counters.light += 1
            elif path == "v1.5":
                self._counters.v15 += 1
            else:
                self._counters.heavy += 1
            if classifier_fell_back:
                self._counters.classifier_fallback += 1
            # keep last 1024 samples
            self._counters.latency_samples_ms.append(latency_ms)
            if len(self._counters.latency_samples_ms) > 1024:
                self._counters.latency_samples_ms = self._counters.latency_samples_ms[-1024:]

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            c = self._counters
            samples = sorted(c.latency_samples_ms)
            p50 = samples[len(samples) // 2] if samples else 0.0
            v1_total = c.light + c.heavy
            split_ratio = (c.heavy / v1_total) if v1_total else 0.0
            return {
                "total": c.total,
                "light": c.light,
                "heavy": c.heavy,
                "v1.5": c.v15,
                "heavy_ratio": round(split_ratio, 4),
                "classifier_fallback": c.classifier_fallback,
                "p50_latency_ms": round(p50, 2),
                "ts": time.time(),
            }
