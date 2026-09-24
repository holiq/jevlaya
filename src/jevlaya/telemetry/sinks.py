"""Telemetry sinks for recording and aggregating decision events."""

from __future__ import annotations

import collections
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevlaya.bench.metrics import compute_percentile
from jevlaya.errors import redact_secrets
from jevlaya.telemetry.events import DecisionEvent


class TelemetrySummary(BaseModel):
    """Aggregated statistical summary of telemetry events."""

    model_config = ConfigDict(extra="ignore")

    total_requests: int = 0
    successful_requests: int = 0
    fallback_requests: int = 0
    error_requests: int = 0
    fallback_rate: float = 0.0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    provider_breakdown: dict[str, int] = Field(default_factory=dict)
    status_breakdown: dict[str, int] = Field(default_factory=dict)


@runtime_checkable
class TelemetrySink(Protocol):
    """Protocol for telemetry destinations."""

    def record(self, event: DecisionEvent) -> None:
        """Record a decision telemetry event."""
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...


class InMemoryTelemetrySink:
    """Thread-safe in-memory sink retaining the last N decision events for analysis."""

    def __init__(self, max_events: int = 10000) -> None:
        self.max_events = max_events
        self._events: collections.deque[DecisionEvent] = collections.deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(self, event: DecisionEvent) -> None:
        with self._lock:
            self._events.append(event)

    def flush(self) -> None:
        pass

    def clear(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()

    def get_events(self, limit: int = 100, offset: int = 0) -> list[DecisionEvent]:
        """Retrieve stored events with pagination (most recent first)."""
        with self._lock:
            all_events = list(reversed(self._events))
            return all_events[offset : offset + limit]

    def get_summary(self) -> TelemetrySummary:
        """Calculate real-time aggregated metrics across stored events."""
        with self._lock:
            events = list(self._events)

        if not events:
            return TelemetrySummary()

        total = len(events)
        success = sum(1 for e in events if e.status == "success")
        fallback = sum(1 for e in events if e.status == "fallback")
        errors = sum(1 for e in events if e.status == "error")

        latencies = [e.latency_ms for e in events]
        avg_latency = round(sum(latencies) / total, 2)
        p50 = compute_percentile(latencies, 50.0)
        p95 = compute_percentile(latencies, 95.0)

        total_cost = round(sum(e.cost_usd or 0.0 for e in events), 6)

        provider_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for e in events:
            provider_counts[e.provider] = provider_counts.get(e.provider, 0) + 1
            status_counts[e.status] = status_counts.get(e.status, 0) + 1

        fallback_rate = round(fallback / total, 4) if total else 0.0
        error_rate = round(errors / total, 4) if total else 0.0

        return TelemetrySummary(
            total_requests=total,
            successful_requests=success,
            fallback_requests=fallback,
            error_requests=errors,
            fallback_rate=fallback_rate,
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            total_cost_usd=total_cost,
            provider_breakdown=provider_counts,
            status_breakdown=status_counts,
        )


class JsonLinesTelemetrySink:
    """Sink that appends serialized decision events to a JSON Lines file."""

    def __init__(self, file_path: str | Path) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: DecisionEvent) -> None:
        raw_json = event.model_dump_json()
        sanitized = redact_secrets(raw_json)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(sanitized + "\n")

    def flush(self) -> None:
        pass


class CallbackTelemetrySink:
    """Sink that delegates each event to an arbitrary callback function."""

    def __init__(self, callback: Callable[[DecisionEvent], Any]) -> None:
        self.callback = callback

    def record(self, event: DecisionEvent) -> None:
        self.callback(event)

    def flush(self) -> None:
        pass


class CompositeTelemetrySink:
    """Sink broadcasting events across multiple child sinks."""

    def __init__(self, sinks: list[TelemetrySink]) -> None:
        self.sinks = list(sinks)

    def add_sink(self, sink: TelemetrySink) -> None:
        self.sinks.append(sink)

    def record(self, event: DecisionEvent) -> None:
        for sink in self.sinks:
            sink.record(event)

    def flush(self) -> None:
        for sink in self.sinks:
            sink.flush()
