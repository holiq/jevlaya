"""Base protocol definition for Jevlaya decision providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jevlaya.protocol.models import DecisionRequest, DecisionResponse


@runtime_checkable
class DecisionProvider(Protocol):
    """Protocol defining the interface for all decision providers."""

    name: str

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Process a decision request and return a canonical normalized response."""
        ...
