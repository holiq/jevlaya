"""Jev provider adapter for hosted System 1 inference via OpenRouter Decisions API."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, ConfigDict

from jevlaya._version import __version__
from jevlaya.errors import (
    InvalidRequest,
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
    redact_secrets,
)
from jevlaya.protocol.models import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    UsageInfo,
)


class JevCapabilities(BaseModel):
    """Declared capabilities and constraints for the hosted TypeSafe Jev model."""

    model_config = ConfigDict(frozen=True)

    supports_choice: bool = True
    supports_score: bool = True
    supports_noul: bool = True
    hosted_api: bool = True
    context_window: int = 32000
    max_recommended_options: int = 255


class JevAdapter:
    """Adapter integrating TypeSafe Jev via the OpenRouter Decisions API."""

    name: str = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "typesafe/jev-1.13",
        endpoint: str = "https://openrouter.ai/api/alpha/decisions",
        fallback_endpoint: str | None = "https://openrouter.ai/api/v1/systemone",
        timeout_s: float = 15.0,
        max_retries: int = 2,
        http_client: Any = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JEV_API_KEY")
        )
        self.model = model_name
        self.endpoint = endpoint
        self.fallback_endpoint = fallback_endpoint
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.capabilities = JevCapabilities()
        self.http_client = http_client

    def translate_request(self, request: DecisionRequest) -> dict[str, Any]:
        """Translate a canonical DecisionRequest to the OpenRouter Decisions API JSON payload."""
        questions: dict[str, dict[str, Any]] = {}

        for q_id, q in request.questions.items():
            if isinstance(q, ChoiceQuestion):
                questions[q_id] = {
                    "type": "choice",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, ScoreQuestion):
                questions[q_id] = {
                    "type": "score",
                    "instructions": q.instructions,
                    "criteria": q.criteria,
                }
            elif isinstance(q, NoulQuestion):
                questions[q_id] = {
                    "type": "noul",
                    "instructions": q.instructions,
                }

        return {
            "model": self.model,
            "state": dict(request.state),
            "questions": questions,
        }

    def _execute_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP request with authentication, timeout, and retry handling."""
        if not self.api_key:
            raise ProviderAuthenticationError(
                "No API key provided for Jev adapter. "
                "Set the OPENROUTER_API_KEY environment variable or pass api_key to JevAdapter."
            )

        if self.http_client is not None:
            try:
                return self.http_client(self.endpoint, payload, self.api_key, self.timeout_s)
            except urllib.error.HTTPError as err:
                if err.code == 404 and self.fallback_endpoint:
                    return self.http_client(
                        self.fallback_endpoint, payload, self.api_key, self.timeout_s
                    )
                raise

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Jevlaya/{__version__}",
        }
        data = json.dumps(payload).encode("utf-8")
        current_endpoint = self.endpoint

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                current_endpoint,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    resp_bytes = resp.read()
                    return json.loads(resp_bytes.decode("utf-8"))
            except urllib.error.HTTPError as http_err:
                error_body = ""
                try:
                    error_body = http_err.read(4096).decode("utf-8", errors="replace")
                except Exception:
                    pass

                masked_msg = redact_secrets(
                    f"HTTP {http_err.code}: {http_err.reason} - {error_body}"
                )

                if (
                    http_err.code == 404
                    and self.fallback_endpoint
                    and current_endpoint != self.fallback_endpoint
                ):
                    current_endpoint = self.fallback_endpoint
                    continue

                if http_err.code in (401, 403):
                    raise ProviderAuthenticationError(
                        f"Jev authentication failed: {masked_msg}"
                    ) from http_err
                if http_err.code == 400:
                    raise InvalidRequest(
                        f"Jev rejected invalid request: {masked_msg}"
                    ) from http_err
                if http_err.code in (429, 502, 503, 504) and attempt < self.max_retries:
                    retry_after = (
                        http_err.headers.get("Retry-After")
                        if hasattr(http_err, "headers") and http_err.headers
                        else None
                    )
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except (ValueError, TypeError):
                            delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.1)
                    else:
                        delay = (0.5 * (2**attempt)) + random.uniform(0.0, 0.1)
                    time.sleep(delay)
                    continue

                raise ProviderResponseError(f"Jev Decisions API error: {masked_msg}") from http_err
            except (TimeoutError, urllib.error.URLError) as net_err:
                if isinstance(net_err, TimeoutError) or "timed out" in str(net_err).lower():
                    if attempt < self.max_retries:
                        time.sleep((0.5 * (2**attempt)) + random.uniform(0.0, 0.1))
                        continue
                    raise ProviderTimeout(
                        f"Jev request timed out after {self.timeout_s}s"
                    ) from net_err

                raise ProviderUnavailable(
                    f"Jev service unreachable: {redact_secrets(str(net_err))}"
                ) from net_err

        raise ProviderResponseError("Jev request failed after maximum retries")

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Process canonical decision request via Jev API and normalize into DecisionResponse."""
        payload = self.translate_request(request)
        raw_result = self._execute_http(payload)

        if not isinstance(raw_result, dict):
            raise NormalizationError(
                f"Jev API returned invalid non-dict payload: {type(raw_result).__name__}"
            )

        raw_answers = raw_result.get("answers")
        if not isinstance(raw_answers, dict):
            raise NormalizationError("Jev API response missing 'answers' dictionary")

        normalized_answers: dict[str, Answer] = {}

        for q_id, question in request.questions.items():
            if q_id not in raw_answers:
                raise NormalizationError(f"Jev API response missing answer for '{q_id}'")

            raw_ans = raw_answers[q_id]
            if not isinstance(raw_ans, dict):
                raise NormalizationError(f"Jev answer for '{q_id}' is not a dictionary")

            normalized_answers[q_id] = self._normalize_answer(q_id, question, raw_ans)

        # Usage mapping
        raw_usage = raw_result.get("usage", {})
        cost_val = raw_usage.get("cost")
        if cost_val is None:
            cost_val = raw_usage.get("cost_usd")

        usage = UsageInfo(
            input_tokens=raw_usage.get("input_tokens"),
            output_tokens=raw_usage.get("output_tokens"),
            cost_usd=float(cost_val) if cost_val is not None else None,
        )

        response_model = str(raw_result.get("model", self.model))

        return DecisionResponse(
            provider=self.name,
            model=response_model,
            answers=normalized_answers,
            usage=usage,
            latency_ms=0.0,
        )

    def _normalize_answer(
        self,
        q_id: str,
        question: Any,
        raw_ans: dict[str, Any],
    ) -> Answer:
        """Normalize Jev raw answer into canonical ChoiceAnswer, ScoreAnswer, or NoulAnswer."""
        try:
            if isinstance(question, ChoiceQuestion):
                choice = str(raw_ans.get("choice", ""))
                probs = {str(k): float(v) for k, v in raw_ans.get("probabilities", {}).items()}
                conf = float(raw_ans.get("confidence", probs.get(choice, 1.0)))
                return ChoiceAnswer(
                    type="choice",
                    choice=choice,
                    probabilities=probs,
                    confidence=conf,
                )

            if isinstance(question, ScoreQuestion):
                # Jev returns score as a float index (e.g. 1.99) or label string
                raw_score = raw_ans.get("score")
                if isinstance(raw_score, (int, float)):
                    # Clamp index to valid criteria range and resolve label
                    idx = int(round(float(raw_score)))
                    idx = max(0, min(len(question.criteria) - 1, idx))
                    score_label = question.criteria[idx]
                else:
                    score_label = str(raw_score)

                # Probabilities can be a dict {'0': p0, '1': p1} or list [p0, p1]
                probs_val = raw_ans.get("probabilities", [])
                if isinstance(probs_val, dict):
                    # Sort by integer keys
                    sorted_items = sorted(probs_val.items(), key=lambda x: int(x[0]))
                    probs_list = [float(v) for _, v in sorted_items]
                elif isinstance(probs_val, list):
                    probs_list = [float(p) for p in probs_val]
                else:
                    probs_list = []

                conf = float(raw_ans.get("confidence", max(probs_list) if probs_list else 1.0))
                return ScoreAnswer(
                    type="score",
                    score=score_label,
                    probabilities=probs_list,
                    confidence=conf,
                )

            if isinstance(question, NoulQuestion):
                noul_val = float(raw_ans.get("noul", 0.0))
                return NoulAnswer(
                    type="noul",
                    noul=noul_val,
                )
        except Exception as exc:
            raise NormalizationError(f"Failed to normalize Jev answer for '{q_id}': {exc}") from exc

        raise NormalizationError(f"Unrecognized question primitive type: {type(question).__name__}")
