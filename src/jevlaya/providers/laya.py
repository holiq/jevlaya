"""Laya provider adapter for local, non-autoregressive System 1 inference."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from jevlaya.errors import (
    NormalizationError,
    ProviderResponseError,
    ProviderUnavailable,
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


class LayaCapabilities(BaseModel):
    """Declared capabilities and constraints for the Laya model family."""

    model_config = ConfigDict(frozen=True)

    supports_choice: bool = True
    supports_score: bool = True
    supports_noul: bool = True
    supports_multilingual: bool = True
    local_inference: bool = True
    max_recommended_options: int = 20


class LayaAdapter:
    """Adapter integrating the open-weights Laya decision model family."""

    name: str = "laya"

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        router: Any = None,
        preload: bool = False,
        device: str | None = None,
    ) -> None:
        self.model = model_name
        self.capabilities = LayaCapabilities()
        self._router = router
        self._preload = preload
        self._device = device

    @property
    def router(self) -> Any:
        """Resolve or lazily initialize the underlying Laya router instance."""
        if self._router is not None:
            return self._router

        try:
            import laya  # type: ignore[import-untyped]

            self._router = laya.Router(preload=self._preload, device=self._device)
            return self._router
        except ImportError as exc:
            raise ProviderUnavailable(
                "The 'laya' package is required for local inference. "
                "Install it via 'pip install laya' or 'uv add laya'."
            ) from exc

    def translate_request(
        self, request: DecisionRequest
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Translate a canonical DecisionRequest into Laya state and questions payloads."""
        state = dict(request.state)
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

        return state, questions

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Execute decision inference using Laya and normalize the response."""
        backend_router = self.router
        state, questions = self.translate_request(request)

        try:
            raw_result = backend_router.predict(state, questions)
        except Exception as exc:
            raise ProviderResponseError(
                f"Laya model inference failed during prediction: {exc}"
            ) from exc

        if not isinstance(raw_result, dict):
            raise NormalizationError(
                f"Laya router returned non-dict output: {type(raw_result).__name__}"
            )

        raw_answers = raw_result.get("answers")
        if not isinstance(raw_answers, dict):
            raise NormalizationError("Laya router output missing valid 'answers' dictionary")

        normalized_answers: dict[str, Answer] = {}

        for q_id, question in request.questions.items():
            if q_id not in raw_answers:
                raise NormalizationError(
                    f"Laya output did not provide an answer for question '{q_id}'"
                )

            raw_ans = raw_answers[q_id]
            if not isinstance(raw_ans, dict):
                raise NormalizationError(f"Laya answer for question '{q_id}' is not a dictionary")

            normalized_answers[q_id] = self._normalize_answer(q_id, question, raw_ans)

        # Inspect routing metadata if provided by Laya Router
        active_model = self.model
        routing_info = raw_result.get("routing")
        if isinstance(routing_info, dict) and "model" in routing_info:
            active_model = f"{self.model}:{routing_info['model']}"

        return DecisionResponse(
            provider=self.name,
            model=active_model,
            answers=normalized_answers,
            usage=UsageInfo(input_tokens=None, output_tokens=None, cost_usd=0.0),
            latency_ms=0.0,
        )

    def _normalize_answer(
        self,
        q_id: str,
        question: Any,
        raw_ans: dict[str, Any],
    ) -> Answer:
        """Normalize a single answer dict into a canonical Answer primitive."""
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
                score = str(raw_ans.get("score", ""))
                probs_raw = raw_ans.get("probabilities", [])
                probs_list = [float(p) for p in probs_raw]
                conf = float(raw_ans.get("confidence", max(probs_list) if probs_list else 1.0))
                return ScoreAnswer(
                    type="score",
                    score=score,
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
            raise NormalizationError(
                f"Failed to normalize Laya answer for '{q_id}': {exc}"
            ) from exc

        raise NormalizationError(f"Unrecognized question primitive type: {type(question).__name__}")
