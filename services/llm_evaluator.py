"""Qualitative LLM risk gate for candidate KXBTC15M trades.

Fails closed: any API error, refusal, truncation, or unparseable response yields a
veto verdict rather than an exception, so a flaky LLM can never green-light a trade.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import anthropic
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from core.schemas import FairValueOutput, LLMVerdict

logger = logging.getLogger(__name__)

# claude-3-5-sonnet-20241022 was retired on 2025-10-28; claude-sonnet-5-5 is its current successor.
DEFAULT_MODEL = os.environ.get("BTC15M_LLM_MODEL", "claude-sonnet-5-5")
DEFAULT_EFFORT = os.environ.get("BTC15M_LLM_EFFORT", "medium")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models without server-side fallback: a fallback list returns 400 on them, so don't send one.
_NO_SERVER_FALLBACK_PREFIXES = ("claude-haiku-5-5",)

SYSTEM_PROMPT = """\
You are the final risk gate for an automated trading system that buys Kalshi KXBTC15M \
contracts: binary options that settle YES if the BTC spot price is above the strike K at \
the end of a 15-minute window.

A quantitative model (Black-Scholes digital option, N(d2) with a momentum adjustment) has \
already found a positive edge versus the Kalshi ask. That model assumes log-normal returns \
with volatility estimated from the last few minutes. Your job is to catch the situations \
where that assumption is likely to break down, and veto the trade when it does.

Before approving, check the supplied context for:
1. Extreme macro news spikes: a scheduled release (CPI, PPI, NFP, FOMC decision or press \
conference, major Fed speeches) inside or just before the contract window, or evidence of \
an unscheduled headline shock (sudden price gap, volume surge, news headlines in context).
2. Liquidation cascades: sharp one-directional 1-minute momentum together with an extreme \
order book imbalance, spiking liquidation volume, open-interest drops, or funding-rate \
dislocations. Cascades produce jumps the volatility estimate has not seen.
3. Abnormal volatility regime shifts: short-window realized volatility far from the \
longer-run baseline, volatility rising quickly, or a volatility input that looks stale \
relative to recent price action.

You have no internet access. Judge only from the context provided. A field that is missing \
is unknown, not safe; weigh how much the missing information matters for this window.

Respond with:
- should_bet: true only if none of the risks above materially threatens the edge.
- adjusted_probability: your estimate of the probability the contract settles YES \
(not the probability of the side being bought), between 0.01 and 0.99. Keep the model's \
value if you see no reason to move it.
- confidence_score: 0.0 to 1.0, how confident you are in this verdict given the data quality.
- reasoning: one to three sentences explaining the decisive factors.
- risk_signals: short labels for each risk you detected, e.g. "FOMC window", \
"Liquidation cascade", "Volatility regime shift". Empty list if none.\
"""


# USD per million tokens: (input, output, cache read). Cache writes (5-minute TTL) bill 1.25x input.
PRICING_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
# Server-side fallbacks may answer from a model not listed above; price those at Opus rates so
# spend is over- rather than under-counted.
_UNKNOWN_MODEL_PRICING = PRICING_PER_MTOK["claude-opus-5-5"]


@dataclass(frozen=True)
class LLMUsage:
    model: str
    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        inp, out, cache_read = PRICING_PER_MTOK.get(self.model, _UNKNOWN_MODEL_PRICING)
        return (
            self.input_tokens * inp
            + self.output_tokens * out
            + self.cache_read_tokens * cache_read
            + self.cache_write_tokens * inp * 1.25
        ) / 1_000_000

    @classmethod
    def from_response(cls, response: Any, requested_model: str) -> Optional[LLMUsage]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return cls(
            model=getattr(response, "model", None) or requested_model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )


def _veto(reason: str, signal: str, fallback_probability: float) -> LLMVerdict:
    return LLMVerdict(
        should_bet=False,
        adjusted_probability=min(max(fallback_probability, 0.01), 0.99),
        confidence_score=0.0,
        reasoning=reason,
        risk_signals=[signal],
    )


def build_user_prompt(candidate: FairValueOutput, context: dict[str, Any]) -> str:
    return (
        "Candidate trade from the quantitative model:\n"
        f"{candidate.model_dump_json(indent=2)}\n\n"
        "Market and risk context:\n"
        f"{json.dumps(context, indent=2, sort_keys=True, default=str)}\n\n"
        "Should the system take this trade?"
    )


class LLMEvaluator:
    def __init__(
        self,
        client: Optional[AsyncAnthropic] = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        timeout_sec: float = 30.0,
    ) -> None:
        # The 15-minute windows make a slow answer worthless, so cap the request time.
        self._client = client or AsyncAnthropic(timeout=timeout_sec, max_retries=2)
        self.model = model
        self.effort = effort
        self.server_fallback = not model.startswith(_NO_SERVER_FALLBACK_PREFIXES)

    async def evaluate(self, candidate: FairValueOutput, context: dict[str, Any]) -> LLMVerdict:
        verdict, _ = await self.evaluate_with_usage(candidate, context)
        return verdict

    async def evaluate_with_usage(
        self, candidate: FairValueOutput, context: dict[str, Any]
    ) -> tuple[LLMVerdict, Optional[LLMUsage]]:
        """Like `evaluate`, plus billed token usage (None when no response was billed)."""
        model_p = candidate.model_probability
        try:
            fallback: dict[str, Any] = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.server_fallback else {}
            response = await self._client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                **fallback,
                # The static system prompt (plus the output schema) is identical on every call;
                # cache it so repeat calls bill it at the cache-read rate.
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": self.effort},
                output_format=LLMVerdict,
                messages=[{"role": "user", "content": build_user_prompt(candidate, context)}],
            )
        except anthropic.APIStatusError as exc:
            logger.warning("LLM evaluator API error %s for %s: %s", exc.status_code, candidate.ticker, exc.message)
            return _veto(f"LLM API error ({exc.status_code}); vetoing by default.", "LLM unavailable", model_p), None
        except anthropic.APIConnectionError as exc:
            logger.warning("LLM evaluator connection error for %s: %s", candidate.ticker, exc)
            return _veto("LLM connection error; vetoing by default.", "LLM unavailable", model_p), None
        except (ValidationError, ValueError) as exc:
            logger.warning("LLM evaluator returned an invalid verdict for %s: %s", candidate.ticker, exc)
            return _veto("LLM returned an invalid verdict; vetoing by default.", "LLM invalid output", model_p), None

        usage = LLMUsage.from_response(response, self.model)
        if response.stop_reason == "refusal":
            return _veto("LLM declined to evaluate; vetoing by default.", "LLM refusal", model_p), usage
        if response.stop_reason == "max_tokens":
            return _veto("LLM response was truncated; vetoing by default.", "LLM invalid output", model_p), usage

        verdict = response.parsed_output
        if verdict is None:
            return _veto("LLM returned no verdict; vetoing by default.", "LLM invalid output", model_p), usage
        return verdict, usage


_default_evaluator: Optional[LLMEvaluator] = None


async def evaluate_btc_trade(
    candidate: FairValueOutput,
    context: dict[str, Any],
    evaluator: Optional[LLMEvaluator] = None,
) -> LLMVerdict:
    """Ask Claude whether a model-approved trade should proceed."""
    global _default_evaluator
    if evaluator is None:
        if _default_evaluator is None:
            _default_evaluator = LLMEvaluator()
        evaluator = _default_evaluator
    return await evaluator.evaluate(candidate, context)
