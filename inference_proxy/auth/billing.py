"""Premium-equivalent cost estimates for tracked token usage (RFE #113).

Self-hosted inference has near-zero marginal cost per token, so the
"token budget saved" figure is the amount the same prompt/completion mix
would have cost at a premium frontier-model API rate. The estimate is an
equivalence figure, not a cash ledger; it deliberately ignores cache-hit
discounts because the usage store records no cache-token split.
"""

from __future__ import annotations


def premium_equivalent_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_rate_per_mtok: float,
    output_rate_per_mtok: float,
) -> float:
    """Estimate the premium-equivalent cost of a token mix in USD.

    Rates are US dollars per million tokens. Both rates must be
    non-negative; zero input or output is valid (e.g. pure generation).
    """
    if input_rate_per_mtok < 0 or output_rate_per_mtok < 0:
        raise ValueError("rates must be non-negative")
    return (prompt_tokens / 1_000_000) * input_rate_per_mtok + (
        completion_tokens / 1_000_000
    ) * output_rate_per_mtok
