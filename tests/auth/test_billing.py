"""Tests for the premium-equivalent cost estimator (RFE #113)."""

from __future__ import annotations

import pytest

from inference_proxy.auth.billing import premium_equivalent_cost

_RATES = {"input_rate_per_mtok": 5.0, "output_rate_per_mtok": 25.0}


class TestPremiumEquivalentCost:
    def test_mixed_mix_matches_reference_example(self) -> None:
        # 1M prompt + 500k completion at $5/$25 per Mtok = $5.00 + $12.50.
        assert premium_equivalent_cost(1_000_000, 500_000, **_RATES) == pytest.approx(
            17.5
        )

    def test_zero_usage_is_zero_cost(self) -> None:
        assert premium_equivalent_cost(0, 0, **_RATES) == 0.0

    def test_output_only_counts_output_rate(self) -> None:
        assert premium_equivalent_cost(0, 1_000_000, **_RATES) == pytest.approx(25.0)

    def test_input_only_counts_input_rate(self) -> None:
        assert premium_equivalent_cost(1_000_000, 0, **_RATES) == pytest.approx(5.0)

    def test_fractional_tokens_scale_linearly(self) -> None:
        assert premium_equivalent_cost(1, 1, **_RATES) == pytest.approx(0.00003)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            premium_equivalent_cost(
                1, 1, input_rate_per_mtok=-1.0, output_rate_per_mtok=25.0
            )
