from __future__ import annotations

from dreamer.demo_seed import DEMO_FACTS


def test_demo_seed_has_promotion_ready_facts():
    promotion_ready = [
        fact for fact in DEMO_FACTS
        if (
            fact["confidence"] == 3
            and fact["stability"] == "long_term"
            and fact["salience"] >= 3
            and fact["use_count"] >= 5
        )
    ]

    assert len(promotion_ready) >= 3
