"""Weighted reciprocal rank fusion."""

from __future__ import annotations

import pytest

from minirag.fusion import reciprocal_rank_fusion


def test_single_ranking_is_preserved() -> None:
    fused = reciprocal_rank_fusion({"a": ["x", "y", "z"]})
    assert [item for item, _ in fused] == ["x", "y", "z"]


def test_scores_follow_one_over_k_plus_rank() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62)


def test_contributions_from_multiple_retrievers_add_up() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x"], "b": ["x"]}, k=60))
    assert fused["x"] == pytest.approx(2 / 61)


def test_consensus_beats_a_single_confident_retriever() -> None:
    # The defining property of RRF: agreement at rank 3 outranks a lone
    # first place. This is why hybrid search is robust.
    fused = reciprocal_rank_fusion(
        {"a": ["lone", "p", "consensus"], "b": ["q", "r", "consensus"]}, k=60
    )
    assert fused[0][0] == "consensus"


def test_lower_k_rewards_a_single_strong_opinion() -> None:
    rankings = {"a": ["lone", "p", "consensus"], "b": ["q", "r", "consensus"]}
    assert reciprocal_rank_fusion(rankings, k=1)[0][0] == "lone"


def test_weights_shift_the_balance() -> None:
    rankings = {"lexical": ["lex"], "dense": ["dns"]}
    assert reciprocal_rank_fusion(rankings, weights={"lexical": 5.0})[0][0] == "lex"
    assert reciprocal_rank_fusion(rankings, weights={"dense": 5.0})[0][0] == "dns"


def test_a_zero_weight_silences_a_retriever() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x"], "b": ["y"]}, weights={"b": 0.0}))
    assert fused["y"] == 0.0
    assert fused["x"] > fused["y"]


def test_lists_may_differ_in_length_and_contents() -> None:
    fused = dict(reciprocal_rank_fusion({"a": [1, 2, 3], "b": [9]}))
    assert set(fused) == {1, 2, 3, 9}


def test_top_k_truncates() -> None:
    assert len(reciprocal_rank_fusion({"a": [1, 2, 3, 4]}, top_k=2)) == 2


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion({}) == ()
    assert reciprocal_rank_fusion({"a": []}) == ()


def test_ties_break_deterministically_on_best_rank() -> None:
    fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "x"]})
    assert [item for item, _ in fused] == ["x", "y"]


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": ["x"]}, k=0)


def test_input_rankings_are_not_mutated() -> None:
    rankings = {"a": ["x", "y"], "b": ["y"]}
    snapshot = {name: list(items) for name, items in rankings.items()}
    reciprocal_rank_fusion(rankings)
    assert rankings == snapshot
