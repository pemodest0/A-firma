from scripts.bench.validation.run_profit_best_of_worlds_matrix import (
    BEST_OF_WORLDS,
    RECOMMENDATIONS,
    _best_of_worlds_rows,
    _flatten_categories,
)


def test_flatten_categories_applies_manual_recommendation() -> None:
    summary = {
        "categories": {
            "official_serious": [
                {
                    "candidate_id": "champion_profit_lock_partial",
                    "net_total_return": 31.0,
                }
            ]
        }
    }
    rows = _flatten_categories(summary)
    assert rows[0]["recommended_action"] == RECOMMENDATIONS["champion_profit_lock_partial"]["action"]
    assert rows[0]["recommended_role"] == RECOMMENDATIONS["champion_profit_lock_partial"]["role"]


def test_best_of_worlds_rows_selects_all_slots() -> None:
    rows = [{"candidate_id": candidate_id} for candidate_id in BEST_OF_WORLDS.values()]
    best = _best_of_worlds_rows(rows)
    assert set(best.keys()) == set(BEST_OF_WORLDS.keys())
    assert best["core"]["candidate_id"] == BEST_OF_WORLDS["core"]
