from __future__ import annotations

import numpy as np
import pandas as pd

from engine.portfolio.hrp import hrp_weights


def test_hrp_weights_sum_to_one_and_are_non_negative() -> None:
    cov = pd.DataFrame(
        [
            [0.04, 0.03, 0.01],
            [0.03, 0.09, 0.015],
            [0.01, 0.015, 0.025],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    w = hrp_weights(cov)
    assert list(w.index) == ["a", "b", "c"]
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0.0).all()


def test_hrp_weights_handle_single_asset() -> None:
    cov = pd.DataFrame([[0.04]], index=["only"], columns=["only"])
    w = hrp_weights(cov)
    assert w["only"] == 1.0
