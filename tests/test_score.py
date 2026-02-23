from __future__ import annotations

import numpy as np

from engine.structural.score import fit_normalizer, structural_score, transform


def test_fit_normalizer_and_transform() -> None:
    train = {"phi": [0.2, 0.3, 0.4], "deff": [4.0, 3.8, 3.5]}
    params = fit_normalizer(train)
    z = transform([0.3, 0.4, 0.5], params, key="phi")
    assert np.isfinite(z).all()
    assert float(z.iloc[-1]) > float(z.iloc[0])


def test_structural_score_increases_with_fragility_pattern() -> None:
    n = 50
    phi_z = np.linspace(-1.0, 1.0, n)
    deff_z = np.linspace(1.0, -1.0, n)
    ac1_z = np.linspace(-0.5, 1.2, n)
    neg_kappa_z = np.linspace(-0.2, 0.8, n)

    score = structural_score(
        {
            "phi": phi_z,
            "deff": deff_z,
            "ac1_phi": ac1_z,
            "neg_kappa_mean": neg_kappa_z,
        }
    )
    assert np.isfinite(score).all()
    assert float(score.iloc[-1]) > float(score.iloc[0])
