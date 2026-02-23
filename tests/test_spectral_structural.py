from __future__ import annotations

import numpy as np

from engine.structural.spectral import effective_dimension, normalize_eigs, order_param_phi, spectral_entropy, spectral_pack


def test_normalize_eigs_clips_negatives_and_sums_to_one() -> None:
    p = normalize_eigs([2.0, -1.0, 1.0, np.nan])
    assert np.isclose(float(p.sum()), 1.0, atol=1e-9)
    assert np.all(p >= 0.0)


def test_entropy_deff_phi_consistency() -> None:
    eigs = np.array([4.0, 1.0, 1.0], dtype=float)
    h = spectral_entropy(eigs)
    deff = effective_dimension(eigs)
    phi = order_param_phi(eigs)
    assert h > 0.0
    assert deff >= 1.0
    assert 0.0 <= phi <= 1.0


def test_spectral_pack_fields() -> None:
    pack = spectral_pack([3.0, 2.0, 1.0], topk=2)
    assert {"phi", "H", "deff", "lambda1", "topk", "n_eigs"}.issubset(pack.keys())
    assert pack["lambda1"] == 3.0
    assert pack["n_eigs"] == 3
