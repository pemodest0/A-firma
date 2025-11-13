from __future__ import annotations

import numpy as np

from gaq.core.two_qubits import chsh_value, singlet


def test_chsh_value_reaches_tsirelson_bound() -> None:
    rho = singlet()
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    b1 = -(z + x) / np.sqrt(2.0)
    b2 = -(z - x) / np.sqrt(2.0)
    s_value = chsh_value(rho, z, x, b1, b2)
    assert abs(s_value - 2 * np.sqrt(2.0)) < 1e-10
