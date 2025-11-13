"""Compare analytic hitting times (Staples 2005) with Monte Carlo estimates."""
from __future__ import annotations

import argparse

from gaq.analysis.hypercube_walks import simulate_hitting_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int, help="Dimension of the hypercube")
    parser.add_argument("source", help="Binary string for source vertex, e.g. 000")
    parser.add_argument("target", help="Binary string for target vertex, e.g. 111")
    parser.add_argument("--trials", type=int, default=5000, help="Number of Monte Carlo trials")
    args = parser.parse_args()

    source = tuple(int(bit) for bit in args.source)
    target = tuple(int(bit) for bit in args.target)
    result = simulate_hitting_time(args.n, source, target, trials=args.trials)
    print(f"H_n({args.source}→{args.target}) analytic: {result.expected_steps:.6f}")
    print(f"Monte Carlo mean ± std: {result.simulated_mean:.6f} ± {result.simulated_std:.6f}")
    print(f"Trials: {result.trials}")


if __name__ == "__main__":
    main()
