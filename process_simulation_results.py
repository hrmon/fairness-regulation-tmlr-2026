"""Summarize the convergence behaviour of the random-transition simulations.

For every ``tv``/``delta`` pair written by ``simulation.py`` this script
classifies each simulated trajectory as

- non-converged (the total-variation distance has not stabilized),
- converged to zero disparity, or
- converged to a strictly positive disparity,

and stores the counts as one row per penalty weight in ``converge/*.csv``.

Run as a script::

    python process_random_tran_band.py --sim-dir outputs/simulations/random_transitions
"""

import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd

DEFAULT_SIM_DIR = "outputs/simulations/random_transitions"
DEFAULT_LAST_STEPS = 10
DEFAULT_CONV_RTOL = 0.01
DEFAULT_CONV_ATOL = 1e-8
DEFAULT_ZERO_ATOL = 1e-2


def read_series(path):
    """Read one time series per column from a space-separated ``.dat`` file.

    The files store a row index in their first column, which is dropped.
    """
    arr = np.loadtxt(path, skiprows=1)
    return arr[:, 1:]


def is_converged(series, n_last, rtol, atol):
    """Return True if the last ``n_last`` values span a small fraction of their scale.

    Parameters
    ----------
    series : np.ndarray
        A single time series.
    n_last : int
        Number of trailing values considered.
    rtol : float
        Relative tolerance on the span of the trailing values.
    atol : float
        Absolute floor used as the scale when the mean is (near) zero.
    """
    last = series[-n_last:]
    span = last.max() - last.min()
    scale = max(abs(last.mean()), atol)
    return span <= rtol * scale


def process_file(tv_path, delta_path, n_last, rtol, atol, zero_atol):
    """Classify the trajectories stored in ``tv_path``/``delta_path``.

    Returns the number of non-converged, converged-to-zero and
    converged-to-nonzero trajectories.
    """
    tv = read_series(tv_path)
    delta = read_series(delta_path)
    non_converged = to_zero = to_nonzero = 0
    for j in range(tv.shape[1]):
        if not is_converged(tv[:, j], n_last, rtol, atol):
            non_converged += 1
        elif abs(delta[-n_last:, j].mean()) <= zero_atol:
            to_zero += 1
        else:
            to_nonzero += 1
    return non_converged, to_zero, to_nonzero


def main(argv=None):
    """Parse the command line and write the per-penalty convergence summaries."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-dir",
        default=DEFAULT_SIM_DIR,
        help="Directory containing tv/ and delta/.",
    )
    parser.add_argument(
        "--n-last",
        type=int,
        default=DEFAULT_LAST_STEPS,
        help="Trailing steps used by the convergence test.",
    )
    parser.add_argument(
        "--conv-rtol",
        type=float,
        default=DEFAULT_CONV_RTOL,
        help="Relative tolerance for the TV convergence test.",
    )
    parser.add_argument(
        "--conv-atol",
        type=float,
        default=DEFAULT_CONV_ATOL,
        help="Absolute floor for the TV convergence test.",
    )
    parser.add_argument(
        "--zero-atol",
        type=float,
        default=DEFAULT_ZERO_ATOL,
        help="Final-disparity threshold treated as zero.",
    )
    args = parser.parse_args(argv)

    tv_dir = os.path.join(args.sim_dir, "tv")
    delta_dir = os.path.join(args.sim_dir, "delta")
    out_dir = os.path.join(args.sim_dir, "converge")
    os.makedirs(out_dir, exist_ok=True)

    funcs = sorted({re.match(r"([A-Za-z]+)_", f).group(1) for f in os.listdir(tv_dir)})
    for func in funcs:
        rows = []
        pattern = re.compile(re.escape(func) + r"_(.*)\.dat$")
        fnames = [f for f in os.listdir(tv_dir) if pattern.match(f)]
        for fname in sorted(fnames, key=lambda f: float(pattern.match(f).group(1))):
            x = float(pattern.match(fname).group(1))
            non_converged, to_zero, to_nonzero = process_file(
                os.path.join(tv_dir, fname),
                os.path.join(delta_dir, fname),
                args.n_last,
                args.conv_rtol,
                args.conv_atol,
                args.zero_atol,
            )
            rows.append(
                {
                    "x": x,
                    "non_converged": non_converged,
                    "converged_to_zero": to_zero,
                    "converged_to_nonzero": to_nonzero,
                }
            )
        out = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"{func}.csv")
        out.to_csv(out_path, sep=" ", index=False)
        print(f"{func}: wrote {out_path}")


if __name__ == "__main__":
    main()
