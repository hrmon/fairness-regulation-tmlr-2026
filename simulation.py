"""Parallel Monte-Carlo simulations over random transition dynamics.

This script runs :func:`utils.run_dynamics_iterations` for many randomly drawn
transition-matrix pairs and several penalty families and weights on the LSAC
bar-pass dataset. For every configuration it saves the total-variation
history and the disparity history of each simulation under
``outputs/simulations/random_transitions/``.

Run as a script::

    python simulation.py --data lsac/bar_pass_prediction.csv --seed 0

or import :func:`run_simulations` for programmatic use.
"""

import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    run_dynamics_iterations,
    generate_random_transition,
    split_distributions,
)

DEFAULT_DATA_PATH = "lsac/bar_pass_prediction.csv"
DEFAULT_OUTPUT_DIR = "outputs/simulations/random_transitions"

GPA_THRESHOLD = 2.95
MAJORITY_RACE = 7


def load_lsac(path=DEFAULT_DATA_PATH):
    """Load the LSAC bar-pass dataset and drop rows with missing key fields."""
    df = pd.read_csv(path)
    return df.dropna(subset=["zfygpa", "gpa", "sex", "race"])


def prepare_lsac(df, threshold=GPA_THRESHOLD, group_a=MAJORITY_RACE):
    """Return the empirical GPA distributions of the two groups.

    Group A is ``race == group_a`` (the majority group in the dataset); the
    GPA values are shifted by ``threshold`` before being discretized.

    Returns
    -------
    a, b : np.ndarray
        Probability vectors for the two groups over the discretized GPA.
    a_size : float
        Fraction of rows belonging to group A.
    Y : np.ndarray
        Sorted array of discretized (shifted) GPA values.
    """
    return split_distributions(
        df,
        value_col="gpa",
        group_col="race",
        group_a=group_a,
        threshold=threshold,
    )


def _run_single_sim(args):
    """Run one simulation and return the tv and disparity histories."""
    Y, T1, T2, a, b, penalty, lamb, a_size, iters = args
    result = run_dynamics_iterations(
        Y, T1, T2, a, b, penalty, lamb=lamb, a_size=a_size, iters=iters, disable=True
    )
    return result.tv, result.delta_hist


def run_simulations(
    data_path=DEFAULT_DATA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    num_simulations=100,
    iters=1000,
    lambs=None,
    penalties=None,
    random_seed=0,
):
    """Run the Monte-Carlo simulation suite and persist its results.

    Parameters
    ----------
    data_path : str
        Path to the LSAC CSV file.
    output_dir : str
        Directory in which ``tv/``, ``delta/`` and the transition matrices
        are written.
    num_simulations : int
        Number of random transition-matrix pairs per configuration.
    iters : int
        Number of iterations per simulation.
    lambs : array-like, optional
        Penalty weights; defaults to ``np.linspace(0, 0.5, 100)``.
    penalties : list of callables, optional
        Penalty functions; defaults to ``[cp.square, cp.abs, cp.exp]``.
    random_seed : int
        Seed for the global numpy random state.
    """
    if lambs is None:
        lambs = np.linspace(0, 0.5, 100)
    if penalties is None:
        penalties = [cp.square, cp.abs, cp.exp]

    np.random.seed(random_seed)

    df = load_lsac(data_path)
    a, b, a_size, Y = prepare_lsac(df)

    T1s = [generate_random_transition(len(Y)) for _ in range(num_simulations)]
    T2s = [generate_random_transition(len(Y)) for _ in range(num_simulations)]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "t1s.npy"), np.array(T1s))
    np.save(os.path.join(output_dir, "t2s.npy"), np.array(T2s))

    delta_dir = os.path.join(output_dir, "delta")
    tv_dir = os.path.join(output_dir, "tv")
    os.makedirs(delta_dir, exist_ok=True)
    os.makedirs(tv_dir, exist_ok=True)

    total = len(penalties) * len(lambs) * len(T1s)
    with (
        ProcessPoolExecutor() as executor,
        tqdm(total=total, desc="simulations") as pbar,
    ):
        for penalty in penalties:
            for lamb in lambs:
                sims = [
                    (Y, T1, T2, a, b, penalty, lamb, a_size, iters)
                    for T1, T2 in zip(T1s, T2s)
                ]
                futures = {
                    executor.submit(_run_single_sim, sim): i
                    for i, sim in enumerate(sims)
                }
                tv_hists = [None] * len(sims)
                delta_hists = [None] * len(sims)
                pbar.set_description(f"{penalty.__name__} lambda={lamb:.3f}")
                for fut in as_completed(futures):
                    i = futures[fut]
                    tv, delta = fut.result()
                    tv_hists[i] = tv
                    delta_hists[i] = delta
                    pbar.update(1)

                prefix = f"{penalty.__name__}_{lamb}"
                pd.DataFrame(delta_hists).transpose().to_csv(
                    os.path.join(delta_dir, f"{prefix}.dat"), sep=" "
                )
                pd.DataFrame(tv_hists).transpose().to_csv(
                    os.path.join(tv_dir, f"{prefix}.dat"), sep=" "
                )


def main(argv=None):
    """Parse the command line and run the simulation suite."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH, help="Path to the LSAC CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for simulation outputs.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Transition-matrix pairs per configuration.",
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations per simulation."
    )
    parser.add_argument(
        "--num-lambs", type=int, default=100, help="Number of penalty weights."
    )
    parser.add_argument(
        "--max-lamb", type=float, default=0.5, help="Largest penalty weight."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args(argv)

    run_simulations(
        data_path=args.data,
        output_dir=args.output_dir,
        num_simulations=args.num_simulations,
        iters=args.iters,
        lambs=np.linspace(0, args.max_lamb, args.num_lambs),
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
