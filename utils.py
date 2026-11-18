"""Shared utilities for the growing-dynamics fairness experiments.

This module provides the numerical building blocks used by
``simulation.py`` and the accompanying reproduction notebook:

- generation of synthetic transition matrices (random, gradual growth);
- the iterative policy algorithm (:func:`run_dynamics_iterations`) and its record
  type :class:`IterationResult`;
- helpers to prepare empirical distributions from tabular data.

All random generators draw from the global ``numpy.random`` state so that
calling ``numpy.random.seed(...)`` fully controls reproducibility.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sc
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Dynamics generation
# --------------------------------------------------------------------------- #


def generate_random_transition(D, rng=None):
    """Return a random row-stochastic tridiagonal transition matrix.

    Each row is built from three positive weights (the diagonal and the two
    neighbouring bands) drawn independently and normalized to sum to one.

    Parameters
    ----------
    D : int
        Number of states (size of the square matrix).
    rng : np.random.Generator or np.random.RandomState, optional
        Random source; defaults to the global ``numpy.random`` state.

    Returns
    -------
    np.ndarray
        A ``(D, D)`` row-stochastic tridiagonal matrix.
    """
    rng = rng if rng is not None else np.random
    T = sc.sparse.diags(
        [rng.uniform(size=(D,)) for _ in range(3)], [-1, 0, 1], shape=(D, D)
    ).toarray()
    return T / T.sum(axis=1, keepdims=True)


def generate_gradual_growth_dynamics(D, params=None, rng=None):
    """Return a pair of tridiagonal matrices modelling gradual growth.

    The two transition matrices share the same tridiagonal structure but use
    different up/down jump probabilities. If ``params`` is not given the
    probabilities are drawn at random with the constraints
    ``0 <= p2 <= p1 <= r[1]`` and ``0 <= q2 <= q1``, giving the second
    matrix a smaller growth advantage than the first.

    Parameters
    ----------
    D : int
        Number of states.
    params : tuple (p1, q1, p2, q2), optional
        Explicit transition probabilities. Defaults to random draws.
    rng : np.random.Generator or np.random.RandomState, optional
        Random source; defaults to the global ``numpy.random`` state.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The two ``(D, D)`` row-stochastic tridiagonal matrices ``(T1, T2)``.
    """
    rng = rng if rng is not None else np.random
    if params is None:
        r = np.sort(rng.uniform(size=2))
        p1, q1 = r[0], 1 - r[1]
        p2 = rng.uniform(0, p1)
        q2 = 1 - rng.uniform(p2, r[1])
    else:
        p1, q1, p2, q2 = params

    T1 = sc.sparse.diags(
        [np.ones(D) * q1, np.ones(D) * (1 - q1 - p1), np.ones(D) * p1],
        [-1, 0, 1],
        shape=(D, D),
    ).toarray()
    T1[0, 0] += q1
    T1[-1, -1] += p1

    T2 = sc.sparse.diags(
        [np.ones(D) * q2, np.ones(D) * (1 - q2 - p2), np.ones(D) * p2],
        [-1, 0, 1],
        shape=(D, D),
    ).toarray()
    T2[0, 0] += q2
    T2[-1, -1] += p2
    return T1, T2


def tridiagonal_transition(down, up):
    """Build a row-stochastic tridiagonal transition matrix.

    The diagonal is chosen so that every row sums to one; the two boundary
    rows only keep their diagonal and the available off-diagonal entry.

    Parameters
    ----------
    down : array-like
        Sub-diagonal weights (transition ``i -> i - 1``) for each row.
    up : array-like
        Super-diagonal weights (transition ``i -> i + 1``) for each row.

    Returns
    -------
    np.ndarray
        A ``(n, n)`` row-stochastic tridiagonal matrix.
    """
    down = np.asarray(down, dtype=float)
    up = np.asarray(up, dtype=float)
    n = len(down)
    diag = np.ones(n) - up
    diag[1:] -= down[:-1]
    T = sc.sparse.diags([down, diag, up], [-1, 0, 1], shape=(n, n)).toarray()
    T[0, 0] = 1 - T[0, 1]
    T[-1, -1] = 1 - T[-1, -2]
    return T


def perturb_transition_matrix(T, scale=0.1):
    """Add Gaussian noise to the positive entries of a stochastic matrix.

    The perturbed entries are clipped to be non-negative and each row is
    renormalized, so the result is again a valid row-stochastic matrix.

    Parameters
    ----------
    T : np.ndarray
        Input (row-stochastic) transition matrix.
    scale : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Perturbed row-stochastic matrix with the same shape as ``T``.
    """
    T = np.asarray(T, dtype=float)
    noise = np.random.normal(loc=0, scale=scale, size=T.shape)
    noise = np.where(T > 0, noise, 0.0)
    perturbed = np.abs(T + noise)
    return perturbed / perturbed.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Empirical distributions
# --------------------------------------------------------------------------- #


def split_distributions(df, value_col, group_col, group_a, threshold=0.0):
    """Split a dataframe into two probability distributions over a discretized column.

    The values of ``value_col`` are shifted by ``threshold`` and then treated
    as a discrete index set. Group A contains the rows where ``group_col``
    equals ``group_a``; the complement forms group B.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    value_col : str
        Column holding the continuous value to discretize (e.g. GPA).
    group_col : str
        Column identifying group membership.
    group_a : object
        Value of ``group_col`` that defines group A.
    threshold : float
        Constant subtracted from ``value_col`` to center the values.

    Returns
    -------
    a, b : np.ndarray
        Probability vectors for group A and B, aligned with ``values``.
    a_size : float
        Fraction of rows belonging to group A.
    values : np.ndarray
        Sorted array of discretized values (the index for ``a`` and ``b``).
    """
    shifted = df[value_col] - threshold
    values = np.sort(shifted.unique())

    is_a = df[group_col] == group_a
    counts_a = shifted[is_a].value_counts()
    counts_b = shifted[~is_a].value_counts()

    a = counts_a.reindex(values, fill_value=0).to_numpy(dtype=float)
    b = counts_b.reindex(values, fill_value=0).to_numpy(dtype=float)
    a /= a.sum()
    b /= b.sum()

    a_size = int(is_a.sum()) / len(df)
    return a, b, a_size, values


# --------------------------------------------------------------------------- #
# Perturbed-state helpers (used by run_dynamics_iterations_with_purturb)
# --------------------------------------------------------------------------- #


def perturb_qualifications(q, noise_scale=0.5, rng=None):
    """Add Gaussian noise to a series of state indices, clipped to the valid range.

    Parameters
    ----------
    q : pd.Series or np.ndarray
        State indices (one per individual).
    noise_scale : float
        Standard deviation of the added noise.
    rng : np.random.Generator or np.random.RandomState, optional
        Random source; defaults to the global ``numpy.random`` state.

    Returns
    -------
    pd.Series
        Noisy state indices, rounded and clipped to ``[0, max(q)]``.
    """
    rng = rng if rng is not None else np.random
    q = q if isinstance(q, pd.Series) else pd.Series(q)
    noisy = q + rng.normal(0, noise_scale, size=q.shape)
    return noisy.clip(lower=0, upper=int(q.max())).round()


def decision(p, v, rng=None):
    """Decide for every individual whether the policy acts.

    Each individual in ``v`` (with state index ``y``) decides independently
    at random with probability ``p[y]``.

    Parameters
    ----------
    p : np.ndarray
        Policy vector (probability of acting) indexed by state.
    v : pd.Series
        State indices, one per individual.
    rng : ..., optional
        Random source; defaults to the global ``numpy.random`` state.

    Returns
    -------
    pd.Series of bool
        Whether each individual is acted upon.
    """
    rng = rng if rng is not None else np.random
    return pd.Series([x < p[int(y)] for x, y in zip(rng.uniform(0, 1, len(v)), v)])


def transform_q(d, v, T0, T1, D, rng=None):
    """Advance each individual's state through ``T1`` or ``T0``.

    Individuals acted upon (``d`` true) transition according to ``T1``,
    the others according to ``T0``.

    Parameters
    ----------
    d : pd.Series of bool
        Whether each individual is acted upon.
    v : pd.Series
        Current state indices, one per individual.
    T0, T1 : np.ndarray
        ``(D, D)`` transition matrices.
    D : int
        Number of states.
    rng : ..., optional
        Random source; defaults to the global ``numpy.random`` state.

    Returns
    -------
    pd.Series
        New state indices, one per individual.
    """
    rng = rng if rng is not None else np.random
    return pd.Series(
        [
            rng.choice(np.arange(D), p=(T1[int(vi)] if di else T0[int(vi)]))
            for di, vi in zip(d, v)
        ]
    )


def get_dist(series, index):
    """Turn a series of state labels into a probability vector over ``index``.

    Parameters
    ----------
    series : pd.Series
        Observed state labels.
    index : array-like
        The full set of possible states.

    Returns
    -------
    pd.Series
        Empirical probability vector (zeros for unobserved states).
    """
    counts = series.value_counts()
    dist = pd.Series(np.zeros(len(index)), index=index)
    dist.update(counts)
    total = counts.sum()
    return dist if total == 0 else dist / total


# --------------------------------------------------------------------------- #
# Iterative policy algorithm
# --------------------------------------------------------------------------- #


@dataclass
class IterationResult:
    """Recorded histories of one run of :func:`run_dynamics_iterations`.

    The object also supports tuple-style indexing (``result[0]`` is
    ``result.tv``, ``result[-1]`` is ``result.value_hist``, ...) for
    backwards compatibility with earlier notebooks.
    """

    tv: np.ndarray
    a_hist: np.ndarray
    b_hist: np.ndarray
    delta_hist: np.ndarray
    p_hist: np.ndarray
    q_hist: np.ndarray
    value_hist: np.ndarray

    def __post_init__(self):
        self.tv = np.asarray(self.tv, dtype=float)
        self.a_hist = np.asarray(self.a_hist, dtype=float)
        self.b_hist = np.asarray(self.b_hist, dtype=float)
        self.delta_hist = np.asarray(self.delta_hist, dtype=float)
        self.p_hist = np.asarray(self.p_hist, dtype=float)
        self.q_hist = np.asarray(self.q_hist, dtype=float)
        self.value_hist = np.asarray(self.value_hist, dtype=float)

    def __iter__(self):
        return iter(
            (
                self.tv,
                self.a_hist,
                self.b_hist,
                self.delta_hist,
                self.p_hist,
                self.q_hist,
                self.value_hist,
            )
        )

    def __getitem__(self, index):
        return tuple(self)[index]


def run_dynamics_iterations(
    Y,
    T1,
    T2,
    a,
    b,
    penalty,
    lamb,
    a_size=0.5,
    iters=200,
    threshold=None,
    disable=False,
):
    """Run the iterative policy-update dynamics and record the histories.

    At every step the two populations are represented by distributions ``a``
    and ``b`` over the states ``Y``. A policy (a vector in ``[0, 1]``) is
    chosen for each group by maximizing a weighted objective minus a penalty
    on the disparity ``|a p - b q|``; the distributions are then advanced
    through the transition matrices ``T1`` (where the policy acts) and ``T2``
    (where it does not).

    Parameters
    ----------
    Y : np.ndarray
        State values (used to size the policy vectors).
    T1, T2 : np.ndarray
        ``(D, D)`` transition matrices.
    a, b : np.ndarray
        Initial probability distributions over the states.
    penalty : callable
        Convex penalty (e.g. ``cvxpy.abs``, ``cvxpy.square``, ``cvxpy.exp``).
    lamb : float
        Penalty weight.
    a_size : float
        Fraction of the population belonging to group A.
    iters : int
        Maximum number of iterations.
    threshold : float, optional
        If given, stop early when the TV distance changes by less than this.
    disable : bool
        Disable the per-iteration progress bar.

    Returns
    -------
    IterationResult
        The recorded histories (see :class:`IterationResult`).
    """
    a_hist, b_hist, tv_hist = [], [], []
    p_hist, q_hist, d_hist, val_hist = [], [], [], []

    for i in tqdm(range(iters), disable=disable):
        a_hist.append(a)
        b_hist.append(b)
        tv_hist.append(np.sum(np.abs(a - b)))

        if (
            len(tv_hist) > 100
            and threshold
            and abs(tv_hist[-1] - tv_hist[-2]) < threshold
        ):
            break

        p = cp.Variable(len(Y))
        q = cp.Variable(len(Y))
        if lamb > 0:
            obj = cp.Maximize(
                a_size * cp.multiply(Y, a) @ p
                + (1 - a_size) * (cp.multiply(Y, b) @ q)
                - lamb * penalty(cp.abs(a @ p - b @ q))
            )
        else:
            obj = cp.Maximize(
                a_size * cp.multiply(Y, a) @ p + (1 - a_size) * (cp.multiply(Y, b) @ q)
            )
        prob = cp.Problem(obj, [p >= 0, p <= 1, q >= 0, q <= 1])
        prob.solve()
        val_hist.append(prob.value)
        p_hist.append(p.value)
        q_hist.append(q.value)
        d_hist.append(np.abs(a @ p.value - b @ q.value))

        a = (a * p.value) @ T1 + (a * (1 - p.value)) @ T2
        b = (b * q.value) @ T1 + (b * (1 - q.value)) @ T2

    return IterationResult(tv_hist, a_hist, b_hist, d_hist, p_hist, q_hist, val_hist)


def run_dynamics_iterations_with_purturb(
    Y, T0, T1, ay, by, penalty, lamb, a_size=0.5, iters=200, threshold=None
):
    """Run the iterative dynamics on perturbed empirical populations.

    Unlike :func:`run_dynamics_iterations`, the populations are tracked as finite
    samples (state indices ``ay``/``by``). At each step the state indices
    are jittered, converted to distributions, and a policy is chosen as
    before; individuals are then advanced through the transition matrices.

    Parameters
    ----------
    Y : np.ndarray
        State values (used to size the policy vectors).
    T0, T1 : np.ndarray
        ``(D, D)`` transition matrices.
    ay, by : pd.Series
        Initial state indices for the two populations.
    penalty : callable
        Convex penalty.
    lamb : float
        Penalty weight.
    a_size : float
        Fraction of the population belonging to group A.
    iters : int
        Maximum number of iterations.
    threshold : float, optional
        Early-stopping tolerance for the TV distance.

    Returns
    -------
    IterationResult
        The recorded histories.
    """
    a_hist, b_hist, tv_hist = [], [], []
    p_hist, q_hist, d_hist, val_hist = [], [], [], []
    D = len(Y)
    index = np.arange(D)

    for i in tqdm(range(iters)):
        ap = perturb_qualifications(ay)
        bp = perturb_qualifications(by)
        a = get_dist(ap, index).to_numpy()
        b = get_dist(bp, index).to_numpy()

        a_hist.append(a)
        b_hist.append(b)
        tv_hist.append(np.sum(np.abs(a - b)))

        p = cp.Variable(len(Y))
        q = cp.Variable(len(Y))
        if lamb > 0:
            obj = cp.Maximize(
                a_size * cp.multiply(Y, a) @ p
                + (1 - a_size) * (cp.multiply(Y, b) @ q)
                - lamb * penalty(cp.abs(a @ p - b @ q))
            )
        else:
            obj = cp.Maximize(
                a_size * cp.multiply(Y, a) @ p + (1 - a_size) * (cp.multiply(Y, b) @ q)
            )
        prob = cp.Problem(obj, [p >= 0, p <= 1, q >= 0, q <= 1])
        prob.solve()
        val_hist.append(prob.value)
        p_hist.append(p.value)
        q_hist.append(q.value)
        d_hist.append(np.abs(a @ p.value - b @ q.value))

        da = decision(p.value, ap)
        db = decision(q.value, bp)

        ay = transform_q(da, ay, T0, T1, D)
        by = transform_q(db, by, T0, T1, D)

        if (
            len(tv_hist) > 1
            and threshold
            and abs(tv_hist[-1] - tv_hist[-2]) < threshold
        ):
            break

    return IterationResult(tv_hist, a_hist, b_hist, d_hist, p_hist, q_hist, val_hist)


def lambda_test(Y, a_init, b_init, a_size, penalty, lambs=None):
    """Sweep the penalty weight and return the resulting expected values.

    For every weight the (single-step) optimal policies are solved and the
    resulting expected values ``a p`` and ``b q`` are recorded.

    Parameters
    ----------
    Y : np.ndarray
        State values.
    a_init, b_init : np.ndarray
        Fixed probability distributions for the two groups.
    a_size : float
        Fraction of the population belonging to group A.
    penalty : callable
        Convex penalty.
    lambs : array-like, optional
        Penalty weights; defaults to ``np.linspace(0, 0.1, 100)``.

    Returns
    -------
    lambs : np.ndarray
        The penalty weights used.
    ps, qs : np.ndarray
        Expected values ``a p`` and ``b q`` for each weight.
    """
    a, b = a_init, b_init
    if lambs is None:
        lambs = np.linspace(0, 0.1, 100)
    ps, qs = [], []

    for lamb in tqdm(lambs):
        p = cp.Variable(len(Y))
        q = cp.Variable(len(Y))
        obj = cp.Maximize(
            a_size * cp.multiply(Y, a) @ p
            + (1 - a_size) * (cp.multiply(Y, b) @ q)
            - lamb * penalty(cp.abs(a @ p - b @ q))
        )
        prob = cp.Problem(obj, [p >= 0, p <= 1, q >= 0, q <= 1])
        prob.solve()
        ps.append(a @ p.value)
        qs.append(b @ q.value)

    return np.array(lambs), np.array(ps), np.array(qs)
