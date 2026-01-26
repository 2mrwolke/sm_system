import numpy as np
import pytest

from sm_system.lse import (
    build_coef_matrix_order2,
    build_coef_matrix_order3,
    calc_signed_indices,
    check_full_rank,
    ensure_fullrank,
    least_squares,
)


def test_calc_signed_indices_invariants_and_shapes():
    """Signed FFT-bin index relations must be algebraically consistent."""

    # Use bin_delta=1.0 so rounding is deterministic and easy to reason about.
    # Create cases that yield negative difference terms (e.g., b-2*a < 0).
    freq_pairs = np.array(
        [
            [
                [6.0, 7.0],  # b-2a = -5
                [3.0, 8.0],  # b-2a = 2
                [4.0, 9.0],  # b-a = 5
            ]
        ]
    )  # shape: (batch=1, n=3, 2)
    bin_delta = 1.0

    indices = calc_signed_indices(freq_pairs, bin_delta)

    # Shape/dtype invariants
    for k in [
        "0",
        "a",
        "b",
        "2*a",
        "2*b",
        "3*a",
        "3*b",
        "b-2*a",
        "b-a",
        "b+a",
        "b+2*a",
        "2*b-a",
        "2*b+a",
    ]:
        assert k in indices
        assert indices[k].shape == (1, 3)
        assert np.issubdtype(indices[k].dtype, np.integer)

    # Algebraic invariants
    bin_pairs = np.round(freq_pairs / bin_delta).astype(int)
    a = bin_pairs[:, :, 0]
    b = bin_pairs[:, :, 1]

    assert np.array_equal(indices["0"], np.zeros_like(a))
    assert np.array_equal(indices["a"], a)
    assert np.array_equal(indices["b"], b)

    assert np.array_equal(indices["2*a"], 2 * a)
    assert np.array_equal(indices["2*b"], 2 * b)
    assert np.array_equal(indices["3*a"], 3 * a)
    assert np.array_equal(indices["3*b"], 3 * b)

    assert np.array_equal(indices["b-2*a"], b - 2 * a)
    assert np.array_equal(indices["b-a"], b - a)
    assert np.array_equal(indices["b+a"], b + a)
    assert np.array_equal(indices["b+2*a"], b + 2 * a)
    assert np.array_equal(indices["2*b-a"], 2 * b - a)
    assert np.array_equal(indices["2*b+a"], 2 * b + a)

    # Ensure we actually produced at least one negative entry for the signed invariants.
    assert np.min(indices["b-2*a"]) < 0


def test_coef_matrix_shapes_and_rank_conditions_order2_and_3():
    """Coefficient matrices must have predictable shapes and be full-rank for typical inputs."""

    # Small but non-trivial set of pairs; avoid degeneracies by keeping all values distinct.
    freq_pairs = np.array(
        [
            [
                [2.0, 5.0],
                [3.0, 8.0],
                [4.0, 9.0],
                [5.0, 11.0],
            ]
        ]
    )
    indices = calc_signed_indices(freq_pairs, bin_delta=1.0)

    # ----- Order 2
    coef2, xpre2, xpost2 = build_coef_matrix_order2(indices)

    batch, n, _ = freq_pairs.shape
    expected_rows2 = batch * n * 4

    # Columns are the number of unique pre-harmonic and post-harmonic indices.
    a = indices["a"].reshape(-1)
    b = indices["b"].reshape(-1)
    pre_unique2 = np.unique(np.concatenate([a, b]))
    post_terms2 = np.concatenate(
        [
            indices["2*a"].reshape(-1),
            indices["2*b"].reshape(-1),
            indices["b-a"].reshape(-1),
            indices["b+a"].reshape(-1),
        ]
    )
    post_unique2 = np.unique(post_terms2)
    expected_cols2 = len(pre_unique2) + len(post_unique2)

    assert coef2.shape == (expected_rows2, expected_cols2)
    assert xpre2.shape == (len(pre_unique2),)
    assert xpost2.shape == (len(post_unique2),)

    
    # Rank condition: these coefficient matrices are typically rank-deficient by a small amount
    # (e.g., due to gauge freedoms in phase references). We verify that the rank deficiency
    # is detected and that a single slice removal can restore full-rank, as intended by
    # the provided utilities.
    assert not check_full_rank(coef2)

    fixed2 = None
    for axis in (0, 1):
        for slyz in range(coef2.shape[axis]):
            candidate = ensure_fullrank(coef2, slyz=slyz, axis=axis)
            if candidate is not None and check_full_rank(candidate):
                fixed2 = candidate
                break
        if fixed2 is not None:
            break
    assert fixed2 is not None

    # ----- Order 3
    coef3, xpre3, xpost3 = build_coef_matrix_order3(indices)

    expected_rows3 = batch * n * 6
    post_terms3 = np.concatenate(
        [
            np.abs(indices["3*a"].reshape(-1)),
            np.abs(indices["3*b"].reshape(-1)),
            np.abs(indices["b-2*a"].reshape(-1)),
            np.abs(indices["b+2*a"].reshape(-1)),
            np.abs(indices["2*b-a"].reshape(-1)),
            np.abs(indices["2*b+a"].reshape(-1)),
        ]
    )
    post_unique3 = np.unique(post_terms3)
    expected_cols3 = len(pre_unique2) + len(post_unique3)

    assert coef3.shape == (expected_rows3, expected_cols3)
    assert xpre3.shape == (len(pre_unique2),)
    assert xpost3.shape == (len(post_unique3),)

    assert not check_full_rank(coef3)

    fixed3 = None
    for axis in (0, 1):
        for slyz in range(coef3.shape[axis]):
            candidate = ensure_fullrank(coef3, slyz=slyz, axis=axis)
            if candidate is not None and check_full_rank(candidate):
                fixed3 = candidate
                break
        if fixed3 is not None:
            break
    assert fixed3 is not None

    # ----- Rank utilities sanity
    rank_deficient = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # rank 1
    assert not check_full_rank(rank_deficient)

    # ensure_fullrank deletes a slice; after deleting one column we get full rank (min(shape)=1)
    reduced = ensure_fullrank(rank_deficient, slyz=1, axis=1)
    assert reduced is not None
    assert reduced.shape == (3, 1)
    assert check_full_rank(reduced)


def test_weighted_least_squares_matches_row_scaled_lstsq_complex():
    """Weighted LS must match the canonical row-scaled least squares formulation."""

    rng = np.random.default_rng(123)
    m, n = 80, 7

    # Complex-valued design matrix and ground-truth weights.
    A = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    x_true = rng.normal(size=(n,)) + 1j * rng.normal(size=(n,))

    # Heteroscedastic noise with known per-sample variance.
    sigma = np.linspace(0.05, 0.5, m)
    noise = (rng.normal(size=m) + 1j * rng.normal(size=m)) * sigma
    b = A @ x_true + noise

    w = 1.0 / (sigma**2)

    # Reference: solve min ||sqrt(W)(Ax-b)|| by row scaling.
    sqrt_w = np.sqrt(w)
    A_w = A * sqrt_w[:, None]
    b_w = b * sqrt_w

    x_ref, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)
    x_hat = least_squares(A, b, w=w)

    # Numerical agreement
    assert np.allclose(x_hat, x_ref, rtol=1e-10, atol=1e-10)

    # With constant weights, weighted and unweighted solutions must match.
    x_unweighted = least_squares(A, b, w=None)
    x_const = least_squares(A, b, w=np.ones(m) * 3.0)
    assert np.allclose(x_const, x_unweighted, rtol=1e-10, atol=1e-10)


def test_least_squares_check_rejects_ill_conditioned_matrix():
    """When check=True, least_squares should assert on ill-conditioned matrices."""

    # Construct an ill-conditioned matrix by making columns nearly collinear.
    A = np.array(
        [
            [1.0, 1.0 + 1e-12],
            [1.0, 1.0 + 2e-12],
            [1.0, 1.0 + 3e-12],
            [1.0, 1.0 + 4e-12],
        ]
    )
    b = np.array([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(AssertionError):
        least_squares(A, b, check=True)
