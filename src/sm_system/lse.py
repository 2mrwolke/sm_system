import numpy as np


def least_squares(A, b, w=None, check=False):
    """
    Solve the least squares problem A * x = b (optional: with weights w)

    DESIGN CHOICE: This implementation prioritizes "derivation traceability" over speed.

    Parameters:
    A (numpy.ndarray): The design matrix of shape (m, n)
    b (numpy.ndarray): The target vector of shape (m,)
    w (numpy.ndarray): The weight vector of shape (m,)
                       (reciprocal of sample variance)
    check (bool): if True, function breaks if A is ill conditioned

    Returns:
    x (numpy.ndarray): The solution vector of shape (n,)
    """

    if check:
        assert np.linalg.cond(A) <= 10

    # without weights
    if w is None:
        return np.linalg.pinv(A) @ b

    # with weights
    else:
        # Create the weight matrix W, which is a diagonal matrix of weights
        W = np.diag(w)

        # Enable complex-valued LSEs
        A_H = np.conj(A.T)

        # Compute the pseudo-inverse of (A.T @ W @ A)
        A_T_W_A = A_H @ W @ A
        A_T_W_A_pseudo_inv = np.linalg.pinv(A_T_W_A)

        # Compute the solution x using the formula
        x = A_T_W_A_pseudo_inv @ (A_H @ W @ b)

        return x


def build_coef_matrix_order2(indices):
    idx_pairs = np.stack([indices["a"], indices["b"]], axis=-1)
    idx_pre = idx_pairs
    idx_pairs = idx_pairs.reshape(-1, 2)

    idx_map_pre = np.unique(idx_pairs)
    idx_map_pre = {val: i for i, val in enumerate(idx_map_pre)}

    idx_post = np.stack(
        [indices["2*a"], indices["2*b"], indices["b-a"], indices["b+a"]],
        axis=-1,
    )
    idx_map_post = np.unique(idx_post)
    idx_map_post = {
        val: i + len(idx_map_pre) for i, val in enumerate(idx_map_post)
    }

    idx_pre = np.vectorize(idx_map_pre.get)(idx_pre.reshape((-1)))
    idx_pre = idx_pre.reshape(-1, 2)
    idx_post = np.vectorize(idx_map_post.get)(idx_post.reshape((-1)))
    idx_post = idx_post.reshape(-1, 4)

    zeros = np.zeros((len(idx_pre) * 4, len(idx_map_pre) + len(idx_map_post)))
    zeros = zeros.reshape((-1, 4, len(idx_map_pre) + len(idx_map_post)))

    rnge = np.arange(len(idx_pre))

    zeros[rnge, 0, idx_pre[:, 0]] = 2
    zeros[rnge, 0, idx_post[:, 0]] = 1

    zeros[rnge, 1, idx_pre[:, 1]] = 2
    zeros[rnge, 1, idx_post[:, 1]] = 1

    zeros[rnge, 2, idx_pre[:, 0]] = -1
    zeros[rnge, 2, idx_pre[:, 1]] = 1
    zeros[rnge, 2, idx_post[:, 2]] = 1

    zeros[rnge, 3, idx_pre[:, 0]] = 1
    zeros[rnge, 3, idx_pre[:, 1]] = 1
    zeros[rnge, 3, idx_post[:, 3]] = 1

    coef_matrix = np.reshape(
        zeros, (len(idx_pre) * 4, len(idx_map_pre) + len(idx_map_post))
    )

    x_idx_pre = np.array(list(idx_map_pre.keys()))
    x_idx_post = np.array(list(idx_map_post.keys()))

    return coef_matrix, x_idx_pre, x_idx_post


def build_coef_matrix_order3(indices):
    idx_pairs = np.stack([indices["a"], indices["b"]], axis=-1)
    idx_pre = idx_pairs
    idx_pairs = idx_pairs.reshape(-1, 2)

    idx_map_pre = np.unique(idx_pairs)
    idx_map_pre = {val: i for i, val in enumerate(idx_map_pre)}

    idx_post = np.stack(
        [
            indices["3*a"],
            indices["3*b"],
            indices["b-2*a"],
            indices["b+2*a"],
            indices["2*b-a"],
            indices["2*b+a"],
        ],
        axis=-1,
    )

    sign = np.sign(idx_post)
    idx_post = np.abs(idx_post)

    idx_map_post = np.unique(idx_post)
    idx_map_post = {
        val: i + len(idx_map_pre) for i, val in enumerate(idx_map_post)
    }

    idx_pre = np.vectorize(idx_map_pre.get)(idx_pre.reshape((-1)))
    idx_pre = idx_pre.reshape(-1, 2)
    idx_post = np.vectorize(idx_map_post.get)(idx_post.reshape((-1)))
    idx_post = idx_post.reshape(-1, 6)

    zeros = np.zeros((len(idx_pre) * 6, len(idx_map_pre) + len(idx_map_post)))
    zeros = zeros.reshape((-1, 6, len(idx_map_pre) + len(idx_map_post)))

    rnge = np.arange(len(idx_pre))

    zeros[rnge, 0, idx_pre[:, 0]] = 3
    zeros[rnge, 0, idx_post[:, 0]] = 1

    zeros[rnge, 1, idx_pre[:, 1]] = 3
    zeros[rnge, 1, idx_post[:, 1]] = 1

    zeros[rnge, 2, idx_pre[:, 0]] = -2
    zeros[rnge, 2, idx_pre[:, 1]] = 1
    zeros[rnge, 2, idx_post[:, 2]] = 1 * sign[:, :, 2].reshape((-1))

    zeros[rnge, 3, idx_pre[:, 0]] = 2
    zeros[rnge, 3, idx_pre[:, 1]] = 1
    zeros[rnge, 3, idx_post[:, 3]] = 1

    zeros[rnge, 4, idx_pre[:, 0]] = -1
    zeros[rnge, 4, idx_pre[:, 1]] = 2
    zeros[rnge, 4, idx_post[:, 4]] = 1

    zeros[rnge, 5, idx_pre[:, 0]] = 1
    zeros[rnge, 5, idx_pre[:, 1]] = 2
    zeros[rnge, 5, idx_post[:, 5]] = 1

    coef_matrix = np.reshape(
        zeros, (len(idx_pre) * 6, len(idx_map_pre) + len(idx_map_post))
    )

    x_idx_pre = np.array(list(idx_map_pre.keys()))
    x_idx_post = np.array(list(idx_map_post.keys()))

    return coef_matrix, x_idx_pre, x_idx_post


def clc_lhs_order_2(levels, phase, indices):
    indices_lhs = np.stack(
        [indices["2*a"], indices["2*b"], indices["b-a"], indices["b+a"]],
        axis=-1,
    )

    b, s = levels.shape[:2]
    idx = indices_lhs.reshape((-1, 4))
    idx = np.abs(idx)

    lvl = levels.reshape((b * s, -1))
    lhs_lvl = np.take_along_axis(lvl, idx, axis=1)
    lhs_lvl = lhs_lvl.reshape((b, -1, 4))
    lhs_offset = 20 * np.log10(np.array([1 / 2, 1 / 2, 1, 1]))
    lhs_offset = lhs_offset.reshape(1, 1, 4)
    lhs_lvl -= lhs_offset

    phase = phase.reshape((b * s, -1))
    phase = np.take_along_axis(phase, idx, axis=1)
    phase = phase.reshape((b, -1, 4))

    return lhs_lvl, phase, idx


def clc_lhs_order_3(levels, phase, indices):
    indices_lhs = np.stack(
        [
            indices["3*a"],
            indices["3*b"],
            indices["b-2*a"],
            indices["b+2*a"],
            indices["2*b-a"],
            indices["2*b+a"],
        ],
        axis=-1,
    )

    b, s = levels.shape[:2]
    lvl = levels.reshape((b * s, -1))
    idx = indices_lhs.reshape((-1, 6))
    sign = np.sign(idx)
    idx = np.abs(idx)
    lhs_lvl = np.take_along_axis(lvl, idx, axis=1)
    lhs_lvl = lhs_lvl.reshape((b, -1, 6))

    lhs_offset = 20 * np.log10(
        np.array([1 / 4, 1 / 4, 3 / 4, 3 / 4, 3 / 4, 3 / 4])
    )
    lhs_offset = lhs_offset.reshape(1, 1, 6)

    lhs_lvl -= lhs_offset

    phase = phase.reshape((b * s, -1))
    phase = np.take_along_axis(phase, idx, axis=1)
    phase = phase.reshape((b, -1, 6))
    phase = phase * sign.reshape(b, -1, 6)

    return lhs_lvl, phase, idx


def check_full_rank(matrix):
    return np.min(np.min(matrix.shape)) == np.linalg.matrix_rank(matrix)


def ensure_fullrank(jakobi, slyz=0, axis=1):
    jakobi = np.delete(jakobi, slyz, axis)
    if check_full_rank(jakobi):
        print("Jakobi-matrix has full rank!")
        return jakobi
    else:
        print("WARNING: Jakobi-matrix has NOT full rank!")


def calc_signed_indices(freq_pairs, bin_delta):
    """
    Calculates indices of fundamentals and harmonics (up to order of 3)
    of fft-bins for real-valued signals. The Indices are signed in order
    to utilize them for phase identification.

    Parameters:
    freq_pairs (numpy.array): Frequency pairs of shape:  [batch_size, n, 2]
    fft_delta (float): Frequency distance of fft-bins
    """

    bin_pairs = np.round(freq_pairs / bin_delta)
    a, b = bin_pairs[:, :, 0], bin_pairs[:, :, 1]

    indices = [
        a * 0,
        a,
        b,
        2 * a,
        2 * b,
        3 * a,
        3 * b,
        b - 2 * a,
        b - a,
        b + a,
        b + 2 * a,
        2 * b - a,
        2 * b + a,
    ]
    indices = np.stack(indices, axis=-1)
    indices = np.round(indices)
    indices = indices.astype(int)
    indices = dict(
        {
            "0": indices[:, :, 0],
            "a": indices[:, :, 1],
            "b": indices[:, :, 2],
            "2*a": indices[:, :, 3],
            "2*b": indices[:, :, 4],
            "3*a": indices[:, :, 5],
            "3*b": indices[:, :, 6],
            "b-2*a": indices[:, :, 7],
            "b-a": indices[:, :, 8],
            "b+a": indices[:, :, 9],
            "b+2*a": indices[:, :, 10],
            "2*b-a": indices[:, :, 11],
            "2*b+a": indices[:, :, 12],
        }
    )

    return indices


def extract_levels(signals, norm=1):
    # for real signals
    n = signals.shape[-1]
    if n == 0:
        print("WARNING n = 0 !!!!!!!!!")
    fft = np.fft.rfft(signals)
    abs = np.abs(fft) / norm
    with np.errstate(divide="ignore"):
        mag = 20 * np.log10(2 * abs / n)
    return mag
