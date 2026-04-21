import numpy as np
from numba import guvectorize, njit

from ..blocks.het_block import het
from .. import interpolate


def hh_init(b_grid, a_grid, z_grid, eis):
    Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    return Va, Vb


def adjustment_costs(a, a_grid, ra, chi0, chi1, chi2):
    chi = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)[0]
    return chi


def marginal_cost_grid(a_grid, ra, chi0, chi1, chi2):
    # precompute Psi1(a', a) on grid of (a', a) for steps 3 and 5
    Psi1 = get_Psi_and_deriv(a_grid[:, np.newaxis],
                             a_grid[np.newaxis, :], ra, chi0, chi1, chi2)[1]
    return Psi1


# policy and bacward order as in grid!
@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)  
def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1, chi2, Psi1):
    # === STEP 2: Wb(z, b', a') and Wa(z, b', a') ===
    # (take discounted expectation of tomorrow's value function)
    Wb = beta * Vb_p
    Wa = beta * Va_p
    W_ratio = Wa / Wb

    # === STEP 3: a'(z, b', a) for UNCONSTRAINED ===

    # for each (z, b', a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio == 1+Psi1
    i, pi = lhs_equals_rhs_interpolate(W_ratio, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_unc = interpolate.apply_coord(i, pi, a_grid)
    c_endo_unc = interpolate.apply_coord(i, pi, Wb) ** (-eis)

    # === STEP 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED ===

    # solve out budget constraint to get b(z, b', a)
    b_endo = (c_endo_unc + a_endo_unc + addouter(-z_grid, b_grid, -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this b' -> b mapping to get b -> b', so we have b'(z, b, a)
    # and also use interpolation to get a'(z, b, a)
    # (note utils.interpolate.interpolate_coord and utils.interpolate.apply_coord work on last axis,
    #  so we need to swap 'b' to the last axis, then back when done)
    i, pi = interpolate.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    a_unc = interpolate.apply_coord(i, pi, a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
    b_unc = interpolate.apply_coord(i, pi, b_grid).swapaxes(1, 2)

    # === STEP 5: a'(z, kappa, a) for CONSTRAINED ===

    # for each (z, kappa, a), linearly interpolate to find a' between gridpoints
    # satisfying optimality condition W_ratio/(1+kappa) == 1+Psi1, assuming b'=0
    lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
    i, pi = lhs_equals_rhs_interpolate(lhs_con, 1 + Psi1)

    # use same interpolation to get Wb and then c
    a_endo_con = interpolate.apply_coord(i, pi, a_grid)
    c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                  * interpolate.apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))

    # === STEP 6: a'(z, b, a) for CONSTRAINED ===

    # solve out budget constraint to get b(z, kappa, a), enforcing b'=0
    b_endo = (c_endo_con + a_endo_con
              + addouter(-z_grid, np.full(len(k_grid), b_grid[0]), -(1 + ra) * a_grid)
              + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)

    # interpolate this kappa -> b mapping to get b -> kappa
    # then use the interpolated kappa to get a', so we have a'(z, b, a)
    # (utils.interpolate.interpolate_y does this in one swoop, but since it works on last
    #  axis, we need to swap kappa to last axis, and then b back to middle when done)
    a_con = interpolate.interpolate_y(b_endo.swapaxes(1, 2), b_grid,
                                      a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)

    # === STEP 7: obtain policy functions and update derivatives of value function ===

    # combine unconstrained solution and constrained solution, choosing latter
    # when unconstrained goes below minimum b
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]

    # calculate adjustment cost and its derivative
    Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)

    # solve out budget constraint to get consumption and marginal utility
    c = addouter(z_grid, (1 + rb) * b_grid, (1 + ra) * a_grid) - Psi - a - b
    uc = c ** (-1 / eis)
    uce = e_grid[:, np.newaxis, np.newaxis] * uc

    # update derivatives of value function using envelope conditions
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    return Va, Vb, a, b, c, uce


'''Supporting functions for HA block'''

@njit(cache=True, fastmath=False)
def _psi_loop(ap_flat, a_flat, ra, chi0, chi1, chi2):
    n = ap_flat.shape[0]
    Psi = np.empty(n)
    Psi1 = np.empty(n)
    Psi2 = np.empty(n)
    ra1 = 1.0 + ra
    chi_exp = chi2 - 1.0
    chi1_over_chi2 = chi1 / chi2
    for i in range(n):
        a_wr = ra1 * a_flat[i]
        dx = ap_flat[i] - a_wr
        if dx >= 0.0:
            adx = dx
            sg = 1.0 if dx > 0.0 else 0.0
        else:
            adx = -dx
            sg = -1.0
        denom = a_wr + chi0
        cf = (adx / denom) ** chi_exp
        p = chi1_over_chi2 * adx * cf
        p1 = chi1 * sg * cf
        p2 = -ra1 * (p1 + chi_exp * p / denom)
        Psi[i] = p
        Psi1[i] = p1
        Psi2[i] = p2
    return Psi, Psi1, Psi2


@njit(cache=True, fastmath=False)
def _psi_loop_last_dim(ap, a_last, ra, chi0, chi1, chi2):
    """Specialized path: ap has shape (..., N), a_last has shape (N,)"""
    ap_flat = ap.reshape(-1, a_last.shape[0])
    outer, n = ap_flat.shape
    Psi = np.empty_like(ap_flat)
    Psi1 = np.empty_like(ap_flat)
    Psi2 = np.empty_like(ap_flat)
    ra1 = 1.0 + ra
    chi_exp = chi2 - 1.0
    chi1_over_chi2 = chi1 / chi2
    for j in range(n):
        a_wr = ra1 * a_last[j]
        denom = a_wr + chi0
        for k in range(outer):
            dx = ap_flat[k, j] - a_wr
            if dx >= 0.0:
                adx = dx
                sg = 1.0 if dx > 0.0 else 0.0
            else:
                adx = -dx
                sg = -1.0
            cf = (adx / denom) ** chi_exp
            p = chi1_over_chi2 * adx * cf
            p1 = chi1 * sg * cf
            Psi[k, j] = p
            Psi1[k, j] = p1
            Psi2[k, j] = -ra1 * (p1 + chi_exp * p / denom)
    return Psi, Psi1, Psi2


@njit(cache=True, fastmath=False)
def _psi_loop_outer(ap_col, a_row, ra, chi0, chi1, chi2):
    """Specialized path: ap_col has shape (N, 1), a_row has shape (1, M)"""
    n = ap_col.shape[0]
    m = a_row.shape[1]
    Psi = np.empty((n, m))
    Psi1 = np.empty((n, m))
    Psi2 = np.empty((n, m))
    ra1 = 1.0 + ra
    chi_exp = chi2 - 1.0
    chi1_over_chi2 = chi1 / chi2
    for j in range(m):
        a_wr = ra1 * a_row[0, j]
        denom = a_wr + chi0
        for i in range(n):
            dx = ap_col[i, 0] - a_wr
            if dx >= 0.0:
                adx = dx
                sg = 1.0 if dx > 0.0 else 0.0
            else:
                adx = -dx
                sg = -1.0
            cf = (adx / denom) ** chi_exp
            p = chi1_over_chi2 * adx * cf
            p1 = chi1 * sg * cf
            Psi[i, j] = p
            Psi1[i, j] = p1
            Psi2[i, j] = -ra1 * (p1 + chi_exp * p / denom)
    return Psi, Psi1, Psi2


def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """Adjustment cost Psi(ap, a) and its derivatives with respect to
    first argument (ap) and second argument (a)"""
    # Fast path 1: a is 1D with length == ap.shape[-1]
    if a.ndim == 1 and ap.ndim >= 1 and ap.shape[-1] == a.shape[0]:
        shape = ap.shape
        ap_c = np.ascontiguousarray(ap)
        Psi, Psi1, Psi2 = _psi_loop_last_dim(ap_c, a, ra, chi0, chi1, chi2)
        return Psi.reshape(shape), Psi1.reshape(shape), Psi2.reshape(shape)
    # Fast path 2: a_grid[:, None], a_grid[None, :] pattern -> outer product
    if (ap.ndim == 2 and a.ndim == 2 and ap.shape[1] == 1 and a.shape[0] == 1):
        return _psi_loop_outer(np.ascontiguousarray(ap), np.ascontiguousarray(a),
                               ra, chi0, chi1, chi2)
    # General fallback
    ap_b, a_b = np.broadcast_arrays(ap, a)
    shape = ap_b.shape
    ap_flat = np.ascontiguousarray(ap_b).reshape(-1)
    a_flat = np.ascontiguousarray(a_b).reshape(-1)
    Psi, Psi1, Psi2 = _psi_loop(ap_flat, a_flat, ra, chi0, chi1, chi2)
    return Psi.reshape(shape), Psi1.reshape(shape), Psi2.reshape(shape)


def matrix_times_first_dim(A, X):
    """Take matrix A times vector X[:, i1, i2, i3, ... , in] separately
    for each i1, i2, i3, ..., in. Same output as A @ X if X is 1D or 2D"""
    # flatten all dimensions of X except first, then multiply, then restore shape
    return (A @ X.reshape(X.shape[0], -1)).reshape(X.shape)


def addouter(z, b, a):
    """Take outer sum of three arguments: result[i, j, k] = z[i] + b[j] + a[k]"""
    return z[:, np.newaxis, np.newaxis] + b[:, np.newaxis] + a


@guvectorize(['void(float64[:], float64[:,:], uint32[:], float64[:])'], '(ni),(ni,nj)->(nj),(nj)')
def lhs_equals_rhs_interpolate(lhs, rhs, iout, piout):
    """
    Given lhs (i) and rhs (i,j), for each j, find the i such that

    lhs[i] > rhs[i,j] and lhs[i+1] < rhs[i+1,j]

    i.e. where given j, lhs == rhs in between i and i+1.

    Also return the pi such that

    pi*(lhs[i] - rhs[i,j]) + (1-pi)*(lhs[i+1] - rhs[i+1,j]) == 0

    i.e. such that the point at pi*i + (1-pi)*(i+1) satisfies lhs == rhs by linear interpolation.

    If lhs[0] < rhs[0,j] already, just return u=0 and pi=1.

    ***IMPORTANT: Assumes that solution i is monotonically increasing in j
    and that lhs - rhs is monotonically decreasing in i.***
    """

    ni, nj = rhs.shape
    assert len(lhs) == ni

    i = 0
    for j in range(nj):
        while True:
            if lhs[i] < rhs[i, j]:
                break
            elif i < nj - 1:
                i += 1
            else:
                break

        if i == 0:
            iout[j] = 0
            piout[j] = 1
        else:
            iout[j] = i - 1
            err_upper = rhs[i, j] - lhs[i]
            err_lower = rhs[i - 1, j] - lhs[i - 1]
            piout[j] = err_upper / (err_upper - err_lower)
            