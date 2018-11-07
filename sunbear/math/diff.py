import numpy as np
from sunbear.math._utils import get_idx, midx, lidx, hidx
from sunbear.math.det import det

def grad(u, axis):
    """
    Obtain the first order gradient of `u` using central difference scheme along
    the axis `axis`. The `ds` for each direction is 1.0.

    Parameters
    ----------
    * `u` : numpy.ndarray
        The ndarray input with shape (n0+2, n1+2, ..., nd1+2).
    * `axis` : int
        The axis where the gradient is calculated along.

    Returns
    -------
    * numpy.ndarray
        The ndarray of the gradient of `u` with shape (n0, n1, ..., nd1).
    """
    ndim = np.ndim(u)

    idx_l = get_idx(ndim, axis, slice(None,-2,None))
    idx_r = get_idx(ndim, axis, slice(2,None,None))
    return 0.5 * (u[idx_r] - u[idx_l])

def grad2(u, axes):
    """
    Return the second grad of the given matrix, `u`.

    Parameters
    ----------
    * `u` : numpy.ndarray
        The ndarray input with shape (n0+2, n1+2, ..., nd1+2).
    * `axes` : sequence, int
        The axes where the gradient is calculated along.

    Returns
    -------
    * numpy.ndarray
        The ndarray of the second grad of `u` with shape (n0, n1, ..., nd1).
    """
    ndim = np.ndim(u)
    if axes[0] == axes[1]:
        axis = axes[0]
        idx_l = get_idx(ndim, axis, lidx) # left
        idx_r = get_idx(ndim, axis, hidx) # right
        idx_m = get_idx(ndim) # middle
        return (u[idx_l] + u[idx_r] - 2*u[idx_m])
    else:
        idx_lr = get_idx(ndim, axes, [hidx, hidx]) # lower right
        idx_ul = get_idx(ndim, axes, [lidx, lidx]) # upper left
        idx_ur = get_idx(ndim, axes, [lidx, hidx]) # upper right
        idx_ll = get_idx(ndim, axes, [hidx, lidx]) # lower left
        return (u[idx_lr] + u[idx_ul] - u[idx_ur] - u[idx_ll]) * 0.25

def det_hess(u):
    """
    Get the determinant of the Hessian matrix of `u`.

    Parameters
    ----------
    * `u` : numpy.ndarray
        The ndarray input with shape (n0+2, n1+2, ..., nd1+2).

    Returns
    -------
    * numpy.ndarray
        The ndarray of the second grad of `u` with shape (n0, n1, ..., nd1).
    """
    ndim = np.ndim(u)
    inshape = np.asarray(u.shape)
    outshape = list(inshape - 2)

    # obtain the second gradient per each pairs of axes
    hess_unarranged = np.zeros([ndim, ndim] + outshape)
    for i in range(ndim):
        for j in range(i,ndim):
            grad2_val = grad2(u, (i,j))
            hess_unarranged[i,j] = grad2_val
            hess_unarranged[j,i] = grad2_val

    return det(hess_unarranged)
