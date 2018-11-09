from functools import reduce
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from sunbear.math.diff import grad, det_hess

__all__ = ["forward", "forward_pos"]

def forward(source, phi, extrapval=0.0):
    """
    Obtain the target density distribution given the source distribution and
    the mapping potential, phi.
    The mapping from source coordinate, $x$, to the target coordinate, $y$, is
    given by:

    $$
    y = x + \nabla phi(x).
    $$

    The coordinate in i-th dimension is given by `np.arange(source.shape[i])`.

    Parameters
    ----------
    * `source` : numpy.ndarray
        The source density distribution in n-dimensional array.
    * `phi` : numpy.ndarray
        The mapping potential given above. It must have the same shape as
        `source`.
    * `extrapval` : float
        The extrapolation value (default: 0.0).

    Returns
    -------
    * numpy.ndarray
        The target density distribution in n-dimensional array.
    """
    # convert to np.ndarray
    source = np.asarray(source)
    phi = np.asarray(phi)
    # check the shapes of inputs
    if source.shape != phi.shape:
        raise ValueError("The source and phi must have the same shape.")

    # calculate the total potential so that $y = \nabla u(x)$
    u0, u, phi_pad = _get_full_potential(phi)
    ndim = np.ndim(phi)

    # calculate the determinant of the hessian
    det_hess_s = det_hess(u)

    # get the displacement in (n x D) format
    x = np.array([grad(u0, axis=i) for i in range(ndim)]).reshape((ndim,-1)).T
    y = np.array([grad(u , axis=i) for i in range(ndim)]).reshape((ndim,-1)).T

    # interpolate the values
    interp = lambda s: griddata(y, s.flatten(), x, "linear").reshape(s.shape)
    target_s = source / det_hess_s
    target = interp(target_s)

    # fill nan values with the extrapolation value
    target[np.isnan(target)] = extrapval
    return target

def forward_pos(pos, phi):
    """
    Returns the position in D-dimension on the target plane, given the mapping
    potential, `phi`.
    The mapping from source coordinate, $x$, to the target coordinate, $y$, is
    given by:

    $$
    y = x + \nabla phi(x).
    $$

    The coordinate in i-th dimension is given by `np.arange(phi.shape[i])`.

    Parameters
    ----------
    * `pos` : array_like
        List of points' positions in `N x D` array where `N` is the number of
        points and `D` is the number of dimensions. All the points must be
        inside the `phi`-defined coordinate (i.e. no extrapolation can be done).
    * `phi` : numpy.ndarray
        The mapping potential given above in `D`-dimensions array, starting from
        coordinate `(0,0,...,0)` and ends in coordinate `(n0-1,n1-1,...,nd1-1)`.

    Returns
    -------
    * array_like
        The position of the transformed points in the normalized coordinates
        in a form of 2D array with shape `N x D`.

    Notes
    -----
    * The displacement on the edge is assumed to be the same as the displacement
        of the interior pixel next to it.
    """
    # get the full potential, u, where $y = \nabla u(x)$ with padded coordinate
    # by 1 pixel in each side.
    u0, u, phi_pad = _get_full_potential(phi, -1)
    ndim = np.ndim(phi)

    # get the new position in (n0, n1, ..., nd1) format
    # y[i] gives the new position at that position in the i-th dimension
    y = np.array([grad(u, axis=i) for i in range(ndim)])
    shape = y[0].shape

    # interpolate the displacement
    points = tuple([np.arange(shape[i]) for i in range(ndim)])
    interps = [RegularGridInterpolator(points, d, "linear") for d in y]
    ypos = np.array([interp(pos) for interp in interps]).T
    return ypos

def _get_full_potential(phi, starts_from=0):
    shape = np.asarray(phi).shape
    ndim = len(shape)
    x_coords = _get_default_expanded_coordinate(np.array(shape)+2, ndim)
    u0 = 0.5 * reduce(lambda x,y: x+(y+starts_from)**2, x_coords, 0.0)
    phi_pad = _pad_conserve_grads(phi)
    u = u0 + phi_pad
    return u0, u, phi_pad

def _pad_conserve_grads(phi):
    # pad by conserving the edge gradients
    ndim = np.ndim(phi)
    pw = [1, 1]
    pp = np.pad(phi, [tuple(pw)]*ndim, mode="constant")
    for dim in range(ndim):
        # get the indices first
        idx_pad0_l = _get_idx(ndim, dim, slice(pw[0], pw[0]+1, None))
        idx_pad0_r = _get_idx(ndim, dim, slice(pw[0]+1, pw[0]+2, None))
        idx_pad0 = _get_idx(ndim, dim, slice(pw[0], pw[0]+1, None))
        idx_pad0_fill = _get_idx(ndim, dim, slice(None, pw[0], None))
        idx_pad1_l = _get_idx(ndim, dim, slice(-pw[1]-2, -pw[1]-1, None))
        idx_pad1_r = _get_idx(ndim, dim, slice(-pw[1]-1, -pw[1], None))
        idx_pad1 = _get_idx(ndim, dim, slice(-pw[1]-1, -pw[1], None))
        idx_pad1_fill = _get_idx(ndim, dim, slice(-pw[1], None, None))

        # now get the padded values
        grad0 = pp[idx_pad0_r] - pp[idx_pad0_l] # (n0, ..., ndim=1, ..., nd1)
        grad1 = pp[idx_pad1_r] - pp[idx_pad1_l]
        pad_arange0 = np.arange(-pw[0],0) # (1, ..., ndim=pw[i], ..., 1)
        pad_arange1 = np.arange(1,pw[0]+1)
        pad0 = pad_arange0 * grad0 + pp[idx_pad0] # (n0,...,ndim=pw[i],...,nd1)
        pad1 = pad_arange1 * grad1 + pp[idx_pad1]
        pp[idx_pad0_fill] = pad0
        pp[idx_pad1_fill] = pad1

    return pp

def _get_default_expanded_coordinate(shape, ndim):
    x_coords = []
    for i in range(ndim):
        idx = [None] * ndim
        idx[i] = slice(None, None, None)
        x_coords.append(np.arange(shape[i])[tuple(idx)])
    return x_coords

def _get_idx(ndim, dim, s, defidx=None):
    defidx = slice(None, None, None) if defidx is None else defidx
    idx = [defidx] * ndim
    idx[dim] = s
    return tuple(idx)
