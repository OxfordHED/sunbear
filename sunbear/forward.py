from functools import reduce
import numpy as np
from scipy.interpolate import griddata
from sunbear.calculus.diff import grad, det_hess

__all__ = ["forward"]

def forward(source, phi):
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
    shape = phi.shape
    ndim = len(shape)
    x_coords = _get_default_expanded_coordinate(np.array(shape)+2, ndim)
    u0 = 0.5 * reduce(lambda x,y: x+y*y, x_coords, 0.0)
    phi_pad = np.pad(phi, [(1,1)]*ndim, mode="constant")
    u = u0 + phi_pad

    # calculate the determinant of the hessian
    det_hess_s = det_hess(u)

    # get the displacement in (n x D) format
    x = np.array([grad(u0, axis=i) for i in range(ndim)]).reshape((ndim,-1)).T
    y = np.array([grad(u , axis=i) for i in range(ndim)]).reshape((ndim,-1)).T

    # interpolate the values
    interp = lambda s: griddata(y, s.flatten(), x, "linear").reshape(s.shape)
    target_s = source / det_hess_s
    target = interp(target_s)

    # fill nan values with zeros
    target[np.isnan(target)] = 0.0
    return target

def _get_default_expanded_coordinate(shape, ndim):
    x_coords = []
    for i in range(ndim):
        idx = [None] * ndim
        idx[i] = slice(None, None, None)
        x_coords.append(np.arange(shape[i])[tuple(idx)])
    return x_coords
