import numpy as np
from scipy.ndimage import map_coordinates
from sunbear.math.diff import grad, det_hess
from sunbear.forward import _get_full_potential, _get_idx
from sunbear.gradopt import Momentum

__all__ = ["inverse"]

def inverse(source, target, phi0=None, gradopt_obj=None, interp="linear",
            return_f=False):
    """
    Get the normalized deflection potential given the source and target density
    distribution.
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
    * `target` : numpy.ndarray
        The target density distribution in n-dimensional array. `source` and
        `target` must have the same shape.
    * `phi0` : numpy.ndarray
        The initial value of the deflection potential to start from. The default
        values are all zeros.
    * `gradopt_obj` : sunbear.gradopt.GradOptInterface obj, optional
        The solver object to solve the gradient descent problem. The default
        is sunbear.gradopt.Momentum.
    * `interp` : str
        Interpolation type to be put on scipy.interpolate.interpn. Supported
        are "linear", "nearest", "splinef2d" (only for 2D data).
    * `return_f`: bool
        If True, it will return the final loss function as the last argument.
        Otherwise, it just returns the mapping potential.

    Returns
    -------
    * numpy.ndarray
        The mapping potential given above. It will have the same shape as
        `source`.
    """
    # convert to numpy array
    source = np.asarray(source)
    target = np.asarray(target)
    # check the shapes
    if source.shape != target.shape:
        raise ValueError("The source and target must have the same shape.")

    # initialize phi0
    if phi0 is None:
        phi0 = np.zeros_like(source)
    ndim = np.ndim(source)
    u0, _, phi0_pad = _get_full_potential(phi0, -1)
    x = np.array([grad(u0, axis=i) for i in range(ndim)])
    pts = tuple([xx[_get_idx(ndim, i, slice(None,None,None), 0)] \
        for i,xx in enumerate(x)])

    # functions to get the loss function and the gradient
    idx_interior = tuple([slice(3,-3,None)]*ndim)
    idx_interior2 = tuple([slice(2,-2,None)]*ndim)
    log_source = np.log(np.abs(source))
    order = {"nearest": 0, "linear": 1, "quadratic": 2}[interp]

    def grad_phi_pad(phi_pad):
        u = u0 + phi_pad

        # calculate the determinant of the hessian
        det_hess_s = det_hess(u)

        # get the new position
        ypts = np.array([grad(u, axis=i) for i in range(ndim)]) # (D x n x n)

        # get the target density on the source plane
        target_s = map_coordinates(target, ypts, order=order, mode="constant")

        # calculate the dudt based on Sulman (2011)
        dudt_interior = log_source - np.log(np.abs(target_s * det_hess_s))
        dudt = np.zeros_like(u)
        # we don't want the gradient on the edge changed
        dudt[idx_interior] = dudt_interior[idx_interior2]

        # handle nan and inf values
        dudt[np.isnan(dudt)] = 0.0
        dudt[np.isinf(dudt)] = 0.0

        # error and the gradient
        f = np.mean(dudt*dudt)
        return f, dudt

    # set up the solver object and solve it
    opt = Momentum() if gradopt_obj is None else gradopt_obj
    phi_pad = opt.solve(grad_phi_pad, phi0_pad)
    idx_interior3 = tuple([slice(1,-1,None)]*ndim)

    if not return_f:
        return phi_pad[idx_interior3]
    else:
        f, _ = grad_phi_pad(phi_pad)
        return phi_pad[idx_interior3], f
