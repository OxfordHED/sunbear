import numpy as np

def det(x):
    """
    Calculate the determinant of ndarray `x`. This is similar to np.linalg.det,
    but the square dimensions are at the beginning instead of at the end.

    Parameters
    ----------
    * `x` : array_like, shape (ndim, ndim, ...)
        An ndarray which determinant to be calculated

    Returns
    -------
    * array_like or float
        The determinant of the `x`, arranged with shape: x.shape[2:]
    """
    # shape checks
    x = np.asarray(x)
    if (len(x.shape) < 2):
        raise ValueError("x must have at least 2 dimensions")
    if x.shape[0] != x.shape[1]:
        raise ValueError("The first two dimensions must equal")

    nd = x.shape[0]
    ndim = len(x.shape)
    if nd == 1:
        return _det_1d(x, ndim)
    elif nd == 2:
        return _det_2d(x, ndim)
    elif nd == 3:
        return _det_3d(x, ndim)
    else:
        return _det_nd(x, nd, ndim)

def _det_1d(x, ndim):
    idx = tuple([0,0] + ([slice(None, None, None)] * (ndim-2)))
    return x[idx]

def _det_2d(x, ndim):
    rest_idx = [slice(None, None, None)] * (ndim-2)
    i00 = tuple([0,0] + rest_idx)
    i01 = tuple([0,1] + rest_idx)
    i10 = tuple([1,0] + rest_idx)
    i11 = tuple([1,1] + rest_idx)
    return x[i00] * x[i11] - x[i01] * x[i10]

def _det_3d(x, ndim):
    rest_idx = [slice(None, None, None)] * (ndim-2)
    i00 = tuple([0,0] + rest_idx)
    i01 = tuple([0,1] + rest_idx)
    i02 = tuple([0,2] + rest_idx)
    i10 = tuple([1,0] + rest_idx)
    i11 = tuple([1,1] + rest_idx)
    i12 = tuple([1,2] + rest_idx)
    i20 = tuple([2,0] + rest_idx)
    i21 = tuple([2,1] + rest_idx)
    i22 = tuple([2,2] + rest_idx)
    return \
        x[i00]*x[i11]*x[i22] + x[i01]*x[i12]*x[i20] + x[i02]*x[i10]*x[i21] -\
        (x[i02]*x[i11]*x[i20] + x[i00]*x[i12]*x[i21] + x[i22]*x[i01]*x[i10])

def _det_nd(x, nd, ndim):
    # rearrange matrix to have shape: outshape + [ndim, ndim]
    perm_idx = list(range(2,ndim)) + list(range(2))
    y = np.transpose(x, perm_idx)
    # calculate and return the determinant
    return np.linalg.det(y)
