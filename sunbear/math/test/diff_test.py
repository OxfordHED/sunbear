import numpy as np
from sunbear.math.diff import grad, grad2, det_hess
from functools import reduce
import pytest

##################################### grad #####################################

def test_grad_1d():
    _test_grad_nd(n=92, ndim=1)

def test_grad_2d():
    _test_grad_nd(n=52, ndim=2)

def test_grad_3d():
    _test_grad_nd(n=12, ndim=3)

def test_grad_4d():
    _test_grad_nd(n=12, ndim=4)

def _test_grad_nd(n, ndim):
    coords = [np.arange(n)] * ndim
    xc = np.meshgrid(*coords, indexing="ij")

    # u = sum_i(xc[i]**2)
    u = reduce(lambda x,y: x+y**2, xc, 0.0)
    ucopy = np.copy(u)

    # check the gradient values
    slices = tuple([slice(1,-1,None)] * ndim)
    for i in range(ndim):
        assert grad(u, axis=i) == pytest.approx(2*xc[i][slices])

    # check if u is unchanged
    assert np.all(u == ucopy)

#################################### grad2 ####################################
def test_grad2_1d():
    _test_grad2_nd(n=32, ndim=1)

def test_grad2_2d():
    _test_grad2_nd(n=32, ndim=2)

def test_grad2_3d():
    _test_grad2_nd(n=12, ndim=3)

def test_grad2_4d():
    _test_grad2_nd(n=12, ndim=4)

def _test_grad2_nd(n, ndim):
    coords = [np.arange(n)] * ndim
    xc = np.meshgrid(*coords, indexing="ij")

    # u = sum_i(xc[i]**2)
    u = reduce(lambda x,y: x+y**2, xc, 0.0)
    ucopy = np.copy(u)

    # check the gradient values
    gu = np.zeros(tuple([n-2]*ndim))
    gu2 = gu + 2.0
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                assert grad2(u, axes=(i,j)) == pytest.approx(gu2)
            else:
                assert grad2(u, axes=(i,j)) == pytest.approx(gu)

    # check if u is unchanged
    assert np.all(u == ucopy)

################################### det_hess ###################################
def test_det_hess_1d():
    _test_det_hess_nd(n=32, ndim=1)

def test_det_hess_2d():
    _test_det_hess_nd(n=32, ndim=2)

def test_det_hess_3d():
    _test_det_hess_nd(n=12, ndim=3)

def test_det_hess_4d():
    _test_det_hess_nd(n=12, ndim=4)

def _test_det_hess_nd(n, ndim):
    coords = [np.arange(n)] * ndim
    xc = np.meshgrid(*coords, indexing="ij")

    # u = sum_i(xc[i]**2)
    u = reduce(lambda x,y: x+y**2, xc, 0.0)
    ucopy = np.copy(u)

    # check the det_hess values
    gu2 = np.zeros(tuple([n-2]*ndim)) + 2.0**ndim
    assert det_hess(u) == pytest.approx(gu2)

    # check if u is unchanged
    assert np.all(u == ucopy)
