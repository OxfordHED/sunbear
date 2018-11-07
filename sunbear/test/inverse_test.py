from functools import reduce
import numpy as np
import pytest
import sunbear as sb
import matplotlib.pyplot as plt

class SourceObj:
    def __init__(self, profile_str):
        self.profile_str = profile_str

    def setparam(self, n, ndim):
        self.ndim = ndim
        self.n = n

    def getprofile(self):
        n = self.n
        ndim = self.ndim
        if self.profile_str == "flat":
            return np.ones([n]*ndim) / (n**ndim)
        elif self.profile_str == "gaussian":
            x = np.linspace(-1, 1, n)
            sigma = 0.6
            xmg = np.meshgrid(*([x]*ndim))
            xsq = reduce(lambda x,y:x+y**2, xmg, 0.0)
            s = np.exp(-xsq / (2*sigma**2))
            return s / np.sum(s)
        else:
            raise ValueError("Unknown source profile: %s" % \
                self.profile_str)

class PhiObj:
    def __init__(self, profile_str):
        self.profile_str = profile_str

    def setparam(self, n, ndim):
        self.ndim = ndim
        self.n = n

    def getprofile(self):
        n = self.n
        ndim = self.ndim
        x = np.linspace(-1, 1, n)
        xmg = np.meshgrid(*([x]*ndim))
        err = ValueError("Unknown phi profile: %s" % self.profile_str)
        if self.profile_str == "no":
            return np.zeros([n] * ndim)
        elif self.profile_str.startswith("gaussian"):
            sigma = 0.2
            xsq = reduce(lambda x,y:x+y**2, xmg, 0.0)
            s = np.exp(-xsq / (2*sigma**2))

            if self.profile_str == "gaussian-1":
                return s * -n # pass

        # if it's not returned at this point, then throw the error
        raise err


@pytest.fixture(params=["flat", "gaussian"])
def source_obj(request):
    return SourceObj(request.param)

@pytest.fixture(params=["no", "gaussian-1"])
def phi_obj(request):
    return PhiObj(request.param)

def test_inverse_1d(source_obj, phi_obj):
    n, ndim = 100, 1
    _test_inverse_nd(n, ndim, source_obj, phi_obj)

def test_inverse_2d(source_obj, phi_obj):
    n, ndim = 100, 2
    _test_inverse_nd(n, ndim, source_obj, phi_obj)

def _test_inverse_nd(n, ndim, source_obj, phi_obj,
        abstarget=None, absphi=None,
        reltarget=None, relphi=None):
    source_obj.setparam(n, ndim)
    phi_obj.setparam(n, ndim)
    source = source_obj.getprofile()
    phi = phi_obj.getprofile()

    # make sure the phi are recoverable
    target = sb.forward(source, phi)
    phi_r = sb.inverse(source, target)
    target_r = sb.forward(source, phi_r)
    # if ndim == 1:
    #     nr = 3
    #     nc = 1
    #     plt.subplot(nr,nc,1)
    #     plt.plot(phi)
    #     plt.plot(phi_r)
    #     plt.subplot(nr,nc,2)
    #     plt.plot(target)
    #     plt.plot(target_r)
    #     plt.subplot(nr,nc,3)
    #     plt.plot(sb.math.diff.grad2(target, axes=(0,0)))
    #     plt.plot(sb.math.diff.grad2(target_r, axes=(0,0)))
    #     plt.show()

    # set the default tolerance
    if abstarget is None:
        abstarget = np.max(np.abs(target)) * 1e-2
    if absphi is None:
        absphi = np.max(np.abs(phi)) * 1e-2
    if reltarget is None:
        reltarget = 2e-2
    if relphi is None:
        relphi = 2e-2

    assert target_r == pytest.approx(target,abs=abstarget,rel=reltarget)
    assert phi_r == pytest.approx(phi, abs=absphi, rel=relphi)
