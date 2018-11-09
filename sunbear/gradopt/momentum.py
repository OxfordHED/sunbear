import time
import numpy as np
from sunbear.gradopt.gradopt_interface import GradOptInterface

__all__ = ["Momentum"]

class Momentum(GradOptInterface):
    def __init__(self,
            refresh_interval = 100,
            rel_tol = 1e-9,
            minstep = 1e-3,
            alpha = 0.1,
            # stopping conditions options
            max_niter = 50000,
            max_time = 60*5, # in s
            max_niter_no_update = 5000,
            verbose=1,
            ):
        self.refresh_interval = refresh_interval
        self.rel_tol = rel_tol
        self.minstep = minstep
        self.alpha = alpha
        self.max_niter = max_niter
        self.max_time = max_time
        self.max_niter_no_update = max_niter_no_update
        self.verbose = verbose

    def solve(self, func, x0):
        self.tinit = time.time()

        # search the step size
        f0, dx0 = func(x0)
        self.finit = f0
        step = _step_search(func, x0, f0, dx0, self.minstep)
        x = x0 - step * dx0

        # iteration
        self.niter = 0
        self.fmin = np.inf
        xmin = 0
        self.last_niter_update = 0
        try:
            while 1:
                f, dx = func(x)
                if f < self.fmin:
                    self.fmin = f
                    xmin = x
                    self.last_niter_update = self.niter

                # step update
                if (f > f0) and (step > self.minstep):
                    step = step / 2.0
                else:
                    f0 = f
                    x0 = x
                    dx0 = dx * self.alpha + dx0 * (1-self.alpha) # momentum update

                # update by stepping to the new value
                x = x0 - step * dx0

                self.niter += 1
                if step < self.minstep:
                    step = self.minstep

                # refresh the step size (i.e. it checks if it can be larger)
                if self.niter % self.refresh_interval == 0 or self.niter == 1:
                    step = _step_search(func, x0, f0, dx0, step)
                    if step < self.minstep:
                        step = self.minstep

                # print the result
                if self.verbose:
                    if self.niter == 1:
                        print("niter  fmin")
                    if (self.niter == 1) or (self.niter % 100 == 0):
                        print("%5d %.3e" % (self.niter, self.fmin))

                # check the stopping conditions
                if self._is_stop():
                    break
        except KeyboardInterrupt:
            pass
        return xmin

    def _is_stop(self):
        def disp(s):
            if self.verbose:
                print("Stopped due to %s" % s)

        # stopping conditions
        if self.niter > self.max_niter:
            disp("niter > max_niter: (%d < %d)" % (self.niter, self.max_niter))
            return True
        if self.fmin < self.rel_tol * self.finit:
            disp("fmin < rel_tol*finit: (%e < %e)" % \
                (self.fmin, self.rel_tol*self.finit))
            return True
        if time.time() - self.tinit > self.max_time:
            disp("time spent > %fs" % self.max_time)
            return True
        if self.niter - self.last_niter_update > self.max_niter_no_update:
            disp("niter with no update > %d" % self.max_niter_no_update)
            return True

        return False

def _step_search(func, x0, f0, dx0, step):
    # search the optimal step size using the golden search algorithm
    golden = 1.618033988749895 # golden ratio
    golden_inv = 1.0 / golden

    # get the bounds
    f, _ = func(x0 - step * dx0)
    step0 = step if (f < f0) else 0.0
    while (f < f0):
        step *= 2.0
        f, _ = func(x0 - step * dx0)

    # now the search is between [0, step]
    a = step0
    b = step
    c = b - (b - a) * golden_inv
    d = a + (b - a) * golden_inv
    fa = f0
    fb = f
    fc, _ = func(x0 - c * dx0)
    fd, _ = func(x0 - d * dx0)
    # golden ratio search algorithm
    for i in range(4):
        if fc < fd:
            b = d
            fb = fd
        else:
            a = c
            fa = fc

        c = b - (b - a) * golden_inv
        d = a + (b - a) * golden_inv
        if fc < fd:
            fd = fc
            fc, _ = func(x0 - c * dx0)
        else:
            fc = fd
            fd, _ = func(x0 - d * dx0)

    # final step selection (select the minimum one)
    step = c if fc < fd else d
    return step
