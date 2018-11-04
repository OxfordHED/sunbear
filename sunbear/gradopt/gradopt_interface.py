from abc import ABCMeta, abstractmethod

class GradOptInterface:
    __metaclass__ = ABCMeta

    def solve(self, func, x0):
        """
        Finding the minimizer of loss function expressed in `func`.

        Parameters
        ----------
        * `func` : callable
            A callable that receives an input like `x0` and returns the loss
            function value, `f`, to be minimized and the gradient of `x`, i.e.
            `df/dx`.
        * `x0` : array_like
            The initial value to `func`.

        Returns
        -------
        * array_like
            The value that minimizes `func`, to have the same shape as `x0`.
        """
        pass
