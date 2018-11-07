"""
This module contains functions to calculate first and second derivatives of a
numpy.ndarray. The first derivatives are calculated with a central difference
scheme while the second derivatives are calculated using the central difference
as well.
The input size is (n0+2, n1+2, ..., nd1+2) and the output size is
(n0, n1, ..., nd1), unless specified.
"""
