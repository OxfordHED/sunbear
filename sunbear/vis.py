import numpy as np
import matplotlib.pyplot as plt
from sunbear.forward import forward_pos

__all__ = ["vis"]

def vis(phi, ngrid=10, line_kwargs={}):
    """
    Plot the grid deformation based on the given deflection potential, `phi`.
    It only works for 2D signal at the moment.

    Parameters
    ----------
    * `phi` : np.ndarray
        The deflection potential.
    * `ngrid` : int or sequential, int
        Number of grid points to be visualized.
    * `line_kwargs` : dict
        Kwargs of the plt.plot to plot the grid lines.
    """
    ndim = np.ndim(phi)
    if ndim != 2:
        raise ValueError("vis function can only take 2D deflection potential")
    if not hasattr(ngrid, "__iter__"):
        ngrid = (ngrid, ngrid)
    if line_kwargs == {}:
        line_kwargs = {"color": "C0"}

    # obtain the mesh position
    x = [np.linspace(0, phi.shape[i]-1, ngrid[i]) for i in range(ndim)]
    meshpos = np.array(np.meshgrid(*x)) # (ndim, ngrid0, ngrid1, ...)
    pos = meshpos.reshape(ndim, -1).T # (N x D)

    # get the new position
    newpos = forward_pos(pos, phi) # (N x D)
    new_shape = list(meshpos.shape[1:]) + [ndim]
    mesh_newpos = newpos.reshape(new_shape) # (ngrid0, ngrid1, ..., ndim)

    if ndim == 2:
        for i in range(mesh_newpos.shape[0]):
            plt.plot(mesh_newpos[i,:,0], mesh_newpos[i,:,1], **line_kwargs)
        for i in range(mesh_newpos.shape[1]):
            plt.plot(mesh_newpos[:,i,0], mesh_newpos[:,i,1], **line_kwargs)
        plt.gca().set_aspect(phi.shape[1] * 1.0 / phi.shape[0])
