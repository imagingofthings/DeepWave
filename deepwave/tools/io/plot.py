# ############################################################################
# plot.py
# =======
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
`Matplotlib <https://matplotlib.org/>`_ helpers.
"""

import matplotlib.axes as axes
import matplotlib.collections as collections
import matplotlib.colors as col
import mpl_toolkits.axes_grid1 as ax_grid
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np


def colorbar(scm, ax):
    """
    Attach colorbar to side of a plot.

    Parameters
    ----------
    scm : :py:class:`~matplotlib.cm.ScalarMappable`
        Intensity scale.
    ax : :py:class:`~matplotlib.axes.Axes`
        Plot next to which the colorbar is placed.

    Returns
    -------
    cbar : :py:class:`~matplotlib.colorbar.Colorbar`

    Examples
    --------
    .. doctest::

       import numpy as np
       import matplotlib.pyplot as plt
       from tools.io.plot import colorbar

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots()
       im = ax.imshow(x + y, cmap='jet')
       cb = colorbar(im, ax)

       fig.show()
    """
    fig = ax.get_figure()
    divider = ax_grid.make_axes_locatable(ax)
    ax_colorbar = divider.append_axes(
        'right', size='5%', pad=0.05, axes_class=axes.Axes)
    cbar = fig.colorbar(scm, cax=ax_colorbar)
    return cbar


def setup_trajectory_plot(subspace,
                          xyz_lim,
                          xyz_kwargs=None,
                          subspace_kwargs=None,
                          ax=None):
    """
    Prepare Axes3D for trajectory plotting/labeling.

    Parameters
    ----------
    subspace : str
        Where to draw the subspace.
        One of ['x', 'y', 'z', 'xy', 'yz', 'xz'].
    xyz_lim : :py:class:`~numpy.ndarray`
        (2, 3) array where columns denote min/max size of each dimension.
    xyz_kwargs : dict
        Keyword arguments to forward to ax.plot3D used to draw the Cartesian axes.
    subspace_kwargs : dict
        Keyword arguments to forward to ax.plot3D used to draw the subspace.
    ax : :py:class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`
        Axis on which to draw.
    """
    valid_subspaces = ['x', 'y', 'z', 'xy', 'yz', 'xz']
    if not (isinstance(subspace, str) and (subspace in valid_subspaces)):
        raise ValueError('Parameter[subspace] is ill-formed.')

    if not (xyz_lim.shape == (2, 3)):
        raise ValueError('Parameter[xyz_lim] is ill-formed.')

    xyz_kw = dict(linewidth=2, marker='.', color='k')
    if xyz_kwargs is not None:
        xyz_kw.update(xyz_kwargs)

    subspace_kw = dict(linewidth=2, color='g', alpha=0.5)
    if subspace_kwargs is not None:
        subspace_kw.update(subspace_kwargs)

    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot Cartesian axes
    len_axis = np.sum(np.abs(xyz_lim), axis=0).min() * 0.2
    ax.plot(np.r_[0, len_axis], np.r_[0, 0], np.r_[0, 0], **xyz_kw)
    ax.plot(np.r_[0, 0], np.r_[0, len_axis], np.r_[0, 0], **xyz_kw)
    ax.plot(np.r_[0, 0], np.r_[0, 0], np.r_[0, len_axis], **xyz_kw)

    # Plot Subspace
    def idx(letter):
        if letter == 'x':
            return 0
        if letter == 'y':
            return 1
        if letter == 'z':
            return 2

    if len(subspace) == 1:
        subspace_x = np.r_[0, xyz_lim[1, idx(subspace)] if (subspace == 'x') else 0]
        subspace_y = np.r_[0, xyz_lim[1, idx(subspace)] if (subspace == 'y') else 0]
        subspace_z = np.r_[0, xyz_lim[1, idx(subspace)] if (subspace == 'z') else 0]
        ax.plot(subspace_x, subspace_y, subspace_z, **subspace_kw)
    elif len(subspace) == 2:
        l0, l1 = subspace
        subspace_vertex = np.array([[0, 0],
                                    [xyz_lim[1, idx(l0)], 0],
                                    [xyz_lim[1, idx(l0)], xyz_lim[1, idx(l1)]],
                                    [0, xyz_lim[1, idx(l1)]]])
        rect = collections.PolyCollection([subspace_vertex], **subspace_kw)

        zdir = ({'x', 'y', 'z'} - set(subspace)).pop()
        ax.add_collection3d(rect, zs=0, zdir=zdir)

    ax.set_xlim(xyz_lim[0, 0], xyz_lim[1, 0])
    ax.set_ylim(xyz_lim[0, 1], xyz_lim[1, 1])
    ax.set_zlim(xyz_lim[0, 2], xyz_lim[1, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = col.LinearSegmentedColormap.from_list(
        f'trunc{cmap.name},{minval:.2f},{maxval:.2f}',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def magma_cmap():
    import matplotlib.pyplot as plt
    cmap = truncate_colormap(plt.get_cmap('magma_r'), 0, 0.75)
    return cmap
