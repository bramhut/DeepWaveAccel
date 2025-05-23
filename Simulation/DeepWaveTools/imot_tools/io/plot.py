# ############################################################################
# plot.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# ############################################################################

"""
`Matplotlib <https://matplotlib.org/>`_ helpers.
"""

import matplotlib.axes as axes
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as ax_grid

import DeepWaveTools.imot_tools.util.argcheck as chk


@chk.check(dict(scm=chk.is_instance(cm.ScalarMappable), ax=chk.is_instance(axes.Axes)))
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

       import matplotlib.pyplot as plt
       import numpy as np

       from DeepWaveTools.imot_tools.io.plot import colorbar

       x, y = np.ogrid[-1:1:100j, -1:1:100j]

       fig, ax = plt.subplots()
       im = ax.imshow(x + y, cmap='jet')
       cb = colorbar(im, ax)

       fig.show()

    .. image:: _img/colorbar_example.png
    """
    fig = ax.get_figure()
    divider = ax_grid.make_axes_locatable(ax)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05, axes_class=axes.Axes)
    cbar = fig.colorbar(scm, cax=ax_colorbar)
    return cbar
