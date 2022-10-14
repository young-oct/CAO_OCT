# -*- coding: utf-8 -*-
# @Time    : 2022-08-02 17:06
# @Author  : young wang
# @FileName: plot.py
# @Software: PyCharm

import numpy as np
import matplotlib as plt
from .proc import index_mid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

matplotlib.rcParams.update(
    {
        'font.size': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'text.usetex': False,
        'font.family': 'sans-serif',
        'mathtext.fontset': 'stix',
    }
)


def heatmap(title, data, ax=None):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data)

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="-3.5%",
                               pad=0.25,
                               pack_start=True)
    ax.figure.add_axes(cax)
    cbar = ax.figure.colorbar(im, cax=cax, orientation="horizontal")

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, y=1, pad=5, fontsize = 20)
    ax.xaxis.set_label_position('top')

    return im, cbar

def line_fit_plot(points,l_txt, ax, order = 1):
    x, y = zip(*points)
    a, b = np.polyfit(x, y, order)

    x_range = np.arange(np.ptp(x))

    ax.scatter(x, y, color='purple')
    ax.set_xlabel('z index')
    ax.set_ylabel(str(l_txt))

    ax.plot(x_range, a * x_range + b, color='steelblue', linestyle='--', linewidth=2)
    ax.text(0.3, 0.15, 'y = ' + '{:.4f}'.format(b) + ' + {:.4f}'.format(a) + 'x',
            size=20, color = 'red', transform=ax.transAxes)

    return ax


def linear_fit_plot(line_list, ax, title):
    x,y = zip(*line_list)
    coef = np.polyfit(x, y, 1)
    slope, intercept = coef[0], coef[1]
    poly1d_fn = np.poly1d(coef)

    ax.scatter(*zip(*line_list), c = 'blue', marker='o', s = 10 )

    # ax.scatter(x, y,  marker = 'o', color = 'blue')
    ax.plot(x, poly1d_fn(x), linestyle = '--', color = 'red' )  # '--k'=black dashed line, 'yo' = yellow circle marker
    ax.text(np.mean(x) * 1.2,np.mean(y) * 0.8, f'$y = {slope:.1f}x {intercept:+.1f}$', c = 'red')
    ax.set_xlabel('z index [pixels]')
    ax.set_ylabel('distance [pixels]')
    ax.set_title(title)

    return ax