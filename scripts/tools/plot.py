# -*- coding: utf-8 -*-
# @Time    : 2022-08-02 17:06
# @Author  : young wang
# @FileName: plot.py
# @Software: PyCharm

import numpy as np
import matplotlib as plt
from .proc import index_mid

def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_axis_off()

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    # ax.set_title('std: %.2f' % np.std(data), y=0, pad=-14)
    ax.xaxis.set_label_position('top')
    # ax.set_xlabel('std: %.2f' % np.std(data))
    # ax.set_title('Manual y', y=1.0, pad=-14)
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