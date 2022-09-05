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
    ax.set_title('std: %.2f' % np.std(data), y=0, pad=-14)
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

def angle_est(x, y, origin, radius, ax):
    xmin_idx, xmax_idx = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    xc, yc = origin[0], origin[1]

    x_mid = index_mid(x)
    y_mid = index_mid(y)

    x_line = [xc, x_mid]
    y_line = [yc, y_mid]

    ax.plot(xmin_idx, ymin, color='green', label='x1', marker='8', ms=5)
    ax.plot(xmax_idx, ymax, color='blue', label='x2', marker='8', ms=5)
    ax.plot(x_mid, y_mid, color='red', label='xc', marker='D', ms=5)
    ax.plot(x_line, y_line, color='red', linestyle ='dashed', linewidth = 1)

    angle_1 = np.degrees(np.arcsin(abs(xc - xmin_idx) / radius))
    angle_2 = np.degrees(np.arcsin(abs(xc - xmax_idx) / radius))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr = '\n'.join((
        r'$\theta_2=%.2f$' % (angle_1,),
        r'$\theta_2=%.2f$' % (angle_2,)))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # ax.annotate('x1', xy=(xmin_idx, ymin), xycoords='data',
    #             xytext=(xmin_idx - radius / 4, ymin + radius / 4), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             horizontalalignment='right', verticalalignment='top',
    #             )
    #
    # ax.annotate('x2', xy=(xmax_idx, ymax), xycoords='data',
    #             xytext=(xmax_idx + radius / 4, ymax + radius / 4), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             horizontalalignment='right', verticalalignment='top',
    #             )
    #
    # ax.annotate('xc', xy=(x_mid, y_mid), xycoords='data',
    #             xytext=(x_mid + radius / 4, y_mid + radius / 4), textcoords='data',
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             horizontalalignment='right', verticalalignment='top',
    #             )

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