# -*- coding: utf-8 -*-
# @Time    : 2022-09-08 18:20
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm

import numpy as np
# import pyfits
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tools.plot import heatmap
from matplotlib.ticker import LinearLocator

import matplotlib
from matplotlib import cm
import cv2 as cv
import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2

from natsort import natsorted
import glob


def Zernike_polar(coefficients, radius, theta):
    z = coefficients

    z1 = z[0] * 1 * (np.cos(theta) ** 2 + np.sin(theta) ** 2)
    z2 = z[1] * 2 * radius * np.cos(theta)
    z3 = z[2] * 2 * radius * np.sin(theta)
    z4 = z[3] * np.sqrt(3) * (2 * radius ** 2 - 1)
    z5 = z[4] * np.sqrt(6) * radius ** 2 * np.sin(2 * theta)
    z6 = z[5] * np.sqrt(6) * radius ** 2 * np.cos(2 * theta)
    z7 = z[6] * np.sqrt(8) * (3 * radius ** 2 - 2) * radius * np.sin(theta)
    z8 = z[7] * np.sqrt(8) * (3 * radius ** 2 - 2) * radius * np.cos(theta)

    zw = z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8  # +  Z9+ Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return zw


def Zernike_plane(coefficients, img):
    x_len, y_len = img.shape

    r = 1  # create unit circle

    [x_idx, y_idx] = np.meshgrid(np.linspace(-r, r, x_len), np.linspace(-r, r, y_len))
    r_idx = np.sqrt(x_idx ** 2 + y_idx ** 2)
    theta_idx = np.arctan2(y_idx, x_idx)

    z_plane = Zernike_polar(coefficients, r_idx, theta_idx)
    z_plane[r_idx > r] = 0
    return preprocessing.normalize(z_plane)


if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    coefficients = np.zeros(8)
    coefficients[1] = 0.1
    coefficients[2] = 0.1
    coefficients[3] = 0.3
    coefficients[4] = 0.2
    coefficients[5] = 0.8
    coefficients[6] = 0.3
    coefficients[7] = 0.2

    img_list = []
    title_list = []

    image_plist = glob.glob('../test_image/*.png')
    img = cv.imread(image_plist[-1])
    img_gray = rgb2gray(img)

    img_list.append(img_gray)
    title_list.append('original image')

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    img_list.append(img_noise)
    title_list.append('noisy image')

    img_fft = np.fft.fftshift(np.fft.fft2(img_noise))
    img_fft_log = 20 * np.log10(abs(img_fft))

    img_list.append(img_fft_log)
    title_list.append('fft image')

    img_zphase = Zernike_plane(coefficients, img_fft_log)
    img_list.append(img_zphase)
    title_list.append('Zernike plane')

    fig, axs = plt.subplots(int(len(img_list) // 2), int(len(img_list) // 2),
                            figsize=(10, 10), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    plt.show()
