# -*- coding: utf-8 -*-
# @Time    : 2022-09-09 16:17
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm

from scipy.special import gamma, factorial

import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import numpy as np
from skimage.color import rgb2gray
import matplotlib
import cv2 as cv
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import glob


def zernike_index(j, k):
    j = j - 1
    n = np.floor(-1 + np.sqrt(8 * j) / 2)
    i = j - n * (n - 1) / 2 + 1
    m = i - np.mod(n + i, 2)
    l = np.power(-1, k * m)
    return m, l


def zernike(noll_no, r, theta):
    n, m = zernike_index(noll_no, noll_no + 1)

    if int(n) == -1:
        zernike_poly = np.zeros(r.shape)
    else:
        if m == 0:
            zernike_poly = np.sqrt(n + 1) * zernike_radial(n, 0, r)
        else:
            if np.mod(m, 2) == 0:
                zernike_poly = np.sqrt(2 * (n + 1)) * zernike_radial(n, 0, r) * np.cos(m * theta)
            else:
                zernike_poly = np.sqrt(2 * (n + 1)) * zernike_radial(n, 0, r) * np.sin(m * theta)
    return zernike_poly


def zernike_radial(n, m, r):
    R = 0
    for s in range(int(np.ptp(np.arange((n - m) / 2)))):
        num = np.power(-1, s) * gamma(n - s + 1)
        denom = gamma(s + 1) * gamma((n + m) / 2 - s + 1) \
                * gamma((n - m) / 2 - s + 1)

        R = R + num / denom * np.power(r, (n - 2 * s))
    return R


def aberrate(img, abe_coes, D=102.4):
    N = img.shape[0]
    [x, y] = np.meshgrid(np.linspace(-N // 2, N // 2 + 1, N),
                         np.linspace(-N // 2, N // 2 + 1, N))

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    z = np.copy(abe_coes)
    W_values = np.zeros((r.shape[0], r.shape[-1], z.shape[-1]))
    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1),
                                           2 * r / D, theta)
    W = np.sum(W_values, axis=-1)

    W_complex = np.exp(complex(0, 1) * 2 * np.pi * W)
    # W_complex_for = np.exp(complex(0,-1) * 2 * np.pi * W)

    P = circ(img, D) * W_complex
    h = ft2(P)
    PSF = abs(h) ** 2
    # aberrated_img = myconv2(PSF, img)
    return myconv2(PSF, img)


def myconv2(A, B):
    return ift2(ft2(A) * ft2(B))


def ift2(f):
    return ifftshift(ifft2(ifftshift(f)))


def ft2(f):
    return fftshift(fft2(fftshift(f)))


def circ(x, pupil):
    pupil_plane = np.empty_like(x)
    x_cent, y_cent = x.shape[0] // 2, x.shape[1] // 2
    for i in range(pupil_plane.shape[0]):
        for j in range(pupil_plane.shape[1]):
            if np.sqrt((i - x_cent) ** 2 + (j - y_cent) ** 2) <= pupil:

                pupil_plane[i, j] = 1
            else:
                pupil_plane[i, j] = 0

    return pupil_plane


if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    image_plist = glob.glob('../test_image/*.png')
    img = cv.imread(image_plist[-1])
    img_gray = rgb2gray(img)

    abe_coes = np.empty(11)
    abe_coes[0] = 0  # piston
    abe_coes[1] = 1  # x tilt
    abe_coes[2] = 0.5  # y tilt
    abe_coes[3] = 0.1  # defocus
    abe_coes[4] = 0.1  # y primary astigmatism
    abe_coes[5] = 0.1  # x primary astigmatism
    abe_coes[5] = 0  # y primary coma
    abe_coes[7] = 0  # x primary coma
    abe_coes[8] = 0  # y trefoil
    abe_coes[9] = 0  # x trefoil
    abe_coes[10] = 0.2  # primary spherical

    aberrate_img = aberrate(img_gray, abe_coes, D=80)
    plt.imshow(abs(aberrate_img), 'gray')
    plt.show()
