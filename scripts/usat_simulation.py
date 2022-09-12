# -*- coding: utf-8 -*-
# @Time    : 2022-09-08 15:55
# @Author  : young wang
# @FileName: usat_simulation.py
# @Software: PyCharm

"""
Simulated aberrated USAF resolution target image
with 3rd-order Zernike polynomials baesd on

Dan Zhu, Ruoyan Wang, Mantas Žurauskas,
Paritosh Pande, Jinci Bi, Qun Yuan, Lingjie Wang,
Zhishan Gao, and Stephen A. Boppart,
"Automated fast computational adaptive optics for
optical coherence tomography based
on a stochastic parallel gradient descent algorithm,"
Opt. Express 28, 23306-23319 (2020)
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from scipy.special import gamma, factorial
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scripts.tools.plot import heatmap
import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import glob

def aberrate(img=None,abe_coes = None,D=None ):
    N = 512
    img = img/np.max(img)
    [x, y] = np.meshgrid(np.linspace(-N / 2, N / 2, N),
                         np.linspace(-N / 2, N / 2, N))

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    z = np.copy(abe_coes)

    W_values = np.zeros((r.shape[0], r.shape[-1], z.shape[-1]))
    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1),
                                           2 * r / D, theta)
    W = np.sum(W_values, axis=-1)

    zernike_plane = circ(x, D) * W
    zernike_plane = zernike_plane/np.max(zernike_plane)

    P = circ(x, D) * np.exp(complex(0, 1) * 2 * np.pi * W)
    Px = np.exp(complex(0, -1) * 2 * np.pi * W)

    psf = normalize_psf(P)

    # zernike_plane = circ(x, D) * W
    # z_plane = zernike_plane/np.max(zernike_plane)

    return zernike_plane,normalize_image(myconv2(psf, img)), Px

def normalize_psf(psf):
    h = ft2(psf)
    psf_norm = abs(h) ** 2
    return psf_norm/np.max(psf_norm)

def normalize_image(image):
    return abs(image) / np.max(abs(image))

def ft2(f):
    g = fftshift(fft2(fftshift(f)))
    return g


def myconv2(img=None, psf=None):
    C = ift2(ft2(img)* ft2(psf))
    return C


def ift2(f=None):
    g = ifftshift(ifft2(ifftshift(f)))
    return g


def zernike_index(j=None, k=None):
    j = j - 1
    n = int(np.floor((- 1 + np.sqrt(8 * j + 1)) / 2))
    i = j - n * (n + 1) / 2 + 1
    m = i - np.mod(n + i, 2)
    l = np.power(-1, k) * m
    return n, l


def zernike(i: object = None, r: object = None, theta: object = None) -> object:
    n, m = zernike_index(i, i + 1)

    if n == -1:
        zernike_poly = np.ones(r.shape)
    else:
        if m == 0:
            zernike_poly = np.sqrt(n + 1) * zernike_radial(n, 0, r)
        else:
            if np.mod(i, 2) == 0:
                zernike_poly = np.sqrt(2 * (n + 1)) * zernike_radial(n, m, r) * np.cos(m * theta)
            else:
                zernike_poly = np.sqrt(2 * (n + 1)) * zernike_radial(n, m, r) * np.sin(m * theta)

    return zernike_poly


def circ(x=None, pupil=None):

    pupil_plane = np.zeros(x.shape)
    x_cent, y_cent = x.shape[0] // 2, x.shape[1] // 2
    for i in range(pupil_plane.shape[0]):
        for j in range(pupil_plane.shape[1]):
            if np.sqrt((i - x_cent) ** 2 + (j - y_cent) ** 2) <= pupil:

                pupil_plane[i, j] = 1
            else:
                pupil_plane[i, j] = 0

    return pupil_plane


def zernike_radial(n=None, m=None, r=None):
    R = 0
    for s in np.arange(0, ((n - m) / 2) + 1):
        num = np.power(-1, s) * gamma(n - s + 1)
        denom = gamma(s + 1) * \
                gamma((n + m) / 2 - s + 1) * \
                gamma((n - m) / 2 - s + 1)
        R = R + num / denom * r ** (n - 2 * s)

    return R

if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    abe_coes = np.zeros(11)
    abe_coes[0] = 1  # piston
    abe_coes[1] = 0  # x tilt
    abe_coes[2] = 0  # y tilt
    abe_coes[3] = 0  # defocus
    abe_coes[4] = 0  # y primary astigmatism
    abe_coes[5] = 0  # x primary astigmatism
    abe_coes[5] = 0  # y primary coma
    abe_coes[7] = 0  # x primary coma
    abe_coes[8] = 0  # y trefoil
    abe_coes[9] = 0  # x trefoil
    abe_coes[10] = 0  # primary spherical

    img_list = []
    title_list = []

    image_plist = glob.glob('../test_image/*.png')
    img = cv.imread(image_plist[-1])
    img_gray = rgb2gray(img)

    # img_list.append(img_gray)
    # title_list.append('original image')

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    img_list.append(img_noise)
    title_list.append('noisy image')

    zernike_plane, aberrated_img, Px\
        = aberrate(img_gray,abe_coes, D=102)
    img_list.append(aberrated_img)
    title_list.append('aberrant image')

    img_fft = np.fft.fftshift(np.fft.fft2(img_noise))
    img_fft_log = 20 * np.log10(abs(img_fft))

    img_list.append(img_fft_log)
    title_list.append('fft image')

    img_list.append(zernike_plane)
    title_list.append('Zernike plane')

    fig, axs = plt.subplots(2, 2,
                            figsize=(10, 10), constrained_layout=True)
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):

        if n == 3:
            heatmap(image,ax)
            ax.set_title(title)

        else:
            ax.imshow(image,'gray')
            ax.set_title(title)
            ax.set_axis_off()
    plt.show()

    ### TBD steps

    # ab_fft = fftshift(fft2(fftshift(aberrated_img))) * Px
    # abr_img = fftshift(ifft2(fftshift(ab_fft)))
    # plt.imshow(abs(abr_img), 'gray')
    # plt.show()
