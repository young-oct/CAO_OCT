# -*- coding: utf-8 -*-
# @Time    : 2022-09-30 17:51
# @Author  : young wang
# @FileName: spicy_method.py
# @Software: PyCharm

from functools import partial
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np


def aberrate(img=None, abe_coes=None):
    N = img.shape[0]
    img = img / np.max(img)
    [x, y] = np.meshgrid(np.linspace(-N / 2, N / 2, N),
                         np.linspace(-N / 2, N / 2, N))

    r = np.sqrt(x ** 2 + y ** 2) / N
    theta = np.arctan2(y, x)

    z = np.copy(abe_coes)

    W_values = np.zeros((r.shape[0], r.shape[-1], z.shape[-1]))
    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1), r, theta)
    W = np.sum(W_values, axis=-1)
    #
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * fftshift(W)
    phi_x = np.conj(phi_o)
    #
    Po = np.exp(phi_o)
    Px = np.exp(phi_x)
    #
    zernike_plane = W / np.max(W)
    zernike_plane = zernike_plane
    #
    ab_img = apply_wavefront(img, Po)
    cj_img = remove_wavefront(ab_img, Px)

    return zernike_plane, Po, Px, normalize_image(ab_img), normalize_image(cj_img)
    #

def normalize_psf(psf):
    h = fft2(psf)
    psf_norm = abs(h) ** 2
    return psf_norm / np.max(psf_norm)


def normalize_image(image):
    return abs(image) / np.max(abs(image))


def apply_wavefront(img=None, Po=None):
    return ifft2(fft2(img) * Po)


def remove_wavefront(aberrant_img=None, Px=None):
    # apply wavefront conjugation to the aberration image in
    # frequency domain
    return ifft2(fft2(aberrant_img) * Px)


def zernike_index(j=None, k=None):
    j -= 1
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

        temp = n + 1
        if m == 0:
            zernike_poly = np.sqrt(temp) * zernike_radial(n, 0, r)
        else:
            if np.mod(i, 2) == 0:
                zernike_poly = np.sqrt(2 * temp) * zernike_radial(n, m, r) * np.cos(m * theta)
            else:
                zernike_poly = np.sqrt(2 * temp) * zernike_radial(n, m, r) * np.sin(m * theta)

    return zernike_poly

def zernike_radial(n=None, m=None, r=None):
    R = 0
    for s in np.arange(0, ((n - m) / 2) + 1):
        num = np.power(-1, s) * gamma(n - s + 1)
        denom = gamma(s + 1) * \
                gamma((n + m) / 2 - s + 1) * \
                gamma((n - m) / 2 - s + 1)
        R = R + num / denom * r ** (n - 2 * s)

    return R


def image_entropy(image=None):
    """
    implementation of image entropy using Eq(10)
    from the reference[1]
    :param image: 2D array
    :return: entropy of the image
    """
    temp_img = normalize_image(image)
    entropy = 0
    for i in range(temp_img.shape[0]):
        for j in range(temp_img.shape[1]):
            intensity_value = temp_img[i, j]
            if intensity_value != 0:  # avoid log(0) error/warning
                entropy -= intensity_value * np.log(intensity_value)
            else:
                pass

    return entropy


def load_zernike_coefficients():
    sigma, mu = np.random.random(size=2)
    z_cos = mu + sigma * 5 * np.random.random(size=10)
    return z_cos

def construct_zernike(zcof, image=None):

    z = np.copy(zcof)
    N = image.shape[0]

    W_values = np.zeros((N, N, z.shape[-1]))

    [x, y] = np.meshgrid(np.linspace(-N / 2, N / 2, N),
                         np.linspace(-N / 2, N / 2, N), indexing='ij')

    r = np.sqrt(x ** 2 + y ** 2) / N
    theta = np.arctan2(y, x)

    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1), r, theta)
    W = np.sum(W_values, axis=-1)
    # phi_x = (complex(0, -1) * 2 * torch.pi * fftshift(W))
    # Px = torch.exp(phi_x)

    return W

np.random.seed(0)

def f(zcof,image):

    W = construct_zernike(zcof, image=image)
    phi_x = complex(0, -1) * 2 * np.pi * fftshift(W)
    Px = np.exp(phi_x)
    cor_img = ifft2(fft2(image) * Px)

    entropy_value = image_entropy(cor_img)

    return entropy_value

if __name__ == '__main__':

    abe_coes = load_zernike_coefficients()
    # img = np.zeros((512,512))
    # W = construct_zernike(abe_coes,img)

    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    _, _, _, ab_img, _ = aberrate(img_noise, abe_coes)
    plt.imshow(ab_img)
    plt.show()

    # np.random.seed(0)
    initial_guess = np.random.random(size=10)

    func = partial(f, image = ab_img)

    options = {
                 'disp': True,
                }
    minResults = minimize.fmin_ncg(func, initial_guess,
                          method='L-BFGS-B',options= options)

    est = minResults.x
    x_axis = np.linspace(0,len(est),len(est))

    plt.bar(x_axis + 0.2, initial_guess, width=0.1,label='initial guess')
    plt.bar(x_axis - 0.2, abe_coes, width=0.1,label='true value')
    plt.bar(x_axis,est, width=0.1,label='estimation')
    plt.legend()
    plt.show()

    Wx = construct_zernike(est, ab_img)
    phi_xx = complex(0, 1) * 2 * np.pi * fftshift(Wx)
    Pxx = np.exp(phi_xx)
    cor_img = remove_wavefront(ab_img,Pxx)

    plt.imshow(normalize_image(cor_img))
    plt.show()


