# -*- coding: utf-8 -*-
# @Time    : 2022-10-16 21:08
# @Author  : young wang
# @FileName: test_recov.py
# @Software: PyCharm

from scipy.special import gamma
from scipy.fftpack import fftshift, fft2, ifft2
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
from tools import plot

def aberrate(img=None, abe_coes=None):
    W_temp = construct_zernike(abe_coes, image=img)
    W = zernike_plane(abe_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * W
    #
    Po = np.exp(phi_o)
    img_ab = apply_wavefront(img=img, Po=Po)

    #
    # #
    # ab_img = apply_wavefront(img, Po)
    # # cj_img = remove_wavefront(ab_img, Px)

    return img_ab

def correct_img(img=None, abe_coes=None):
    W_temp = construct_zernike(abe_coes, image=img)
    W = zernike_plane(abe_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_x = complex(0, -1) * 2 * np.pi * W
    # phi_x = np.conj(phi_o)
    #
    Px = np.exp(phi_x)
    # cor_img = remove_wavefront(aberrant_img=img, Px=Px)

    return Px


def normalize_image(image):
    return abs(image) / np.max(abs(image))


def apply_wavefront(img=None, Po=None):
    return ifft2(fft2(img) * fftshift(Po))


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


def load_zernike_coefficients(no_terms):
    np.random.seed(20)

    sigma, mu = np.random.random(size=2)
    A = mu + sigma * 20 * np.random.random(size=no_terms)
    return A


def construct_zernike(z, image=None):
    N = image.shape[0]

    W_values = np.zeros((N, N, z.shape[-1]))

    [x, y] = np.meshgrid(np.linspace(-N / 2, N / 2, N),
                         np.linspace(-N / 2, N / 2, N), indexing='ij')

    r = np.sqrt(x ** 2 + y ** 2) / N
    theta = np.arctan2(y, x)

    for i in range(z.shape[-1]):
        W_values[:, :, i] = zernike(int((i + 3) + 1), r, theta)

    return W_values


def zernike_plane(zcof, W_values):
    W = np.zeros((512, 512))

    for i in range(zcof.shape[-1]):
        W += zcof[i] * W_values[:, :, i]

    return W


def cost_func(Az, image):
    phi_x = complex(0, -1) * 2 * np.pi * fftshift(Az)
    Px = np.exp(phi_x)
    cor_img = ifft2(fft2(image) * Px)

    entropy_value = image_entropy(cor_img)

    return entropy_value


def add_noise(img, g_var=0.005, s_var=0.1):
    img_noise = random_noise(img, mode='gaussian', var=g_var)
    return random_noise(img_noise, mode='speckle', var=s_var)

if __name__ == '__main__':
    # np.random.seed(2022)
    no_terms = 1
    # A_true = load_zernike_coefficients(no_terms = no_terms)

    A_true = np.ones(1)
    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    img_noise = add_noise(img_gray)

    fig, ax = plt.subplots(1,3, figsize = (16,9))

    ab_img = aberrate(img_noise,A_true)
    # ab_img = ifft2(fft2(img_noise) * po)

    # ab_img = normalize_image(cj_img)
    # # new = ifft2(fft2(ab_img) * px)
    ax[0].imshow(normalize_image(ab_img))
    ax[0].axis('off')

    Px = correct_img(ab_img,A_true)
    # cor_img = ifft2(fft2(ab_img) * px)

    # cj_img = normalize_image(cj_img)
    new = ifft2(fft2(ab_img) * fftshift(Px))
    cor_img = normalize_image(new)

    ax[1].imshow(cor_img)
    ax[1].axis('off')

    ax[2].imshow(img_noise)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
