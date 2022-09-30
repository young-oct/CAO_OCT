# -*- coding: utf-8 -*-
# @Time    : 2022-09-20 17:36
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm


import torch
from scipy.special import gamma
from torch.fft import fftshift, ifftshift, fft2, ifft2
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import glob
import numpy as np
from torch import nn
from torch.functional import F


def aberrate(img=None, abe_coes=None):
    img = torch.from_numpy(img)
    N = img.shape[0]
    img = img / torch.max(img)
    [x, y] = torch.meshgrid(torch.linspace(-N / 2, N / 2, N),
                         torch.linspace(-N / 2, N / 2, N),indexing='ij')

    r = torch.sqrt(x ** 2 + y ** 2) / N
    theta = torch.arctan2(y, x)

    z = torch.clone(abe_coes)

    W_values = torch.zeros((r.shape[0], r.shape[-1], z.shape[-1]))
    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1), r, theta)
    W = torch.sum(W_values, axis=-1)
    #
    # # shift zero frequency to align with wavefront
    phi_o = (complex(0, 1) * 2 * torch.pi * fftshift(W))
    phi_x = torch.conj(phi_o)
    #
    Po = torch.exp(phi_o)
    Px = torch.exp(phi_x)
    #
    zernike_plane = W / torch.max(W)
    zernike_plane = zernike_plane
    #
    ab_img = apply_wavefront(img, Po)
    cj_img = remove_wavefront(ab_img, Px)

    return zernike_plane, Po, Px, normalize_image(ab_img), normalize_image(cj_img)
    #

def normalize_psf(psf):
    h = fft2(psf)
    psf_norm = abs(h) ** 2
    return psf_norm / torch.max(psf_norm)


def normalize_image(image):
    return abs(image) / torch.max(abs(image))


def apply_wavefront(img=None, Po=None):
    # img = torch.from_numpy(img)
    return ifft2(fft2(img) * Po)


def remove_wavefront(aberrant_img=None, Px=None):
    # apply wavefront conjugation to the aberration image in
    # frequency domain
    return ifft2(fft2(aberrant_img) * Px)


def zernike_index(j=None, k=None):
    j -= 1
    n = int(torch.floor((- 1 + torch.sqrt(torch.tensor( 8 * j + 1))) / 2))
    i = j - n * (n + 1) / 2 + 1
    m = torch.tensor(i) - torch.remainder(torch.tensor( n + i), 2)
    l = torch.pow(-1, torch.tensor(k)) * m
    return n, l


def zernike(i: object = None, r: object = None, theta: object = None) -> object:
    n, m = zernike_index(i, i + 1)

    if n == -1:
        zernike_poly = torch.ones(r.shape)
    else:

        temp = torch.tensor(n + 1)
        if m == 0:
            zernike_poly = torch.sqrt(temp) * zernike_radial(n, 0, r)
        else:
            if np.mod(i, 2) == 0:
                zernike_poly = torch.sqrt(2 * temp) * zernike_radial(n, m, r) * torch.cos(m * theta)
            else:
                zernike_poly = torch.sqrt(2 * temp) * zernike_radial(n, m, r) * torch.sin(m * theta)

    return zernike_poly

def zernike_radial(n=None, m=None, r=None):
    R = 0
    for s in torch.arange(0, ((n - m) / 2) + 1):
        num = torch.pow(-1, s) * gamma(n - s + 1)
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
    # temp = image.numpy()
    # normalize image with l2
    img_mag = F.normalize(image, p=2.0, eps=1e-12)
    # img_mag = preprocessing.normalize(abs(temp), norm='l2')

    entropy = 0
    for i in range(img_mag.shape[0]):
        for j in range(img_mag.shape[1]):
            intensity_value = img_mag[i, j]
            if intensity_value != 0:  # avoid log(0) error/warning
                entropy = entropy - (intensity_value * torch.log(intensity_value))
            else:
                pass

    return abs(entropy)


def load_zernike_coefficients():
    torch.manual_seed(0)
    sigma, mu = torch.distributions.Uniform(0, 2).sample((2,))
    z_cos = mu + sigma * torch.distributions.Uniform(0, 1).sample((10,))
    return z_cos

def construct_zernike(zcof, image=None):

    z = torch.clone(zcof)
    N = image.shape[0]

    W_values = torch.zeros((N, N, z.shape[-1]))

    [x, y] = torch.meshgrid(torch.linspace(-N / 2, N / 2, N),
                         torch.linspace(-N / 2, N / 2, N), indexing='ij')

    r = torch.sqrt(x ** 2 + y ** 2) / N
    theta = torch.arctan2(y, x)

    for i in range(z.shape[-1]):
        W_values[:, :, i] = z[i] * zernike(int(i + 1), r, theta)
    W = torch.sum(W_values, axis=-1)
    # phi_x = (complex(0, -1) * 2 * torch.pi * fftshift(W))
    # Px = torch.exp(phi_x)

    return W

if __name__ == '__main__':


    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)


    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    abe_coes = load_zernike_coefficients()
    img = torch.zeros((512,512))
    W = construct_zernike(abe_coes,img)

    N = 512
    [x, y] = torch.meshgrid(torch.linspace(-N / 2, N / 2, N),
                            torch.linspace(-N / 2, N / 2, N),indexing='ij')

    r = torch.sqrt(x ** 2 + y ** 2) / N
    theta = torch.arctan2(y, x)

    W_values = torch.zeros((N, N, abe_coes.shape[-1]))
    for i in range(abe_coes.shape[-1]):
        W_values[:, :, i] = zernike(int(i + 1), r, theta)

    plt.imshow(W)
    plt.show()

    plt.imshow(W_values[:,:,4])
    plt.show()
    # b = W_values.T