# -*- coding: utf-8 -*-
# @Time    : 2022-10-13 17:40
# @Author  : young wang
# @FileName: aberration_spsa.py
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


class spsa:
    def __init__(self, loss_function, a, c, alpha_val, gamma_val, max_iter, img, zernike):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.alpha_val = alpha_val
        self.gamma_val = gamma_val

        self.loss = loss_function

        self.max_iter = max_iter
        self.A = max_iter / 10
        self.img = img
        self.zernike = zernike

    def calc_loss(self, current_Aw):
        Az = self.update_AZ(current_Aw)
        """Evalute the cost/loss function with a value of theta"""
        return self.loss(Az, self.img)

    def update_AZ(self, current_Aw):
        return zernike_plane(current_Aw, self.zernike)

    def minimise(self, current_Aw):
        k = 0  # initialize count

        cost_func_val = []
        while k < self.max_iter:
            # get the current values for gain sequences
            a_k = self.a / (k + 1 + self.A) ** self.alpha_val
            c_k = self.c / (k + 1) ** self.gamma_val

            # get the random perturbation vector Bernoulli distribution with p=0.5
            delta = (np.random.randint(0, 2, current_Aw.shape) * 2 - 1)

            Aw_plus = current_Aw + delta * c_k
            Aw_minus = current_Aw - delta * c_k

            # measure the loss function at perturbations
            loss_plus = self.calc_loss(Aw_plus)
            loss_minus = self.calc_loss(Aw_minus)

            # compute the estimate of the gradient
            g_hat = (loss_plus - loss_minus) / (2.0 * delta * c_k)

            # update the estimate of the parameter
            current_Aw += - a_k * g_hat

            cost_val = self.calc_loss(current_Aw)
            cost_func_val.append(cost_val.squeeze())

            k += 1

        return current_Aw, cost_func_val


def aberrate(img=None, abe_coes=None):
    W_temp = construct_zernike(abe_coes, image=img)
    W = zernike_plane(abe_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * fftshift(W)
    phi_x = np.conj(phi_o)
    #
    Po = np.exp(phi_o)
    Px = np.exp(phi_x)
    #
    zernike_val = W / np.max(W)
    #
    ab_img = apply_wavefront(img, Po)
    cj_img = remove_wavefront(ab_img, Px)

    return zernike_val, Po, Px, normalize_image(ab_img), normalize_image(cj_img)


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


def load_zernike_coefficients(no_terms):
    sigma, mu = np.random.random(size=2)
    A = mu + sigma * 10 * np.random.random(size=no_terms)
    return A


def construct_zernike(z, image=None):
    N = image.shape[0]

    W_values = np.zeros((N, N, z.shape[-1]))

    [x, y] = np.meshgrid(np.linspace(-N / 2, N / 2, N),
                         np.linspace(-N / 2, N / 2, N), indexing='ij')

    r = np.sqrt(x ** 2 + y ** 2) / N
    theta = np.arctan2(y, x)

    for i in range(z.shape[-1]):
        W_values[:, :, i] = zernike(int(i + 1), r, theta)

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


def correct_image(img=None, cor_coes=None):
    W_temp = construct_zernike(cor_coes, image=img)
    W = zernike_plane(cor_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * fftshift(W)
    phi_x = np.conj(phi_o)

    Px = np.exp(phi_x)

    return normalize_image(remove_wavefront(ab_img, Px))


if __name__ == '__main__':
    np.random.seed(14)
    no_terms = 10
    A_true = load_zernike_coefficients(no_terms = no_terms)

    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    img_noise = add_noise(img_gray)

    _, _, _, ab_img, _ = aberrate(img_noise, A_true)

    Zo = construct_zernike(A_true, img)
    A_initial = np.random.random(size=no_terms)

    optimizer = spsa(loss_function=cost_func,
                     a=9e-1, c=1.0,
                     alpha_val=4,
                     gamma_val=1,
                     max_iter=2000, img=ab_img, zernike=Zo)
    #
    A_estimate, costval = optimizer.minimise(A_initial)
    print('done')
    #
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(ab_img, vmin=0, vmax=1)

    ax[0, 0].axis('off')
    ax[0, 0].set_title('aberrant image')

    cor_img = correct_image(img=ab_img, cor_coes=A_estimate)
    ax[0, 1].imshow(cor_img, vmin=0, vmax=1)

    ax[0, 1].axis('off')
    ax[0, 1].set_title('corrected image')

    x_axis = np.linspace(0, A_estimate.shape[-1], A_estimate.shape[-1])

    ax[1, 0].bar(x_axis + .25, A_estimate.squeeze(), width=0.5, label='guess value')
    ax[1, 0].bar(x_axis - 0.25, A_true.squeeze(), width=0.5, label='ground truth')

    ax[1, 0].legend()
    ax[1, 0].set_xlabel('zernike terms')
    ax[1, 0].set_ylabel('numerical values')

    image_entropy(img_gray)
    ax[1, 1].plot(np.arange(len(costval)), costval)
    # ax[1,1].axhline(y= image_entropy(img_gray), xmin=0, xmax=1,color = 'red',linestyle ='--' )
    ax[1, 1].set_xlabel('iterations')
    ax[1, 1].set_ylabel('cost function values')
    fig.suptitle('SPSA based computational adaptive optics(CAO)')

    plt.tight_layout()
    plt.show()
