# -*- coding: utf-8 -*-
# @Time    : 2022-11-02 17:55
# @Author  : young wang
# @FileName: aberration_spgd.py
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
    def __init__(self, loss_function,
                 # a, c,
                 alpha_val,
                 gamma_val, max_iter, img_target, zernike,
                 # momentum = 0.2,
                 cal_tolerance = 1e-6):
        # Initialize gain parameters and decay factors
        # self.a = a
        # self.c = c
        self.alpha_val = alpha_val
        self.gamma_val = gamma_val

        self.loss = loss_function

        self.max_iter = max_iter
        self.A = max_iter / 10
        self.img_target = img_target
        self.zernike = zernike

        self.initial_entropy = image_entropy(self.img_target)
        # self.momentum = momentum
        self.cal_tolerance = cal_tolerance

    def calc_loss(self, current_Aw):
        Az = self.update_AZ(current_Aw)
        """Evalute the cost/loss function with a value of theta"""
        return self.loss(Az, self.img_target)

    def update_AZ(self, current_Aw):
        return zernike_plane(current_Aw, self.zernike)

    def minimise(self, current_Aw):
                 # optimizer_type='vanilla'):
        k = 0  # initialize count

        cost_func_val = []
        Aw_values = []
        vk = 0

        previous_Aw = 0

        while k < self.max_iter and \
                np.linalg.norm(previous_Aw - current_Aw) > self.cal_tolerance:

            previous_Aw = current_Aw
            # get the current values for gain sequences
            # a_k = self.a / (k + 1 + self.A) ** self.alpha_val
            # c_k = self.c / (k + 1) ** self.gamma_val

            # get the random perturbation vector Bernoulli distribution with p=0.5
            delta_intermediate = (np.random.randint(0, 2, current_Aw.shape) * 2 - 1)
            delta = delta_intermediate/np.max(delta_intermediate)

            Aw_plus = current_Aw + delta * self.gamma_val
            Aw_minus = current_Aw - delta * self.gamma_val

            # measure the loss function at perturbations
            loss_plus = self.calc_loss(Aw_plus)
            loss_minus = self.calc_loss(Aw_minus)

            # loss_delta = (loss_plus - loss_minus) / 2.0
            # Aw_delta = Aw_plus - Aw_minus

            # compute the estimate of the gradient
            g_hat = (loss_plus - loss_minus) * delta / np.abs(delta)

            current_Aw = current_Aw - self.alpha_val * g_hat

            # update the estimate of the parameter
            # if optimizer_type == 'vanilla':
            #     current_Aw = current_Aw - a_k * g_hat
            # elif optimizer_type == 'momentum':
            #
            #     vk_next = a_k * g_hat + self.momentum * vk
            #     current_Aw = current_Aw - vk_next
            #     vk = vk_next
            # elif optimizer_type == 'SPGD':
            #     current_Aw = current_Aw + loss_delta * Aw_delta
            #
            cost_val = self.calc_loss(current_Aw)
            cost_func_val.append(cost_val.squeeze())
            Aw_values.append(current_Aw)
            #
            k += 1

        return current_Aw, cost_func_val, Aw_values


def aberrate(img_target=None, abe_coes=None):
    W_temp = construct_zernike(abe_coes, N=512)
    W = zernike_plane(abe_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * W
    Po = np.exp(phi_o)

    return apply_wavefront(img_target=img_target, z_poly=Po)


def normalize_image(img_target):
    return abs(img_target) / np.max(abs(img_target))


def apply_wavefront(img_target=None, z_poly=None):
    return ifft2(fft2(img_target) * fftshift(z_poly))


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


def construct_zernike(z, N=512):
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


def load_zernike_coefficients(no_terms, A_true_flat):
    if A_true_flat:
        np.random.seed(20)
        sigma, mu = np.random.random(size=2)
        A = mu + sigma * 20 * np.random.random(size=no_terms)
    else:
        np.random.seed(None)
        A = np.random.random(size=no_terms)
    return A


def correct_image(img_target=None, cor_coes=None):
    W_temp = construct_zernike(cor_coes, N=512)
    W = zernike_plane(cor_coes, W_temp)

    phi_x = complex(0, -1) * 2 * np.pi * W
    Px = np.exp(phi_x)

    return apply_wavefront(img_target=img_target, z_poly=Px)


def add_noise(img, g_var=0.005, s_var=0.1):
    img_noise = random_noise(img, mode='gaussian', var=g_var)
    return random_noise(img_noise, mode='speckle', var=s_var)


def image_entropy(img_target=None):
    """
    implementation of image entropy using Eq(10)
    from the reference[1]
    :param image: 2D array
    :return: entropy of the image
    """
    temp_img = normalize_image(img_target)
    entropy = 0
    for i in range(temp_img.shape[0]):
        for j in range(temp_img.shape[1]):
            intensity_value = temp_img[i, j]
            if intensity_value != 0:  # avoid log(0) error/warning
                entropy -= intensity_value * np.log(intensity_value)
            else:
                pass

    return entropy


def cost_func(Az, image):
    phi_x = complex(0, -1) * 2 * np.pi * fftshift(Az)
    Px = np.exp(phi_x)

    cor_img = apply_wavefront(img_target=image, z_poly=Px)

    return image_entropy(cor_img)


if __name__ == '__main__':
    no_terms = 2
    A_true = load_zernike_coefficients(no_terms=no_terms,
                                       A_true_flat=True)

    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    # img_noise = add_noise(img_gray)

    ab_img = aberrate(img_gray, A_true)

    Zo = construct_zernike(A_true, N=512)

    A_initial = load_zernike_coefficients(no_terms=no_terms,
                                          A_true_flat=False)

    tolerance = 1e-6

    optimizer = spsa(loss_function=cost_func,
                     # a=9e-1, c=1.0,
                     # alpha_val=0.601,
                     # gamma_val=0.101,
                     alpha_val=2,
                     gamma_val=1,
                     max_iter=200,
                     img_target=ab_img,
                     zernike=Zo,
                     # momentum = 0.15,
                     cal_tolerance=tolerance)
    #
    # vanilla or momentum
    optimizer_type = 'momentum'
    A_estimate, costval, A_values = optimizer.minimise(current_Aw=A_initial)
                                                       # optimizer_type=optimizer_type)
    print('done')

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    ax[0, 0].imshow(normalize_image(ab_img), vmin=0, vmax=1)

    ax[0, 0].axis('off')
    ax[0, 0].set_title('aberrant image')

    cor_img = correct_image(img_target=ab_img, cor_coes=A_estimate)
    ax[0, 1].imshow(normalize_image(cor_img), vmin=0, vmax=1)

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
    fig.suptitle('SPSA based computational adaptive optics(CAO): %s' % optimizer_type)

    plt.tight_layout()
    plt.show()

    plt.plot(A_values)
    plt.show()
