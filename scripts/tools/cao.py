# -*- coding: utf-8 -*-
# @Time    : 2022-12-01 21:12
# @Author  : young wang
# @FileName: cao.py
# @Software: PyCharm

import numpy as np
from scipy.special import gamma
from scipy.fftpack import fftshift, fft2, ifft2
from scipy import ndimage

class optimization:
    def __init__(self, loss_function,
                 a,
                 c,
                 alpha_val,
                 gamma_val,
                 max_iter,
                 img_target,
                 zernike,
                 momentum=0.2,
                 cal_tolerance=1e-6):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.alpha_val = alpha_val
        self.gamma_val = gamma_val

        self.loss = loss_function

        self.max_iter = max_iter
        self.A = max_iter / 10
        self.img_target = img_target
        self.zernike = zernike

        self.initial_entropy = image_entropy(self.img_target)
        self.momentum = momentum
        self.cal_tolerance = cal_tolerance

    def calc_loss(self, current_Aw):
        # print('current_Aw: %s' %str(current_Aw))

        Az = self.update_AZ(current_Aw)
        # print(Az[0][0])
        # print('Az: %s' %str(Az))
        """Evaluate the cost/loss function with a value of theta"""
        return self.loss(Az, self.img_target)

    def update_AZ(self, current_Aw):
        return zernike_plane(current_Aw, self.zernike)

    def minimizer(self, current_Aw, optimizer_type='spgd',
                  beta1=0.9,
                  beta2=0.99,
                  epsilon=1e-8,
                  verbose=False):
        k = 0  # initialize count
        cost_func_val = []
        Aw_values = []
        vk = 0

        previous_Aw = 0

        # paramaters for the adam method
        beta1 = beta1
        beta2 = beta2
        epsilon = epsilon

        adam_m = 0
        adam_v = 0

        # previous_ghat = 0

        while k < self.max_iter and \
                np.linalg.norm(previous_Aw - current_Aw) > self.cal_tolerance:

            previous_Aw = current_Aw
            cost_val = self.calc_loss(current_Aw)

            if verbose:
                print('iteration %d: %s with cost function value %.2f' % (k, str(current_Aw), cost_val))
            else:
                pass

            # get the random perturbation vector Bernoulli distribution with p=0.5
            delta = (np.random.randint(0, 2, current_Aw.shape) * 2 - 1) * self.gamma_val

            Aw_plus = current_Aw + delta
            Aw_minus = current_Aw - delta

            # measure the loss function at perturbations
            loss_plus = self.calc_loss(Aw_plus)
            loss_minus = self.calc_loss(Aw_minus)

            # compute the estimate of the gradient

            loss_delta = (loss_plus - loss_minus) / 2
            g_hat = loss_delta * delta

            # compute the estimate of the gradient
            # g_hat = loss_delta * delta
            # delta_g = g_hat - previous_ghat
            # print(k,g_hat, delta_g/delta)
            # print((k,np.diff((g_hat - previous_ghat),delta)))

            # previous_ghat = g_hat
            # # update the estimate of the parameter
            if optimizer_type == 'spgd':

                # compute the estimate of the gradient
                current_Aw = current_Aw - self.alpha_val * g_hat

            elif optimizer_type == 'spgd-momentum':

                # compute the estimate of the gradient

                vk_next = self.alpha_val * g_hat + self.momentum * vk
                current_Aw = current_Aw - vk_next
                vk = vk_next

            elif optimizer_type == 'spgd-adam':

                adam_m = beta1 * adam_m + (1 - beta1) * g_hat
                adam_v = beta2 * adam_v + (1 - beta2) * (g_hat ** 2)

                # bias correction for both first and second moments
                alpha_updated = self.alpha_val * np.sqrt(1 - beta2 ** (k + 1)) / (1 - beta1 ** (k + 1))
                current_Aw = current_Aw - alpha_updated * adam_m / (np.sqrt(adam_v) + epsilon)


            elif optimizer_type == 'spsa':

                # get the current values for gain sequences
                a_k = self.a / (k + 1 + self.A) ** self.alpha_val
                c_k = self.c / (k + 1) ** self.gamma_val

                current_Aw = current_Aw - a_k * (g_hat * c_k)

            else:
                raise ValueError('please input the right optimizer')

            cost_val = self.calc_loss(current_Aw)
            cost_func_val.append(np.log10(cost_val.squeeze()))
            Aw_values.append(current_Aw)

            k += 1

        sol_idx = np.argmin(cost_func_val)
        Aw_estimate = Aw_values[sol_idx]
        # print('optimal solution is found at %d iteration' % sol_idx)

        return Aw_estimate, cost_func_val, sol_idx


def aberrate(img_target=None, abe_coes=None):
    W_temp = construct_zernike(abe_coes, N=img_target.shape[0])
    W = zernike_plane(abe_coes, W_temp)
    # # shift zero frequency to align with wavefront
    phi_o = complex(0, 1) * 2 * np.pi * W
    Po = np.exp(phi_o)

    return apply_wavefront(img_target=img_target, z_poly=Po)


def normalize_image(img_target):
    return abs(img_target) / np.max(abs(img_target))


def apply_wavefront(img_target=None, z_poly=None):
    return ifft2(fft2(img_target) * fftshift(z_poly))
    # return ifft2(fft2(img_target) * z_poly)


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
    W = np.zeros((W_values.shape[0], W_values.shape[1]))

    for i in range(zcof.shape[-1]):
        W += zcof[i] * W_values[:, :, i]

    return W


def load_zernike_coefficients(no_terms, A_true_flat, repeat=False):
    if A_true_flat and repeat == False:
        np.random.seed(20)
        sigma, mu = np.random.random(size=2)
        A = mu + sigma * 20 * np.random.random(size=no_terms)
    elif not repeat:
        np.random.seed(None)
        A = np.random.random(size=no_terms)
    else:
        np.random.seed(2022)
        A = np.random.random(size=no_terms)
        factor = np.random.randint(1, 5)
        A = A*factor

    return A


def correct_image(img_target=None, cor_coes=None):
    W_temp = construct_zernike(cor_coes, N=img_target.shape[0])
    W = zernike_plane(cor_coes, W_temp)

    phi_x = complex(0, -1) * 2 * np.pi * W
    Px = np.exp(phi_x)

    return apply_wavefront(img_target=img_target, z_poly=Px)


def image_entropy(img_target=None):
    """
    implementation of image entropy using Eq(10)
    from the reference[1]
    :param image: 2D array
    :return: entropy of the image
    """

    # temp_img = ndimage.median_filter(complex2int(img_target), size=3)
    temp_img = ndimage.median_filter(abs(img_target), size=3)

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
    phi_x = complex(0, -1) * 2 * np.pi * Az

    Px = np.exp(phi_x)

    cor_img = apply_wavefront(img_target=image, z_poly=Px)
    return image_entropy(cor_img)


def square_crop(image):
    raduis = image.shape[0] // 2
    x_c, y_c = image.shape[0] // 2, image.shape[1] // 2

    lenght = int(raduis / np.sqrt(2))

    return image[int(x_c - lenght):int(x_c + lenght),
           int(y_c - lenght):int(y_c + lenght)]


def inten2pixel(image):
    temp = 20 * np.log10(abs(image))
    return (temp - np.min(temp)) / np.ptp(temp)


def complex2int(Aline_vol):
    """
    converts complex volume into intensity volume for imshow
    :param Aline_vol: 3D array
    :return: normalized to [0,1] intensity volume
    """
    mag_vol = abs(Aline_vol)
    return (mag_vol - np.min(mag_vol)) / np.ptp(mag_vol)