# -*- coding: utf-8 -*-
# @Time    : 2022-11-18 19:43
# @Author  : young wang
# @FileName: OCT_target(CAO).py
# @Software: PyCharm


from tools.proc import surface_index, mip_stack
from tools.pre_proc import load_from_oct_file
import matplotlib
from skimage.morphology import (erosion, dilation, opening, disk)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import copy
from scipy.special import gamma
from scipy.fftpack import fftshift, fft2, ifft2
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
import matplotlib.gridspec as gridspec


def oct_enface(file_name, p_factor=0.25, shift=5, thickness = 20):
    en_face = {}
    data = load_from_oct_file(file_name)
    idx = surface_index(data)[-1][-1]
    img_ori = mip_stack(data, index=int(330 - idx - shift), thickness=thickness)

    en_face['original'] = normalize_image(img_ori)

    img_norm = prep.normalization_fft(img_ori, sigma=0.5)

    en_face['denoised'] = normalize_image(img_norm)

    img_bin = prep.binarization(img_norm, ratio=0.005, thres=p_factor * 255)
    img_cls = opening(img_bin, disk(3))
    #
    # # mask to break down the connected pixels
    mask = erosion(img_cls, np.ones((21, 1)))
    mask = dilation(mask, np.ones((1, 3)))

    img_cls = cv.bitwise_xor(img_cls, mask)
    #
    en_face['feature'] = np.where(img_cls <= p_factor*np.max(img_cls), 0, img_cls)

    return en_face

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

    def minimizer(self, current_Aw, optimizer_type='vanilla',
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

        previous_ghat = 0

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
        W_values[:, :, i] = zernike(int((i + 2) + 1), r, theta)

    return W_values


def zernike_plane(zcof, W_values):
    W = np.zeros((512, 512))

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

    return A


def correct_image(img_target=None, cor_coes=None):
    W_temp = construct_zernike(cor_coes, N=512)
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
    phi_x = complex(0, -1) * 2 * np.pi * Az

    Px = np.exp(phi_x)

    cor_img = apply_wavefront(img_target=image, z_poly=Px)
    return image_entropy(cor_img)

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 16,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    dset_lst = ['../data/Ossiview intensity/*.oct']
    data_sets = natsorted(glob.glob(dset_lst[-1]))
    p_factor = 0.25
    imgs = oct_enface(file_name=data_sets[-1], p_factor=p_factor)
    ab_img = imgs['denoised']

    no_terms = 6
    A_initial = copy.deepcopy(load_zernike_coefficients(no_terms=no_terms,
                                                        A_true_flat=False, repeat=True))
    A_initial *= np.random.randint(10)

    Zo = construct_zernike(A_initial, N=512)

    alpha_val,gamma_val = 0.01, 0.05
    tolerance = gamma_val/100

    optimizer = optimization(loss_function=cost_func,
                             a=9e-1, c=1.0,
                             alpha_val=alpha_val,
                             gamma_val=gamma_val,
                             max_iter=500,
                             img_target=ab_img,
                             zernike=Zo,
                             momentum=0.15,
                             cal_tolerance=tolerance)

    optimizer_types = ['spgd','spgd-momentum','spgd-adam','spsa']

    optimizer_type = optimizer_types[2]

    A_estimate, costval, sol_idx = optimizer.minimizer(current_Aw=A_initial,
                                                       optimizer_type=optimizer_type,
                                                       beta1=0.9,
                                                       beta2=0.99,
                                                       verbose=False)

    img_cor = correct_image(ab_img, A_estimate)

    fig = plt.figure(figsize=([16, 9]))
    gs = gridspec.GridSpec(2, 6)

    ax1 = plt.subplot(gs[0, 0:2])

    ax1.imshow(normalize_image(ab_img), vmin=p_factor, vmax=1)
    ax1.axis('off')
    ax1.set_title('aberrant image')

    ax2 = plt.subplot(gs[0, 2:4])

    cor_img = correct_image(img_target=ab_img, cor_coes=A_estimate)
    ax2.imshow(normalize_image(cor_img), vmin=p_factor, vmax=1)
    ax2.axis('off')
    ax2.set_title('corrected image')

    ax3 = plt.subplot(gs[0, 4:6])

    ax3.imshow(normalize_image(ab_img-cor_img), vmin=p_factor, vmax=1)
    ax3.axis('off')
    ax3.set_title('difference image')


    x_axis = np.linspace(0, A_estimate.shape[-1], A_estimate.shape[-1])

    ax4 = plt.subplot(gs[1, 0:3])

    ax4.bar(x_axis - 0.15, A_initial.squeeze(), width=0.15, label='initial guess')
    ax4.bar(x_axis, A_estimate.squeeze(), width=0.15, label='estimated value')
    ax4.legend(loc='best')
    ax4.set_xlabel('zernike terms')
    ax4.set_ylabel('numerical values')

    ax5 = plt.subplot(gs[1, 3:6])

    ax5.plot(np.arange(len(costval)), costval)
    ax5.set_xlabel('iterations')
    ax5.set_ylabel('cost function values')

    fig.suptitle('%s based computational adaptive optics(CAO)\n'
                 'solution found at iteration %d' % (optimizer_type, sol_idx))

    plt.tight_layout()
    plt.show()
    print('done')