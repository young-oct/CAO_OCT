# -*- coding: utf-8 -*-
# @Time    : 2022-09-15 12:58
# @Author  : young wang
# @FileName: gradient_descent.py
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
    j -= - 1
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
    img_mag = torch.abs(img_mag)
    # img_mag = preprocessing.normalize(abs(temp), norm='l2')

    entropy = 0
    for i in range(img_mag.shape[0]):
        for j in range(img_mag.shape[1]):
            intensity_value = img_mag[i, j]
            if intensity_value != 0:  # avoid log(0) error/warning
                entropy = entropy - (intensity_value * torch.log(intensity_value))
            else:
                pass

    return entropy


def load_zernike_coefficients(order=20, radmon_sel=True):
    if radmon_sel:
        torch.manual_seed(0)
        sigma, mu = torch.randint(order, size=(2,))
        z_cos = mu + sigma * torch.distributions.Uniform(0, 0.1).sample((10,))
    else:
        z_cos = torch.zeros(11)
        z_cos[0] = 1  # piston
        z_cos[1] = 0  # x tilt
        z_cos[2] = 1000  # y tilt
        z_cos[3] = 10  # defocus
        z_cos[4] = 1e4  # y primary astigmatism
        z_cos[5] = 0  # x primary astigmatism
        z_cos[6] = 0  # y primary coma
        z_cos[7] = 0  # x primary coma
        z_cos[8] = 0  # y trefoil
        z_cos[9] = 0.2  # x trefoil
        z_cos[10] = 1e5  # primary spherical

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
    phi_x = (complex(0, -1) * 2 * torch.pi * fftshift(W))
    Px = torch.exp(phi_x)

    return ifft2(fft2(image) * Px)


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        torch.manual_seed(0)
        weights = torch.distributions.Uniform(0, 1).sample((10,))
        weights.requires_grad=True
        # make weights torch parameters
        print(weights)
        self.weights = nn.Parameter(weights)

    def forward(self, image):

        zcof = self.weights
        return construct_zernike(zcof, image)


def training_loop(model, optimizer, image, n=10):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        c_image = model(image)
        loss = image_entropy(c_image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().numpy())
        print(i)
    return losses

if __name__ == '__main__':

    image_plist = glob.glob('../test images/*.png')

    # test image selector: -1: USA target; 0: dot target
    img_no = -1

    if img_no == -1 or img_no == 1:
        # simulate a low order, high value aberration for the USA target
        abe_coes = load_zernike_coefficients(order=5, radmon_sel=True)
    elif img_no == 0:
        # simulate a high order, high value aberration for the dot target
        abe_coes = load_zernike_coefficients(order=10, radmon_sel=False)
    else:
        raise ValueError('please input either 0 or -1 or 1')

    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    _, _, _, ab_img, _ = aberrate(img_noise, abe_coes)
    print('applied weights:',abe_coes.detach().numpy())
    m = Model()
    # Instantiate optimizer
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    losses = training_loop(m, opt, ab_img)
    print('done')
    est_coe = m.weights.detach().numpy()
    print('estimated weights:',est_coe)

    fig,ax = plt.subplots(1,2, figsize = (16,9))
    X_axis = np.arange(len(abe_coes))

    ax[0].bar(X_axis , abe_coes.detach().numpy(), 0.2, label='ground truth')
    initi_guess =[0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341,
                  0.4901, 0.8964, 0.4556,0.6323]
    ax[0].bar(X_axis + 0.2,initi_guess, 0.2, label='initial guess')
    ax[0].bar(X_axis + 0.4, m.weights.detach().numpy(), 0.2, label='estimation')

    ax[0].legend()
    ax[1].plot(abs(np.array(losses)))
    ax[1].set_title('loss function')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(16, 9),layout="constrained")
    ax[0].imshow(abs(img_noise))
    ax[0].set_title('original')
    ax[0].axis('off')

    ax[1].imshow(abs(ab_img))
    ax[1].set_title('aberrant image')
    ax[1].axis('off')

    cor_img = construct_zernike(m.weights, image=ab_img)
    ax[2].imshow(abs(cor_img.detach().numpy()))
    ax[2].set_title('corrected image')
    ax[2].axis('off')
    # plt.tight_layout()
    plt.show()

    # tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074, 0.6341, 0.4901, 0.8964, 0.4556,
    #         0.6323])
