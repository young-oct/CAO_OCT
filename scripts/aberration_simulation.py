# -*- coding: utf-8 -*-
# @Time    : 2022-09-08 15:55
# @Author  : young wang
# @FileName: aberration_simulation.py
# @Software: PyCharm

"""
Simulated aberrated USAF resolution target image
with 3rd-order Zernike polynomials based on the following
references

1. Dan Zhu, Ruoyan Wang, Mantas Žurauskas,
Paritosh Pande, Jinci Bi, Qun Yuan, Lingjie Wang,
Zhishan Gao, and Stephen A. Boppart,
"Automated fast computational adaptive optics for
optical coherence tomography based
on a stochastic parallel gradient descent algorithm,"
Opt. Express 28, 23306-23319 (2020)
"""

from scipy.special import gamma
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scripts.tools.plot import heatmap
import numpy as np
from sklearn import preprocessing
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import glob


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

    # shift zero frequency to align with wavefront
    phi_o = fftshift(complex(0, 1) * 2 * np.pi * W)
    phi_x = np.conjugate(phi_o)

    Po = np.exp(phi_o)
    Px = np.exp(phi_x)

    zernike_plane = W / np.max(W)
    zernike_plane = zernike_plane * circ(W, pupil=int(N / 2))

    ab_img = apply_wavefront(img, Po)
    cj_img = remove_wavefront(ab_img, Px)

    return zernike_plane, Po, Px, normalize_image(ab_img), normalize_image(cj_img)


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


def image_entropy(image=None):
    """
    implementation of image entropy using Eq(10)
    from the reference[1]
    :param image: 2D array
    :return: entropy of the image
    """

    # normalize image with l2
    img_mag = preprocessing.normalize(abs(image), norm='l2')

    entropy = 0
    for i in range(img_mag.shape[0]):
        for j in range(img_mag.shape[1]):
            intensity_value = img_mag[i, j]
            if intensity_value != 0:  # avoid log(0) error/warning
                entropy = entropy - (intensity_value * np.log(intensity_value))
            else:
                pass

    return entropy


def load_zernike_coefficients(order=20, radmon_sel=True):
    if radmon_sel:
        np.random.seed(2022)
        sigma, mu = np.random.randint(order, size=2)
        z_cos = mu + sigma * np.random.randn(20)
    else:
        z_cos = np.zeros(11)
        z_cos[0] = 1  # piston
        z_cos[1] = 0  # x tilt
        z_cos[2] = 1000  # y tilt
        z_cos[3] = 10  # defocus
        z_cos[4] = 1e4  # y primary astigmatism
        z_cos[5] = 0  # x primary astigmatism
        z_cos[5] = 0  # y primary coma
        z_cos[7] = 0  # x primary coma
        z_cos[8] = 0  # y trefoil
        z_cos[9] = 0.2  # x trefoil
        z_cos[10] = 1e5  # primary spherical

    return z_cos


if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    img_list = []
    title_list = []

    image_plist = glob.glob('../test images/*.png')

    # test image selector: -1: USA target; 0: dot target
    img_no = 2

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

    img_list.append(img_gray)
    title_list.append('original image')

    ori_img_fft = np.fft.fftshift(np.fft.fft2(img_gray))
    ori_img_fft_log = 20 * np.log10(abs(ori_img_fft))

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)

    img_list.append(img_noise)
    title_list.append('noisy image')

    zernike_plane, Po, Px, aberrant_img, conjugate_img \
        = aberrate(img_gray, abe_coes)
    img_list.append(aberrant_img)
    title_list.append('aberrant image')

    img_list.append(conjugate_img)
    title_list.append('recovered image')

    img_fft = np.fft.fftshift(np.fft.fft2(img_noise))
    img_fft_log = 20 * np.log10(abs(img_fft))

    img_list.append(ori_img_fft_log)
    title_list.append('fft image: original')

    img_list.append(img_fft_log)
    title_list.append('fft image: noisy')

    ab_img_fft = np.fft.fftshift(np.fft.fft2(aberrant_img))
    ab_img_fft_log = 20 * np.log10(abs(ab_img_fft))

    img_list.append(ab_img_fft_log)
    title_list.append('fft image: aberrant)')

    img_list.append(zernike_plane)
    title_list.append('Zernike plane')

    fig, axs = plt.subplots(2, 4,
                            figsize=(16, 9))
    for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, title_list)):

        if n == len(title_list) - 1:
            heatmap(title, image, ax)
            ax.set_axis_off()

        else:
            ax.imshow(image, 'gray', aspect=image.shape[1] / image.shape[0],
                      vmin=np.mean(image), vmax=np.max(image))
            ax.set_title(title)

            ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    gray_entropy = image_entropy(img_gray)
    noise_entropy = image_entropy(img_noise)
    aberrant_entropy = image_entropy(aberrant_img)
    recovered_entropy = image_entropy(conjugate_img)

    x = ['original\nimage', 'noisy\nimage',
         'aberrant\nimage', 'recovered\nimage']

    y = [gray_entropy, noise_entropy,
         aberrant_entropy, recovered_entropy]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.bar(x, y, width=0.5)
    ax.set_ylabel('entropy [a.u]')
    plt.title('image entropy')
    plt.tight_layout()
    plt.show()
