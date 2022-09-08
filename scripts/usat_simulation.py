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
import os
import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
from natsort import natsorted
import glob

if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    image_plist = glob.glob('../test_image/*.png')
    img = cv.imread(image_plist[-1])
    img_gray = rgb2gray(img)

    img_noise = random_noise(img_gray, mode='gaussian', var=0.005)
    img_noise = random_noise(img_noise, mode='speckle', var=0.1)
    plt.imshow(img_noise)
    plt.show()

    img_fft = np.fft.fftshift(np.fft.fft2(img_noise))
    img_fft_log = 20 * np.log10(abs(img_fft))
    plt.imshow(img_fft_log, 'gray')
    plt.show()
