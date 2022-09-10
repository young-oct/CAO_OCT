# -*- coding: utf-8 -*-
# @Time    : 2022-09-09 18:33
# @Author  : young wang
# @FileName: MTf.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from sklearn import preprocessing
import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib
import cv2 as cv
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from natsort import natsorted
import glob


def Zernike_polar(coefficients, radius, theta):
    z = coefficients

    z1 = z[0] * 1 * (np.cos(theta) ** 2 + np.sin(theta) ** 2)
    z2 = z[1] * 2 * radius * np.cos(theta)
    z3 = z[2] * 2 * radius * np.sin(theta)
    z4 = z[3] * np.sqrt(3) * (2 * radius ** 2 - 1)
    z5 = z[4] * np.sqrt(6) * radius ** 2 * np.sin(2 * theta)
    z6 = z[5] * np.sqrt(6) * radius ** 2 * np.cos(2 * theta)
    z7 = z[6] * np.sqrt(8) * (3 * radius ** 2 - 2) * radius * np.sin(theta)
    z8 = z[7] * np.sqrt(8) * (3 * radius ** 2 - 2) * radius * np.cos(theta)

    zw = z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8
    return zw


def pupil_size(D,lam,pix,size):
    pixrad = pix*np.pi/(180*3600)  # Pixel-size in radians
    nu_cutoff = D/lam      # Cutoff frequency in rad^-1
    deltanu = 1./(size*pixrad)     # Sampling interval in rad^-1
    rpupil = nu_cutoff/(2*deltanu) #pupil size in pixels
    return int(rpupil)


def phase(coefficients, rpupil):
    r = 1
    x = np.linspace(-r, r, 2 * rpupil)
    y = np.linspace(-r, r, 2 * rpupil)

    [X, Y] = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    Z = Zernike_polar(coefficients, R, theta)
    Z[R > 1] = 0
    return Z

def center(coefficients,size = 512,rpupil = 102):
    A = np.zeros([size,size])
    A[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= phase(coefficients,rpupil)
    return A


def mask(rpupil, size):
    r = 1
    x = np.linspace(-r, r, 2*rpupil)
    y = np.linspace(-r, r, 2*rpupil)

    [X,Y] = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    theta = np.arctan2(Y, X)
    M = 1*(np.cos(theta)**2+np.sin(theta)**2)
    M[R>1] = 0
    Mask =  np.zeros([size,size])
    Mask[size//2-rpupil+1:size//2+rpupil+1,size//2-rpupil+1:size//2+rpupil+1]= M
    return Mask

def complex_pupil(A,Mask):
    abbe =  np.exp(1j*A)
    abbe_z = np.zeros((len(abbe),len(abbe)),dtype=np.complex)
    abbe_z = Mask*abbe
    return abbe_z

def PSF(complx_pupil):
    PSF = ifftshift(fft2(fftshift(complx_pupil)))
    PSF = (np.abs(PSF))**2 #or PSF*PSF.conjugate()
    PSF = PSF/PSF.sum() #normalizing the PSF
    return PSF

def OTF(psf):
    otf = ifftshift(psf) #move the central frequency to the corner
    otf = fft2(otf)
    otf_max = float(otf[0,0]) #otf_max = otf[size/2,size/2] if max is shifted to center
    otf = otf/otf_max #normalize by the central frequency signal
    return otf

def MTF(otf):
    mtf = np.abs(otf)
    return mtf


def ft2(g):
    G = fftshift(fft2(ifftshift(g)))
    return G


def ift2(G):
    numPixels = G.shape[0]

    g = fftshift(ifft2(ifftshift(G)))
    return g


def conv2(g1, g2):
    G1 = ft2(g1)
    G2 = ft2(g2)
    G_out = G1 * G2

    numPixels = g1.shape[0]

    g_out = ift2(G_out)
    return g_out

if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    coefficients = np.zeros(8)
    coefficients[1] = 0.1
    coefficients[2] = 0
    coefficients[3] = 0.2
    coefficients[4] = 0

    D = 140  # diameter of the aperture
    lam = 617.3 * 10 ** (-6)  # wavelength of observation
    pix = 0.5  # plate scale
    f = 4125.3  # effective focal length
    size = 512  # size of detector in pixels

    # coefficients = np.zeros(8)
    # coefficients[1] = 0.1
    # coefficients[2] = 0.1
    # coefficients[3] = 0.3
    # coefficients[4] = 0.2
    # coefficients[5] = 0.8
    # coefficients[6] = 0.3
    # coefficients[7] = 0.2

    rpupil = pupil_size(D, lam, pix, size)
    sim_phase = center(coefficients, size, rpupil)
    Mask = mask(rpupil, size)

    pupil_com = complex_pupil(sim_phase, Mask)
    psf = PSF(pupil_com)
    otf = OTF(psf)
    mtf = MTF(otf)

    image_plist = glob.glob('../test_image/*.png')
    img = cv.imread(image_plist[-1])
    img_gray = rgb2gray(img)

    a = conv2(img_gray**2,abs(mtf)**2)
    c = abs(a)/np.max(abs(a))
    plt.imshow(abs(c))
    plt.show()
