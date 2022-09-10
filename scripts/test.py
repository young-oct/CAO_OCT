# -*- coding: utf-8 -*-
# @Time    : 2022-09-09 16:17
# @Author  : young wang
# @FileName: test.py
# @Software: PyCharm

from scipy.special import gamma, factorial
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
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import glob

def zernike_index(j,k):
    j = j-1
    n = np.floor(-1+np.sqrt(8*j)/2)
    i = j- n*(n-1)/2+1
    m = i-np.mod(n+i,2)
    l = np.power(-1, k*m)
    return m,l

def zernike(noll_no, r, theta):
    n,m = zernike_index(noll_no, noll_no+1)

    if m == 0:
        zernike_poly = np.sqrt(n+1)*zernike_radial(n,0,r)
    else:
        if np.mod(m,2) == 0:
            zernike_poly = np.sqrt(2*(n + 1))* zernike_radial(n, 0, r) * np.cos(m*theta)
        else:
            zernike_poly = np.sqrt(2*(n + 1)) * zernike_radial(n, 0, r) * np.sin(m * theta)
    return zernike_poly

def zernike_radial(n,m, r):
    R = 0
    for s in range(int(np.ptp(np.arange((n-m)/2)))):
        num = np.power(-1,s) * gamma(n-s+1)
        denom = gamma(s + 1) * gamma((n + m) / 2 - s + 1)\
                * gamma((n - m) / 2 - s + 1)

        R = R + num / denom * np.power(r, (n-2*s))
    return R

def  aberrate(img,defocus,ast,coma,sph, D = 102.4 ):
    N = img.shape[0]
    [x, y] = np.meshgrid(np.linspace(-N//2, N//2 + 1,N),
                         np.linspace(-N//2, N//2 + 1,N))

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    W = defocus*zernike(4,2*r/D,theta)+\
        ast*zernike(5,2*r/D,theta) +\
        ast*zernike(6,2*r/D,theta) +\
        coma*zernike(7,2*r/D,theta) + \
        coma*zernike(8,2*r/D,theta) + \
        sph*zernike(11,2*r/D,theta)

    W_complex = np.exp(complex(0,1) * 2 * np.pi * W)

    P = circ(img,D) * W_complex

    h = ft2(P)
    PSF = abs(h) ** 2

    aberrated_img = myconv2(PSF,img)
    return aberrated_img

def myconv2(A,B):
    return ift2(ft2(A) * ft2(B))

def ift2(f):

    return ifftshift(ifft2(ifftshift(f)))

def ft2(f):
    return fftshift(fft2(fftshift(f)))

def circ(x,pupil):

    pupil_plane = np.empty_like(x)
    x_cent, y_cent = x.shape[0]//2, x.shape[1]//2
    for i in range(pupil_plane.shape[0]):
        for j in range(pupil_plane.shape[1]):
            if np.sqrt((i-x_cent) ** 2 + (j-y_cent) ** 2) <= pupil:

                pupil_plane[i,j] = 1
            else:
                pupil_plane[i,j] = 0

    return pupil_plane

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
    img = cv.imread(image_plist[0])
    img_gray = rgb2gray(img)

    defocus, ast, coma, sph = 0.2,-1,0.5,-0.4

    PSF = aberrate(img_gray, defocus, ast, coma, sph, D=100)
    plt.imshow(abs(PSF),'gray')
    plt.show()

