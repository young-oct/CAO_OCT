# -*- coding: utf-8 -*-
# @Time    : 2022-11-18 19:43
# @Author  : young wang
# @FileName: OCT_target(CAO).py
# @Software: PyCharm


import cv2 as cv
from tools.proc import filter_mask, \
    surface_index, sphere_fit, frame_index, max_slice, despecking, mip_stack
import glob

import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
import matplotlib
from skimage.morphology import (erosion, dilation, opening, closing,
                                white_tophat, disk, black_tophat, square, skeletonize)
from natsort import natsorted
import discorpy.prep.preprocessing as prep


def oct_enface(file_name, p_factor=0.25, shift=5, thickness = 20):
    en_face = {}
    data = load_from_oct_file(file_name)
    idx = surface_index(data)[-1][-1]
    p_factor = p_factor

    img_ori = mip_stack(data, index=int(330 - idx - shift), thickness=thickness)

    en_face['original'] = np.where(img_ori <= p_factor*np.max(img_ori),0, img_ori)

    img_norm = prep.normalization_fft(img_ori, sigma=0.5)

    en_face['denoised'] = np.where(img_norm <= p_factor*np.max(img_norm), 0,img_norm)

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


if __name__ == '__main__':

    matplotlib.rcParams.update(
        {
            'font.size': 20,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )


    dset_lst = ['../data/*.oct']
    data_sets = natsorted(glob.glob(dset_lst[-1]))

    a = oct_enface(file_name=data_sets[-1])
    b = a['feature']
