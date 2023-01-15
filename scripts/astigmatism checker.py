# -*- coding: utf-8 -*-
# @Time    : 2022-09-05 19:59
# @Author  : young wang
# @FileName: astigmatism checker.py
# @Software: PyCharm

from tools.proc import surface_index,mip_stack
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tools.pre_proc import load_from_oct_file
from tools.pos_proc import image_export
from natsort import natsorted
from tools import plot
import discorpy.prep.preprocessing as prep

"""
check astigmatism and defocusing following method described here 
Lu, Y., Zhang, X., &amp; Li, H. (2018). 
A simplified focusing and astigmatism correction method for 
a scanning electron microscope.
AIP Advances, 8(1), 015124. https://doi.org/10.1063/1.5009683 
"""

if __name__ == '__main__':

    data_path = '../data/2002.10.19(MEEI)\corrected\*.oct'
    data_sets = natsorted(glob.glob(data_path))

    for volume in range(len(data_sets)):
        data = load_from_oct_file(data_sets[volume])

        idx = surface_index(data)[-1][-1]
        p_factor = 0.53
        vmin, vmax = p_factor * 255, 255

        img_list, tit_list = [], []

        img_ori = mip_stack(data, index=idx, thickness=idx)
        img_list.append(img_ori)
        tit_list.append('original image')

        img_norm = prep.normalization_fft(img_ori, sigma=0.5)
        img_list.append(img_norm)
        tit_list.append('denoised image')
    
        exportFile = "D:/Repos/CAO_OCT/CAO_OCT/calibration images/calibrated_" + str(volume) + ".png"
        image_export(img_norm, vmin, vmax, exportFile)

    # threshold = prep.calculate_threshold(img_norm, bgr="dark", snr=0.05)
    # img_bin = prep.binarization(img_norm, ratio=0.05, thres=threshold)
    # img_list.append(img_bin)
    # tit_list.append('binary image')
    #
    # img_fft = np.fft.fftshift(np.fft.fft2(img_bin))
    # img_fft_log = 20*np.log10(abs(img_fft))
    #
    # img_bw = np.where(img_fft_log <= img_fft_log.mean(), 0, 1)
    #
    # img_list.append(img_bw)
    # tit_list.append('fft log image')
    #
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10),constrained_layout=True)
    # for n, (ax, image, title) in enumerate(zip(axs.flat, img_list, tit_list)):
    #
    #     ax.imshow(image, 'gray', vmin=np.mean(image), vmax=np.max(image))
    #     ax.set_title(title)
    #     ax.set_axis_off()
    #
    # figure_folder = '../figure'
    # if not os.path.isdir(figure_folder):
    #     os.mkdir(figure_folder)
    # fig_path = os.path.join(figure_folder,'figure1.jpeg')
    # plt.savefig(fig_path, format='jpeg',
    #     bbox_inches=None, pad_inches=0)
    # plt.show()