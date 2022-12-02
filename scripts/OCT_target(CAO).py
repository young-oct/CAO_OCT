# -*- coding: utf-8 -*-
# @Time    : 2022-11-18 19:43
# @Author  : young wang
# @FileName: OCT_target(CAO).py
# @Software: PyCharm

from tools import cao
from tools.proc import surface_index, mip_stack
from tools.pre_proc import load_from_oct_file
import matplotlib
from skimage.morphology import (erosion, dilation, opening, disk)
from natsort import natsorted
import discorpy.prep.preprocessing as prep
import copy
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

    en_face['original'] = cao.normalize_image(img_ori)

    img_norm = prep.normalization_fft(img_ori, sigma=0.5)

    en_face['denoised'] = cao.normalize_image(img_norm)

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
    A_initial = copy.deepcopy(cao.load_zernike_coefficients(no_terms=no_terms,
                                                        A_true_flat=False, repeat=True))
    A_initial *= np.random.randint(1,5)

    Zo = cao.construct_zernike(A_initial, N=512)

    alpha_val,gamma_val = 0.01, 0.05
    tolerance = gamma_val/100

    optimizer = cao.optimization(loss_function=cao.cost_func,
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

    img_cor = cao.correct_image(ab_img, A_estimate)

    fig = plt.figure(figsize=([16, 9]))
    gs = gridspec.GridSpec(2, 6)

    ax1 = plt.subplot(gs[0, 0:2])

    ax1.imshow(cao.normalize_image(ab_img), vmin=p_factor, vmax=1)
    ax1.axis('off')
    ax1.set_title('aberrant image')

    ax2 = plt.subplot(gs[0, 2:4])

    cor_img = cao.correct_image(img_target=ab_img, cor_coes=A_estimate)
    ax2.imshow(cao.normalize_image(cor_img), vmin=p_factor, vmax=1)
    ax2.axis('off')
    ax2.set_title('corrected image')

    ax3 = plt.subplot(gs[0, 4:6])

    ax3.imshow(cao.normalize_image(ab_img-cor_img), vmin=p_factor, vmax=1)
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