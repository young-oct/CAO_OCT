# -*- coding: utf-8 -*-
# @Time    : 2022-11-02 17:55
# @Author  : young wang
# @FileName: CAO_optimzation.py
# @Software: PyCharm
import copy
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2 as cv
import glob
import numpy as np
import matplotlib.gridspec as gridspec
from tools import cao

if __name__ == '__main__':
    no_terms = 6
    A_true = cao.load_zernike_coefficients(no_terms=no_terms,
                                       A_true_flat=True)

    image_plist = glob.glob('../test images/*.png')
    img_no = -1
    img = cv.imread(image_plist[int(img_no)])
    img_gray = rgb2gray(img)

    ab_img = cao.aberrate(img_gray, A_true)

    Zo = cao.construct_zernike(A_true, N=512)
    tolerance = 1e-4

    optimizer = cao.optimization(loss_function=cao.cost_func,
                             a=9e-1, c=1.0,
                             alpha_val = 0.75,
                             gamma_val = 0.02,
                             max_iter=100,
                             img_target=ab_img,
                             zernike=Zo,
                             momentum=0.15,
                             cal_tolerance=tolerance)
    #
    optimizer_types = ['spgd','spgd-momentum','spgd-adam','spsa']
    # optimizer_types = ['spgd', 'spgd-momentum', 'spgd-adam']

    for i in range(len(optimizer_types)):
        optimizer_type = optimizer_types[i]
        A_initial = copy.deepcopy(cao.load_zernike_coefficients(no_terms=no_terms,
                                                            A_true_flat=False, repeat=True))

        A_estimate, costval, sol_idx = optimizer.minimizer(current_Aw=A_initial,
                                                           optimizer_type=optimizer_type,
                                                           beta1=0.9,
                                                           beta2=0.99,
                                                           verbose=False)

        discrepancy = np.std(A_estimate - A_true)
        img_cor = cao.correct_image(ab_img, A_true)
        loss_groud_truth = cao.image_entropy(img_cor)

        fig = plt.figure(figsize=([16, 9]))
        gs = gridspec.GridSpec(2, 6)

        ax1 = plt.subplot(gs[0, 0:2])
        ax1.imshow(cao.normalize_image(img_gray), vmin=0, vmax=1)
        ax1.axis('off')
        ax1.set_title('original image')

        ax2 = plt.subplot(gs[0, 2:4])
        ax2.imshow(cao.normalize_image(ab_img), vmin=0, vmax=1)
        ax2.axis('off')
        ax2.set_title('aberrant image')

        ax3 = plt.subplot(gs[0, 4:6])
        cor_img = cao.correct_image(img_target=ab_img, cor_coes=A_estimate)
        ax3.imshow(cao.normalize_image(cor_img), vmin=0, vmax=1)
        ax3.axis('off')
        ax3.set_title('corrected image')

        x_axis = np.linspace(0, A_estimate.shape[-1], A_estimate.shape[-1])
        ax4 = plt.subplot(gs[1, 0:3])
        ax4.bar(x_axis - 0.15, A_initial.squeeze(), width=0.15, label='initial guess')
        ax4.bar(x_axis, A_estimate.squeeze(), width=0.15, label='guess value')
        ax4.bar(x_axis + 0.15, A_true.squeeze(), width=0.15, label='ground truth')
        ax4.legend(loc='best')
        ax4.set_xlabel('zernike terms')
        ax4.set_ylabel('numerical values')

        ax5 = plt.subplot(gs[1, 3:6])
        ax5.plot(np.arange(len(costval)), costval)
        ax5.set_xlabel('iterations')
        ax5.set_ylabel('cost function values')

        fig.suptitle('%s based computational adaptive optics(CAO)\n'
                     'solution found at iteration %d with a discrepancy of %.4f' % (
                         optimizer_type, sol_idx, discrepancy))

        plt.tight_layout()
        plt.show()
