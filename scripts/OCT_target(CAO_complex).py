from tools.spiral_loader import loader
from scipy import ndimage
import matplotlib
from natsort import natsorted
import copy
import matplotlib.pyplot as plt
import glob
import numpy as np
import matplotlib.gridspec as gridspec
import time
from tools import cao

if __name__ == '__main__':
    matplotlib.rcParams.update(
        {
            'font.size': 16,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )

    folder_path = '../data/Ossiview complex/*.bin'
    file_path = natsorted(glob.glob(folder_path))

    Aline_vol = loader(file_path[-1], radius=256)

    norm_vol = cao.complex2int(Aline_vol)

    c_idx = norm_vol.shape[0] // 2
    center_slice = norm_vol[:, c_idx, :]

    lx = []
    for i in range(center_slice.shape[0]):
        lx.append(np.argmax(center_slice[i, :]))

    z_idx = ndimage.median_filter(lx, size=5)
    z_idx = ndimage.gaussian_filter1d(z_idx, sigma=5)

    idx = int(np.mean(z_idx))

    start_time = time.time()
    ab_img = cao.square_crop(Aline_vol[:, :, idx])
    #
    no_terms = 4

    A_initial = copy.deepcopy(cao.load_zernike_coefficients(no_terms=no_terms,
                                                        A_true_flat=False, repeat=False))


    Zo = cao.construct_zernike(A_initial, N=ab_img.shape[0])
    # # alpha_val is the learning rate
    # # gamma_val is the perturbation amount rate
    alpha_val, gamma_val = 0.05, 0.05
    tolerance = gamma_val / 100

    optimizer = cao.optimization(loss_function=cao.cost_func,
                             a=9e-1, c=1.0,
                             alpha_val=alpha_val,
                             gamma_val=gamma_val,
                             max_iter=1000,
                             img_target=ab_img,
                             zernike=Zo,
                             momentum=0.15,
                             cal_tolerance=tolerance)

    optimizer_types = ['spgd', 'spgd-momentum', 'spgd-adam', 'spsa']
    optimizer_type = optimizer_types[2]
    #
    A_estimate, costval, sol_idx = optimizer.minimizer(current_Aw=A_initial,
                                                       optimizer_type=optimizer_type,
                                                       beta1=0.9,
                                                       beta2=0.99,
                                                       verbose=False)
    img_cor = cao.correct_image(ab_img, A_estimate)
    fig = plt.figure(figsize=([16, 9]))
    gs = gridspec.GridSpec(2, 6)

    ax1 = plt.subplot(gs[0, 0:2])
    p_factor = 0.5

    ax1.imshow(ndimage.median_filter(cao.inten2pixel(ab_img), size=3), vmin=p_factor, vmax=1)

    ax1.axis('off')
    ax1.set_title('aberrant image')

    ax2 = plt.subplot(gs[0, 2:4])

    cor_img = cao.correct_image(img_target=ab_img, cor_coes=A_estimate)

    ax2.imshow(ndimage.median_filter(cao.inten2pixel(cor_img), size=3), vmin=p_factor, vmax=1)
    ax2.axis('off')
    ax2.set_title('corrected image')

    ax3 = plt.subplot(gs[0, 4:6])

    ax3.imshow(ndimage.median_filter(cao.inten2pixel(ab_img - cor_img), size=3), vmin=p_factor, vmax=1)
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

    end_time = time.time()

    proc_time = (end_time-start_time)/60
    fig.suptitle('%s based computational adaptive optics(CAO)\n'
                 'learning rate: %.3f; perturbation amount:%.3f \n'
                 'solution found at iteration %d in %.2f min' % (optimizer_type, alpha_val,
                                                                 gamma_val, sol_idx, proc_time))

    plt.tight_layout()
    plt.show()
    print('done')

