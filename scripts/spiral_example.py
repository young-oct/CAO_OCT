from tools.spiral_loader import loader,complex2int
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from natsort import natsorted
import glob
from scipy import signal

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

    norm_vol = complex2int(Aline_vol)




    # ROI_median = ndimage.median_filter(ROI_slice,size=3)

    # mag_vol = 20*np.log10(abs(Aline_vol))
    #
    # norm_vol = (mag_vol - np.min(mag_vol)) / np.ptp(mag_vol)

    # ROI_slice = norm_vol[:,:,245]

    # ROI_median = ndimage.median_filter(ROI_slice,size=3)
    # plt.imshow(ROI_median, 'gray',vmin=0.5, vmax=1)
    # plt.show()
    #
    print('done')
    #
