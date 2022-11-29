from tools.spiral_loader import loader
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from natsort import natsorted
import glob

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

    Aline_vol = loader(file_path[0])

    mag_vol = 20*np.log10(abs(Aline_vol))

    norm_vol = (mag_vol - np.min(mag_vol)) / np.ptp(mag_vol)

    plt.imshow(norm_vol[:,128,:], 'gray',vmin=0.8, vmax=1)
    plt.show()

