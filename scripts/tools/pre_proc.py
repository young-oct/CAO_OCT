# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 8:18 a.m.
# @Author  : young wang
# @FileName: pre_proc.py
# @Software: PyCharm
"""preprocessing module"""

from skimage import measure,exposure
from skimage.morphology import closing,disk,dilation,square
from scipy import ndimage
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
import os
from tools.proc import clean_removal,circle_cut,despecking
from tools.pos_proc import imag2uint,convert
import numpy as np
from tools.OssiviewBufferReader import OssiviewBufferReader

def folder_creator(folder_path):
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        # directory already exists
        pass

def load_from_oct_file(oct_file, clean = False):
    """
    read .oct file uising OssiviewBufferReader
    export an array in the shape of [512,512,330]
    the aarry contains pixel intensity data(20log) in float16 format
    """
    obr = OssiviewBufferReader(oct_file)
    data_fp16 = np.squeeze(obr.data)

    data = imag2uint(data_fp16)

    if clean:
        data = clean_removal(data)
    else:
        pass
    return data

def arrTolist(volume, Yflag=False):
    '''
    convert volume array into list for parallel processing
    :param volume: complex array
    :return:
    volume_list with 512 elements long, each element is Z x X = 330 x 512
    '''

    volume_list = []

    if not Yflag:
        for i in range(volume.shape[0]):
            volume_list.append(volume[i, :, :])
    else:
        for i in range(volume.shape[1]):
            volume_list.append(volume[:, i, :])

    return volume_list

def listtoarr(volume_list, Yflag=False):
    '''convert the volume list back to array format
    :param volume_list: complex array
    :return:
    volume with 512 elements long, each element is Z x X = 330 x 512
    '''

    if not Yflag:
        volume = np.empty((len(volume_list), 512, 330))
        for i in range(len(volume_list)):
            volume[i, :, :] = volume_list[i]
    else:
        volume = np.empty((512, len(volume_list), 330))
        for i in range(len(volume_list)):
            volume[:, i, :] = volume_list[i]

    return volume


def pre_volume(volume,p_factor = 0.6):

    new_volume = np.zeros_like(volume)
    volume = circle_cut(volume)

    vmin, vmax = int(p_factor * 255), 255
    s_vol = np.where(volume <= vmin, 0, 255)

    for i in range(s_vol.shape[-1]):
        temp_img = s_vol[:,:,i]
        temp_img = despecking(temp_img, sigma=1, size=3)
        new_volume[:,:,i] = closing(temp_img, diamond(5))

    new_volume = np.where(new_volume < np.max(new_volume) * p_factor, 0, 255)
    return new_volume



def clean_small_object(volume):
    new_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        label_im, nb_labels = ndimage.label(c_slice)
        sizes = ndimage.sum(c_slice, label_im, range(nb_labels + 1))

        mask_size = sizes < np.mean(sizes)
        remove_pixel = mask_size[label_im]

        label_im[remove_pixel] = 0
        new_volume[:, :, i] = label_im
    return new_volume

def obtain_inner_edge(volume):

    iedge_volume = np.zeros_like(volume)
    for i in range(volume.shape[-1]):
        c_slice = volume[:,:,i]
        contours = measure.find_contours(c_slice)
        #1 is the inner edge, 0 is the outer edge
        edge_arr = np.zeros_like(c_slice)
        if len(contours) > 1:
            for j in range(len(contours[1])):
                x, y = contours[1][j]
                edge_arr[int(x), int(y)] = 255
        elif len(contours) == 1:
            for j in range(len(contours[-1])):
                x, y = contours[-1][j]
                edge_arr[int(x), int(y)] = 255
        else:
            pass

        iedge_volume[:,:,i] = edge_arr
    return iedge_volume