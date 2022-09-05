# -*- coding: utf-8 -*-
# @Time    : 2022-07-29 09:52
# @Author  : young wang
# @FileName: pos_proc.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pickle
from os.path import join
from os.path import isfile
from pydicom.uid import generate_uid

def export_list(corr_list,file_path):

    idx_map_size = np.uint32(corr_list.size / 2)
    # corr_list.insert(0,coe_map_len)

    with open(file_path, 'wb') as f:
        f.write(idx_map_size)
        f.write(corr_list)
    return None


def export_map(coe_map, file_path):
    # export correction map
    # check if coe_map is 2D array
    assert coe_map.ndim == 2

    # coe_map /= 512
    coe_map = coe_map.astype(np.float32)
    coe_map_size = np.uint32(coe_map.size)

    # Save correction maps to disk
    with open(file_path, 'wb') as f:
        f.write(coe_map_size)
        f.write(coe_map)
    return None

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def imag2uint(data, lt=0, ut=255):
    '''
    convert pixel data from the 255 range to unit16 range(0-65535)

    :param data:  oct 3d data 512x512x330
    :param lt: lower threshold of pixel values to be removed
    :param ut: upper threshold of pixel values to be removed
    :return: oct 3d data 512x512x330
    '''

    # remove the low and high bounds of the pixel intensity data points
    data = np.clip(data, lt, np.max(data))
    # pixel intensity normalization
    # for detail, please see wiki page
    # https://en.wikipedia.org/wiki/Normalization_(image_processing)

    data = (data - np.min(data)) * ut / (np.max(data) - np.min(data))

    return np.uint16(np.around(data, 0))


def oct_to_dicom(data, resolutionx, resolutiony,resolutionz,
                 dicom_folder,**patient_info):
    """
    convert pixel array [512,512,330] to DICOM format
    using MRI template, this template will be deprecated
    in the future once OCT template is received
    """
    dss = []

    template_file = '../template data/template.dcm'
    ds = pydicom.dcmread(template_file)

    # SeriesInstanceUID refers to each series, and should be
    # unqie for each sesssion, and generate_uid() provides an unique
    # identifier
    ds.SeriesInstanceUID = generate_uid()

    all_files_exist = False
    dicom_prefix = 'oct'
    # looping through all 330 slice of images with [512(row) x 512(column)]
    for i in range(data.shape[2]):
        # UID used for indexing slices
        ds.SOPInstanceUID = generate_uid()

        # update row and column numbers to be 512
        ds.Rows = data.shape[0]
        ds.Columns = data.shape[1]

        # define the bottom(assuming the middle plane to be zero,
        # that -165 * 30um(axial resolution) = -4.95 mm)
        # DICOM assumes physical dimension to be in mm unit
        bottom = -4.95
        # elevate the z by its axial resolution at a time
        z = bottom + (i * resolutionz)
        # update meta properties

        # 1cm / 512 = 0.02 mm, needs to check with rob
        # this spacing should be calculated as radiant/pixel then mm to pixel

        ds.PixelSpacing = [resolutionx, resolutiony]  # pixel spacing in x, y planes [mm]
        ds.SliceThickness = resolutionz  # slice thickness in axial(z) direction [mm]
        ds.SpacingBetweenSlices = resolutionz # slice spacing in axial(z) direction [mm]
        ds.SliceLocation = '%0.2f' % z  # slice location in axial(z) direction
        ds.InstanceNumber = '%0d' % (i + 1,)  # instance number, 330 in total
        ds.ImagePositionPatient = [z, 0, 0]  # patient physical location
        ds.Manufacturer = 'Audioptics Medical Inc'
        ds.InstitutionName = 'Audioptics Medical'
        ds.InstitutionAddress = '1344 Summer St., #55, Halifax, NS, Canada'

        ds.PatientName = patient_info['PatientName']
        ds.PatientBirthDate = patient_info['PatientBirthDate']
        ds.PatientSex = patient_info['PatientSex']
        ds.PatientAge = patient_info['PatientAge']
        ds.PatientID = patient_info['PatientID']
        ds.SeriesDescription = patient_info['SeriesDescription']
        ds.StudyDescription = patient_info['StudyDescription']
        ds.StationName = 'Unit 1'
        ds.PhysiciansOfRecord = ''
        ds.PerformingPhysicianName = ''
        ds.InstitutionalDepartmentName = ''
        ds.ManufacturerModelName = 'Mark II'
        ds.PatientAddress = ''

        # setting the dynamic range with WindowCenter and WindowWidth
        # lowest_visible_value = window_center — window_width / 2
        # highest_visible_value = window_center + window_width / 2

        ds.WindowCenter = '215'
        ds.WindowWidth = '225'

        # # set highest and lowest pixel values
        ds.LargestImagePixelValue = 255
        ds.SmallestImagePixelValue = 0
        dicom_file = join(dicom_folder, "%s%04d.dcm" % (dicom_prefix, i))

        pixel_data = data[:, :, i]
        ds.PixelData = pixel_data.tobytes()
        ds.save_as(dicom_file)
        dss.append(ds)
        all_files_exist = all_files_exist and isfile(dicom_file)
    return all_files_exist

def image_export(img,vmin, vmax, filename):
    plt.axis("off")
    fig = plt.imshow(img, 'gray', vmin = vmin, vmax = vmax,
                     interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.savefig(filename, transparent=True,
                bbox_inches='tight', pad_inches=0)
    return None
