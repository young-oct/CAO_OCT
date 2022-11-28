# -*- coding: utf-8 -*-
# @Time    : 2022-11-28 15:53
# @Author  : young wang
# @FileName: spiral_loader.py
# @Software: PyCharm


import numpy as np
import struct
from scipy.fft import fft, ifft, rfft, irfft
from scipy import signal

eps = 1e-14


def AverageAlineGroups(alineData, navg=5):
    # Calculate the complex average within groups
    imavg = np.zeros([alineData.shape[0], alineData.shape[1] // navg, alineData.shape[2]], dtype=np.complex64)
    for j in range(0, alineData.shape[0]):
        for n, k in enumerate(range(0, alineData.shape[1], navg)):
            imavg[j, n, :] = np.mean(alineData[j, k:(k + navg), :], axis = 0)

    return imavg


def BackgroundSubtraction(interferogramData):
    im = np.float64(interferogramData)

    for k in range(im.shape[0]):
        im[k, :, :] = im[k, :, :] - np.mean(im[k, :, :], axis=0)

    return im


def ConstructAlines(interferogramData):
    # Apply background subtraction
    im = BackgroundSubtraction(interferogramData)

    # Apply FFT to create alines
    window = signal.windows.nuttall(im.shape[2])
    imfft = np.zeros([im.shape[0], im.shape[1], im.shape[2] // 2 + 1], dtype=np.complex64)
    for k in range(im.shape[0]):
        imfft[k, :, :] = rfft(window * im[k, :, :], axis=1)

    return imfft


class spiral_coordinates:
    def __init__(self, numQuadrants=4, radius=128):
        self.numQuadrants = numQuadrants
        self.radius = radius
        self.xyloc = self.sprial2raster()

    def CircleSegment(self, quadrant, radius, center):
        x = -1 * radius
        y = 0
        error = 2 - 2 * radius
        xCoords = []
        yCoords = []

        while x < 0:
            if quadrant == 1:
                xCoords.append(center - x)
                yCoords.append(center + y)
            elif quadrant == 2:
                xCoords.append(center - y)
                yCoords.append(center - x)
            elif quadrant == 3:
                xCoords.append(center + x)
                yCoords.append(center - y)
            elif quadrant == 4:
                xCoords.append(center + y)
                yCoords.append(center + x)

            radius = error

            if radius <= y:
                y = y + 1
                error += 2 * y + 1

            if radius > x or error > y:
                x = x + 1
                error += 2 * x + 1

        return xCoords, yCoords

    def GenerateCircle(self, radius, center):

        xCoords = []
        yCoords = []

        for quadrant in range(1, self.numQuadrants + 1):
            x, y = self.CircleSegment(quadrant, radius, center)
            xCoords += x
            yCoords += y

        return xCoords, yCoords

    def sprial2raster(self):

        # Running coordinate arrays
        x_coordinates = []
        y_coordinates = []

        # Running point counter
        numPoints = 0
        for r in range(self.radius - 1, 0, -1):
            x_temp, y_temp = self.GenerateCircle(r, self.radius - 1)

            x_coordinates.append(x_temp)
            y_coordinates.append(y_temp)

            numPoints += len(x_coordinates[self.radius - r - 1])

        x_course_spiral = np.hstack(x_coordinates)
        y_course_spiral = np.hstack(y_coordinates)

        xy_list = [(x_course_spiral[i], y_course_spiral[i])
                   for i in range(len(x_course_spiral))]

        return np.asarray(xy_list)


def loader(file_path, radius = 128,top = 30):

    Aline_coords = spiral_coordinates(radius=radius).xyloc

    # Load in header info
    headerLen = 8
    with open(file_path, 'rb') as f:
        datastr = f.read(headerLen)
    dataHeader = struct.unpack(r'II', datastr)


    numRecords = dataHeader[0]
    numSamples = dataHeader[1]
    numFrames = int(np.ceil(Aline_coords.shape[0] * 5 / numRecords))


    with open(file_path, 'rb') as f:
        daqData = np.fromfile(f, dtype=np.uint16,
                              count=numFrames * numSamples *
                                    numRecords).reshape((-1, numRecords, numSamples))

        imfft = ConstructAlines(daqData)
        zdim = 330
        bottom = (top + zdim)

        alineData = imfft[:, :, top:bottom]

        Aline_average = AverageAlineGroups(alineData)
        temp = Aline_average.reshape((-1, Aline_average.shape[-1]))

        vol = np.zeros((int(radius*2), int(radius*2), zdim), dtype=temp.dtype)

        for i in range(Aline_coords.shape[0]):
            x_idx = Aline_coords[i][0]
            y_idx = Aline_coords[i][1]

            vol[x_idx, y_idx, :] = temp[i, :]

        vol += eps

    return vol/np.linalg.norm(vol)


