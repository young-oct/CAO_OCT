# -*- coding: utf-8 -*-
# @Time    : 2022-07-29 09:54
# @Author  : young wang
# @FileName: proc.py
# @Software: PyCharm
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter, median_filter
import cv2 as cv
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import math
from numba import jit
import matplotlib.patches as patches
from numba import njit

class circle_fit:
    def __init__(self, pts):
        self.x, self.y = zip(*pts)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        self.A = self.form_A()
        self.B = self.form_B()

        self.radius, self.origin = self.cal_sphere()

    def form_A(self):
        A = np.zeros((len(self.x), 3))
        A[:, 0] = self.x
        A[:, 1] = self.y
        A[:, 2] = 1

        return A

    def form_B(self):
        #   Assemble the f matrix
        B = np.zeros((len(self.x), 1))
        B[:, 0] = (self.x * self.x) + (self.y * self.y)
        return B

    def cal_sphere(self):
        c, residules, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)

        x, y = float(c[0] / 2), float(c[1] / 2)
        origin = (x,y)
        radius = np.sqrt(c[2] + x ** 2 + y ** 2)
        #
        return radius, origin

    def plot(self, ax):

        circle = patches.Circle(self.origin, self.radius,
                                transform=ax.transData, fill=False,
                                linestyle='dashed', edgecolor='red',label='estimated circle')
        ax.add_patch(circle)
        ax.plot(self.origin[0], self.origin[1], marker='o', color='red', label='origin')
        ax.set_title('R: %.2f \n origin (x y): %.2f %.2f' % (self.radius, self.origin[0], self.origin[1]))
        ax.set_aspect('equal')

        return ax


class sphere_fit:
    def __init__(self, pts, centre=None, fixed_origin=True):

        self.x, self.y, self.z = zip(*pts)
        self.fixed_origin = fixed_origin

        if fixed_origin:
            self.centre = centre
            self.x = np.array(self.x) - self.centre[0]
            self.y = np.array(self.y) - self.centre[1]
            self.z = np.array(self.z) - self.centre[2]

        else:
            self.x = np.array(self.x)
            self.y = np.array(self.y)
            self.z = np.array(self.z)

        self.A = self.form_A()
        self.B = self.form_B()

        self.radius, self.origin = self.cal_sphere()

    def form_A(self):
        A = np.zeros((len(self.x), 4))
        A[:, 0] = self.x * 2
        A[:, 1] = self.y * 2
        A[:, 2] = self.z * 2
        A[:, 3] = 1

        return A

    def form_B(self):
        #   Assemble the f matrix
        B = np.zeros((len(self.x), 1))
        B[:, 0] = (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
        return B

    #
    def cal_sphere(self):
        c, residules, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)
        radius = np.sqrt((c[0] * c[0]) + (c[1] * c[1]) + (c[2] * c[2]) + c[3])
        if self.fixed_origin:
            origin = (c[0] + self.centre[0], c[1] + self.centre[1], c[2] + self.centre[2])
        else:
            origin = (c[0], c[1], c[2])

        return radius, origin

    def plot(self, ax):
        x0, y0, z0 = self.origin[0], self.origin[1], self.origin[2]

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v) * self.radius
        y = np.sin(u) * np.sin(v) * self.radius
        z = np.cos(v) * self.radius
        if self.fixed_origin:

            x = x + x0 - self.centre[0]
            y = y + y0 - self.centre[1]
            z = z + z0 - self.centre[2]
        else:
            x = x + x0
            y = y + y0
            z = z + z0

        ax.scatter(self.x, self.y, self.z, zdir='z', s=0.1, c='b', alpha=0.3, rasterized=True)
        ax.plot_wireframe(x, y, z, color="r", )
        origin = np.asarray(self.origin).flatten()
        ax.set_title('radius = %.2f \n origin(x,y,z) is %s' % (self.radius, origin))
        return ax


class plane_fit:

    def __init__(self, pts, order):

        self.x, self.y, self.z = zip(*pts)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

        self.order = order

        self.A = self.form_A()
        self.B = self.form_B()

        self.xx, self.yy = np.arange(0, 512, 1), np.arange(0, 512, 1)

        self.x_plane, self.y_plane = np.meshgrid(self.xx, self.yy)
        self.zc = self.cal_plane()

    def form_A(self):
        # the  equation for a linear plane is: ax+by+c = z s.t Ax = B
        # A = [x0, y0, 1,......xn, yn, 1].T
        if self.order == 1:
            A = np.c_[self.x, self.y, np.ones(self.x.shape[0])]

        elif self.order == 2:
            # the  equation for a quadratic(n=2) plane is: ax^2+bxy+cy^2+dx+ey+f = z s.t Ax = B
            # A = [x0^2, x0y0, y0^2, x0, y0, 1,......xn^2, xnyn, yn^2, xn, yn, 1].T
            A = np.c_[self.x ** 2, self.x * self.y, self.y ** 2,
                      self.x, self.y, np.ones(self.x.shape[0])]

        elif self.order == 3:
            # the  equation for a cubic(n=3) plane is:
            # ax^3+ by^3 + cx^2*y + dx*y^2 + ex^2 + fy^2 + gx*y+ hx+iy + j= z s.t Ax = B
            # A = [x0^3, y0^3, x0^2*y0, + x0*y0^2 + x0^2 + y0^2 +
            # x0y0 + x0 + y0 + 1, ...... xn ^ 3, yn ^ 3, xn ^ 2 * yn, + xn * yn ^ 2 + xn ^ 2 +
            # yn ^ 2 + xnyn + xn + yn + 1].T
            # A = np.c_[self.x ** 3, self.y ** 3,
            #
            #
            #           (self.x ** 2)*self.y, self.x * (self.x ** 2),
            #           self.x ** 2,self.y ** 2,
            #           self.x * self.y, self.x, self.y,
            #           np.ones(self.x.shape[0])]
            A = np.c_[self.x ** 3,
                      self.y ** 3,
                      np.prod(np.c_[self.x ** 2, self.y], axis=1),
                      np.prod(np.c_[self.x, self.y ** 2], axis=1),
                      self.x ** 2,
                      self.y ** 2,
                      self.x * self.y,
                      self.x, self.y,
                      np.ones(self.x.shape[0])]
        else:
            pass

        return A

    def form_B(self):

        B = self.z

        return B

    def cal_plane(self):

        c, _, _, _ = np.linalg.lstsq(self.A, self.B, rcond=None)  # coefficients

        # x = [a, b, c].T
        if self.order == 1:
            zc = c[0] * self.x_plane + c[1] * self.y_plane + c[2]

        # x = [a, b, c, d, e, f].T
        elif self.order == 2:
            zc = c[0] * self.x_plane ** 2 + \
                 c[1] * self.x_plane * self.y_plane + \
                 c[2] * self.y_plane ** 2 + c[3] * self.x_plane + \
                 c[4] * self.y_plane + c[5]

        # x = [a, b, c, d, e, f, h, i, j].T
        elif self.order == 3:

            zc = c[0] * self.x_plane ** 3 + \
                 c[1] * self.y_plane ** 3 + \
                 c[2] * np.prod(np.c_[self.x_plane ** 2, self.y_plane], axis=1) + \
                 c[3] * np.prod(np.c_[self.x_plane, self.y_plane ** 2], axis=1) + \
                 c[4] * self.x_plane ** 2 + c[5] * self.y_plane ** 2 + \
                 c[6] * self.x_plane * self.y_plane + \
                 c[7] * self.x_plane + c[8] * self.y_plane + c[9]

            pass
        return zc

    def plot(self, ax, low=0, high=330):

        ax.scatter(self.x, self.y, self.z, zdir='z', s=0.1, c='b', alpha=0.3, rasterized=True)
        ax.plot_surface(self.x_plane, self.y_plane, self.zc, color='r', alpha=0.5)

        ax.set_zlabel('z')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 512])
        ax.set_ylim([0, 512])
        ax.set_zlim([low, high])
        return ax


def frame_index(volume, dir, index, shift=0):
    peak_loc = []

    if dir == 'x':
        slice = volume[index, :, :]
        for i in range(slice.shape[0]):
            a_line = slice[i, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, int(peaks[0][-1] - shift)))
            else:
                pass
    elif dir == 'y':
        slice = volume[:, index, :]
        for i in range(slice.shape[0]):
            a_line = slice[i, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, int(peaks[0][-1] - shift)))
            else:
                pass
    else:
        print('please enter the correct direction')
    return peak_loc


def slice_index(slice,shift=0):
    peak_loc = []
    for i in range(slice.shape[1]):
        a_line = slice[:, i]
        peaks = np.where(a_line == 255)
        if len(peaks[0]) >= 1:
            peak_loc.append((i, int(peaks[0][-1] - shift)))
        else:
            pass

    return peak_loc


def surface_index(volume, shift=0):
    '''get the index for the peaks in the each volume'''

    peak_loc = []
    for i in range(volume.shape[0]):
        slice = volume[i, :, :]
        for j in range(slice.shape[0]):
            a_line = slice[j, :]
            peaks = np.where(a_line == 255)
            if len(peaks[0]) >= 1:
                peak_loc.append((i, j, int(peaks[0][-1] - shift)))
            else:
                pass

    return peak_loc


def clean_removal(data, top=5, radius=230):
    '''

    :param data: oct 3d data 512x512x330
    :param top: top index to be removed
    :param radius: radius to remove the artfact as a result of scanning
    :return: oct 3d data 512x512x330
    '''

    data[:, :, 0:top] = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.sqrt((i - 256) ** 2 + (j - 256) ** 2) >= radius:
                data[i, j, :] = 0

    return data


def despecking(frame, sigma=0.8, size=3):
    """
    :param frame: 512x 330 or 330x512 oct b mode frame
    :param sigma: sigma for gaussian filter
    :param size: median filter kernel size """

    frame = gaussian_filter(frame, sigma=sigma)

    return median_filter(frame, size=size)


def binary_mask(slice, vmin, vmax):
    ret, msk = cv.threshold(slice, vmin, vmax, cv.THRESH_BINARY)
    krn = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    return cv.bitwise_and(cv.dilate(msk, krn, iterations=100), msk)


def filter_mask(slice, vmin, vmax):
    mask = binary_mask(slice, vmin, vmax)
    mask = median_filter(mask, size=10)
    mask = gaussian_filter(mask, sigma=0.2)
    return mask


def max_slice(volume):
    # take a volume and find the index of the maximum intensity
    # slice
    assert volume.ndim == 3

    slice = np.sum(volume, axis=0)
    line = np.sum(slice, axis=0)

    return np.argmax(line)


def mip_stack(volume, index, thickness):
    assert volume.ndim == 3

    low_b, high_b = int(index - thickness), int(index + thickness)

    if low_b >= 0 or high_b <= volume.shape[-1]:
        return np.amax(volume[:, :, low_b:high_b], axis=2)


@jit(nopython=True)
def getPolarco(f_zmax=1.7, degree=10.5):
    '''obtian correct polar coordinates from the distorted image

    since X and Y correction can be done independently with respect to Z,
    here we replace X_dim, Y_dim mentioend in Josh's proposal as i_dim
    for detailed math, see johs's proposal
    we can do this because
    (1) i_dim = X_dim = Y_dim = 512
    (2) azimuth and elevation are roughly the same 10 degrees (according to Dan)
    (3) 3D geometirc correction can be decomposed into two independent 2D correction
    please see "Real-time correction of geometric distortion artifact
     in large-volume optical coherence tomography paper'''

    i_dim, zdim, zmax = 512, 330, int(330 * f_zmax)

    _iz = np.zeros((i_dim, zdim, 2))  # construct iz plane
    i0, z0 = int(i_dim / 2), zmax  # i0 is half of the i dimension

    i_phi = math.radians(degree)  # converting from degree to radiant

    ki = i_dim / (2 * i_phi)  # calculate pixel scaling factor for i dimension
    # kz = 1.5 # calculate pixel scaling factor for z dimension, it should be Zmax/D, this is
    # a magic number kind works,
    kz = 1
    for i in range(i_dim):
        for z in range(zdim):  # pixel coordinates conversion
            _iz[i, z, :] = [
                (z + kz * z0) * math.sin((i - i0) / ki) * math.cos((i - i0) / ki) + i0,
                (z + kz * z0) * math.cos((i - i0) / ki) * math.cos((i - i0) / ki) - kz * z0]

        # _iz.reshape(i_dim * zdim, 2): numpy stores arrays in row-major order
        # This means that the resulting two-column array will first contain all the x values,
        # then all the y values rather than containing pairs of (x,y) in each row
    _iz = _iz.reshape(i_dim * zdim, 2)
    return _iz


@jit(nopython=True)
def valueRemap(dis_image):
    """remap the data to match with the correct orientation"""

    _v = np.zeros(dis_image.shape)
    for i in range(dis_image.shape[0]):
        for z in range(dis_image.shape[1]):  # pixel coordinates conversion

            _v[i, z] = dis_image[i, -z]  # store the pixel date temporally and flip along the colume
            # axis
    return np.ravel(_v)


def polar2cart(tri, xq, zq, values):
    values = valueRemap(values)

    """interpolate values from the target grid points"""

    # initilize interpolator
    interpolator = LinearNDInterpolator(tri, values)

    # interpolate values from from with respect to the targeted
    # cartisan coordinates
    valueUpdate = interpolator(xq, zq)

    return np.fliplr(valueUpdate)
    # return valueUpdate


def iniTri(polrcoordinate):
    '''initialize triangulation'''
    return Delaunay(polrcoordinate)

def index_mid(input_list):
    mid_pts = []
    mid = float(len(input_list)) / 2
    if mid % 2 != 0:
        mid_pts.append(input_list[int(mid - 0.5)])
    else:
        mid_pts.append((input_list[int(mid)], input_list[int(mid - 1)]))

    return np.mean(mid_pts)

def map_index(img, xcenter, ycenter, radial_list, perspective_list):

    c1, c2, c3, c4, c5, c6, c7, c8 = perspective_list

    (height, width) = img.shape
    xu_list = np.arange(width) - xcenter
    yu_list = np.arange(height) - ycenter
    xu_mat, yu_mat = np.meshgrid(xu_list, yu_list)
    ru_mat = np.sqrt(xu_mat ** 2 + yu_mat ** 2)

    # apply radial model
    fact_mat = np.sum(np.asarray(
        [factor * ru_mat ** i for i, factor in enumerate(radial_list)]), axis=0)

    # shift the distortion center
    xd_mat = xcenter + fact_mat * xu_mat
    yd_mat = ycenter + fact_mat * yu_mat

    # apply  perspective model
    mat_tmp = (c7 * xd_mat + c8 * yd_mat + 1.0)
    xd_mat = (c1 * xd_mat + c2 * yd_mat + c3) / mat_tmp
    yd_mat = (c4 * xd_mat + c5 * yd_mat + c6) / mat_tmp
    xd_mat = np.float32(np.clip(xd_mat, 0, width - 1))
    yd_mat = np.float32(np.clip(yd_mat, 0, height - 1))

    indices = np.vstack((np.ndarray.flatten(yd_mat), np.ndarray.flatten(xd_mat)))

    xd_mat = xd_mat.flatten()/img.shape[0]
    yd_mat = yd_mat.flatten()/img.shape[1]

    idx_map = np.empty((xd_mat.size + yd_mat.size), dtype=xd_mat.dtype)
    idx_map[0::2] = xd_mat
    idx_map[1::2] = yd_mat

    # map img to new indices
    c_img = map_coordinates(img, indices).reshape(img.shape)
    # index normalize to [0,1] for GPU texture

    return c_img, idx_map


def wall_index(img, distance = 80, height = 0.3 ):
    mid_index = int(img.shape[0]/2)
    v_loc, h_loc = [], []
    v_line = img[:, mid_index]
    h_line = img[mid_index,:]

    pk_heights = np.max(img) * height

    vpeaks, _ = find_peaks(v_line, distance=distance, height = pk_heights)
    hpeaks, _ = find_peaks(h_line, distance=distance, height = pk_heights)

    if len(vpeaks) >=2 and len(hpeaks) >=2:
        v_loc.append((mid_index,vpeaks[0]))
        v_loc.append((mid_index,vpeaks[-1]))

        h_loc.append((hpeaks[0],mid_index))
        h_loc.append((hpeaks[-1],mid_index))

    else:
        pass

    return v_loc,h_loc

def line_fit(points, order=1):
    x, y = zip(*points)
    a, b = np.polyfit(x, y, order)

    return a, b


@njit
def circle_cut(vol,cut_ori = (256,256),
               inner_radius =40, edge_radius = 250):
    x, y = cut_ori
    assert vol.ndim == 3

    for i in range(vol.shape[-1]):
        for j in range(vol.shape[0]):
            for k in range(vol.shape[1]):
                radius = np.sqrt((j-x) ** 2 + (k-y) ** 2)

                inner_criteria = radius - inner_radius
                edge_criteria = radius - edge_radius

                if inner_criteria < 0 or edge_criteria > 0:
                    vol[j,k,i] = 0
                else:
                    pass
    return vol


