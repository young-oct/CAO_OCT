# -*- coding: utf-8 -*-
# @Time    : 2022-09-15 12:58
# @Author  : young wang
# @FileName: gradient_descent.py
# @Software: PyCharm

'''
f(x) = w0 + w1 * x
'''

from scipy.special import gamma
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scripts.tools.plot import *
from sklearn import preprocessing
from skimage.util import random_noise
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import glob
import numpy as np
import copy
from numpy.random import permutation


class Line():
    """
        Linear Model with two weights w0 (intercept) and w1 (slope)
    """

    def __init__(self):
        self.weights = [np.random.uniform(0, 1, 1) for _ in range(2)]
        self.derivative_funcs = [self.dx_w0, self.dx_w1]

    def evaluate(self, x):
        """
            evaluate: will evaluate the line yhate given x
            x: a point in the plane

            return the result of the function evalutation
        """
        return self.weights[0] + self.weights[1] * x

    def derivate(self, x, y):
        """
            derivate: will calculate all partial derivatives and return them
            input:
            x: a point in the plane
            y: the response of the point x

            output:
            partial_derivatives: an array of partial derivatives
        """
        partial_derivatives = []

        yhat = self.evaluate(x)
        partial_derivatives.append(self.dx_w0(x, y, yhat))
        partial_derivatives.append(self.dx_w1(x, y, yhat))

        return partial_derivatives

    def dx_w0(self, x, y, yhat):
        """
            dx_w0: partial derivative of the weight w0
            x: a point in the plane
            y: the response of the point x
            yhat: the current approximation of y given x and the weights

            return the gradient at that point for this x and y for w0
        """
        return 2 * (yhat - y)

    def dx_w1(self, x, y, yhat):
        """
            dx_w1: partial derivative of the weight w1 for a linear function
            x: a point in the plane
            y: the response of the point x
            yhat: the current approximation of y given x and the weights

            return the gradient at that point for this x and y for w1
        """
        return 2 * x * (yhat - y)

    def __str__(self):
        return f"y = {self.weights[0]} + {self.weights[1]}*x"


#################### Helper functions ######################
def stochastic_sample(xs, ys):
    """
        stochastic_sample: sample with replacement one x and one y
        xs: all point on the plane
        ys: all response on the plane

        return the randomly selected x and y point
    """
    perm = permutation(len(xs))
    x = xs[perm[0]]
    y = ys[perm[0]]

    return x, y


def gradient(dx, evaluate, xs, ys):
    """
        gradient: estimate mean gradient over all point for w1
        evaluate: the evaulation function from the model
        dx: partial derivative function used to evaluate the gradient
        xs: all point on the plane
        ys: all response on the plane

        return the mean gradient all x and y for w1
    """
    N = len(ys)

    total = 0
    for x, y in zip(xs, ys):
        yhat = evaluate(x)
        total = total + dx(x, y, yhat)

    gradient = total / N
    return gradient


################## Optimization Functions #####################

def gd(model, xs, ys, learning_rate=0.01, max_num_iteration=1000):
    """
        gd: will estimate the parameters w1 and w2 (here it uses least square cost function)
        model: the model we are trying to optimize using gradient descent
        xs: all point on the plane
        ys: all response on the plane
        learning_rate: the learning rate for the step that weights update will take
        max_num_iteration: the number of iteration before we stop updating
    """

    for i in range(max_num_iteration):
        # Updating the model parameters
        model.weights = [weight - learning_rate * gradient(derivative_func, model.evaluate, xs, ys) for
                         weight, derivative_func in zip(model.weights, model.derivative_funcs)]

        # if i % 100 == 0:
        #     # print(f"Iteration {i}")
        #     print(model)


def sgd(model, xs, ys, learning_rate=0.01, max_num_iteration=1000):
    """
        sgd: will estimate the parameters w0 and w1
        (here it uses least square cost function)
        model: the model we are trying to optimize using sgd
        xs: all point on the plane
        ys: all response on the plane
        learning_rate: the learning rate for the step that weights update will take
        max_num_iteration: the number of iteration before we stop updating
    """

    for i in range(max_num_iteration):

        # Select a random x and y
        x, y = stochastic_sample(xs, ys)

        # Updating the model parameters
        model.weights = [weight - learning_rate * derivative for weight, derivative in
                         zip(model.weights, model.derivate(x, y))]

        # if i % 100 == 0:
        #     # print(f"Iteration {i}")
        #     print(model)

if __name__ == '__main__':
    xs = np.linspace(start = 1, stop= 10, num=200)
    noise = np.random.normal(0, 1, xs.shape[-1])
    ys = xs * 2 + 1 + noise
    # Gradient Descent
    model_gd = Line()
    print("Gradient Descent: ")
    gd(model_gd, xs, ys)
    print(model_gd)

    model_sgd = Line()
    print("Stochastic Gradient Descent: ")
    sgd(model_sgd, xs, ys)
    print(model_sgd)

    fig,ax = plt.subplots(1,1, figsize = (16,9))

    line1, = ax.plot(xs, ys,label='true value')
    line2, = ax.plot(xs, xs *model_gd.weights[1] + model_gd.weights[0], label = 'gradient descent')
    line3, = ax.plot(xs, xs *model_sgd.weights[1] + model_sgd.weights[0], label ='stochastic gradient descent')
    ax.legend(handles=[line1, line2, line3])
    plt.tight_layout()
    plt.show()
