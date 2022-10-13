# -*- coding: utf-8 -*-
# @Time    : 2022-10-13 10:10
# @Author  : young wang
# @FileName: spsa_test.py
# @Software: PyCharm

"""
this script implements Simultaneous Perturbation
Stochastic Approximation (SPSA) and uses a dummy
function as a benchmark for testing purpose
"""

import numpy as np
import matplotlib.pyplot as plt
class spsa:
    def __init__(self, a, c, A, alpha, gamma, loss_function):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.loss_function = loss_function

        # counters
        self.k = 0

    def step(self, current_theta, *args):

        # get the current values for gain sequences
        a_k = self.a / (self.k + 1 + self.A)**self.alpha
        c_k = self.c / (self.k + 1)**self.gamma

        # get the random perturbation vector Bernoulli distribution with p=0.5
        delta = (np.random.randint(0, 2, current_theta.shape) * 2 - 1)

        theta_plus = current_theta + delta * c_k
        theta_minus = current_theta - delta * c_k

        # measure the loss function at perturbations
        loss_plus = self.loss_function(theta_plus, args)
        loss_minus = self.loss_function(theta_minus, args)

        # compute the estimate of the gradient
        g_hat = (loss_plus - loss_minus) / (2.0 * delta * c_k)

        # update the estimate of the parameter
        current_theta = current_theta - a_k * g_hat

        # increment the counter
        self.k += 1

        return current_theta

class LinearModel:
    def __init__(self, input_d, output_d, optimizer):
        self.input_d = input_d
        self.output_d = output_d
        self.optimizer = optimizer

        # initialize the weights
        self.W_estimate = np.random.randn(self.output_d, self.input_d) * 0.001

    def forward(self, x_value):
        return np.matmul(self.W_estimate, x_value)

    def backward(self, x_value, y_target):
        """
        :param input: inputs
        :param target: targets
        :param selection: selected weight
        :return: updated weight for the selected

        Minimizes the squared loss

        """
        # Update the weight
        self.W = self.optimizer.step(self.W_estimate, x_value, y_target)

class L2Loss(object):
    def __call__(self, W_estimate, *args):
        # extract the parameters
        x_value = args[0][0]
        y_target = args[0][1]

        # data loss
        pred = np.matmul(W_estimate,x_value)
        squared_loss = np.sum((y_target - pred)**2, axis=1).reshape((y_target.shape[0],-1))
        average_squared_loss = squared_loss / x_value.shape[1]
        return average_squared_loss

if __name__ == "__main__":
    # Generate sample points
    N = 500
    input_dim = 10
    output_dim = 1

    np.random.seed(13)
    x = np.random.rand(input_dim, N)
    W_true = np.random.rand(output_dim, input_dim)
    print("The true value of W is: \n "+str(W_true))
    y = np.matmul(W_true, x)

    # create the optimizer class
    max_iter = 2000
    optimizer = spsa(a=9e-1, c=1.0, A=max_iter/10,
                     alpha=0.6, gamma=0.1, loss_function=L2Loss())

    # create the linear model
    linear_model = LinearModel(input_d=input_dim,
                               output_d=output_dim,
                               optimizer=optimizer)

    # the main loop
    for i in range(max_iter):
        linear_model.backward(x,y)

    # finally print W
    W_estimate = linear_model.W
    print("The solution is: \n"+str(W_estimate))

    x_axis = np.linspace(0,W_estimate.shape[-1],W_estimate.shape[-1])
    plt.bar(x_axis + .25, W_estimate.squeeze(), width=0.5,label='guess value')
    plt.bar(x_axis - 0.25, W_true.squeeze(), width=0.5,label='ground truth')
    plt.legend()
    plt.show()