# -*- coding: utf-8 -*-
# @Time    : 2022-10-19 08:25
# @Author  : young wang
# @FileName: multivariate optimization benchmark.py
# @Software: PyCharm


"""
this script implements Simultaneous Perturbation
Stochastic Approximation (SPSA) and uses a dummy
function as a benchmark for testing purpose
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class optimization:
    def __init__(self, loss_function,
                 a, c,
                 alpha_val,
                 gamma_val,
                 max_iter,
                 momentum=0.2,
                 cal_tolerance=1e-6,
                 args=()):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.alpha_val = alpha_val
        self.gamma_val = gamma_val

        self.loss = loss_function

        self.max_iter = max_iter
        self.A = max_iter / 10

        # self.adam_m = adam_m

        self.momentum = momentum
        self.cal_tolerance = cal_tolerance

        self.args = args

    def calc_loss(self, current_theta):
        """Evalute the cost/loss function with a value of theta"""
        return self.loss(current_theta, self.args)

    def minimizer(self,
                  current_theta,
                  optimizer_type='vanilla',
                  beta1=0.9,
                  beta2=0.99,
                  epsilon=1e-8,
                  verbose=False):

        k = 0  # initialize count
        vk = 0

        adam_m = 0
        adam_v = 0

        beta1 = beta1
        beta2 = beta2
        epsilon = epsilon

        previous_theta = 0
        cost_func_val = []

        while k < self.max_iter and \
                np.linalg.norm(previous_theta - current_theta) > self.cal_tolerance:

            # # get the current values for gain sequences
            # a_k = self.a / (k + 1 + self.A) ** self.alpha_val
            # c_k = self.c / (k + 1) ** self.gamma_val

            previous_theta = current_theta
            cost_val = self.calc_loss(current_theta)

            if verbose:
                print('iteration %d: %s with cost function value %.2f' % (k, str(current_theta), cost_val))
            else:
                pass

            # get the random perturbation vector Bernoulli distribution with p=0.5
            delta = (np.random.randint(0, 2, current_theta.shape) * 2 - 1) * self.gamma_val

            theta_plus = current_theta + delta
            theta_minus = current_theta - delta

            # measure the loss function at perturbations
            loss_plus = self.calc_loss(theta_plus)
            loss_minus = self.calc_loss(theta_minus)

            loss_delta = (loss_plus - loss_minus)
            # # update the estimate of the parameter
            if optimizer_type == 'spgd':
                # compute the estimate of the gradient
                g_hat = loss_delta * delta
                current_theta = current_theta - self.alpha_val * g_hat

            elif optimizer_type == 'spgd-momentum':
                # compute the estimate of the gradient
                g_hat = loss_delta * delta
                vk_next = self.alpha_val * g_hat + self.momentum * vk
                current_theta = current_theta - vk_next
                vk = vk_next
            elif optimizer_type == 'spgd-adam':

                # calculate moment and second moment
                # compute the estimate of the gradient
                g_hat = loss_delta * delta
                adam_m = beta1 * adam_m + (1 - beta1) * g_hat
                adam_v = beta2 * adam_v + (1 - beta2) * (g_hat ** 2)

                # bias correction for both first and second moments

                alpha_updated = self.alpha_val * np.sqrt(1 - beta2 ** (k + 1)) / (1 - beta1 ** (k + 1))
                current_theta = current_theta - alpha_updated * adam_m / (np.sqrt(adam_v) + epsilon)

            elif optimizer_type == 'spsa':
                alpha_val = 0.602
                gamma_val = 0.101

                # get the current values for gain sequences
                a_k = self.a / (k + 1 + self.A) ** alpha_val
                c_k = self.c / (k + 1) ** gamma_val

                g_hat = loss_delta * delta / (2.0 * c_k)
                current_theta = current_theta - a_k * g_hat

            else:
                raise ValueError('please input the right optimizer')

            k += 1

            cost_func_val.append(cost_val.squeeze())

        return current_theta, cost_func_val


def calc_loss(W_estimate, *args):
    x_value = args[0][0]
    y_target = args[0][1]

    # data loss
    pred = np.matmul(W_estimate, x_value)
    squared_loss = np.sum((y_target - pred) ** 2, axis=1).reshape((y_target.shape[0], -1))
    average_squared_loss = squared_loss / x_value.shape[1]
    return average_squared_loss


if __name__ == "__main__":

    matplotlib.rcParams.update(
        {
            'font.size': 10,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'mathtext.fontset': 'stix',
        }
    )
    # Generate sample points
    N = 500
    input_dim = 10
    output_dim = 1

    np.random.seed(13)
    x = np.random.rand(input_dim, N)
    W_true = 2 * np.random.rand(output_dim, input_dim)
    noise = np.random.rand(N)
    y = np.matmul(W_true, x) + noise * 0.5

    W_initial = np.random.rand(output_dim, input_dim)
    tolerance = 1e-7

    optimizer = optimization(loss_function=calc_loss,
                             a=9e-1, c=1.0,
                             alpha_val=0.75,
                             gamma_val=0.02,
                             # alpha_val=0.602,
                             # gamma_val=0.101,
                             max_iter=200,
                             momentum=0.15,
                             cal_tolerance=tolerance,
                             args=(x, y))
    #
    # #spgd or spgd-momentum or adam

    optimizer_types = ['spgd','spgd-momentum','spgd-adam','spsa']

    fig, ax = plt.subplots(2, len(optimizer_types), figsize=(16, 9))

    for i in range(len(optimizer_types)):

        optimizer_type = optimizer_types[i]
        W_estimate, costval = optimizer.minimizer(current_theta=W_initial,
                                                  optimizer_type=optimizer_type)

        discrepancy = np.std(W_estimate - W_true)

        width = 0.25
        if output_dim == 1:
            # fig, ax = plt.subplots(1, 2, figsize=(16, 9))
            x_axis = np.linspace(0, W_estimate.shape[-1], W_estimate.shape[-1])
            ax[0,i].bar(x_axis, W_initial.squeeze(), width=width, label='initial guess')
            ax[0,i].bar(x_axis + width, W_estimate.squeeze(), width=width, label='guess value')
            ax[0,i].bar(x_axis + 2 * width, W_true.squeeze(), width=width, label='ground truth')
            ax[0,i].set_title('%s\n discrepancy value: %.4f' % (optimizer_type,discrepancy))

            ax[0,i].legend()

            ax[1,i].plot(np.arange(len(costval)), costval)
            ax[1,i].set_xlabel('iteration')
            ax[1,i].set_ylabel('cost function values')
            ax[1,i].set_ylabel('cost function values')
            # fig.suptitle('benchmark performance for SPSA: %s\n discrepancy value: %.4f' % (optimizer_type,
            #                                                                                discrepancy))

        else:
            pass

    fig.suptitle('benchmark performance comparison')

    plt.tight_layout()
    plt.show()