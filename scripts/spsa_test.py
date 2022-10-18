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
import matplotlib
class spsa:
    def __init__(self, loss_function, a, c, alpha, gamma, max_iter, args=()):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma

        self.loss = loss_function

        self.max_iter = max_iter
        self.A = max_iter / 10
        self.args = args

    def calc_loss(self, current_theta):
        """Evalute the cost/loss function with a value of theta"""
        return self.loss(current_theta, self.args)

    def minimise(self, current_theta, optimizer_type = 'vanilla'):
        k = 0  # initialize count

        cost_func_val = []
        vk = 0
        while k < self.max_iter:
            # get the current values for gain sequences
            a_k = self.a / (k + 1 + self.A) ** self.alpha
            c_k = self.c / (k + 1) ** self.gamma

            # get the random perturbation vector Bernoulli distribution with p=0.5
            delta = (np.random.randint(0, 2, current_theta.shape) * 2 - 1)

            theta_plus = current_theta + delta * c_k
            theta_minus = current_theta - delta * c_k

            # measure the loss function at perturbations
            loss_plus = self.calc_loss(theta_plus)
            loss_minus = self.calc_loss(theta_minus)

            # compute the estimate of the gradient
            g_hat = (loss_plus - loss_minus) / (2.0 * delta * c_k)

            if optimizer_type == 'vanilla':
                current_theta = current_theta + - a_k * g_hat
            elif optimizer_type == 'momentum':

                vk_next = - a_k * g_hat + 0.3 * vk
                current_theta = current_theta + vk_next
                vk = vk_next
            else:
                pass

            k += 1

            cost_val = self.calc_loss(current_theta)
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
            'font.size': 18,
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
    W_true = np.random.rand(output_dim, input_dim)
    print("The true value of W is: \n " + str(W_true))
    noise = np.random.rand(N)
    y = np.matmul(W_true, x) + noise * 0.2

    W_initial = np.random.rand(output_dim, input_dim)

    optimizer = spsa(loss_function=calc_loss,
                     a=9e-1, c=1.0, alpha=0.602, gamma=0.101,
                     max_iter=200, args=(x, y))

    #vanilla or momentum
    optimizer_type = 'momentum'
    W_estimate, costval = optimizer.minimise( current_theta = W_initial, optimizer_type =optimizer_type)


    # print("The estimate value of W is: \n " + str(W_estimate))

    discrepancy = np.std(W_estimate - W_true)
    print("The discrepancy value of estimation is: \n " + str(discrepancy))


    if output_dim == 1:
        fig, ax = plt.subplots(1, 2, figsize=(16, 9))
        x_axis = np.linspace(0, W_estimate.shape[-1], W_estimate.shape[-1])
        ax[0].bar(x_axis + .25, W_estimate.squeeze(), width=0.5, label='guess value')
        ax[0].bar(x_axis - 0.25, W_true.squeeze(), width=0.5, label='ground truth')
        ax[0].legend()

        ax[1].plot(np.arange(len(costval)),  costval)
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('cost function values')
        ax[1].set_ylabel('cost function values')
        fig.suptitle('benchmark performance for SPSA: %s\n discrepancy value: %.4f'%(optimizer_type,discrepancy))
        plt.tight_layout()
        plt.show()
    else:
        pass

