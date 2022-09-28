import numpy as np
from numpy.random import standard_t as t
from matplotlib import pyplot as plt
from scipy.stats import norm


def simulate_l(distribution_type, degrees_of_freedom=0, mean=0, std=0, alpha=0):
    """
    The function simulates 10000 realization of the random variable L(t, t+delta_t)
    :param distribution_type: "student"/"normal": the distribution type of alpha*X for a specific alpha to find
    :param degrees_of_freedom: the degrees of freedom of the student distribution if distribution_type == "student"
    :param mean: the mean of the normal distribution if distribution_type == "normal"
    :param std: the std of the normal distribution if distribution_type == "normal"
    :param alpha: the parameter alpha that needed to be found if distribution_type == "student"
    :return: L, a vector of 10000 realization of the random variable L(t, t+delta_t)
    """
    lambda_ = 1
    S = 100
    if distribution_type == "student":
        # simulate alpha*x
        alpha_times_x = t(df=degrees_of_freedom, size=10000)
        # retrieve x
        x = alpha_times_x / alpha
        # simulate L
        L = -lambda_ * S * (np.exp(x) - 1)
    elif distribution_type == "normal":
        # simulate x
        x = np.random.normal(mean, std, 10000)
        # simulate L
        L = -lambda_ * S * (np.exp(x) - 1)
    else:
        raise ValueError('error in distribution type')
    return L


def plot_overlapped_distribution(Ls, nus, bins):
    """
    The function plots the distributions of L (as an histogram) together with the distribution of
    the standard normal with mean = mean(L) and std = std(L)
    :param nus: vector of degrees of freedom
    :param Ls: a vector of vectors of 10000 realization of the random variable L(t, t+delta_t) for each point of the exercise
    :param bins: a vector of number of bins in the histogram of L for each point of the exercise
    :return: None
    """
    # compute the means and stds of Ls
    means = [np.mean(Ls[i]) for i in range(4)]
    stds = [np.std(Ls[i]) for i in range(4)]

    # plot everything with a single subplot

    x_axis = np.arange(-4, 4, 0.01)
    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.flatten()):
        ax.set_xlim(-4, 4)
        ax.hist(Ls[i], bins=bins[i], density=True, ec="k")
        # plot density of the gaussian distribution together with the histogram
        ax.plot(x_axis, norm.pdf(x_axis, means[i], stds[i]), "r-")
        if i <= 2:
            ax.legend(["normal density", "x student with degrees of freedom = " + str(nus[i])], prop={'size': 7})
        else:
            ax.legend(["normal density", "x is normal distribution"], prop={'size': 7})
        ax.set_xlabel("loss")
        ax.set_ylabel("density function")

    plt.show()
    return None


# parameters setting
np.random.seed(0)

# point A
alpha_a = np.sqrt(3)/0.01
print(f"alpha_a = {alpha_a}")
L_a = simulate_l("student", 3, alpha =alpha_a)

# Point B
alpha_b = np.sqrt(10)/(0.01*(np.sqrt(8)))
print(f"alpha_b = {alpha_b}")
L_b = simulate_l("student", 10, alpha=alpha_b)

# Point C
alpha_c = np.sqrt(50)/(0.01*(np.sqrt(48)))
print(f"alpha_c = {alpha_c}")
L_c = simulate_l("student", 50, alpha=alpha_c)

# point D
L_d = simulate_l("normal", mean=0, std=0.01)
Ls = [L_a, L_b, L_c, L_d]
print(f"in case a) the mean and std of the loss are: mean = {np.mean(L_a)}, std = {np.std(L_a)}")
print(f"in case b) the mean and std of the loss are: mean = {np.mean(L_b)}, std = {np.std(L_b)}")
print(f"in case c) the mean and std of the loss are: mean = {np.mean(L_c)}, std = {np.std(L_c)}")
print(f"in case d) the mean and std of the loss are: mean = {np.mean(L_d)}, std = {np.std(L_d)}")
plot_overlapped_distribution(Ls, nus=[3, 10, 50], bins=[500, 60, 60, 60])
