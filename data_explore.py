import sys
import argparse
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from utils import *


def plot_runtime_cdfs(env, save_path, x_max=None):
    means = []
    for i in range(env.get_num_configs()):
        means.append(np.mean(env._runtimes[i, :]))
    cmap = matplotlib.cm.get_cmap('cividis')
    norm = matplotlib.colors.Normalize(vmin=min(means), vmax=max(means))
    for i in range(env.get_num_configs()):
        plt.plot(*ecdf(env._runtimes[i, :]), linewidth=lw['tiny'], color=cmap(norm(means[i])))
    if x_max:
        plt.xlim(0, x_max)
    plt.ylim(-.005, 1.005)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\Pr(runtime \leq t)$")
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='(empirical) average runtime')
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


def plot_utility_cdfs(env, u, save_path):
    utils = []
    means = []
    vect_u = np.vectorize(u, otypes=[np.float])
    for i in range(env.get_num_configs()):
        us = vect_u(env._runtimes[i, :])
        utils.append(us)
        means.append(np.mean(us))
    cmap = matplotlib.cm.get_cmap('cividis_r')
    norm = matplotlib.colors.Normalize(vmin=min(means), vmax=max(means))
    for i in range(env.get_num_configs()):
        plt.plot(*ecdf(utils[i]), linewidth=lw['tiny'], color=cmap(norm(means[i])))
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='(empirical) average utility')
    plt.xlim(-.005, 1.005)
    plt.ylim(-.005, 1.005)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\Pr(utility \leq x)$")
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
    args = parser.parse_args()

    if args.dataset == "minisat":
        env = LBEnvironment('icar/dataset_icar/minisat_cnfuzzdd/measurements.dump', 900)
    elif args.dataset == "cplex_rcw":
        env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
    elif args.dataset == "cplex_region":
        env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
    else:
        print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
        exit()

    k_max = 1.05 * env.get_max_timeout()
    print("data_explore: {}, plotting runtime cdfs".format(args.dataset))
    plot_runtime_cdfs(env, "img/data_explore_runtime_cdf_{}.pdf".format(args.dataset), x_max=k_max)

    utility_functions = []
    utility_functions.append((u_ll, {'k0': 60, 'a': 1}))
    utility_functions.append((u_unif, {'k0': 60}))
    utility_functions.append((u_ll, {'k0': 300, 'a': 1}))

    plot_us(utility_functions, 1000, "img/data_explore_utility_functions.pdf", xscale='linear')
    plot_us(utility_functions, 1e6, "img/data_explore_utility_functions_log.pdf", xscale='log')

    for u in utility_functions:
        u_fn, u_params = u

        print("data_explore: {}, plotting utility cdfs for {}".format(args.dataset, u_to_str(u)))
        plot_utility_cdfs(env, lambda t: u_fn(t, **u_params), "img/data_explore_utiliy_cdf_{}_{}.pdf".format(args.dataset, u_to_str(u)))











