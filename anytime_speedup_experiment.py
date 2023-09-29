import sys
import argparse
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


from environment import Environment, LBEnvironment
from up import up
from naive import naive
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('-plot', action='store_true', help="only plot, no running")
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

delta = .1
epsilons = np.linspace(.1, .25, 6)

ks = [60, 300, 600]

utility_functions = []
utility_functions.append((u_ll, {'k0': 60, 'a': 1}))
utility_functions.append((u_unif, {'k0': 60}))

for u in utility_functions:

    u_fn, u_params = u

    for k in ks:

        data_save_path = "dat/anytime_speedup_{}_{}_captime={}.p".format(args.dataset, u_to_str(u), k)
        image_save_path = "img/anytime_speedup_{}_{}_captime={}.pdf".format(args.dataset, u_to_str(u), k)

        if not args.plot:

            data = {'dataset_name': args.dataset,
                    'epsilons': epsilons,
                    'k': k,
                    'naive': [],
                    'epsilons_naive': [],
                    'up': []}

            for epsilon in epsilons:
                try:
                    _ = naive(env, lambda t: u_fn(t, **u_params), epsilon, delta, k)
                    data['naive'].append(env.get_time_spent_running_all() / day_in_s)
                    data['epsilons_naive'].append(epsilon)
                except AssertionError:
                    print("anytime_speedup_experiment: naive failed for epsilon={}, k={}".format(epsilon, k))            
                env.reset()

                _ = up(env, lambda t: u_fn(t, **u_params), delta, k0=1, epsilon_min=epsilon)
                data['up'].append(env.get_time_spent_running_all() / day_in_s)
                env.reset()

            pickle.dump(data, open(data_save_path, 'wb'))

        data = pickle.load(open(data_save_path, 'rb'))
        dataset_name = data['dataset_name']
        epsilons = data['epsilons']
        epsilons_naive = data['epsilons_naive']

        plt.plot(epsilons_naive, data['naive'], label="Naive(k={})".format(data['k']), c=colors[4], linewidth=lw['main'])
        plt.plot(epsilons, data['up'], label="UP", c=colors[5], linewidth=lw['main'])

        plt.legend()
        plt.xlabel(r"$\epsilon$", fontsize=fs['axis'])
        plt.ylabel("Total Runtime (CPU days)", fontsize=fs['axis'])
        plt.title(dataset_name, fontsize=fs['title'])
        plt.savefig(image_save_path, bbox_inches='tight')
        plt.clf()



