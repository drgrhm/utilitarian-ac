import sys
import argparse
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment, LBEnvironment
from naive import naive
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="choose one of [minisat | cplex_rcw | cplex_region]")
parser.add_argument('-plot', action='store_true', help="only plot, no running")
args = parser.parse_args()

if args.dataset == "minisat":
    env = LBEnvironment('icar/dataset_icar/measurements.dump', 900)
elif args.dataset == "cplex_rcw":
    env = Environment('icar/dataset_icar/cplex_rcw/cplex_rcw_rt_seed{}.npy'.format(520), 10000)
elif args.dataset == "cplex_region":
    env = Environment('icar/dataset_icar/cplex_region/cplex_region_rt_seed{}.npy'.format(520), 10000)
else:
    print("\nERROR: dataset should be one of [minisat | cplex_rcw | cplex_region]\n")
    exit()

np.random.seed(987)

gridsize = 21

if args.dataset == "minisat":
    ks = np.round(np.linspace(10, 1000, gridsize))
else:
    ks = np.linspace(50, 10500, gridsize)

epsilons = [.1, .15, .2]
delta = .1

u = (u_ll, {'k0': 60, 'a': 1})
u_fn, u_params = u

data_save_path = "dat/naive_captime_{}_{}_{}.p".format(args.dataset, u_to_str(u), gridsize)
image_save_path = "img/naive_captime_{}_{}_{}.pdf".format(args.dataset, u_to_str(u), gridsize)

total_times = np.zeros((len(epsilons), len(ks)))
failures = []

if not args.plot:

    for ei, epsilon in enumerate(epsilons):
        for ki, k in enumerate(ks):

            print("naive_captime_experiment: k={}, epsilon={}".format(k, epsilon))            
            
            try:
                i_star = naive(env, lambda t: u_fn(t, **u_params), epsilon, delta, k)
                total_times[ei][ki] = env.get_time_spent_running_all() / day_in_s
                print("naive_captime_experiment: i_star={}, time_spent_running_all={}".format(i_star, env.get_time_spent_running_all() / day_in_s))
            except (AssertionError, IndexError) as e:
                print("naive_captime_experiment: failed on epsilon={}".format(epsilon))
                total_times[ei][ki] = None
                failures.append((epsilon, k))
            env.reset()

    pickle.dump({'total_times': total_times,
                 'failures': failures,
                 'ks': ks,
                 'epsilons': epsilons,
                 'delta': delta,
                 'u': u,                 
                 'dataset_name': args.dataset}, open(data_save_path, 'wb'))


data = pickle.load(open(data_save_path, 'rb'))
total_times = data['total_times']
failures = data['failures']
ks = data['ks']
epsilons = data['epsilons']
delta = data['delta']
u = data['u']
dataset_name = data['dataset_name']

print("Failures: ")
for p in failures:
    print(p)

if dataset_name == "minisat":
    xmax = 1000
else:
    xmax = 11000
    
for ei, epsilon in enumerate(epsilons):
    plt.plot(ks, total_times[ei], label=r"$\epsilon$={}".format(epsilon), c=colors[ei], linewidth=lw['main'])

plt.legend()
plt.xlabel(r"$\kappa$", fontsize=fs['axis'])
plt.ylabel("Total Runtime (CPU days)", fontsize=fs['axis'])
plt.title(dataset_name, fontsize=fs['title'])
plt.savefig(image_save_path, bbox_inches='tight')
plt.clf()





