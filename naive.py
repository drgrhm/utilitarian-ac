import argparse
import json
import math
import pickle
import numpy as np  

from environment import Environment, LBEnvironment
from utils import *


def naive(env, u, epsilon, delta, k):
    """ Naive algorithm """
    assert u(k) < epsilon, "ERROR: captime must be large enough. k={}, u(k)={}, epsilon={}".format(k, u(k), epsilon)
    
    n = env.get_num_configs()
    m = int(2 * math.log(2 * n / delta) / (epsilon - u(k))**2) + 1
    
    assert m < env.get_num_instances(), "ERROR: cannot do m={} runs, only {} instances. u(k)={}, epsilon={}".format(m, env.get_num_instances(), u(k), epsilon)
    
    print("naive: doing m={} runs of n={} algorithms at captime k={}. u(k)={}, epsilon={}".format(m, n, k, u(k), epsilon))

    U = {}
    for i in range(n):
        U[i] = sum(u(env.run(i, j, k)) for j in range(m)) / m
    i_star = max(U, key=U.get)
    return i_star, m

