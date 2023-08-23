import math
import numpy as np
import matplotlib.pyplot as plt

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

color_schemes = [['#377eb8', '#629fd0', '#9dc3e2'], 
                ['#ff7f00', '#ffa64d', '#ffcc99'], 
                ['#4daf4a', '#72c36f', '#a7d9a5'],
                ['#f781bf', '#f99fcf', '#fccfe7'],
                ['#a65628', '#d27a46', '#e1a684'],
                ['#984ea3', '#bd87c5', '#d9bade']]
                # ['#999999', '#bfbfbf', '#d9d9d9']]

fs = {'axis': 15,
      'title': 20}

lw = {'main': 5,
      'small': 2,
      'tiny': .5}

day_in_s = 60 * 60 * 24


# Utility functions and their derivatives: 
def u_unif(t, k0): # Uniform
    if t < k0:
        return 1 - t / k0
    else:
        return 0

def uinv_unif(u, k0): # Uniform inverse 
    return k0 * (1 - u)

def u_exp(t, k0): # Exponential 
    return math.exp(- t / k0)

def uinv_exp(u, k0): # Exponential inverse 
    return -k0 * math.log(u)

# def uprime_exp(t, k0):
#     return - math.exp(- t / k0) / k0

def u_pareto(t, k0, a): # Pareto 
    if t < k0:
        return 1
    else:
        return (k0 / t) ** a

def uprime_pareto(t, k0, a): # Pareto derivative 
    if t < k0:
        return 0
    else:
        return - a * k0**a / t**(a + 1)

def uinv_pareto(u, k0, a): # Pareto inverse 
    return k0 / u**(1 / a)

def u_ll(t, k0, a): # Log-Laplace
    if  t < k0:
        return 1 - (t / k0) ** a / 2
    else:
        return (k0 / t) ** a / 2


def u_gll(t, k0, a, b): # generalized log-Laplace
    if  t < k0:
        return 1 - a * (t / k0) ** b / (a + b)
    else:
        return b * (k0 / t) ** a / (a + b)


def u_poly(t, k0, a):
    if t < k0:
        return 1 - (t / k0)**a
    else:
        return 0

# Uniform
def pdf_uniform(t, t0):
    if t < t0:
        return 1 / t0
    else:
        return 0


def cdf_uniform(t, t0):
    if t < t0:
        return t / t0
    else:
        return 1


# Exponential
def pdf_exponential(t, t0):
    return math.exp(- t / t0) / t0


def cdf_exponential(t, t0):
    return 1 - math.exp(- t / t0)


# Pareto
def pdf_pareto(t, t0, a):
    if t < t0:
        return 0
    else:
        return a * t0 ** a / t ** (a + 1)


def cdf_pareto(t, t0, a):
    if t < t0:
        return 0
    else:
        return 1 - (t0 / t) ** a


# (gen) log Laplace
def pdf_genlogLaplace(t, t0, a, b):
    if t < t0:
        return a * b * (t / t0) ** (b - 1) / (a + b) / t0
    else:
        return a * b * (t0 / t) ** (a + 1) / (a + b) / t0


def cdf_genlogLaplace(t, t0, a, b):
    if t < t0:
        return a / (a + b) * (t / t0) ** b
    else:
        return 1 - b / (a + b) * (t0 / t) ** a


# log Normal
def pdf_logNormal(t, t0, sigma, eps=.0001):
    t = t + eps
    return math.exp(- math.log(t / t0) ** 2 / 2 / sigma ** 2) / t / sigma / math.sqrt(2 * math.pi)


def cdf_logNormal(t, t0, sigma, eps=.0001):
    t = t + eps
    return 1 / 2 + math.erf(math.log(t / t0) / math.sqrt(2) / sigma) / 2


# Piecewise
def pdf_piecewise(t, t0, t1, delta):
    if t < t0:
        if t < t1:
            return delta / t1
        else:
            return (1 - delta) / (t0 - t1)
    else:
        return 0


def cdf_piecewise(t, t0, t1, delta):
    if t < t0:
        if t < t1:
            return delta * t / t1
        else:
            return delta + (1 - delta) * (t - t1) / (t0 - t1)
    else:
        return 1


# Cost
def cost_constant(t, a=1.0):
    return a


def cost_linear(t, a=1.0):
    return a * t


def cost_log(t, a=1.0):
    return a * math.log(1 + t)


def cost_step(t, ts, cs, b):
    for i in range(len(ts)):
        if t >= ts[-i-1]:
            return b + cs[-i-1]
    return b


# Misc
def integrate(fn, x0, x1, steps=None):
    """ If fn has discontinuities at points in <steps>, breaks integral up into pieces """
    if steps is None:
        steps = []
    val = 0
    errors = []
    _x0 = x0
    for i, x in enumerate(steps):
        val += _integrate.quad(fn, _x0, x)[0]
        errors.append(_integrate.quad(fn, _x0, x)[1])
        _x0 = x
    val += _integrate.quad(fn, _x0, x1)[0]
    errors.append(_integrate.quad(fn, _x0, x1)[1])
    return (val, errors)


def ecdf(data):
    """ empirical CDF """
    x = np.sort(data)
    y = np.arange(len(x)) / float(len(x))
    return x, y


def u_to_str(utility_function):
    """ return a nice string representation of a utility function and its parameters """
    u_fn, u_params = utility_function
    if type(u_params) is dict:
        param_str = ",".join("{}={}".format(k, v) for k, v in u_params.items())
    else:
        param_str = ",".join("{}".format(v) for v in u_params)
    return "{}(".format(u_fn.__name__) + param_str + ")"


def plot_us(us, k_max, save_path, xscale='linear'):
    for u in us:
        u_fn, u_params = u
        if xscale == 'log':
            ts = np.logspace(-2, math.log(k_max), 1000)
        else:
            ts = np.linspace(0, k_max, 1000)
        plt.plot(ts, [u_fn(t, **u_params) for t in ts], label=u_to_str(u), linewidth=lw['main'])
        plt.legend()

    plt.xscale(xscale)
    plt.xlim(.01, k_max)
    plt.ylim(-.005, 1.005)
    plt.xlabel("runtime", fontsize=fs['axis'])
    plt.ylabel("utility", fontsize=fs['axis'])
    plt.title("Utility Functions", fontsize=fs['title'])
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()



    