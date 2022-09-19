from math import inf, log, sqrt
from random import normalvariate
import numpy as np
import numpy
from numpy import choose
# hyper-parameters
num_bandits = 15
num_runs = 1000
mu = np.linspace(1,15,15)
epsilon = 0.1

# data collectors
Q = np.zeros(num_bandits)
N = np.zeros(num_bandits)


# i:index of a bandit, 0 for the first one.
def Reward(i):
    return normalvariate(mu[i], 1)

def ucb(t, i):
    return inf if N[i] == 0 else sqrt(log(t)/N[i])+Q[i]

def try_ucb(T):
    R = np.empty(num_runs)
    A = np.empty(num_runs)

    for t in range(T):
        UCBs = [ucb(t, i) for i in range(num_bandits)]
        UCBs = np.array(UCBs)
        select



#print(Q)
test = numpy.empty(10)
test[0:4] = inf
print(np.where(test == np.max(test)))