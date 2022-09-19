from os import chdir
chdir('./deep_rl')


from math import inf, log, sqrt
from random import choice, normalvariate, uniform
import numpy as np
import matplotlib.pyplot as plt
# hyper-parameters
num_runs = 2000
num_bandits = 15
num_steps = 1000
mu = np.linspace(1,15,15)
epsilon = 0.1


# explicit results collectors
ucb_RR = np.empty((num_runs, num_steps))
ucb_AA = np.empty((num_runs, num_steps))
greedy_RR = np.empty((num_runs, num_steps))
greedy_AA = np.empty((num_runs, num_steps))
# i:index of a bandit, 0 for the first one.
def Reward(i):
    return normalvariate(mu[i], 1)

def ucb(t, i, N, Q):
    return inf if N[i] == 0 else sqrt(log(t)/N[i])+Q[i]

def try_ucb(T):
    R = np.empty(T)
    A = np.empty(T)

    # internal data collectors
    Q = np.zeros(num_bandits)
    N = np.zeros(num_bandits)

    for t in range(T):
        UCBs = [ucb(t, i, N, Q) for i in range(num_bandits)]
        UCBs = np.array(UCBs)
        pick = choice(np.where(UCBs == np.max(UCBs))[0])
        A[t] = pick
        R[t] = Reward(pick)
        N[pick] += 1
        Q[pick] += (R[t]-Q[pick])/N[pick]

    return (R, A==num_bandits-1)

def try_eps_greedy(T):
    R = np.empty(T)
    A = np.empty(T)

    # internal data collectors
    Q = np.zeros(num_bandits)
    N = np.zeros(num_bandits)

    for t in range(T):
        if uniform(0,1) < epsilon:
            pick = choice(range(num_bandits))
        else:
            pick = choice(np.where(Q == np.max(Q))[0])
        A[t] = pick
        R[t] = Reward(pick)
        N[pick] +=1
        Q[pick] += (R[t]-Q[pick])/N[pick]

    return (R, A==num_bandits-1)



for j in range(num_runs):
    ucb_result = try_ucb(num_steps)
    ucb_RR[j,:] = ucb_result[0]
    ucb_AA[j,:] = ucb_result[1]

    greedy_result = try_eps_greedy(num_steps)
    greedy_RR[j,:] = greedy_result[0]
    greedy_AA[j,:] = greedy_result[1]

plt.figure(1)
plt.plot(np.mean(ucb_RR, 0), color='b', label='UCB')
plt.plot(np.mean(greedy_RR, 0), color='g', label=r'$\varepsilon$-greedy')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('./img/reward.png', dpi=300)

plt.figure(2)
plt.plot(np.mean(ucb_AA, 0), color='b', label='UCB')
plt.plot(np.mean(greedy_AA, 0), color='g', label=r'$\varepsilon$-greedy')
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.legend()
plt.savefig('./img/optimal.png', dpi=300)