import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class BernoulliArm() :
    def __init__(self, p):
        self.p = p

    def draw(self) :
        if random.random() > self.p :
            return 0.0
        else:
            return 1.0

class EpsilonGreedy() :
    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    def initialize(self):
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
    def select_arm(self):
        if random.random() > self.epsilon:
            # argmax needs for values to be unique?
            return np.argmax(self.values)
        else:
            return np.random.randint(0, len(self.values))
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n, value = self.counts[chosen_arm], self.values[chosen_arm]

        new_value = float(n-1) / n * value + (1.0 / n) * reward
        self.values[chosen_arm] = new_value

# num_sim represents the number of simulations
# horizon represents ?
def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros((num_sims, horizon))
    rewards = np.zeros((num_sims, horizon))
    cumulative_rewards = np.zeros((num_sims, horizon))
    for sim in range(num_sims):
        algo.initialize()
        for t in range(horizon):
            chosen_arm = algo.select_arm()
            chosen_arms[sim][t] = chosen_arm
            reward = arms[chosen_arm].draw()
            rewards[sim][t] = reward
            if t == 0:
                cumulative_rewards[sim][t] = reward
            else:
                cumulative_rewards[sim][t] = cumulative_rewards[sim][t-1] + reward
            algo.update(chosen_arm, reward)
    return (chosen_arms, rewards, cumulative_rewards)

means = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
n_arms = len(means)
random.shuffle(means)
arms = map(lambda x: BernoulliArm(x), means)

for epsilon in np.arange(0.1, 0.6, 0.1):
    algo = EpsilonGreedy(epsilon, n_arms)
    chosen_arms, rewards, cumulative_rewards = test_algorithm(algo, arms, 5000, 250)
    plt.plot(np.mean(rewards, axis=0), label="epsilon="+str(epsilon))

plt.legend(loc="best")
plt.show()
