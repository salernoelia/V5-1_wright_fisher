import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

class WrightFisherModel:
    
    def __init__(self, population_size: int, alpha: float = 0, beta: float = 0):
        self.N = population_size
        self.alpha = alpha
        self.beta = beta
        self.has_mutation = (alpha > 0 or beta > 0)
    
    def get_transition_prob(self, j):
        if not self.has_mutation:
            p = j / self.N
        else:
            # With mutation (equation 3.4)
            # $p = \frac{j (1 - \alpha) + (N - j) \beta}{N}$
            p = (j * (1 - self.alpha) + (self.N - j) * self.beta) / self.N
        return p
    
    def step(self, current_a_genes):
        p = self.get_transition_prob(current_a_genes)
        next_a_genes = np.random.binomial(self.N, p)
        return next_a_genes
    
    def simulate(self, initial_a_genes, generations):
        trajectory = np.zeros(generations + 1, dtype=int)
        trajectory[0] = initial_a_genes
        
        for t in range(generations):
            trajectory[t + 1] = self.step(trajectory[t])
        
        return trajectory
    
    def simulate_multiple(self, initial_a_genes, generations, n_simulations):
        trajectories = []
        for _ in range(n_simulations):
            traj = self.simulate(initial_a_genes, generations)
            trajectories.append(traj)
        return np.array(trajectories)
    
    def get_steady_state_distribution(self, n_simulations=10000, generations=1000):
        if not self.has_mutation:
            print("Warning: Steady state only meaningful with mutation")
        
        final_states = []
        for _ in range(n_simulations):
            initial = np.random.randint(0, self.N + 1)
            traj = self.simulate(initial, generations)
            final_states.append(traj[-1])
        
        frequencies = np.bincount(final_states, minlength=self.N + 1) / n_simulations
        return frequencies

