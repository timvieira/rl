import pylab as pl
import numpy as np
from scipy import linalg

from arsenal.maths import onehot, random_dist, sample
from arsenal import viz


class MarkovChain(object):
    "γ-discounted Markov chain."
    def __init__(self, s0, P, gamma):
        [self.S] = s0.shape
        self.s0 = s0
        self.P = P
        self.gamma = gamma

    #___________________________________________________________________________
    # Simulation

    def run(self):
        "Simulate the Markov chain with (1-γ)-resetting dynamics"
        s = self.start()
        while True:
            sp = self.step(s)
            yield s, sp
            s = sp

    def start(self):
        return sample(self.s0)

    def step(self, s):
        if np.random.uniform(0,1) <= 1-self.gamma:
            return self.start()
        return sample(self.P[s,:])

    #___________________________________________________________________________
    # Important quantities

    def successor_representation(self):
        "Dayan's successor representation."
        #equivalently: linalg.solve(self.M, np.eye(self.S))
        return np.linalg.inv(self.M)

    def d(self):
        "Stationary distribution."
        # The stationary distribution, much like other key quantities in MRPs,
        # is the solution to a linear recurrence,
        #            d = (1-γ) s0 + γ Pᵀ d    # transpose because P is s->s' and we want s'->s.
        #   d - γ Pᵀ d = (1-γ) s0
        # (I - γ Pᵀ) d = (1-γ) s0
        # See also: stationarity condition in the linear programming solution
        return (1-self.gamma) * self.solve_t(self.s0)   # note the transpose

    def d_by_eigen(self):
        """
        Compute the stationary distribution via eigen methods.
        """
        # Markov chain has this transition matrix, which makes the transition to
        # the start state explicit.
        t = self.P_with_reset()          # TODO: compare with transition to an absorbing state.

        # Transition matrix is from->to, so it sum-to-one over rows so we
        # transpose it.  Alternatively, we can get do the left eig decomp.
        [S, U] = linalg.eig(t.T)

        ss = np.argsort(S)
        S = S[ss]
        U = U[:,ss]

        s = U[:,-1].real
        s /= s.sum()
        return s

    def d_by_power_iteration(self, b0=None, iterations=1000):
        "Produce the sequence iterates of power iterations."
        A = self.P_with_reset().T
        b = b0 if b0 is not None else self.s0
        for _ in range(iterations):
            b = A @ b
            b /= np.sum(b)
            yield b

    @property
    def M(self):
        "Transition matrix with γ-leaking dynamics."
        return (np.eye(self.S) - self.gamma * self.P)

    def P_with_reset(self):
        "Transition matrix with (1-γ)-resetting dynamics."
        return (1-self.gamma)*self.s0[None,:] + self.gamma*self.P

    #___________________________________________________________________________
    # Operators

    def solve(self, b):
        "Solve linear system, (I - γ P) x = b"
        return linalg.solve(self.M, b)

    def solve_t(self, b):
        "Solve linear system, xᵀ (I - γ P) = b"
        return linalg.solve(self.M.T, b)
