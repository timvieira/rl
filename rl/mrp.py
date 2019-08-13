import numpy as np
from scipy import linalg
from rl.markovchain import MarkovChain


class MRP(MarkovChain):
    "Markov reward process is Markov chain with a state-dependent reward function."

    def __init__(self, s0, P, R, gamma):
        super(MRP, self).__init__(s0, P, gamma)
        self.R = R
        self.gamma = gamma
        [self.S, _] = P.shape
        assert R.ndim == 1 and R.shape[0] == P.shape[0] == P.shape[1]

    def __iter__(self):
        return iter([self.s0, self.P, self.gamma, self.R])

    #___________________________________________________________________________
    # Simulation

    def run(self):
        "Simulate the MRP"
        s = self.start()
        while True:
            sp = self.step(s)
            yield s, self.R[s], sp
            s = sp

    #___________________________________________________________________________
    # Important quantities

    def J(self):
        "Expected value of the MRP"
        return self.s0 @ self.V()

    def V(self):
        "Value function"
        return self.solve(self.R)

    #___________________________________________________________________________
    # Operators

    def projected_bellman_error(self, F, v, w):
        """Average projected Bellman error under the MRP's stationary distribution.
        This quantity will go to zero in all solution methods, which minimize it.
        """
        w = np.diag(w)
        proj = F @ linalg.inv(F.T @ w @ F) @ F.T @ w
        resid = proj @ self.bellman_residual(v)
        return resid @ resid

    def bellman_residual(self, v):
        return (self.R + self.gamma*self.P @ v) - v

    #___________________________________________________________________________
    # Properties

    def epsilon_return_mixing_time(self):
        """The epsilon-return mixing time is the smallest truncated
        value function with a epsilon bounded estimation at all state (i.e.,
        under infinity norm).

        The quantity is upper bounded by

        H(epsilon) <= log_\gamma( epsilon * (1-gamma) / Rmax )

        """
        t = 0
        V = self.V()            # true value function
        Vt = np.zero(self.S)    # truncated-time estimate of value function
        while True:
            t += 1
            [Vt, _] = self.B(V)  # how many times do we have to apply the Bellman operator
            err = np.abs(V - Vt).max()   # infinity-norm
            yield t, err, Vt
