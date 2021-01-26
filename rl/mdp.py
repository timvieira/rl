import numpy as np
import cvxpy as cp
from arsenal.maths import sample, random_dist, logsumexp, softmax
from numpy.random import uniform
from scipy import linalg

from rl.mrp import MRP


#def random_MDP(S, A, γ):
#    "Randomly generate an MDP."
#    return DiscountedMDP(
#        s0 = random_dist(S),
#        R = np.random.uniform(0,1,size=(S,A,S)),
#        P = random_dist(S,A,S),
#        γ = γ,
#    )


def random_MDP(S, A, γ=0.95, b=None, r=None):
    """Randomly generated MDP

    Text taken from http://www.jmlr.org/papers/volume15/geist14a/geist14a.pdf

      "... we consider Garnet problems (Archibald et al., 1995), which are a
      class of randomly constructed finite MDPs. They do not correspond to any
      specific application, but are totally abstract while remaining
      representative of the kind of MDP that might be encountered in
      practice. In our experiments, a Garnet is parameterized by 3 parameters
      and is written G(S, A, b): S is the number of states, A is the number of
      actions, b is a branching factor specifying how many possible next states
      are possible for each state-action pair (b states are chosen uniformly at
      random and transition probabilities are set by sampling uniform random b −
      1 cut points between 0 and 1). The reward is state-dependent: for a given
      randomly generated Garnet problem, the reward for each state is uniformly
      sampled between 0 and 1."

      "The discount factor γ is set to 0.95 in all experiments."

    We consider two types of problems, “small” and “big”, respectively
    corresponding to instances G(30, 2, p=2, dim=8) and G(100, 4, p=3, dim=20)

    """

    if b is None: b = S
    if r is None: r = S

    P = np.zeros((S,A,S))
    states = np.array(list(range(S)))

    #rs = np.random.choice(states, size=r, replace=False)

    for s in range(S):
        for a in range(A):
            # pick b states to be connected to.
            connected = np.random.choice(states, size=b, replace=False)
            P[s,a,connected] = random_dist(b)

    R = np.zeros((S,A,S))
    rstates = np.random.choice(states, size=r, replace=False)
    R[rstates,:,:] = np.random.uniform(0,1,r)

    M = DiscountedMDP(
        s0 = random_dist(S),
        R = R,
        P = P,
        γ = γ,
    )

    return M


class MDP(object):
    def __init__(self, s0, P, R):
        # P: Probability distribution p(S' | A S) stored as an array S x A x S'
        # R: Reward function r(S, A, S) -> Reals stored as an array S x A x S'
        # s0: Distribution over the initial state.
        self.s0 = s0
        [self.S, self.A, _] = P.shape
        self.P = P
        self.R = R

        # Compute `r(s,a)` from `r(s,a,s')`
        self.r = np.einsum('sap,sap->sa', self.P, self.R)


class FiniteHorizonMDP(MDP):
    "Finite-horizon MDP."

    def __init__(self, s0, P, R, T):
        super(FiniteHorizonMDP, self).__init__(s0, P, R)
        self.T = T

    def value(self, π):
        "Compute `T`-step value functions for policy `π`."
        S = self.S; A = self.A; P = self.P; R = self.R; T = self.T
        Q = np.zeros((T+1,S,A))
        V = np.zeros((T+1,S))
        for t in reversed(range(T)):
            for s in range(S):
                for a in range(A):
                    Q[t,s,a] = P[s,a,:] @ (R[s,a,:] + V[t+1,:])
                V[t,s] = π[s,:] @ Q[t,s,:]
        J = self.s0 @ V[0,:]
        # Value functions: No conditioning, state conditioned, state-action conditioned
        return J, V, Q

    def d(self, π):
        "Probability of state `s` under the given policy `π` conditioned on each time `t <= T`."
        S = self.S; A = self.A; P = self.P; T = self.T
        d = np.zeros((T, S))
        d[0,:] = self.s0
        for t in range(1, T):
            for sp in range(S):
                d[t,sp] = sum(d[t-1,s] * π[s,a] * P[s,a,sp] for s in range(S) for a in range(A))
            d[t] /= d[t].sum()   # normalized per-time step.
        return d


class DiscountedMDP(MDP):
    "γ-discounted, infinite-horizon Markov decision process."
    def __init__(self, s0, P, R, γ):
        # γ: Temporal discount factor
        super(DiscountedMDP, self).__init__(s0, P, R)
        self.γ = γ

    @property
    def gamma(self):
        return self.γ

    def __iter__(self):
        return iter((self.s0, self.P, self.R, self.γ))

    def copy(self):
        return DiscountedMDP(self.s0.copy(),
                             self.P.copy(),
                             self.R.copy(),
                             self.γ * 1)

    #___________________________________________________________________________
    # Simulation

    # XXX: add analogous methods to Markov chain
    def simulate(self, π, s=None, a=None, mode='direct'):
        """Simulation is under various `modes` -- i.e., interpretations of γ:

        - "direct": episodes never terminate and might not mix (i.e., get stuck
          in a subset of the state space if the underlying Markov chain is not
          ergodic conditioned on π).

        - "terminate": with probability (1-γ) the episode terminates.

        - "reset": with probability (1-γ) we reset the trajectory to an initial
          state drawn from `s0`.

        """
        # Coerce arrays into functions
        if isinstance(π, np.ndarray): π = lambda s, π=π: sample(π[s,:])
        assert mode in ('direct', 'terminate', 'reset'), mode
        if s is None: s = sample(self.s0)
        if a is None: a = π(s)
        while True:
            sp = sample(self.P[s,a,:])
            r = self.R[s,a,sp]            # this is consistent with r[s,a] = E[R[s,a,s']]
            yield (s,a,r)
            if mode != 'direct':
                if np.random.uniform(0,1) <= (1-self.γ):
                    if mode == 'terminate':
                        return
                    else:
                        sp = sample(self.s0)
            s = sp
            a = π(s)

#    # TODO: this method does not belong on this class. It should be part of some
#    # sort of a learning agent base class.
#    def learn(self, learner, max_iterations=None, callback=None, verbosity=False):
#        for t,(s,a,r,sp) in enumerate(self.run(learner), start=1):
#            if t >= max_iterations: break
#            if learner.update(s, a, r, sp): break
#            if verbosity and t % verbosity == 0:
#                callback(t, learner)

#    def run(self, π, s=None, a=None):
#        yield from self.simulate(π, s=s, a=a, mode='reset')

#    def step(self, s, a):
#        sp = sample(self.P[s,a,:])
#        if uniform(0,1) <= 1-self.γ:
#            sp = self.start()
#        r = self.R[s,a,sp]
#        return r, sp
#
#    def start(self):
#        return sample(self.s0)

    #___________________________________________________________________________
    # Conditioning

    def mrp(self, π):
        "MDP becomes an `MRP` when we condition on policy `π`."
        return MRP(self.s0,
                   np.einsum('sa,sap->sp', π, self.P),
                   np.einsum('sa,sap,sap->s', π, self.P, self.R),
                   self.γ)

    __or__ = mrp   # alias M | π

    #___________________________________________________________________________
    # Implicit functions

    def J(self, π):
        "Expected value of policy `π`."
        return (self | π).J()

    def d(self, π, jac=False):
        "same a `d(π)`, but includes a Jacobian-vector product callback."

        if not jac:
            return (self | π).d()
        else:
            γ = self.γ
            Φ = self.successor_representation(π)
            d = (1-γ) * self.s0 @ Φ

            def grad(adj):
                dM = np.einsum('sp,p,z->zs', Φ, adj, d)
                return γ * np.einsum('sap,sp->sa', self.P, dM)
            return d, grad

    def V(self, π, jac=False):
        "same a `V(π)`, but includes a vector-Jacobian product callback."

        if not jac:
            return (self | π).V()

        else:
            γ = self.γ
            Φ = self.successor_representation(π)
            r = np.einsum('sa,sa->s', self.r, π)
            v = Φ @ r

            def grad(adj):
                dM = np.einsum('sp,s,z->pz', Φ, adj, v)
                return np.einsum('s,sa->sa', adj @ Φ, self.r) + γ * np.einsum('sap,sp->sa', self.P, dM)
            return v, grad

        return v

    def successor_representation(self, π, normalize=False):
        "Dayan's successor representation."
        F = (self | π).successor_representation()
        if normalize: F /= (1 - self.γ)
        return F

    def sasa_matrix(self, π, normalize=True):
        # TODO: create a general operation that conditions the MDP on π such
        # that we get a Markov reward process with states that are the
        # state--action pairs of the MDP.
        #
        #   Under that view, This method computes the equivalent of the
        #   normalized successor reorientation of the Markov chain.
        #
        # Wang normalizes, we make that optional.
        # Without normalizatoin,
        #   Q[i,a] = sum(W[i,a,k,b] * R[k,b] for k in S for b in A))
        # With normalizaiton, we have to divide the rhs by (1-γ)

        S, A = self.S, self.A
        # Wang08's H matrix, H = (1-γ)I + γ Π P H   [ normalized case]
        I = np.eye(S*A)
        H = linalg.inv(I - self.γ * self.P.reshape(S*A, S) @ self.Π(π))
        H = H.reshape((S, A, S, A))
        if normalize: H /= (1 - self.γ)
        return H

    def Q(self, π):
        "Compute the action-value function `Q(s,a)` for a π."
        # See also: Q_by_linalg
        return self.Q_from_V((self | π).V())

    def Q_by_linalg(self, π):
        """
        Compute the action-value function `Q(s,a)` for a policy `π` by solving
        a linear system of equations.
        """
        # Notes: the implementation of the method is kind of messy (because of
        # all the reshaping business).  Also, the linear system for `V` is much
        # cleaner and more efficient because the linear system is smaller by a
        # factor of A.  So I've made computing Q by V the defualt method.
        r = self.r.ravel()
        return (linalg.solve(self.M(π), r).reshape((self.S, self.A)))

    def M(self, π):
        P = self.P.reshape((self.S*self.A, self.S))
        return np.eye(self.S*self.A) - self.γ * P @ self.Π(π)

    def Advantage(self, π):
        "Advantage function for policy π."
        return self.bellman_residual(self.V(π))

    def Π(self, π):
        """
        The policy matrix Π is an |S| × |S||A| matrix representation of π.

        It is convenient because,
          Π P = P_π = P(s' | s; π)
          Π R = R_π = E[ r | π ]
          P Π = Pr[ <s', a'> | s, a ]

        This definition is used in Lagoudakis and Parr (2003) and Wang et al. (2008).

        """

        Π = np.zeros((self.S, self.S, self.A))
        for s in range(self.S):
            Π[s, s, :] = π[s, :]
        return Π.reshape((self.S, self.S * self.A))

    def epsilon_return_mixing_time(self, π):
        """The epsilon-return mixing time (for a policy `π`) is the smallest truncated
        value function with a epsilon bounded estimation at all state (i.e.,
        under infinity norm).

        """
        yield from (self | π).epsilon_return_mixing_time()

    def P_with_reset(self):
        "Transition matrix with (1-γ)-resetting dynamics."
        # TODO: we might need to be careful when using this to make sure that
        # the reward is given before the state is restarted.
        return (1-self.γ)*self.s0[None,None,:] + self.γ*self.P

    #___________________________________________________________________________
    # Operators

    def B(self, V):
        "Bellman operator."
        # Act greedily according to one-step lookahead on V.
        Q = self.Q_from_V(V)
        if 1:
            π = np.zeros((self.S, self.A))
            π[range(self.S), Q.argmax(axis=1)] = 1
        else:
            # π will be uniformly random over equally good actions.
            π = Q == Q.max(axis=1)[:,None]
            π = π / π.sum(axis=1)[:,None]
        v = Q.max(axis=1)
        return v, π

    # TODO: There are many other smooth operators, e.g., sparsemax, mellowmean
    # (Asadi & Littman), Boltzman.

    def SB(self, V, τ):
        """Soft-Bellman operator: A smooth approximation to the Bellman operator.

        As the temperature parameter τ→0, the smoothed Bellman operator SB(V)
        approaches the nonsmooth Bellman operator B(V).
        """
        Q = self.Q_from_V(V) / τ
        v = logsumexp(Q, axis=1) * τ
        π = softmax(Q, axis=1)
        return v, π

    def Q_from_V(self, V):
        "Lookahead by a single action from value function estimate `V`."
        r = self.r
        Q = np.zeros((self.S, self.A))
        for s in range(self.S):
            for a in range(self.A):
                Q[s,a] = r[s,a] + self.γ*self.P[s,a,:] @ V
        return Q

    def bellman_residual(self, V):
        "The control case of the Bellman residual."
        return self.Q_from_V(V) - V[:,None]

    def apply_potential_based_shaping(self, ϕ):
        "Apply potential-based reward shaping"
        self.R[...] = self.shaped_reward(ϕ)
        self.r[...] = np.einsum('sap,sap->sa', self.P, self.R)

    def shaped_reward(self, ϕ):
        """
        Potential-based shaping augments the reward function with an extra
        term:  R'(s,a,s') = R(s,a,s') + γ ϕ(s') - ϕ(s)
        """
        # See also: performance-difference lemma
        γ = self.γ
        return self.R + γ*ϕ[None,None,:] - ϕ[:,None,None]

    #___________________________________________________________________________
    # Algorithms

    def solve_by_policy_iteration(self, max_iter=50):
        "Solve the MDP with the policy iteration algorithm."
        V = np.zeros(self.S)
        π_prev = np.zeros((self.S, self.A))
        for _ in range(max_iter):
            # Policy iteration does not take the value function from Bellman
            # operator (the variable `_` below). Instead, it uses the greedy
            # policy. (the greedy value function doesn't satisfy the first
            # Bellman equation). We find a new value function for the improved
            # policy, which satisfies the first Bellman equation.
            _, π = self.B(V)   # Bellman equation 2: the optimal policy is greedy with its value function
            V = self.V(π)      # Bellman equation 1: definition of value function for a policy
                               # Note: the value function returned by `B` is not the same as policy evaluation
            if (π_prev == π).all(): break
            π_prev = π
        else:
            print(f'Warning: policy iteration did not converge in {max_iter} iterations.')
        return {
            'obj': V @ self.s0,
            'policy': π,
            'V': V,
        }

    def solve_by_value_iteration(self, tol=1e-10):
        "Solve the MDP with the value iteration algorithm."
        V = np.zeros(self.S)
        while True:
            V1, π = self.B(V)
            if np.abs(V1 - V).max() < tol: break
            V = V1
        # Bounding the difference ||V_t - V_{t-1}||_inf < tol
        # bounds ||V_{greedy policy wrt V_t} - V*||_inf < 2*tol*γ/(1-γ),
        # which is a more meaningful bound.
        return {
            'obj': V @ self.s0,
            'policy': π,
            'V': V,
        }

    def solve_by_lp_dual(self):
        "Solve the MDP by dual linear programming (Wang et al., 2008)."

        # Note: The straightforward formulation of policy optimization is not an LP
        #
        #    maximize \sum_{s,a} r(s,a) π(a ∣ s) ⋅ μ(s)
        #
        # because μ (the distribution over states) is multiplied by the policy π.
        #
        # Luckily, there is a simple trick to avoid this nonlinearity.  What we
        # do is merge `π(a ∣ s)` and `μ(s)` into `μ(s,a) = π(a ∣ s) ⋅ μ(s)`.
        #
        #   linear objective:   maximize ∑_{s,a} r(s,a) μ(s,a)
        #
        # can recover the original variables by marginalization,
        #
        #    μ(s) = ∑_a μ(s,a)  and  π(a ∣ s) = μ(s,a) / μ(s)
        #
        # References:
        # http://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/slides/mdps.pdf

        P = self.P; S = list(range(self.S)); A = list(range(self.A)); γ = self.γ; r = self.r

        # Declare optimization variables (and constrain. μ(s,a) ≥ 0).
        μ = cp.Variable(shape=(self.S, self.A), nonneg=True)

        # Stationarity of μ(s,a):
        #  - Notes: In this constraint, `s0` is arbitrary: all you /actually/ need
        #    there is *some* distribution over s', e.g., a uniform distribution is
        #    fine. Actually, any positive function is fine, but then you have to
        #    renormalize to interpret μ as a joint distribution.
        constraints = [
            sum(μ[sp, ap] for ap in A)
            == (1-γ) * self.s0[sp] + γ * sum(μ[s,a] * P[s,a,sp] for s in S for a in A)
        for sp in S]

        # Note: Sum-to-one constraints are not necessary. Furthermore, if we do
        # these constraints they (superficially) mess up the interpertation of
        # the dual variables as value functions. this constraint, it shifts the
        # value-function interpretation of the Lagrange multipliers.
        # Specifically, if this is the last constraint (i.e., in the -1
        # position), V = λ[:-1] + λ[-1]

        # Note: that the LP objective is (1-γ)*J.  This is consistent with the
        # the primal.  See comments on the primal formulation.
        m = cp.Problem(
            cp.Maximize(sum(μ[s,a] * r[s,a] for s in S for a in A)),
            constraints
        )

        m.solve()

        # Extract solution
        mm = np.zeros((self.S, self.A))
        π = np.zeros((self.S, self.A))
        for s in S:
            for a in A:
                mm[s,a] = μ[s,a].value
        for s in S:
            if mm[s].sum() == 0:
                π[s,:] = 1/self.A
            else:
                π[s,:] = mm[s,:] / mm[s].sum()

        return {
            'obj': m.value / (1-γ),
            'policy': π,
            'mu': mm,
            'V': np.array([c.dual_value for c in constraints]),
        }

    def solve_by_lp_primal(self):
        "Solve the MDP by primal linear programming (Wang et al., 2008; Manne, 1960)."
        P = self.P; S = list(range(self.S)); A = list(range(self.A)); γ = self.γ; r = self.r

        V = cp.Variable(shape=self.S)  # note: no lb

        c = {(s,a):
             V[s] >= r[s,a] + γ * P[s,a,:] @ V
        for s in S for a in A}

        # Note: Wang+08, define their primal LP objective differently from the
        # usual definition of J (in terms of V).  Specifically, their version is
        # (1-γ)*J.  (why?  Probably to make the primal-dual relationship
        # cleaner). this is not the usual definition of J.
        m = cp.Problem(
            cp.Minimize((1-γ) * self.s0 @ V),
            c.values()
        )

        m.solve()

        # Extract solution, V*
        v = np.array([V[s].value for s in S])

        # Extracting the optimal policy requires 'one step of lookahead' on V*
        _, π = self.B(v)

        μ = np.zeros((self.S, self.A))
        for s in S:
            for a in A:
                μ[s,a] = c[s,a].dual_value

        return {
            'obj': m.objective.value / (1-γ),
            'policy': π,
            'mu': μ,
            'V': v,
        }

    #solve = solve_by_lp
    solve = solve_by_policy_iteration
