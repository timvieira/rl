"""Tests for various MDP-related bounds and identities (and our implementation
of those identities).

Note that some tests occasionally fail because of numerical issues. (In most
cases, these issues could be handled in a more principled way by using the right
types of numerical tolerances.  Suggestions welcome.)

"""
import numpy as np
import pylab as pl
from arsenal.maths import assert_equal, compare, onehot, random_dist, is_distribution, spherical, fdcheck
from arsenal import colors
ok = colors.green % 'ok'

from rl.mdp import DiscountedMDP, MRP, random_MDP


def random_mrp(S, γ=0.3):
    # Randomly generate and MDP.
    return MRP(
        s0 = random_dist(S),
        R = np.random.uniform(0,1,size=S),
        P = random_dist(S, S),
        γ = γ,
    )


def test():
    M = random_MDP(20, 5, 0.7)

    test_dual_representation(M)
    print('done')

    test_solve(M)
    test_policy_matrix(M)
    test_stationary(M)
    test_lp_solver(M)
    test_performance_difference_lemma_discounted(M)
    test_potential_based_shaping(M)
    test_gradients(M)
    test_smooth(M)
    test_J(M)


def test_dual_representation(mdp):
    # Wang et al. 2008. "Dual Representations for Dynamic Programming" JMLR.
    # https://webdocs.cs.ualberta.ca/~dale/papers/dualdp.pdf

    S = range(mdp.S); A = range(mdp.A)
    γ = mdp.γ

    π = random_dist(mdp.S, mdp.A)
    Q = mdp.Q(π)
    V = mdp.V(π)
    R = mdp.r
    P = mdp.P

    Π = mdp.Π(π)

    # Wang08's H matrix, which I'll call W, is a Markov chain over (s,a) ->
    # (s'', a'')
    W = mdp.sasa_matrix(π, normalize=True)
    F = mdp.successor_representation(π, normalize=True)

    # Lemma 4
    assert np.all(F >= 0)
    assert np.allclose(1.0, np.einsum('ik->i', F))

    # Lemma 10 W ≥ 0 and W @ 1 = 1
    assert np.all(W >= 0)
    assert np.allclose(1.0, np.einsum('iakc->ia', W))

    # Q as a function of W
    assert np.allclose(Q*(1-γ), np.einsum('iakb,kb->ia',W,R))

    # Check that W solves our equations
    for k in S:
        for c in A:
            for i in S:
                for a in A:
                    np.allclose(
                        W[i,a,k,c],
                        (1-γ)*((i,a)==(k,c)) + γ*sum(W[j,b,k,c] * π[j,b] * P[i,a,j] for j in S for b in A)
                    )

    # Lemma 13
    assert np.allclose(V, Π @ Q.flat)

    # Lemma 14
    assert np.allclose(F @ Π, Π @ W.reshape(mdp.S*mdp.A, mdp.S*mdp.A))


def test_lp_solver(M):
    # primary testing strategy is to compare the linear programming solver (LP)
    # to anther solver (e.g., value iteration or policy iteration). In addition
    # to checking equivalence of the policies found by each, we also compare
    # equivalence of other quantities found by the LP. The dual variables should
    # be value functions (equal to VI's). The primal variables should be the
    # joint state-action distribution of the policy.
    vi = M.solve_by_policy_iteration()

    D = M.solve_by_lp_dual()
    P = M.solve_by_lp_primal()

    π = P['policy']
    assert np.allclose(P['policy'], vi['policy'])
    print('[lp-solver] policy', ok)

    # Objective value matches the solution found by VI.
    assert abs(D['obj'] - vi['obj']) / abs(vi['obj']) < 0.01
    print('[lp-solver] objective value', ok)

    d = D['mu'].sum(axis=1)
    assert is_distribution(d), 'stationary distribution is not a valid distribution.'
    assert compare(D['mu'].sum(axis=1), M.d(π), verbose=False).max_err < 1e-5
    print('[lp-solver] stationary distribution', ok)

    assert np.allclose(vi['V'], D['V'])
    print('[lp-solver] value function', ok)

    # Test the relationships between primal and dual LPs
#    assert np.allclose(P['policy'], D['policy'])    # behavior with ties is different.
    assert np.allclose(P['mu'],     D['mu'])
    print('[dual-lp-solver]', ok)

    # Test that the objectives match
    assert np.allclose(D['obj'], M.J(π))
    assert np.allclose(P['obj'], M.J(π))
    print('[lp-objectives]', ok)


def test_potential_based_shaping(M0):
    S = M0.S; A = M0.A; s0 = M0.s0

    opt_π = M0.solve()['policy']

    # generate a random potential function
    ϕ = np.random.uniform(-1, 1, size=S)

    M1 = M0.copy()    # use a copy!
    M1.apply_potential_based_shaping(ϕ)

    # Check that both methods found the same policy
    original = M0.solve()
    shaped = M1.solve()
    assert np.allclose(shaped['policy'], original['policy'])

#    opt_π = M0.solve()['policy']

    π = random_dist(S, A)

    v0 = M0.V(π)
    v1 = M1.V(π)

    # Corollary 2 of Ng et al. (1999).
    assert np.allclose(v0, v1 + ϕ)

    # Advantage is invariant to shaping.
    assert np.allclose(M0.Advantage(π), M1.Advantage(π))

    # The difference in J only depends on the initial state
    assert np.allclose(M0.J(π), M1.J(π) + s0 @ ϕ)
    print('[potential-based shaping] relationship between expected values and value functions', ok)

    # shaping with the optimal value function
    # TODO: are there other interesting things to say about this setting?
    vstar = original['V']
    M2 = M0.copy()  # use a copy of R!
    M2.apply_potential_based_shaping(vstar)

    assert np.allclose(0, M2.V(opt_π))  # optimal policy as V=0 everywhere and everything else in negative
    assert (M2.V(π) <= 0).all()         # suboptimal policies have negative value everywhere

    # optimal policy in the "optimally shapped MDP" can be found with γ=0!
    M2.γ *= 0
    assert (M2.solve()['policy'] == opt_π).all()

    # The optimal policy in M2 requires *zero* steps of lookahead (i.e., just
    # optimize immediate reward). The proof is pretty trivial.
    #
    # Given the V*-shaped reward R':
    #    R'[s,a,s'] def= R[s,a,s'] + γ V*[s'] - V*[s]
    #
    # R'[s,a] = sum_{s'} p(s' | s, a) * R'[s,a,s']
    #         = sum_{s'} p(s' | s, a) * (R[s,a,s'] + γ V*[s'] - V*[s])
    #         = A*(s,a)
    #
    # Acting greedily according to A*(s,a) is clearly optimal. Nonetheless, we
    # have an explict test below.
    assert np.allclose(M2.r, M0.Advantage(opt_π))
    M2_r = M2.r
    myopic_π = (M2_r == M2_r.max(axis=1)[:,None]) * 1.0
    assert np.allclose(myopic_π, opt_π)

    print('[potential-based shaping] "optimal shaping"', ok)


# TODO: PD-lemma as a functional derivative of a policy mixture ∇ₐ J[ (1-α) p + α q ]
# TODO: PD lemma can be used to give PG thm https://twitter.com/neu_rips/status/1180466116444987392
def test_performance_difference_lemma_discounted(M):
    """
    Evaluate performance difference of `p` over `q` based on roll-outs from on
    `q` and roll-ins from `p`.
    """

    p = random_dist(M.S, M.A)
    q = random_dist(M.S, M.A)

    dp = M.d(p)           # Roll-in with p
    Aq = M.Advantage(q)   # Roll-out with q
    # Accumulate advantages of p over q.
    z = 1/(1-M.γ) * sum(dp[s] * p[s,:] @ Aq[s,:] for s in range(M.S))

    assert np.allclose(M.J(p) - M.J(q), z)
    print('[pd-lemma]', ok)


    # The PD lemma is just potential-based shaping.
    #   See `test_potential_based_shaping` to read about potential-based shaping.
    #
    # Let `ϕ(s) = Vq(s)` where `Vq(s)` is the value function of some policy `q`.
    # The shaped reward is
    #
    #   R'(s,a,s') = R(s,a,s') + γ Vq(s') - Vq(s)
    #
    # Now take the expectation over s',
    #
    #   E_{s'}[ R'(s,a,s') ]
    #     = E_{s'}[ R(s,a,s') + γ Vq(s') - Vq(s) ]
    #     = E_{s'}[ R(s,a,s') + γ Vq(s')  ]  - Vq(s)
    #     = Qq(s,a) - Vq(s).
    #     = Aq(s, a)
    #
    # We see that the shaped reward function is the advantage of policy `q`.

    ϕ = M.V(q)
    M1 = M.copy()
    M1.apply_potential_based_shaping(ϕ)

    assert_equal(M1.J(p), M.J(p) - M.J(q), verbose=True)

    # Sanity check: q should have no advantive over itself.
    assert abs(M1.J(q)) < 1e-10



def test_policy_matrix(M):
    π = random_dist(M.S, M.A)
    Π = M.Π(π)
    m = M.mrp(π)

    np.allclose(Π @ M.r.reshape(M.S*M.A), m.R.flatten())
    np.allclose(Π @ M.P.reshape((M.S*M.A, M.S)), m.P)

    # Markov chain over state-action pairs <s,a> -> <s',a'>
    X = np.zeros((M.S, M.A, M.S, M.A))
    for s in range(M.S):
        for a in range(M.A):
            for sp in range(M.S):
                for ap in range(M.A):
                    X[s,a,sp,ap] = M.P[s,a,sp] * π[sp, ap]

    X = X.reshape((M.S*M.A, M.S*M.A))
    np.allclose(M.P.reshape((M.S*M.A, M.S)) @ Π, X)

    print('[policy matrix]', colors.light.green % 'ok')


def test_solve(M):
    vi = M.solve_by_value_iteration()
    PI = M.solve_by_policy_iteration()
    assert np.allclose(vi['V'], PI['V'])
    assert np.allclose(vi['policy'], PI['policy'])
    assert np.allclose(vi['obj'], PI['obj'])
    print('policy iteration == value iteration', ok)

    π = vi['policy']
    d = M.d(π)
    assert (d >= 0).all() and abs(d.sum() - 1) < 1e-10, 'stationary distribution did not a valid distribution.'
    print('stationary distribution is valid', ok)

    J = M.J(π)
    assert abs(J - vi['obj']) / abs(J) < 0.01
    print('value of policy matches VI', ok)

    v = M.V(π)
    assert np.allclose(v, vi['V'])
    print('value function', ok)

    Q = M.Q(π)
    assert np.allclose(v, np.einsum('sa,sa->s', Q, π))
    print('check V is average Q', ok)

    assert np.allclose(Q, M.Q_by_linalg(π))
    print('check Q by linalg', ok)


def test_J(M):
    # Test a single-state MRP
    # Sanity check: Why is there a 1/(1-γ) here?
    # if there is 1 state {
    #   rewards    = [r]
    #   stationary = [1]
    #   value      = r + γ value
    #              = r / (1-γ)
    #   J          = r / (1-γ)
    # }
    m1 = random_mrp(1)
    assert np.allclose(m1.J(), m1.R / (1-m1.γ))

    # Test equivalence of various methods for computing J.
    π = random_dist(M.S, M.A)
    [α, _, γ, r] = m = M | π

    T = 1 / (1-γ)

    J_by_V = α @ m.V()
    J_by_d = T * m.d() @ r
    J_by_S = α @ m.successor_representation() @ r

    J = m.J()
    assert np.allclose(J_by_d, J)
    assert np.allclose(J_by_S, J)
    assert np.allclose(J_by_V, J)

    # The reason why we have this equivalence is simply because of where we put
    # the parentheses
    #   (α @ m.successor_representation()) @ r
    #     = T dᵀ @ r
    # vs
    #   α @ (m.successor_representation() @ r)
    #     = α @ v

    assert np.allclose(α @ m.successor_representation(), T * m.d())
    assert np.allclose(m.successor_representation() @ r, m.V())

    # [2018-09-26 Wed] The following idea was tempting, but wrong! Here is where
    # my logic broke down: In the case of MDPs, we can use the performance
    # difference lemma (PD) to create a similar equation.  However, PD relates
    # the expected advantage function under a stationary distribution to the
    # difference of J's.  In the special case of a single PD of a policy versus
    # itself, we have that J'-J should be zero.  Note that the advantage of a
    # policy against itself is just the reward function.
    #
    # J_by_dV = T @ M.d() @ M.V()   # <=== INCORRECT!

    print('[test J]', ok)


def test_stationary(M):
    print('[test stationary]')

    π = random_dist(M.S, M.A)
    [_, _, γ, r] = M = M | π
    T = 1 / (1-γ)

    d1 = M.d()
    d2 = M.d_by_eigen()
    assert compare(d1, d2).max_relative_error < 1e-5

    J0 = M.J()
    d0 = M.d()

    def estimate(N):
        d = np.zeros(M.S)
        J = 0.0
        for t, [s, r, _] in enumerate(M.run(), start=1):
            if t >= N: break

            d += (onehot(s, M.S) - d) / t

            # Note the 'importance sampling correction' T, which accounts for
            # the (1-γ)-resetting dynamics.
            J += (r * T - J) / t

            if t % 1000 == 0:
                yield [
                    t,
                    0.5*abs(J - J0),
                    0.5*abs(d - d0).sum(),
                ]

    ns, J_err, d_err = np.array(list(estimate(1_000_000))).T

    dmax = 1
    Jmax = T * r.max()   # scaled by T because of the importance sampling correction.

    # Very loose bounds on total variation distance
    J_bnd = Jmax/np.sqrt(ns)
    d_bnd = M.S * dmax/np.sqrt(ns)

    if 0:
        # Error decays at a rate of 1/sqrt(N)
        pl.title('performance estimate')
        pl.loglog(ns, J_bnd, label='error bound')
        pl.loglog(ns, J_err, label='error observed')
        pl.show()

        pl.title('distribution estimate')
        pl.loglog(ns, d_bnd, label='error bound')
        pl.loglog(ns, d_err, label='error observed')
        pl.show()

    assert (J_err <= J_bnd).all()
    assert (d_err <= d_bnd).all()


def test_gradients(M):

    J = lambda: M.J(π)

    π = random_dist(M.S, M.A)
    r = M.r

    # The policy gradient theorem
    fdcheck(J, π,
            1/(1-M.γ) * M.d(π)[:,None] * M.Q(π),    # Note: not Q is not interchangeable with Advantage!
    ) #.show(title='policy gradient v1.')

    print('[policy gradient theorem]', ok)

    # Jacobians of the implicit d(p) and v(p) functions.
    z = spherical(M.S)
    _d, d_grad = M.d(π, jac=True)
    fdcheck(lambda: z @ M.d(π), π, d_grad(z))#.show(title='implicit d')
    _v, v_grad = M.V(π, jac=True)
    fdcheck(lambda: z @ M.V(π), π, v_grad(z))#.show(title='implicit v')

    # check that the implicit functions are consistent with the other methods for computing them.
    assert np.allclose(_d, M.d(π))
    assert np.allclose(_v, M.V(π))

    # The policy gradient theorem
#    fdcheck(J, p,
#            1/(1-M.γ) * (
#                np.einsum('s,sa->sa', M.d(p), M.Advantage(p))
#                + (M.d(p) * M.V(p))[:,None]
#            )
#    ) .show(title='policy gradient v1.')

    # Extract the full Jacobian, flatten SA dim of policy
    Jdp = np.zeros((M.S, M.S*M.A))
    for s in range(M.S):
        Jdp[s, :] = d_grad(onehot(s, M.S)).flat

    # The stuff below is the chaining from J to derivatives thru π
    fdcheck(J, π,
            1/(1-M.γ)*(np.einsum('sa,sa->s', r, π) @ Jdp
                           + np.einsum('s,sa->sa', M.d(π), r).flatten())
    ) #.show(title='policy gradient v2.')


    # Extract the full Jacobian, flatten SA dim of policy
    Jvp = np.zeros((M.S, M.S*M.A))
    for s in range(M.S):
        Jvp[s, :] = v_grad(onehot(s, M.S)).flat
    fdcheck(J, π, M.s0 @ Jvp) #.show(title='policy gradient v2a.')
    fdcheck(J, π, v_grad(M.s0)) #.show(title='policy gradient v2b.')


def test_smooth(M):

    # TODO: Should we make policy iteration and value iteration generic in the
    # fixed-point operator?

    v = np.random.uniform(-1,1,size=M.S)

    τ = 1e-10
    v1, π1 = M.SB(v, τ)
    v2, π2 = M.B(v)

    assert np.allclose(v1, v2)
    assert np.allclose(π1, π2)

    τ = 1e10
    v1, π1 = M.SB(v, τ)

    assert np.allclose(π1, np.ones((M.S, M.A))/M.A), π1
    #assert np.allclose(v1, τ + np.log(M.A)), v1
    #from IPython import embed; embed()

    tol = 1e-8

    # Very crude integration test, check that SB's fixed point roughly matches
    # B's fixed point at low temperatures.
    τ = 1e-5

    while True:
        vp, π = M.SB(v, τ)
        diff = np.abs(vp - v).max()
        if diff < tol:
            break
        v = vp

    opt = M.solve()
    opt_π = opt['policy']
    opt_v = opt['V']

    #print()
    #print(π)
    #print(opt_π)

    assert np.abs(π - opt_π).max() <= 1e-10
    assert np.max(np.abs(v - opt_v) / np.abs(opt_v)) <= 0.001

    #print(v)
    #print(opt_v)
    #from IPython import embed; embed()


if __name__ == '__main__':
    test()
