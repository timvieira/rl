import numpy as np
import pylab as pl
from arsenal.maths import compare, onehot, random_dist, is_distribution, spherical, fdcheck
from arsenal import colors, iterview
ok = colors.green % 'ok'

from rl.mdp import DiscountedMDP, FiniteHorizonMDP, MRP, random_MDP


def test():
    print()
    print('Finite-horizon tests:', ok)

    S = 10
    A = 3
    M = FiniteHorizonMDP(
        s0 = random_dist(S),
        R = np.random.uniform(0,1,size=(S,A,S)),
        P = random_dist(S,A,S),
        T = 20,
    )

    p = random_dist(M.S, M.A)
    assert abs(M.d(p).sum() - M.T)/M.T < 1e-5

    test_pd_lemma_finite_horizon(M)


def test_pd_lemma_finite_horizon(M):
    """
    Evaluate performance difference of `p` over `q` based on roll-outs from on
    `q` and roll-ins from `p`.
    """
    p = random_dist(M.S, M.A)
    q = random_dist(M.S, M.A)

    Jq,Vq,Qq = M.value(q)   # Roll-out with q
    dp = M.d(p)             # Roll-in with p. Note that dp sums to T, not 1.
    #assert dp.sum() == M.T

    Jp,_,_ = M.value(p)     # Value p.
    # Accumulate advantages of p over q.
    z = 0.0
    for t in range(M.T):
        for s in range(M.S):
            A = p[s,:] @ Qq[t,s,:] - Vq[t,s]
            z += dp[t,s] * A
    assert np.allclose(Jp - Jq, z)
    print('[pd-lemma]', ok)


if __name__ == '__main__':
    test()
