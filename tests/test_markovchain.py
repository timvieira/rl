import pylab as pl
import numpy as np
from scipy import linalg
from arsenal.maths import random_dist, onehot
from arsenal import viz


def d_via_eigen(P):
    """
    Compute the stationary distribution via eigen methods.
    """
    [S, U] = linalg.eig(P.T)

    ss = np.argsort(S)
    S = S[ss]
    U = U[:,ss]

    s = U[:,-1].real
    s /= s.sum()
    return s




def test():
    S = 10

    gamma = 0.7
    p0 = random_dist(S)
    P = random_dist(S,S)

    R = onehot(0, S)

    from notes.rl.mdp import MRP
    M = MRP(s0=p0, P=P, R=R, gamma=gamma)
#    A = 1 - M.P_with_reset().T

    d = M.d()
    #d = d_via_eigen(A)

    lc = viz.lc['power']
    lc.yscale = 'log'
    lc.xscale = 'log'
    for t, x in enumerate(M.d_by_power_iteration(iterations=50), start=1):
        err = linalg.norm(x - d)**2
        print(t, err)
        lc.update(t, power = err)

    print('done')
    lc.draw()
    pl.ioff(); pl.show()


if __name__ == '__main__':
    test()
