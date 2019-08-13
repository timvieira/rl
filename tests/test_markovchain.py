import pylab as pl
import numpy as np
from scipy import linalg
from arsenal.maths import random_dist, onehot
from arsenal import viz
from rl.markovchain import MarkovChain


def test():
    S = 10

    gamma = 0.7
    p0 = random_dist(S)
    P = random_dist(S,S)

    M = MarkovChain(s0=p0, P=P, gamma=gamma)

    d = M.d()

    assert np.allclose(d, M.d_by_eigen())

    lc = viz.lc['power']

    lc.baselines = {'eps_mach': np.finfo(np.float64).eps ** 2}

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
