import pylab as pl
import numpy as np


class Halfspaces:
    "A x <= b"

    def __init__(self, A, b):
        self.A = np.array(A)
        self.b = np.array(b)
        [self.m, self.n] = self.A.shape
        assert self.b.shape == (self.m,), [A.shape, b.shape]

    def __call__(self, x):
        "Feasible?"
        return self.A @ x <= self.b

    def viz(self, xlim, ylim, ax=None):
        if ax is None: ax = pl.gca()
        assert self.n == 2   # todo: support 1 and 3 dimensional cases.
        for i in range(self.m):
            # a x <= b
            a = self.A[i]
            b = self.b[i]

            # x0 will be the x-axis
            [p,q] = a

            # p*x + q*y <= b
            # y <= (b - p*x)/q

            # y <= (b - p*x)/q
            # -(y*q-b)/p <= x

            xs = np.linspace(*xlim, 2)
            y2 = (b - p*xs)/q

            # Which side of the line do we fill?  The two-argument arc-tangent
            # function `arctan2(p, q)` gives the angle of the vector from
            # `<0,0>` to the point `<p,q>`.  The a positive angle tells us
            # whether to fill y values above, and a negative angle tells us to
            # fill the y values below.
            #
            # TODO: Handle the special case when the line is completely verical
            # (i.e., q=0)
            if q == 0:
                assert False, 'vertical lines not yet supported.'

            else:
                if np.arctan2(p, q) < 0:
                    y1 = ylim[1]*np.ones_like(xs)
                else:
                    y1 = ylim[0]*np.ones_like(xs)
                ax.plot(xs, y2, alpha=1, color='k')
                ax.fill_between(xs, y1, y2, alpha=1, color='grey')

        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)


def test():
    from arsenal.maths import spherical

    A = np.array([
        [ 0.06052216,  1.93689366],
        [-0.84213754,  1.19210556],
        [-0.16304193,  1.48302313]
    ])
    B = np.array([-0.6249662,  -1.85593953, -0.82883232])

    A = [[0.22032854, 0.65650877],
         [0.78670232, 0.24127292],
         [0.08878384, 0.68543539]]
    B = [1.47872332, 0.24802336, 1.60300663]

#    A *= -1
#    B *= -1

    A = np.array(A)
    B = np.array(B)

    for _ in range(10):
        [a,b,c] = spherical(3)

        print([a,b,c])

#    for i in range(len(B)):
#        [a,b],c = A[i], B[i]
#        print('>>>>', a, b, c)

        H = Halfspaces(
            [[a,b]],
            [c],
        )

        r = 10
        H.viz([-r,r], [-r,r])

        pts = np.random.uniform(-r,r,size=(10, 2))

        for x,y in pts:
            pl.scatter([x], [y], c = 'g' if H([x, y]) else 'r')

        pl.title(f'{a} x + {b} y <= {c}')
        pl.show()


if __name__ == '__main__':
    test()
