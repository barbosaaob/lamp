"""
Force-Scheme
"""

from __future__ import print_function
from projection import projection
import numpy as np


class Force(projection.Projection):
    def __init__(self, data, data_class, dtype="data", delta_frac=8,
                 niter=50, tol=1.0e-6):
        projection.Projection.__init__(self, data, data_class)
        self.dtype = dtype
        self.delta_frac = delta_frac
        self.niter = niter
        self.tol = tol

    def project(self):
        """
        Projection
        """
        assert type(self.data) is np.ndarray, \
            "*** ERROR (Force-Scheme): project input must be of numpy.array" \
            "type."

        # number of instances, dimension of the data
        ninst = self.data_ninstances

        Y = np.random.random((ninst, 2))    # random initialization

        # computes distance in R^n
        if self.dtype == "data":
            distRn = self.pdist(self.data)
        elif self.dtype == "dmat":
            distRn = self.data
        else:
            print("*** ERROR (Force-Scheme): Undefined data type.")
        assert type(distRn) is np.ndarray and distRn.shape == (ninst, ninst), \
            "*** ERROR (Force-Scheme): project input must be numpy.array type."

        idx = np.random.permutation(ninst)

        for k in range(self.niter):
            # for each x'
            for i in range(ninst):
                inst1 = idx[i]
                # for each q' != x'
                for j in range(ninst):
                    inst2 = idx[j]
                    if inst1 != inst2:
                        # computes direction v
                        v = Y[inst2] - Y[inst1]
                        distR2 = np.hypot(v[0], v[1])
                        if distR2 < self.tol:
                            distR2 = self.tol
                        delta = (distRn[inst1][inst2] - distR2)/self.delta_frac
                        v /= distR2
                        # move q' = Y[j] in the direction of v by a fraction
                        # of delta
                        Y[inst2] += delta * v
        self.projection = Y

    def pdist(self, x):
        """
        Pairwise distance between pairs of objects
        TODO: find a faster function
        """
        n, d = x.shape
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.linalg.norm(x[i] - x[j])
        return dist


def test():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data_file = np.loadtxt("mammals.data", delimiter=",")
    print("Done.")
    n, dim = data_file.shape
    data = data_file[:, range(dim - 1)]
    data_class = data_file[:, dim - 1]
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    force = Force(data, data_class)
    force.project()
    print("Done. (" + str(time.time() - start_time) + "s)")
    force.plot()

if __name__ == "__main__":
    test()
