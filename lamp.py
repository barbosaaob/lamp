"""
LAMP projection
"""
from __future__ import print_function
import numpy as np
from projection import projection
from force import force


class Lamp(projection.Projection):
    def __init__(self, data, data_class, sample=None, sample_projection=None):
        assert type(data) is np.ndarray, "*** ERROR (LAMP): Data is of wrong \
                type!"

        projection.Projection.__init__(self, data, data_class)
        self.sample = sample
        self.sample_projection = sample_projection
        # self.initialization()

    def initialization(self):
        """
        TODO: fix sample condition
        """
        # ninst = self.data_ninstances
        sample_condition = self.sample and (not self.sample_projection)

        print(bool(self.sample), bool(not self.sample_projection))
        print(bool(sample_condition))
        print(bool(not sample_condition))

        if sample_condition or (not sample_condition):
            print("*** WARNING (LAMP): Using random sample!")
            self.sample = None
            self.sample_projection = None

        # if not self.sample:
        #     self.sample = np.random.permutation(ninst)
        # else:
        #     self.sample = sample

        # if not self.sample_projection:
        #     force_proj = force.Force(self.data[self.sample, :], [])
        #     force_proj.project()
        #     self.sample_projection = force_proj.get_projection()
        # else:
        #     self.sample_projection = sample_projection

    def project(self):
        tol = 1e-6
        ninst, dim = self.data.shape    # number os instances, data dimension
        k = len(self.sample)            # number os sample instances
        p = self.projection_dim         # visual space dimension
        x = self.data
        xs = self.data[self.sample, :]
        ys = self.sample_projection

        for pt in range(ninst):
            # compute alphas
            alpha = np.zeros(k)
            for i in range(k):
                # verify if the point to be projectec is a control point
                # avoids division by zero
                if np.linalg.norm(xs[i] - x[pt]) < tol:
                    alpha[i] = 1e14
                else:
                    alpha[i] = 1.0 / np.linalg.norm(xs[i] - x[pt])**2

            # computes x~ and y~ (eq 3)
            xtilde = np.zeros(dim)
            ytilde = np.zeros(p)
            for i in range(k):
                xtilde += alpha[i] * xs[i]
                ytilde += alpha[i] * ys[i]
            xtilde /= np.sum(alpha)
            ytilde /= np.sum(alpha)

            A = np.zeros((k, dim))
            B = np.zeros((k, p))
            xhat = np.zeros((k, dim))
            yhat = np.zeros((k, p))
            # computation of x^ and y^ (eq 6)
            for i in range(k):
                xhat[i] = xs[i] - xtilde
                yhat[i] = ys[i] - ytilde
                A[i] = np.sqrt(alpha[i]) * xhat[i]
                B[i] = np.sqrt(alpha[i]) * yhat[i]

            U, D, V = np.linalg.svd(np.dot(A.T, B))  # (eq 7)
            # VV is the matrix V filled with zeros
            VV = np.zeros((dim, p))  # size of U = dim, by SVD
            for i in range(p):  # size of V = p, by SVD
                VV[i, range(p)] = V[i]

            M = np.dot(U, VV)  # (eq 7)

            self.projection[pt] = np.dot(x[pt] - xtilde, M) + ytilde  # (eq 8)


def run():
    import time
    import sys
    print("Loading data set... ", end="")
    sys.stdout.flush()
    data_file = np.loadtxt("iris.data")
    print("Done.")
    ninst, dim = data_file.shape
    sample_size = int(np.ceil(np.sqrt(ninst)))
    data = data_file[:, range(dim - 1)]
    data_class = data_file[:, dim - 1]
    sample = np.random.permutation(ninst)
    sample = sample[range(sample_size)]
    # force
    start_time = time.time()
    print("Projecting samples... ", end="")
    sys.stdout.flush()
    f = force.Force(data[sample, :], [])
    f.project()
    sample_projection = f.get_projection()
    print("Done. (" + str(time.time() - start_time) + "s.)")
    # lamp
    start_time = time.time()
    print("Projecting... ", end="")
    sys.stdout.flush()
    lamp = Lamp(data, data_class, sample, sample_projection)
    lamp.project()
    print("Done. (" + str(time.time() - start_time) + "s.)")
    lamp.plot()


if __name__ == "__main__":
    run()
