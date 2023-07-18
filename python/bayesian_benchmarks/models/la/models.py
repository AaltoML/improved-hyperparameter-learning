import numpy as np
from scipy.stats import norm
import GPy


class ClassificationModel(object):
    def __init__(self, K, is_test=False, seed=0):
        if is_test:

            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                initial_likelihood_var = 0.01

        else:  # pragma: no cover

            class ARGS:
                num_inducing = 100
                iterations = 10000
                small_iterations = 1000
                initial_likelihood_var = 0.01

        self.ARGS = ARGS
        self.K = K
        self.model = None

    def fit(self, X, Y, ARD):
        if ARD:
            k = GPy.kern.Matern52(
                X.shape[1],
                ARD=True,
                variance=1.0,
                lengthscale=np.ones(
                    X.shape[1],
                ),
            )
        else:
            k = GPy.kern.Matern52(X.shape[1], variance=1.0, lengthscale=1.0)

        lik = GPy.likelihoods.Bernoulli()

        self.model = GPy.models.GPClassification(
            X=X,
            Y=Y,
            kernel=k,
            inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
            likelihood=lik,
        )

        stopped = 0
        self.model.optimize(max_iters=self.ARGS.iterations)

        return stopped

    def predict(self, Xs):
        m, v = self.model.predict(Xs)  # , session=self.sess)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m
