import gpflow
import numpy as np
from scipy.stats import norm
from gpflow.utilities.misc import set_trainable
from bayesian_benchmarks.models.new_bernoulli.logphi import NewProbitBernoulli

import tensorflow as tf


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
            k = gpflow.kernels.Matern52(
                variance=1.0,
                lengthscales=tf.ones(
                    [
                        tf.shape(X)[1],
                    ]
                ),
            )
        else:
            k = gpflow.kernels.Matern52(variance=1.0, lengthscales=1.0)

        self.model = gpflow.models.VGP(
            data=(X, Y),
            kernel=k,
            likelihood=NewProbitBernoulli(),  # gpflow.likelihoods.Bernoulli(),
        )

        stopped = 0

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            self.model.training_loss,
            variables=self.model.trainable_variables,
            options=dict(maxiter=self.ARGS.iterations, disp=False),
        )

        return stopped

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)  # , session=self.sess)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m
