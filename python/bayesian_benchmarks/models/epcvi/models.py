import gpflow
import numpy as np
from scipy.stats import norm
import tensorflow as tf

from gpflow.utilities.misc import set_trainable


import sys

sys.path.append("../../../")
from src.tvgp import t_VGP
from src.new_bernoulli.logphi import NewProbitBernoulli


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
        optimizer = tf.optimizers.Adam(0.01)

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

        self.model = t_VGP(
            data=(X, Y),
            kernel=k,
            likelihood=NewProbitBernoulli(),  # gpflow.likelihoods.Bernoulli(),
        )

        old_variables = gpflow.optimizers.Scipy.pack_tensors(
            self.model.variables
        ).numpy()
        old_params = gpflow.utilities.deepcopy(
            gpflow.utilities.parameter_dict(self.model)
        )

        len_trajectory = []
        var_trajectory = []

        total_ep = []
        total_elbo = []

        len_trajectory.append(self.model.kernel.lengthscales.numpy())
        var_trajectory.append(self.model.kernel.variance.numpy())
        total_elbo.append(self.model.elbo().numpy())
        total_ep.append(self.model.ep_estimation().numpy())

        for step in range(self.ARGS.iterations):
            # E step
            for _ in range(20):
                self.model.update_variational_parameters(beta=0.1)

            # M step
            set_trainable(self.model.sites.lambda_1, False)
            set_trainable(self.model.sites.lambda_2, False)

            for _ in range(20):
                optimizer.minimize(
                    lambda: -self.model.ep_estimation(), self.model.trainable_variables
                )

            new_variables = gpflow.optimizers.Scipy.pack_tensors(
                self.model.variables
            ).numpy()
            new_params = gpflow.utilities.deepcopy(
                gpflow.utilities.parameter_dict(self.model)
            )

            len_trajectory.append(self.model.kernel.lengthscales.numpy())
            var_trajectory.append(self.model.kernel.variance.numpy())
            total_elbo.append(self.model.elbo().numpy())
            total_ep.append(self.model.ep_estimation().numpy())

            ep_diff = total_ep[-1] - total_ep[-2]

            stopped = 0

            # print(step, ep_diff)

            if np.allclose(old_variables, new_variables) or ep_diff < 0:
                stopped = 1
                gpflow.utilities.multiple_assign(self.model, old_params)
                break

            else:
                old_variables = new_variables
                old_params = new_params

        result = {
            "len_trajectory": len_trajectory,
            "var_trajectory": var_trajectory,
            "total_elbo": total_elbo,
            "total_ep": total_ep,
            "stopped": stopped,
        }

        return result

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)  # , session=self.sess)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], 1)
        else:
            return m
