import gpflow
import tensorflow as tf
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from gpflow.utilities.misc import set_trainable

import sys

sys.path.append("../../../")
from src.tsvgp import t_SVGP
from src.new_bernoulli.logphi import NewProbitBernoulli

try:
    from tqdm import trange
except ImportError:
    trange = range


class ClassificationModel:
    def __init__(self, K, is_test=False, seed=0):
        if is_test:

            class ARGS:
                num_inducing = 2
                iterations = 3
                small_iterations = 1
                adam_lr = 0.01
                minibatch_size = 100

        else:  # pragma: no cover

            class ARGS:
                num_inducing = 500
                iterations = 1000
                small_iterations = 1000
                adam_lr = 0.01
                minibatch_size = 100

        self.ARGS = ARGS

        self.K = K
        self.model = None
        self.model_objective = None
        self.opt = tf.optimizers.Adam(self.ARGS.adam_lr)

    def fit(self, X, Y, ARD):
        num_data, input_dim = X.shape

        if num_data > self.ARGS.num_inducing:
            np.random.seed(42)
            shuffle_index = np.random.permutation(num_data)

            Z, _ = kmeans2(
                X[shuffle_index][:50000], self.ARGS.num_inducing, minit="points"
            )
        else:
            Z = X.copy()

        if self.model is None:
            if self.K == 2:
                lik = NewProbitBernoulli()  # gpflow.likelihoods.Bernoulli(),
                num_latent_gps = 1
            else:
                lik = gpflow.likelihoods.MultiClass(self.K)
                num_latent_gps = self.K

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

            self.model = t_SVGP(
                kernel=k, likelihood=NewProbitBernoulli(), inducing_variable=Z
            )

        set_trainable(self.model.inducing_variable.Z, False)

        data = (X, Y)

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
        total_elbo.append(self.model.elbo(data).numpy())
        total_ep.append(self.model.ep_estimation(data).numpy())

        for _ in trange(self.ARGS.iterations):
            # E step
            for _ in range(20):
                self.model.natgrad_step(data, 0.1)

            # M step
            set_trainable(self.model.sites.lambda_1, False)
            set_trainable(self.model.sites.lambda_2_sqrt, False)
            for _ in range(20):
                self.opt.minimize(
                    lambda: -self.model.ep_estimation(data),
                    self.model.trainable_variables,
                )

            new_variables = gpflow.optimizers.Scipy.pack_tensors(
                self.model.variables
            ).numpy()
            new_params = gpflow.utilities.deepcopy(
                gpflow.utilities.parameter_dict(self.model)
            )

            len_trajectory.append(self.model.kernel.lengthscales.numpy())
            var_trajectory.append(self.model.kernel.variance.numpy())
            total_elbo.append(self.model.elbo(data).numpy())
            total_ep.append(self.model.ep_estimation(data).numpy())

            ep_diff = total_ep[-1] - total_ep[-2]

            stopped = 0

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
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], axis=1)
        else:
            return m
