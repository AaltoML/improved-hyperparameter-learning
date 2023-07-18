import gpflow
import tensorflow as tf
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from scipy.stats import norm

from bayesian_benchmarks.models.new_bernoulli.logphi import NewProbitBernoulli

from gpflow.utilities.misc import set_trainable

try:
    from tqdm import trange
except ImportError:
    trange = range


class ClassificationModel:
    def __init__(self, K, is_test=False, seed=0):
        if is_test:

            class ARGS:
                num_inducing = 2
                iterations = 1
                small_iterations = 1
                adam_lr = 0.01
                minibatch_size = 100

        else:  # pragma: no cover

            class ARGS:
                num_inducing = 500
                iterations = 3000
                small_iterations = 1000
                adam_lr = 0.01
                minibatch_size = 100

        self.ARGS = ARGS

        self.K = K
        self.model = None
        self.model_objective = None
        self.opt = None

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
                lik = NewProbitBernoulli()  # gpflow.likelihoods.Bernoulli()
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

            self.model = gpflow.models.SVGP(
                kernel=k,
                likelihood=lik,
                inducing_variable=Z,
                num_latent_gps=num_latent_gps,
                whiten=True,
            )

            self.opt = gpflow.optimizers.Scipy()

            iters = self.ARGS.iterations

        else:
            iters = self.ARGS.small_iterations

        set_trainable(self.model.inducing_variable.Z, False)

        self.opt.minimize(
            self.model.training_loss_closure((X, Y)),
            variables=self.model.trainable_variables,
            options=dict(maxiter=self.ARGS.iterations, disp=False),
        )

        return (
            self.model.kernel.lengthscales.numpy(),
            self.model.kernel.variance.numpy(),
            self.model.elbo((X, Y)).numpy(),
        )

    def predict(self, Xs):
        m, v = self.model.predict_y(Xs)
        if self.K == 2:
            # convert Bernoulli to onehot
            return np.concatenate([1 - m, m], axis=1)
        else:
            return m
