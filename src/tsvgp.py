"""
Module for the t-SVGP model class

The code is based on AaltoML/t-SVGP repo, we add implementation for EP estimation of
marginal likelihood in ep_estimation()

"""

import abc

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import kullback_leiblers
from gpflow.conditionals import conditional
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuf, Kuu
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import inducingpoint_wrapper

from gpflow.quadrature.gauss_hermite import NDiagGHQuadrature

quadrature = NDiagGHQuadrature(1, 20)

from src.utils.sites import DenseSites
from src.utils.util import (
    conditional_from_precision_sites,
    gradient_transformation_mean_var_to_expectation,
    posterior_from_dense_site,
)


class base_SVGP(GPModel, ExternalDataTrainingLossMixin, abc.ABC):
    """
    Modified gpflow.svgp.SVGP class to accommodate
    for different paramaterization of q(u)
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    @abc.abstractmethod
    def get_mean_chol_cov_inducing_posterior(self):
        """Returns the mean and cholesky factor of the covariance matrix of q(u)"""
        raise NotImplementedError

    def prior_kl(self) -> tf.Tensor:
        """Returns the KL divergence KL[q(u)|p(u)]"""
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, q_mu, q_sqrt, whiten=False
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        """
        The variational lower bound
        :param data: input data
        """
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        :param data: input data
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(
        self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        """
        Posterior prediction at new input Xnew
        :param Xnew: N x D Tensor
        """
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=False,
            full_output_cov=full_output_cov,
        )
        tf.debugging.assert_positive(var)
        return mu + self.mean_function(Xnew), var


class t_SVGP(base_SVGP):
    """
    Class for the t-SVGP model
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        ep_alpha=1.0,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        lambda_1=None,
        lambda_2_sqrt=None,
        num_data=None,
        force=False,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps)

        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self.num_inducing = self.inducing_variable.num_inducing

        self.ep_alpha = ep_alpha

        self._init_variational_parameters(self.num_inducing, lambda_1, lambda_2_sqrt)
        self.whiten = False
        self.force = force

    def _init_variational_parameters(
        self, num_inducing, lambda_1, lambda_2_sqrt, **kwargs
    ):
        """
        Constructs the site parameters λ₁, Λ₂.
        for site t(u) = exp(uᵀλ₁ - ½ uᵀΛ₂u)

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically referred to as M.
        :param lambda_1: np.array or None
            First order natural parameter of the variational site.
        :param lambda_2_sqrt: np.array or None
            Second order natural parameter of the variational site.
        """

        lambda_1 = (
            np.zeros((num_inducing, self.num_latent_gps))
            if lambda_1 is None
            else lambda_1
        )
        if lambda_2_sqrt is None:
            lambda_2_sqrt = [
                -tf.eye(num_inducing, dtype=default_float()) * 1e-10
                for _ in range(self.num_latent_gps)
            ]
            lambda_2_sqrt = np.array(lambda_2_sqrt)
        else:
            assert lambda_2_sqrt.ndim == 3
            self.num_latent_gps = lambda_2_sqrt.shape[0]

        self.sites = DenseSites(lambda_1, lambda_2_sqrt)

    @property
    def lambda_1(self):
        """first natural parameter"""
        return self.sites.lambda_1

    @property
    def lambda_2_sqrt(self):
        """Cholesky factor of the second natural parameter"""
        return self.sites.lambda_2_sqrt

    @property
    def lambda_2(self):
        """second natural parameter"""
        return tf.matmul(self.lambda_2_sqrt, self.lambda_2_sqrt, transpose_b=True)

    def get_mean_chol_cov_inducing_posterior(self):
        """
        Computes the mean and cholesky factor of the posterior
        on the inducing variables q(u) = 𝓝(u; m, S)
        S = (K⁻¹ + Λ₂)⁻¹ = (K⁻¹ + L₂L₂ᵀ)⁻¹ = K - KL₂W⁻¹L₂ᵀK , W = (I + L₂ᵀKL₂)⁻¹
        m = S λ₁
        """
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        return posterior_from_dense_site(K_uu, self.lambda_1, self.lambda_2_sqrt)

    # todo : make broadcastable
    def new_predict_f(
        self, Xnew: InputData, full_cov=False, full_output_cov=False
    ) -> MeanAndVariance:
        """
        Posterior prediction at new input Xnew
        :param Xnew: N x D Tensor
        """
        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        K_uf = Kuf(self.inducing_variable, self.kernel, Xnew)  # [P, M, M] or [M, M]
        K_ff = self.kernel.K_diag(Xnew)[..., None]

        mu, var = conditional_from_precision_sites(
            K_uu, K_ff, K_uf, self.lambda_1, L=self.lambda_2_sqrt
        )
        tf.debugging.assert_positive(
            var
        )  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def natgrad_step(self, data, lr=0.1, jitter=1e-9):
        """Takes natural gradient step in Variational parameters in the local parameters
        λₜ = rₜ▽[Var_exp] + (1-rₜ)λₜ₋₁
        Input:
        :param: X : N x D
        :param: Y:  N x 1
        :param: lr: Scalar

        Output:
        Updates the params
        """
        X, Y = data
        mean, var = self.predict_f(X)

        # todo : hack to get heterokedastic demo to run
        if isinstance(
            self.inducing_variable,
            gpflow.inducing_variables.SharedIndependentInducingVariables,
        ):
            meanZ, _ = self.predict_f(self.inducing_variable.inducing_variables[0].Z)
        else:
            meanZ, _ = self.predict_f(self.inducing_variable.Z)

        with tf.GradientTape() as g:
            g.watch([mean, var])
            ve = self.likelihood.variational_expectations(mean, var, Y)
        grads = g.gradient(ve, [mean, var])

        # cropping grads to stay negative
        eps = 1e-8
        grads[1] = tf.minimum(grads[1], -eps * tf.ones_like(grads[1]))

        Id = tf.eye(self.num_inducing, dtype=tf.float64)

        # Compute the projection matrix A from prior information
        K_uu = Kuu(self.inducing_variable, self.kernel)
        K_uf = Kuf(self.inducing_variable, self.kernel, X)  # [P, M, M] or [M, M]
        chol_Kuu = tf.linalg.cholesky(K_uu + Id * jitter)
        A = tf.transpose(tf.linalg.cholesky_solve(chol_Kuu, K_uf))

        # ▽μ₁[Var_exp] = aₙαₙ ,
        # ▽μ2[Var_exp] = λₙaₙaₙᵀ

        if tf.rank(A) == 2:
            A = tf.tile(A[..., None], [1, 1, self.num_latent_gps])
        grads = [
            tf.einsum("nml,nl->ml", A, grads[0]),
            tf.einsum("nml,nol,nl->lmo", A, A, grads[1]),
        ]

        # chain rule at f
        grad_mu = gradient_transformation_mean_var_to_expectation(meanZ, grads)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, dtype=tf.float64)
            minibatch_size = tf.cast(tf.shape(X)[0], dtype=tf.float64)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, dtype=tf.float64)

        lambda_2 = -0.5 * self.lambda_2
        lambda_1 = self.lambda_1
        # compute update in natural form
        lambda_1 = (1 - lr) * lambda_1 + lr * scale * grad_mu[0]
        lambda_2 = (1 - lr) * lambda_2 + lr * scale * grad_mu[1]

        # transform and perform update
        lambda_2_sqrt = -tf.linalg.cholesky(-2.0 * lambda_2 + Id * jitter)
        # To match SVGP you need to eliminate this jitter for minibatching
        self.lambda_1.assign(lambda_1)
        self.lambda_2_sqrt.assign(lambda_2_sqrt)
        self.get_mean_chol_cov_inducing_posterior()

    def ep_estimation(self, data) -> tf.Tensor:
        """
        This gives the log marginal likelihood estimated by EP sites
        """

        X, Y = data

        ep_alpha = self.ep_alpha

        # obtain natural parameters for data

        mean, var = self.predict_f(X)

        with tf.GradientTape() as g:
            g.watch([mean, var])
            ve = self.likelihood.variational_expectations(mean, var, Y)

        grads = g.gradient(ve, [mean, var])

        # cropping grads to stay negative
        eps = 1e-8
        grads[1] = tf.minimum(grads[1], -eps * tf.ones_like(grads[1]))

        lambda_1 = grads[0] - 2.0 * mean * grads[1]
        lambda_2 = grads[1]

        m_u, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        q_sqrt = tf.squeeze(q_sqrt)
        V_u = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)

        K_uu = Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [P, M, M] or [M, M]
        K_uf = Kuf(self.inducing_variable, self.kernel, X)  # [P, M, M] or [M, M]
        Luu_prior = tf.linalg.cholesky(K_uu)
        Ws = tf.linalg.cholesky_solve(
            Luu_prior, K_uf
        )  # [M, N], W = [w_1; w_2; ...; w_N]

        tau = -2.0 * lambda_2
        taun = tf.squeeze(tau, axis=1)
        g = -0.5 * lambda_1 / lambda_2
        gn = tf.squeeze(g, axis=1)
        gn_taun = gn * taun

        wnT_Vu_wn = tf.einsum("in,ij,jn->n", Ws, V_u, Ws)

        # 1 - w_n^T * α * τ_n * V_u * w_n
        constant_factor_n = 1.0 - ep_alpha * taun * wnT_Vu_wn  # [N, 1, 1]
        constant_factor_n = tf.maximum(
            constant_factor_n, 1e-5 * tf.ones_like(constant_factor_n)
        )

        part_1 = -0.5 * tf.math.log(constant_factor_n) / ep_alpha

        wnT_mu = tf.einsum("in,i->n", Ws, tf.squeeze(m_u, axis=1))

        part_2 = 0.5 * taun * wnT_mu**2 / constant_factor_n

        wnT_Vcavnu_wn = wnT_Vu_wn + ep_alpha * taun * wnT_Vu_wn**2 / constant_factor_n
        part_3_a = 0.5 * gn_taun**2 * ep_alpha * wnT_Vcavnu_wn
        part_3_b = -gn_taun * (
            wnT_mu + wnT_Vu_wn * wnT_mu * ep_alpha * taun / constant_factor_n
        )
        part_3 = part_3_a + part_3_b

        wnT_wn = tf.reduce_sum(Ws**2, axis=0)
        m_f_cavn = (
            wnT_mu
            + ep_alpha * wnT_Vu_wn * (wnT_mu * taun - gn_taun) / constant_factor_n
        )
        Kff = self.kernel(X, full_cov=False)
        V_f_cavn = Kff - tf.einsum("in,ij,jn->n", Ws, K_uu, Ws) + wnT_Vcavnu_wn

        N = tf.shape(X)[0]

        quad = quadrature.logspace(
            [lambda F, Y: self.likelihood.log_prob(F, Y)[..., tf.newaxis] ** ep_alpha],
            tf.reshape(m_f_cavn, [N, 1]),
            tf.reshape(V_f_cavn, [N, 1]),
            Y=Y,
        )

        log_Z = tf.reduce_sum(tf.squeeze(quad, axis=0))

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, log_Z.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], log_Z.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, log_Z.dtype)

        sparse_ep_energy = (
            tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_sqrt)))
            + 0.5 * tf.transpose(m_u) @ tf.linalg.cholesky_solve(q_sqrt, m_u)
            - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Luu_prior)))
            + scale * (log_Z / ep_alpha + tf.reduce_sum(part_1 + part_2 + part_3))
        )
        return tf.squeeze(sparse_ep_energy)
