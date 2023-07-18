import numpy as np
import argparse
import pickle

import gpflow
from gpflow.utilities.misc import set_trainable
import tensorflow as tf

import sys

sys.path.append("../..")
from bayesian_benchmarks.data import get_regression_data


sys.path.append("../../../")

from src.tvgp_crop import t_VGP


def return_cur_index(X, ID, kf):
    index = 0
    for train_index, test_index in kf.split(X):
        if ID == index:
            return train_index, test_index
        else:
            index += 1


def load_data(dataset_name):
    try:
        data = np.loadtxt(
            f"../../bayesian_benchmarks/data/{dataset_name}.txt", delimiter=","
        )

        X = data[:, 0].reshape(-1, 1)
        Y = data[:, 1].reshape(-1, 1)

    except:
        data = get_regression_data("boston")
        X_train = data.X_train
        Y_train = data.Y_train
        X_test = data.X_test
        Y_test = data.Y_test

        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((Y_train, Y_test))

    return X, Y


def define_model(init_variance, init_lengthscale, train_X, train_Y, model_name):
    if model_name == "cvi":
        model = t_VGP(
            data=(train_X, train_Y),
            kernel=gpflow.kernels.Matern52(
                variance=init_variance, lengthscales=init_lengthscale
            ),
            likelihood=gpflow.likelihoods.StudentT(),
        )

    if model_name == "epcvi":
        model = t_VGP(
            data=(train_X, train_Y),
            kernel=gpflow.kernels.Matern52(
                variance=init_variance, lengthscales=init_lengthscale
            ),
            likelihood=gpflow.likelihoods.StudentT(),
        )

    if model_name == "vi":
        model = gpflow.models.VGP(
            data=(train_X, train_Y),
            kernel=gpflow.kernels.Matern52(
                variance=init_variance, lengthscales=init_lengthscale
            ),
            likelihood=gpflow.likelihoods.StudentT(),
        )

    return model


def fit_model(model, steps, e_step, m_step, natgrad_lr, model_name):
    optimizer = tf.optimizers.Adam()

    if model_name == "cvi":
        old_variables = gpflow.optimizers.Scipy.pack_tensors(model.variables).numpy()

        for i in range(steps):
            # E step
            for _ in range(e_step):
                model.update_variational_parameters(beta=natgrad_lr)

            # M step
            set_trainable(model.sites.lambda_1, False)
            set_trainable(model.sites.lambda_2, False)
            for _ in range(m_step):
                optimizer.minimize(lambda: -model.elbo(), model.trainable_variables)

            new_variables = gpflow.optimizers.Scipy.pack_tensors(
                model.variables
            ).numpy()
            stopped = 0

            if np.allclose(old_variables, new_variables):
                stopped = 1
                break
            else:
                old_variables = new_variables

        return model

    if model_name == "epcvi":
        old_variables = gpflow.optimizers.Scipy.pack_tensors(model.variables).numpy()

        for i in range(steps):
            # E step
            for _ in range(e_step):
                model.update_variational_parameters(beta=natgrad_lr)

            # M step
            set_trainable(model.sites.lambda_1, False)
            set_trainable(model.sites.lambda_2, False)
            for _ in range(m_step):
                optimizer.minimize(
                    lambda: -model.ep_estimation(), model.trainable_variables
                )

            new_variables = gpflow.optimizers.Scipy.pack_tensors(
                model.variables
            ).numpy()
            stopped = 0

            if np.allclose(old_variables, new_variables):
                stopped = 1
                break
            else:
                old_variables = new_variables

        return model

    if model_name == "vi":
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss,
            variables=model.trainable_variables,
            options=dict(maxiter=steps, disp=False),
        )

        return model
