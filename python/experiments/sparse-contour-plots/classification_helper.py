import numpy as np
import argparse
import pickle

import gpflow
import tensorflow as tf

import GPy


import sys

sys.path.append("../..")
from bayesian_benchmarks.data import get_classification_data


sys.path.append("../../../")
from src.tsvgp import t_SVGP
from src.tvgp import t_VGP
from src.new_bernoulli.logphi import NewProbitBernoulli


############################## general helper functions #############################


def return_cur_index(X, ID, kf):
    index = 0
    for train_index, test_index in kf.split(X):
        if ID == index:
            return train_index, test_index
        else:
            index += 1


def load_data(dataset_name):
    data = get_classification_data(dataset_name)
    X_train = data.X_train
    Y_train = data.Y_train
    X_test = data.X_test
    Y_test = data.Y_test

    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))

    Y[np.where(Y == 0)] = -1

    return X, Y


def define_sparse_model(init_variance, init_lengthscale, Z, model_name):
    likelihood = NewProbitBernoulli()
    kernel = gpflow.kernels.Matern52(
        variance=init_variance, lengthscales=init_lengthscale
    )

    if model_name == "svgp":
        model = gpflow.models.SVGP(
            kernel=kernel, likelihood=likelihood, inducing_variable=Z
        )

    if model_name == "tsvgp":
        model = t_SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=Z)

    return model


def do_inference_sparse(model, train_X, train_Y, test_X, test_Y):
    N = train_X.shape[0]

    [model.natgrad_step((train_X, train_Y), 0.1) for _ in range(200)]

    log_marginal = model.elbo((train_X, train_Y)).numpy()

    ep_est = model.ep_estimation((train_X, train_Y)).numpy()

    lpd = tf.reduce_mean(model.predict_log_density((test_X, test_Y))).numpy()

    return log_marginal / N, ep_est / N, lpd
