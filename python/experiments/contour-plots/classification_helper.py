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


def define_model(init_variance, init_lengthscale, train_X, train_Y, model_name):
    likelihood = NewProbitBernoulli()
    kernel = gpflow.kernels.Matern52(
        variance=init_variance, lengthscales=init_lengthscale
    )

    if model_name == "cvi":
        model = t_VGP(
            data=(train_X, train_Y),
            kernel=kernel,
            likelihood=likelihood,
        )

    if model_name == "epcvi":
        model = t_VGP(
            data=(train_X, train_Y),
            kernel=kernel,
            likelihood=likelihood,
        )

    if model_name == "ep":
        model = GPy.models.GPClassification(
            X=train_X,
            Y=train_Y,
            kernel=GPy.kern.Matern52(
                train_X.shape[1], variance=init_variance, lengthscale=init_lengthscale
            ),
            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
            likelihood=GPy.likelihoods.Bernoulli(),
        )

    if model_name == "la":
        model = GPy.models.GPClassification(
            X=train_X,
            Y=train_Y,
            kernel=GPy.kern.Matern52(
                train_X.shape[1], variance=init_variance, lengthscale=init_lengthscale
            ),
            inference_method=GPy.inference.latent_function_inference.laplace.Laplace(),
            likelihood=GPy.likelihoods.Bernoulli(),
        )

    return model


def do_inference(model, model_name, train_X, train_Y, test_X, test_Y):
    N = train_X.shape[0]

    if model_name == "cvi":
        [model.update_variational_parameters(beta=0.1) for _ in range(200)]
        log_marginal = model.elbo().numpy()
        ep_est = model.ep_estimation().numpy()
        lpd = tf.reduce_mean(model.predict_log_density((test_X, test_Y))).numpy()
        return log_marginal / N, ep_est / N, lpd

    if model_name == "la" or model_name == "ep":
        _, log_marginal, _ = model.inference_method.inference(
            model.kern, train_X, model.likelihood, train_Y
        )
        lpd = np.mean(model.log_predictive_density(test_X, test_Y))
        return log_marginal / N, lpd
