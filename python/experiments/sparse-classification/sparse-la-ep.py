iters = 500

import numpy as np
import bayesnewton
import objax
import time
import jax.numpy as jnp

import pickle
import argparse
import numpy as np

import json

from scipy.stats import multinomial

from bayesian_benchmarks.data import get_classification_data

from sklearn.model_selection import KFold

from scipy.cluster.vq import kmeans2

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="ep", nargs="?", type=str)
parser.add_argument("--split", default=0, nargs="?", type=int)
parser.add_argument("--seed", default=0, nargs="?", type=int)
parser.add_argument("--dataset_id", default=1, nargs="?", type=int)
parser.add_argument("--num_inducing", default=500, nargs="?", type=int)
parser.add_argument("--cv_seed", default=42, nargs="?", type=int)
parser.add_argument("--exp_name", default="", nargs="?", type=str)

ARGS = parser.parse_args()

with open("dataset_list.json", "r") as f:
    dataset_list = json.load(f)
ARGS.dataset = dataset_list[ARGS.dataset_id][0]


def onehot(Y, K):
    Y[np.where(Y == -1)] = 0  ## convert label back to {0,1}
    return np.eye(K)[Y.flatten().astype(int)].reshape(Y.shape[:-1] + (K,))


data = get_classification_data(ARGS.dataset, split=ARGS.split)

X_train = data.X_train
Y_train = data.Y_train
X_test = data.X_test
Y_test = data.Y_test

X = np.concatenate((X_train, X_test))
Y = np.concatenate((Y_train, Y_test))


kf = KFold(n_splits=5, shuffle=True, random_state=ARGS.cv_seed)

lr_adam = 0.1
lr_newton = 0.1


def train(model):
    opt_hypers = objax.optimizer.Adam(model.vars())
    energy = objax.GradValues(model.energy, model.vars())

    @objax.Function.with_vars(model.vars() + opt_hypers.vars())
    def train_op():
        model.inference(
            lr=lr_newton
        )  # perform inference and update (variational) params
        dE, E = energy()  # compute energy and its gradients w.r.t. hypers
        opt_hypers(lr_adam, dE)
        return E

    train_op = objax.Jit(train_op)

    t0 = time.time()
    for i in range(1, iters + 1):
        loss = train_op()
        if i % 10 == 0:
            print("iter %2d, energy: %1.4f" % (i, loss[0]))
    t1 = time.time()
    # print('Optimisation time: %2.2f seconds' % (t1-t0))


num_inducing = ARGS.num_inducing

cv_result = []
for train_index, test_index in kf.split(X):
    data.X_train = X[train_index]
    data.Y_train = Y[train_index]
    data.X_test = X[test_index]
    data.Y_test = Y[test_index]

    num_data = data.X_train.shape[0]

    np.random.seed(42)
    shuffle_index = np.random.permutation(num_data)

    Z, _ = kmeans2(X[shuffle_index][:50000], num_inducing, minit="points")

    # Likelihood (Bernoulli/classification)
    lik = bayesnewton.likelihoods.Bernoulli(link="logit")

    if ARGS.model == "ep":
        model = bayesnewton.models.SparseExpectationPropagationGP(
            kernel=bayesnewton.kernels.Matern52(variance=1.0, lengthscale=1.0),
            likelihood=lik,
            X=X,
            Y=Y.reshape(-1),
            Z=Z.copy(),
            opt_z=False,
        )
    if ARGS.model == "la":
        model = bayesnewton.models.SparseLaplaceGP(
            kernel=bayesnewton.kernels.Matern52(variance=1.0, lengthscale=1.0),
            likelihood=lik,
            X=X,
            Y=Y,
            Z=Z.copy(),
            opt_z=False,
        )

    train(model)

    Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # [1 x N_test x K]
    Y_oh_train = onehot(data.Y_train, data.K)[None, :, :]

    p, _ = model.predict_y(data.X_test)
    p_train, _ = model.predict_y(data.X_train)

    p = p.reshape(-1, 1)
    p_train = p_train.reshape(-1, 1)

    p = np.concatenate([1 - p, p], 1)
    p_train = np.concatenate([1 - p_train, p_train], 1)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p = p / np.expand_dims(np.sum(p, -1), -1)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)

    # evaluation metrics
    res = {}

    logp = multinomial.logpmf(Y_oh, n=1, p=p)  # [N_test]
    res["test_loglik"] = np.average(logp)

    logp_train = multinomial.logpmf(Y_oh_train, n=1, p=p_train)  # [N_test]
    res["train_loglik"] = np.average(logp_train)

    pred = np.argmax(p, axis=-1)
    pred_train = np.argmax(p_train, axis=-1)

    res["test_acc"] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))
    res["train_acc"] = np.average(
        np.array(pred_train == data.Y_train.flatten()).astype(float)
    )
    # res["fold_res"] = fold_res
    cv_result.append(res)


cv_result.append(ARGS.__dict__)

DATASET = cv_result[-1]["dataset"]
METHOD = cv_result[-1]["model"]


with open(
    f"results/isotropic/{DATASET}_{METHOD}_{ARGS.exp_name}_{ARGS.cv_seed}.pkl", "wb"
) as result_file:
    pickle.dump(cv_result, result_file)
