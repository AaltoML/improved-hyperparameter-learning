"""
Binary Classification
"""

import pickle
import argparse
import numpy as np

import json

from scipy.stats import multinomial

import sys

sys.path.append("../..")

from bayesian_benchmarks.data import get_classification_data
from bayesian_benchmarks.models.get_model import get_classification_model
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.tasks.utils import meanlogsumexp

from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cvi", nargs="?", type=str)
parser.add_argument("--split", default=0, nargs="?", type=int)
parser.add_argument("--seed", default=0, nargs="?", type=int)
parser.add_argument("--cv_seed", default=42, nargs="?", type=int)
parser.add_argument("--dataset_id", default=1, nargs="?", type=int)
parser.add_argument("--ARD", default=0, nargs="?", type=int)

ARGS = parser.parse_args()

with open("dataset_list.json", "r") as f:
    dataset_list = json.load(f)
ARGS.dataset = dataset_list[ARGS.dataset_id][0]


def onehot(Y, K):
    Y[np.where(Y == -1)] = 0  ## convert label back to {0,1}
    return np.eye(K)[Y.flatten().astype(int)].reshape(Y.shape[:-1] + (K,))


is_test = False

data = get_classification_data(ARGS.dataset, split=ARGS.split)

X_train = data.X_train
Y_train = data.Y_train
X_test = data.X_test
Y_test = data.Y_test

X = np.concatenate((X_train, X_test))
Y = np.concatenate((Y_train, Y_test))

## NewBernoulli only support {-1,1}
Y[np.where(Y == 0)] = -1

kf = KFold(n_splits=5, shuffle=True, random_state=ARGS.cv_seed)

cv_result = []

for train_index, test_index in kf.split(X):
    data.X_train = X[train_index]
    data.Y_train = Y[train_index]
    data.X_test = X[test_index]
    data.Y_test = Y[test_index]

    model = get_classification_model(ARGS.model)(
        data.K, is_test=is_test, seed=ARGS.seed
    )

    Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # [1 x N_test x K]

    fold_res = model.fit(data.X_train, data.Y_train, ARGS.ARD)
    p = model.predict(data.X_test)  # [N_test x K] or [samples x N_test x K]

    assert p.ndim in {
        2,
        3,
    }  # 3-dim in case of approximate predictions (multiple samples per each X)

    # clip very large and small probs
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    p = p / np.expand_dims(np.sum(p, -1), -1)

    assert np.all(p >= 0.0) and np.all(p <= 1.0)

    # evaluation metrics
    res = {}

    if p.ndim == 2:  # keep analysis as in the original code in case 2-dim predictions
        logp = multinomial.logpmf(Y_oh, n=1, p=p)  # [N_test]

        res["test_loglik"] = np.average(logp)

        pred = np.argmax(p, axis=-1)

    else:  # compute metrics in case of 3-dim predictions
        res["test_loglik"] = []

        for n in range(p.shape[0]):  # iterate through samples
            logp = multinomial.logpmf(Y_oh, n=1, p=p[n])  # [N_test]
            res["test_loglik"].append(logp)

        # Mixture test likelihood (mean over per data point evaluations)
        res["test_loglik"] = meanlogsumexp(res["test_loglik"])

        p = np.mean(p, axis=0)
        pred = np.argmax(p, axis=-1)

    res["test_acc"] = np.average(np.array(pred == data.Y_test.flatten()).astype(float))
    res["fold_res"] = fold_res
    cv_result.append(res)

cv_result.append(ARGS.__dict__)

DATASET = cv_result[-1]["dataset"]
METHOD = cv_result[-1]["model"]

with open(
    f"results/isotropic/{DATASET}_{METHOD}_{ARGS.cv_seed}.pkl", "wb"
) as result_file:
    pickle.dump(cv_result, result_file)
