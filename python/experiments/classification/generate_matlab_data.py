import json
import sys

sys.path.append("../..")
from bayesian_benchmarks.data import get_classification_data
from scipy.io import savemat
from sklearn.model_selection import KFold
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split", default=0, nargs="?", type=int)
parser.add_argument("--seed", default=0, nargs="?", type=int)
parser.add_argument("--cv_seed", default=42, nargs="?", type=int)
parser.add_argument("--dataset_id", default=1, nargs="?", type=int)

ARGS = parser.parse_args()

with open("dataset_list.json", "r") as f:
    dataset_list = json.load(f)
ARGS.dataset = dataset_list[ARGS.dataset_id][0]

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

split_data = {"fold" + str(k): [] for k in range(5)}

i = 0
for train_index, test_index in kf.split(X):
    data.X_train = X[train_index]
    data.Y_train = Y[train_index]
    data.X_test = X[test_index]
    data.Y_test = Y[test_index]

    split_data["fold" + str(i)] = {
        "train_X": data.X_train,
        "train_Y": data.Y_train,
        "test_X": data.X_test,
        "test_Y": data.Y_test,
    }
    i += 1

savemat(f"../../../matlab/experiment/data/{ARGS.dataset}{ARGS.cv_seed}.mat", split_data)
