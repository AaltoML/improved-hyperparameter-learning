import numpy as np
import argparse

import tensorflow as tf

import pickle

from sklearn.model_selection import KFold

from classification_helper import (
    load_data,
    return_cur_index,
    define_model,
    do_inference,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--model_name_id", type=int, help="model name")
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument(
    "--sigma_grid_min",
    type=float,
    default=-1,
    help="specify the start point of sigma grid range",
)
parser.add_argument(
    "--sigma_grid_max",
    type=float,
    default=5.0,
    help="specify the ending point of sigma grid range",
)
parser.add_argument(
    "--l_grid_min",
    type=float,
    default=-1,
    help="specify the start point of l grid range",
)
parser.add_argument(
    "--l_grid_max",
    type=float,
    default=5.0,
    help="specify the ending point of l grid range",
)
parser.add_argument("--grid_num", type=int, default=21, help="number of the grid")
parser.add_argument("--k_fold_id", default=3, type=int, help="which k fold to use")

args = parser.parse_args()

model_name_list = ["la", "ep", "cvi"]

model_name = model_name_list[args.model_name_id]

try:
    X, Y = load_data(args.dataset_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = return_cur_index(X, args.k_fold_id, kf)
    train_X, train_Y = X[train_index], Y[train_index]
    test_X, test_Y = X[test_index], Y[test_index]

    train_Y[np.where(train_Y == 0)] = -1
    test_Y[np.where(test_Y == 0)] = -1

except:
    with open(f"../../bayesian_benchmarks/data/{args.dataset_name}.pkl", "rb") as f:
        data = pickle.load(f)
    train_X = data["train_X"]
    train_Y = data["train_Y"]
    test_X = data["test_X"]
    test_Y = data["test_Y"]

SIGMA_GRID_MIN = args.sigma_grid_min
SIGMA_GRID_MAX = args.sigma_grid_max
L_GRID_MIN = args.l_grid_min
L_GRID_MAX = args.l_grid_max
GRID_NUM = args.grid_num

lengthscales = np.logspace(L_GRID_MIN, L_GRID_MAX, GRID_NUM, base=np.e)
sigmas = np.logspace(SIGMA_GRID_MIN, SIGMA_GRID_MAX, GRID_NUM, base=np.e)

mean_lml = np.zeros((GRID_NUM, GRID_NUM))
mean_lpd = np.zeros((GRID_NUM, GRID_NUM))
mean_lml_ep = np.zeros((GRID_NUM, GRID_NUM))

for i in range(GRID_NUM):
    for j in range(GRID_NUM):
        model = define_model(
            sigmas[-1 - i] ** 2, lengthscales[j], train_X, train_Y, model_name
        )
        if model_name == "cvi":
            mean_lml[i][j], mean_lml_ep[i][j], mean_lpd[i][j] = do_inference(
                model, model_name, train_X, train_Y, test_X, test_Y
            )
        else:
            mean_lml[i][j], mean_lpd[i][j] = do_inference(
                model, model_name, train_X, train_Y, test_X, test_Y
            )

result = {
    "mean_lml": mean_lml,
    "mean_lpd": mean_lpd,
    "mean_lml_ep": mean_lml_ep,
    "grid_min_max_num": (
        SIGMA_GRID_MIN,
        SIGMA_GRID_MAX,
        L_GRID_MIN,
        L_GRID_MAX,
        GRID_NUM,
    ),
}

with open(
    f"experiment_results/{args.dataset_name}_{model_name}_fold_{args.k_fold_id}.pkl",
    "wb",
) as result_file:
    pickle.dump(result, result_file)
