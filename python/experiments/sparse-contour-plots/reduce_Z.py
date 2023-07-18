import numpy as np
import argparse

from gpflow.utilities import print_summary
import tensorflow as tf

import pickle

from sklearn.model_selection import KFold

from classification_helper import (
    define_sparse_model,
    do_inference_sparse,
    load_data,
    return_cur_index,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="dataset name")
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
parser.add_argument("--ratio_id", type=int, help="ratio index of inducing point")
parser.add_argument("--k_fold_id", default=3, type=int, help="which k fold to use")

args = parser.parse_args()

model_name = "tsvgp"

X, Y = load_data(args.dataset_name)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = return_cur_index(X, args.k_fold_id, kf)
train_X, train_Y = X[train_index], Y[train_index]
test_X, test_Y = X[test_index], Y[test_index]

N = train_X.shape[0]

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

np.random.seed(42)
inducing_index = np.random.permutation(N)

ratio_id = args.ratio_id

ratio_index = [0.1, 0.25, 0.5, 0.75, 1.0]
ratio = ratio_index[ratio_id]

Z = train_X[inducing_index[: int(N * ratio)]]

for i in range(GRID_NUM):
    for j in range(GRID_NUM):
        model = define_sparse_model(sigmas[-1 - i] ** 2, lengthscales[j], Z, model_name)
        mean_lml[i][j], mean_lml_ep[i][j], mean_lpd[i][j] = do_inference_sparse(
            model, train_X, train_Y, test_X, test_Y
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
    f"experiment_results/{args.dataset_name}_Z_{args.ratio_id}_fold_3.pkl", "wb"
) as result_file:
    pickle.dump(result, result_file)
