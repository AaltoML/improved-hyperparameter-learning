import scipy.io
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name", type=str, default="ionosphere", help="dataset name"
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
    default=5,
    help="specify the ending point of l grid range",
)
parser.add_argument("--grid_num", type=int, default=21, help="number of the grid")


args = parser.parse_args()

L_GRID_MIN = args.l_grid_min
L_GRID_MAX = args.l_grid_max


mean_lml = np.zeros((21, 21))
mean_lpd = np.zeros((21, 21))

for i in range(21):
    for j in range(21):
        lml = scipy.io.loadmat(
            "../../../matlab/contour-plots/lml/"
            + args.dataset_name
            + "/i="
            + str(i + 1)
            + "j="
            + str(j + 1)
            + ".mat"
        )["lml"][0][0]
        lp = scipy.io.loadmat(
            "../../../matlab/contour-plots/lp/"
            + args.dataset_name
            + "/i="
            + str(i + 1)
            + "j="
            + str(j + 1)
            + ".mat"
        )["lp"][0][0]
        mean_lml[i][j] = lml
        mean_lpd[i][j] = lp

results_data = {
    "mean_lml": mean_lml,
    "mean_lpd": mean_lpd,
    "grid_min_max_num": (-1, 5, L_GRID_MIN, L_GRID_MAX, 21),
}
results_file = open(f"experiment_results/{args.dataset_name}_mcmc_fold_3.pkl", "wb")
pickle.dump(results_data, results_file)
