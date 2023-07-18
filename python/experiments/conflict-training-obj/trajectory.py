import numpy as np
import argparse

import tensorflow as tf

import pickle
from sklearn.model_selection import KFold
from gpflow.utilities.misc import set_trainable

import gpflow

gpflow.config.set_default_positive_bijector("exp")

from classification_helper import load_data, return_cur_index, define_model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--model_name_id", type=int, help="model name")
parser.add_argument("--setting", default=1, nargs="?", type=int)
parser.add_argument("--steps", default=1, nargs="?", type=int)
parser.add_argument("--skip_len", default=1, nargs="?", type=int)
parser.add_argument("--k_fold_id", default=0, type=int, help="which k fold to use")

args = parser.parse_args()

X, Y = load_data(args.dataset_name)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = return_cur_index(X, args.k_fold_id, kf)
train_X, train_Y = X[train_index], Y[train_index]
test_X, test_Y = X[test_index], Y[test_index]


model_name = "epcvi"

init_variance, init_lengthscale = 1.0, 1.0
model = define_model(init_variance, init_lengthscale, train_X, train_Y, model_name)
optimizer = tf.optimizers.Adam(0.01)

len_trajectory = []
var_trajectory = []

total_elbo = []
total_ep = []

if args.setting == 0:
    E_step = 1000
    M_step = 1
else:
    E_step = 20
    M_step = 20

print(E_step, M_step)

for step in range(args.steps):
    if step % args.skip_len == 0:
        len_trajectory.append(model.kernel.lengthscales.numpy())
        var_trajectory.append(model.kernel.variance.numpy())
        total_elbo.append(model.elbo().numpy())
        total_ep.append(model.ep_estimation().numpy())

    # E step
    for _ in range(E_step):
        model.update_variational_parameters(beta=0.1)

    # M step
    set_trainable(model.sites.lambda_1, False)
    set_trainable(model.sites.lambda_2, False)

    for _ in range(M_step):
        optimizer.minimize(lambda: -model.ep_estimation(), model.trainable_variables)

    result = {
        "len_trajectory": len_trajectory,
        "var_trajectory": var_trajectory,
        "total_elbo": total_elbo,
        "total_ep": total_ep,
    }

    with open(
        f"experiment_results/old_fold_adam_exp_{args.setting}_{args.dataset_name}_{model_name}.pkl",
        "wb",
    ) as result_file:
        pickle.dump(result, result_file)
