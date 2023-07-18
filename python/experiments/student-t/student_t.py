import numpy as np
import argparse

from gpflow.utilities import print_summary
import tensorflow as tf

import pickle

from sklearn.model_selection import KFold

from student_t_helper import fit_model, return_cur_index, define_model, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="dataset name")
parser.add_argument("--model_name", type=str, help="model name")
parser.add_argument("--total_step", default=2, type=int, help="total step")
parser.add_argument(
    "--e_step", default=20, type=int, help="number of iterations in each e step"
)
parser.add_argument(
    "--m_step", default=20, type=int, help="number of iterations in each m step"
)
parser.add_argument(
    "--num_k_fold", default=5, type=int, help="number of cross validation folds"
)
parser.add_argument("--k_fold_id", default=0, type=int, help="for parallel cv running")

args = parser.parse_args()

init_variance = 1.0
init_lengthscale = 1.0
natgrad_lr = 0.001


X, Y = load_data(args.dataset_name)
kf = KFold(n_splits=args.num_k_fold, shuffle=True, random_state=42)
train_index, test_index = return_cur_index(X, args.k_fold_id, kf)
train_X, train_Y = X[train_index], Y[train_index]
test_X, test_Y = X[test_index], Y[test_index]

model = define_model(init_variance, init_lengthscale, train_X, train_Y, args.model_name)

model = fit_model(
    model, args.total_step, args.e_step, args.m_step, natgrad_lr, args.model_name
)

lpd = tf.reduce_mean(model.predict_log_density((test_X, test_Y))).numpy()

print_summary(model)

result = {
    "nlpd": lpd,
    "scale": model.likelihood.scale.numpy(),
    "fold_id": args.k_fold_id,
}

with open(
    f"student_t_results/{args.dataset_name}_{args.model_name}_{args.k_fold_id}.pkl",
    "wb",
) as result_file:
    pickle.dump(result, result_file)
