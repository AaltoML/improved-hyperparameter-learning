import json
import pickle


with open("dataset_list.json", "r") as f:
    dataset_list = json.load(f)

metric = ["test_acc", "test_loglik"]

all_dataset_result = {}
missing_result = {}

dataset_list = dataset_list[:27]

method_list = ["vgp", "epcvi", "ep", "la"]

loc = "results/isotropic/"

for DATASET in dataset_list:
    DATASET = DATASET[0]
    single_dataset_res = {}
    missing_method = []
    for METHOD in method_list:
        single_method_res = {}
        for METRIC in metric:
            res = []
            for seed_id in range(1, 10):
                with open(loc + f"{DATASET}_{METHOD}_{seed_id}.pkl", "rb") as f:
                    res_f = pickle.load(f)
                for i in range(5):
                    res.append(res_f[i][METRIC])

            with open(loc + f"{DATASET}_{METHOD}_42.pkl", "rb") as f:
                res_f = pickle.load(f)
            for i in range(5):
                res.append(res_f[i][METRIC])

            single_method_res[METRIC] = res
        single_dataset_res[METHOD] = single_method_res
    all_dataset_result[DATASET] = single_dataset_res

with open(f"10seed_result.json", "w") as f:
    json.dump(all_dataset_result, f, indent=2)
