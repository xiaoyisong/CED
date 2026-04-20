import os
import subprocess

project_dir = ""
result_base_dir = os.path.join(project_dir, "results/benchmark")
base_test_file = os.path.join(project_dir, "data/Vbias/occ_test_base_ask_person.json")
cf_test_file = os.path.join(project_dir, "data/Vbias/occ_test_cf_ask_person.json")


model_names = []


base_datasets = ["OccBaseAskPerson", "OccBaseAskPersonSwapOption"]
cf_datasets = ["OccCfAskPerson", "OccCfAskPersonSwapOption"]
eval_datasets = ["Vbias", "Vbias_swap"]



if __name__ == "__main__":
    for model_name in model_names:
        for base_dataset, cf_dataset, eval_dataset in zip(base_datasets, cf_datasets, eval_datasets):
            base_sub_dir = os.path.join(result_base_dir, model_name, "Vbias")
            cf_sub_dir = os.path.join(result_base_dir, model_name, "Vbias")
            if eval_dataset == "Vbias":
                base_result = os.path.join(base_sub_dir, "occ_test_base_ask_person.json")
                cf_result = os.path.join(cf_sub_dir, "occ_test_cf_ask_person.json")
            elif eval_dataset == "Vbias_swap":
                base_result = os.path.join(base_sub_dir, "occ_test_base_ask_person_swap_option.json")
                cf_result = os.path.join(cf_sub_dir, "occ_test_cf_ask_person_swap_option.json")

            print(f"model_name: {model_name}, eval_dataset: {eval_dataset}")

            command = [
                "python", "cal_bias.py",
                "--model_name", model_name,
                "--eval_dataset", eval_dataset,
                "--inferencer_type", "PPL",
                "--base_result", base_result,
                "--cf_result", cf_result,
                "--base_test_file", base_test_file,
                "--cf_test_file", cf_test_file
            ]

            subprocess.run(command)

