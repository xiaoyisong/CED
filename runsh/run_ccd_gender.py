from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import subprocess
import os

os.chdir("code")

available_cuda_ids = ["1"]

def run_command(args: dict):
    command = [
        "python", "benchmark_ced.py",
        "--model", args["model"],
        "--json_path", str(args["json_path"]),
        "--debias", str(args["debias"]),
        "--ced_lambda", str(args.get("ced_lambda", 10.0)),
        "--direction", args["direction"],
        "--entropy_weight", str(args.get("entropy_weight", 1.0)), 
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args["cuda_id"]

    try:
        subprocess.run(command, env=env, check=True)
        return f"[✓] Task '{args['json_path']}' completed on GPU {args['cuda_id']}"
    except subprocess.CalledProcessError as e:
        return f"[✗] Task '{args['json_path']}' failed on GPU {args['cuda_id']} with error: {e}"


if __name__ == "__main__":
    model_names = [
        "llava-1.5-7b",
    ]
    
    test_list = [
        "data/Vbias/occ_test_base_ask_person.json",
        "data/Vbias/occ_test_cf_ask_person.json",
        "data/Vbias/occ_test_base_ask_person_swap_option.json",
        "data/Vbias/occ_test_cf_ask_person_swap_option.json",
         
    ]

    max_workers = len(available_cuda_ids)
    
    debias = "ced"  
    futures = []
    current_cuda_index = 0  
    lambda_list = [3]
    entropy_weight = 1.0
    

    for model_name in model_names:
        direction = f"results/direction/{model_name}"
        for lambda_para in lambda_list:
            for entropy_weight in [entropy_weight]:
                for test_json_path in test_list:
                    
                    
                    cuda_id = available_cuda_ids[current_cuda_index % len(available_cuda_ids)]

                    args = {
                        "cuda_id": cuda_id,
                        "model": model_name,
                        "json_path": test_json_path,
                        "debias": debias,
                        "direction": direction,
                        "ced_lambda": lambda_para, 
                        "entropy_weight": entropy_weight,
                    }
            
                    futures.append(copy.deepcopy(args))
                    current_cuda_index += 1
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        for args in futures:
            print(f"Submitting task '{args['json_path']}' on GPU {args['cuda_id']}")
            executor.submit(run_command, args)
        