import os
import re
import json
import torch
import warnings
import argparse
import pandas as pd

from tqdm import tqdm
from typing import Optional
from vlms import load_model
from bias_eval_utils import Prompt
from bias_eval_utils import BiasPrompt
from utils.benchmark_utils import make_dataloader, encode_option_letter
from baukit import TraceDict
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from util_ced import build_intervene_layers, MODEL_TO_LAYERS
# Use pdb to detect errors
import pdb
import math
# Silence warnings
warnings.filterwarnings("ignore")

from functools import partial

last_layer_mapping = {
    "qwen-chat-7b": ["transformer.visual.transformer.resblocks.47.mlp"],
    "llava-1.5-7b": ["model.mm_projector"],
    "internvl2-8b": ["vision_model.encoder.layers.23.mlp"]
}

model_to_layers_mapping = {
    "qwen-chat-7b": {
        "VIT": [f"transformer.visual.transformer.resblocks.{i}.mlp" for i in range(0,48)],
        "LLM": [f"transformer.h.{i}.mlp" for i in range(0,32)]
    },
    "llava-1.5-7b": {
        "VIT": [f"model.vision_tower.vision_tower.vision_model.encoder.layers.{i}.mlp" for i in range(0,24)],
        "PROJ": ["model.mm_projector"],
        "LLM": [f"model.layers.{i}.mlp" for i in range(0,32)]
    },
    "internvl2-8b": {
        "VIT": [f"vision_model.encoder.layers.{i}.mlp" for i in range(0,24)],
        "LLM": [f"language_model.model.layers.{i}.feed_forward" for i in range(0,32)]
    }
}

def make_prompts(test_json_path: str) -> list[BiasPrompt]:
    print(test_json_path)
    # Get prompts
    with open(test_json_path, "r") as f:
        prompts = json.load(f)
    
    prompts = [BiasPrompt(**prompt) for prompt in prompts]
    return prompts


def get_cmd_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--debias", type=str, default='ced')
    parser.add_argument("--direction", type=str, default="male_to_female")
    parser.add_argument("--ced_lambda", type=float, default=10.0)
    parser.add_argument("--acc_threshold", type=float, default=0.7)
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    
    return parser.parse_args()


def prompt_to_keys(prompt: Prompt) -> dict[str, str]:
    return {
        "query": prompt.query,
        "image": prompt.image,
        "gt_choice": prompt.gt_choice,
        "gt_choices": prompt.gt_choices,
        "id": prompt.id,
        "filename": prompt.filename,
        "occ": prompt.occ,
        "occ_sim": prompt.occ_sim,
        "gender": prompt.gender,
        "image_type": prompt.image_type,
    }


def fair_fuse_logits(
    base_logits: torch.Tensor,
    cf_logits: torch.Tensor,
    tau: float = 1.0,
    return_logits: bool = True,
    entropy_weight: float = 1.0,  
    eps: float = 1e-8
) -> torch.Tensor:
    
    base_logits = base_logits.to(dtype=torch.float32)
    cf_logits = cf_logits.to(dtype=torch.float32)
    
    base_probs = F.softmax(base_logits, dim=-1) + eps  
    cf_probs = F.softmax(cf_logits, dim=-1) + eps      

    base_entropy_token = -base_probs * torch.log(base_probs)  
    cf_entropy_token = -cf_probs * torch.log(cf_probs)       

    base_entropy_scaled = base_entropy_token * entropy_weight  
    cf_entropy_scaled = cf_entropy_token * entropy_weight     

    base_weight_token = torch.exp(-base_entropy_scaled / tau)
    cf_weight_token = torch.exp(-cf_entropy_scaled / tau)    
    total_weight_token = base_weight_token + cf_weight_token  
    total_weight_token = torch.clamp(total_weight_token, min=eps) 
    base_weight_norm = base_weight_token / total_weight_token  
    cf_weight_norm = cf_weight_token / total_weight_token    

    fused_logits = base_weight_norm * base_logits + cf_weight_norm * cf_logits

    if return_logits:
        return fused_logits, None
    else:
        return F.softmax(fused_logits, dim=-1), None
    



def _compute_layerwise_jsd(base_logits: torch.Tensor, cf_logits: torch.Tensor, eps=1e-12):

    base_logits = base_logits.clone().detach()
    cf_logits = cf_logits.clone().detach()

    base_logits = base_logits.to(dtype=torch.float32)
    cf_logits = cf_logits.to(dtype=torch.float32)
    
    p_base = F.softmax(base_logits, dim=-1)              

    p_cf = F.softmax(cf_logits, dim=-1)                 
    M = 0.5 * (p_base + p_cf)  
    
    M = torch.clamp(M, min=eps)  
    M = M / M.sum(dim=-1, keepdim=True)  
    p_base = torch.clamp(p_base, min=eps) 
    p_cf = torch.clamp(p_cf, min=eps)      
    kl_base = torch.sum(p_base * (p_base.log() - M.log()), dim=-1)  
    kl_cf   = torch.sum(p_cf   * (p_cf.log()   - M.log()), dim=-1)  

    js_divs = 0.5 * (kl_base + kl_cf) 

    return js_divs

def get_results(test_json_path: str, save_file: str, 
                layers_to_intervene: dict, intervene_func: callable,
                args: argparse.Namespace, last_layer_clf: dict) -> list[dict]:

    model = load_model(args.model)
    swap_flag = False
    if "swap_option" in test_json_path:
        swap_flag = True
    print(f"Swap flag: {swap_flag}")
    prompts = make_prompts(test_json_path)

    dataloader = make_dataloader(prompts, model, args.model)


    ced_lambda = args.ced_lambda
    entropy_weight = args.entropy_weight

    lm_early_exit_layers = [
        20,
        22,
        24,
        26,
        28,
        30,
        32,
    ]
    mature_layer = lm_early_exit_layers[-1]
    candidate_premature_layers = lm_early_exit_layers[:-1]


    A_index = encode_option_letter("A", model, args.model)
    B_index = encode_option_letter("B", model, args.model)

    results = []
    flag = 1
    batch_id = 0

    last_layers = last_layer_mapping[args.model]
    selected_layer_count = dict()
    for layer_id in candidate_premature_layers + [mature_layer]:
        selected_layer_count[layer_id] = 0

    for prompt, metadata in tqdm(iter(dataloader)):
        batch_id += 1
        
        with torch.no_grad():
            with TraceDict(model.model, last_layers) as ret:
                _image_features = model.get_image_features(prompt)
                last_layer_wise_hidden_states = ret[last_layers[0]].output.to(torch.float16) 
                
                if "qwen" in args.model:
                    last_layer_wise_hidden_states = last_layer_wise_hidden_states.permute(1, 0, 2)  
                coef = last_layer_clf['coef'].to(device=last_layer_wise_hidden_states.device, dtype=last_layer_wise_hidden_states.dtype) 
                intercept = last_layer_clf['intercept'].to(device=last_layer_wise_hidden_states.device, dtype=last_layer_wise_hidden_states.dtype) 

                coef_expanded = coef.unsqueeze(0).expand(last_layer_wise_hidden_states.size(0), -1, -1) 

                gender_scores = torch.sum(last_layer_wise_hidden_states * coef_expanded, dim=-1) 

                gender_scores += intercept 
                gender_predictions = (gender_scores > 0).float() * 2 -1 
                print(gender_predictions)
                del last_layer_wise_hidden_states, ret

        func_ced_cf = partial(intervene_func, ced_lambda=ced_lambda, gender_value=gender_predictions)
        list_layers = list(layers_to_intervene.keys())
        with torch.no_grad():
            logits_base, dict_outputs_base = model.get_next_token_logits_with_early_exit(prompt, early_exit_layers=lm_early_exit_layers)

            with TraceDict(model.model, list_layers, edit_output=func_ced_cf) as ret:
                logits_cf, dict_outputs_cf = model.get_next_token_logits_with_early_exit(prompt, early_exit_layers=lm_early_exit_layers)
 
        
        base_stacked_premature_layers = torch.stack(
            [dict_outputs_base[i] for i in candidate_premature_layers + [mature_layer]], dim=0
        )

        cf_stacked_premature_layers = torch.stack(
            [dict_outputs_cf[i] for i in candidate_premature_layers + [mature_layer]], dim=0
        )
        js_divs = _compute_layerwise_jsd(base_stacked_premature_layers, cf_stacked_premature_layers)

        max_layer_indices  = js_divs.argmax(dim=0) 
        __candidate_premature_layers = candidate_premature_layers + [mature_layer]
        per_sample_layer_ids = [__candidate_premature_layers[i] for i in max_layer_indices.cpu().tolist()]


        for layer_id in per_sample_layer_ids:
            if layer_id not in selected_layer_count:
                selected_layer_count[layer_id] = 0
            selected_layer_count[layer_id] += 1

        B = len(per_sample_layer_ids)
        base_logits_4ced = []
        cf_logits_4ced = []

        for b in range(B):
            layer_id = per_sample_layer_ids[b] 
            base_logits_4ced.append(dict_outputs_base[layer_id][b])    
            cf_logits_4ced.append(dict_outputs_cf[layer_id][b])   

        base_logits_4ced = torch.stack(base_logits_4ced, dim=0)
        cf_logits_4ced = torch.stack(cf_logits_4ced, dim=0)
        final_logits = dict_outputs_base[mature_layer]
        final_logits_cf = dict_outputs_cf[mature_layer]

    
        logits, _weight = fair_fuse_logits(base_logits_4ced, cf_logits_4ced, tau=1.0, return_logits=True
            , entropy_weight=entropy_weight)
        next_token_logits  = logits

        probs = torch.softmax(next_token_logits, dim=-1)  
        
        for i, prompt_metadata in enumerate(metadata):
            prob_A = probs[i, A_index].item()
            prob_B = probs[i, B_index].item()

            log_prob_A = math.log(prob_A)
            log_prob_B = math.log(prob_B)
            ppl_results = [log_prob_A, log_prob_B]
            logits = torch.tensor([log_prob_A, log_prob_B], dtype=torch.float32)
            
            normalized_probs = torch.softmax(logits, dim=0).tolist()

            _chosen_index = 0 if normalized_probs[0] > normalized_probs[1] else 1
            chosen_letter = "A" if _chosen_index == 0 else "B"
            chosen_prob = prob_A if chosen_letter == "A" else prob_B
            metric_result = False
            if swap_flag:
                if chosen_letter == "B":
                    metric_result = True
            else:
                if chosen_letter == "A":
                    metric_result = True
            if flag:
                print(f"A {normalized_probs[0]}, B {normalized_probs[1]}")  
                flag = 0
            keys_to_save = prompt_to_keys(prompt_metadata)
            results.append({
                **keys_to_save,
                "ppl_results": ppl_results,
                "probs_norm": normalized_probs,
                "probs": [prob_A, prob_B],
                "prob": chosen_prob,
                "answer": chosen_letter,
                "metric_result": metric_result,
            })

        if batch_id % 2 == 0:
            print(f"Processed {batch_id} batches")
            with open(save_file, "w") as f:
                json.dump(results, f, indent=4)

    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Layer selection frequency:")
    for layer, count in sorted(selected_layer_count.items()):
        print(f"Layer {layer}: {count}")

    layer_count_file = save_file.replace(".json", "_layer_count.json")
    with open(layer_count_file, "w") as f:
        json.dump(selected_layer_count, f, indent=4)
    print(f"Layer count saved to {layer_count_file}")

    return results


if __name__ == '__main__':
    args = get_cmd_arguments()

    model_name = args.model
    direction = args.direction
    layers_to_intervene, last_layer_clf = build_intervene_layers(
        model_name=model_name,
        direction_dir=direction,
        acc_threshold=args.acc_threshold
    )
    def intervene_func(value, layer_name, ced_lambda=0, gender_value=None):
        if "llava" in model_name:
            pass
        if layer_name in layers_to_intervene:
            direction = layers_to_intervene[layer_name]  
            direction = direction.to(device=value.device, dtype=value.dtype)
            direction = direction.unsqueeze(0)
            if gender_value is None:
                return value + ced_lambda * direction
            else:
                if "llava" in model_name:
                    gender_value_expanded = gender_value.unsqueeze(-1).to(dtype=value.dtype)
                    gender_value_expanded = gender_value_expanded.expand(-1, -1, direction.shape[-1])
                    if "llava" in model_name and "mm_projector" not in layer_name:
                        value[:,1:,:] = value[:,1:,:] - ced_lambda * gender_value_expanded * direction[:,1:,:]
                    else:
                        value = value - ced_lambda * gender_value_expanded * direction
                    return value
                elif "qwen" in model_name:
                    gender_value_expanded = gender_value.unsqueeze(-1).to(dtype=value.dtype)
                    gender_value_expanded = gender_value_expanded.expand(-1, -1, direction.shape[-1])
                    value_ = value.permute(1, 0, 2)
                    value_ = value_ - ced_lambda * gender_value_expanded * direction
                    value = value_.permute(1, 0, 2)
                    return value
                elif "internvl2" in model_name:
                    gender_value_expanded = gender_value.unsqueeze(-1).to(dtype=value.dtype)
                    gender_value_expanded = gender_value_expanded.expand(-1, -1, direction.shape[-1])
                    value = value - ced_lambda * gender_value_expanded * direction
                    return value

    test_json_path = args.json_path


    save_path = os.path.join(f"./results/benchmark", f"{args.model}_{args.debias}", f"{args.debias}_entropy_{args.entropy_weight}_lambda_{args.ced_lambda}")
        
    save_path = os.path.join(save_path, "Vbias")

    os.makedirs(save_path, exist_ok=True)
    save_filename = os.path.basename(test_json_path)
    save_file = os.path.join(save_path, save_filename)

    results = get_results(test_json_path, save_file, 
                          layers_to_intervene=layers_to_intervene, 
                          intervene_func=intervene_func, args=args,
                          last_layer_clf=last_layer_clf)
