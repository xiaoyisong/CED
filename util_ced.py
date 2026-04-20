import os
import pickle
import numpy as np
import torch

MODEL_TO_LAYERS = {
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

def build_intervention_directions(
    token_clf_dict,
    acc_threshold=0.7,
    patch_num=1024,
    hidden_dim=1664,
    is_proj=False,
    device="cuda",
    model_name="llava-1.5-7b"  
):
    layers_to_intervene = {}
    strong_token_indices = {}

    layer_patch_dict = {}

    for key, val in token_clf_dict.items():
        acc = val["acc"]
        coef = val["coef"].flatten()
        std_proj = val.get("std_proj", 1.0) 

        if is_proj:
            layer_idx = "PROJ"
            patch_idx = key
        else:
            layer_idx, patch_idx = key

        if 'llava' in model_name and patch_idx == 0 and layer_idx != "PROJ":
            direction = np.zeros_like(coef)
        elif acc > acc_threshold:
            norm = np.linalg.norm(coef)
            unit_dir = coef / norm if norm > 1e-6 else np.zeros_like(coef)
            direction = unit_dir * std_proj
        else:
            direction = np.zeros_like(coef)


        if layer_idx not in layer_patch_dict:
            layer_patch_dict[layer_idx] = {}
        layer_patch_dict[layer_idx][patch_idx] = direction

    for layer_idx, patch_dict in layer_patch_dict.items():
        direction_matrix = np.zeros((patch_num, hidden_dim))
        strong_patches = []

        for patch_idx, direction in patch_dict.items():
            direction_matrix[patch_idx] = direction
            if not np.allclose(direction, 0.0):
                strong_patches.append(patch_idx)

        layers_to_intervene[layer_idx] = torch.tensor(direction_matrix, dtype=torch.float16).to(device)
        strong_token_indices[layer_idx] = strong_patches    
    return layers_to_intervene, strong_token_indices

def build_intervene_layers_llava(model_name, direction_dir):
    with open(os.path.join(direction_dir, "vit_token_clf_dict.pkl"), "rb") as f:
        vit_token_clf_dict = pickle.load(f)
    with open(os.path.join(direction_dir, "proj_token_clf_dict.pkl"), "rb") as f:
        proj_token_clf_dict = pickle.load(f)

    layers_to_intervene = {}
    VIT_layers = MODEL_TO_LAYERS[model_name]["VIT"]
    PROJ_layers = MODEL_TO_LAYERS[model_name]["PROJ"]

    sample_val = list(vit_token_clf_dict.values())[0]
    D_vit = sample_val["coef"].shape[-1]
    P_vit = max(p[1] for p in vit_token_clf_dict.keys()) + 1

    vit_layers_to_intervene, vit_strong_token_indices = build_intervention_directions(
        vit_token_clf_dict,
        acc_threshold=0.7,
        patch_num=P_vit,
        hidden_dim=D_vit,
        is_proj=False,
        device="cuda",
        model_name=model_name
    )

    sample_val = list(proj_token_clf_dict.values())[0]
    D_proj = sample_val["coef"].shape[-1]
    P_proj = max(proj_token_clf_dict.keys()) + 1

    proj_layers_to_intervene, proj_strong_token_indices = build_intervention_directions(
        proj_token_clf_dict,
        acc_threshold=0.7,
        patch_num=P_proj,
        hidden_dim=D_proj,
        is_proj=True,
        device="cuda",
        model_name=model_name
    )

    for layer_idx, direction_matrix in vit_layers_to_intervene.items():
        if layer_idx >= len(VIT_layers):
            print(f"[Warning] Layer index {layer_idx} exceeds VIT mapping list for {model_name}")
            continue
        layer_name = VIT_layers[layer_idx]
        layers_to_intervene[layer_name] = direction_matrix  
    for proj_name in PROJ_layers:
        layers_to_intervene[proj_name] = proj_layers_to_intervene["PROJ"]  



    coef_list = []
    intercept_list = []

    for key, val in proj_token_clf_dict.items():
        
        coef = torch.tensor(val["coef"], dtype=torch.float16).flatten()  
        intercept = torch.tensor(val["intercept"], dtype=torch.float16).flatten() 
        
        coef_list.append(coef)  
        intercept_list.append(intercept)  
    coef_stack = torch.stack(coef_list, dim=0)  
    intercept_stack = torch.cat(intercept_list, dim=0) 

    print(f"coef_stack.shape is {coef_stack.shape}")
    print(f"intercept_stack.shape is {intercept_stack.shape}")
    last_layer_clf = {
        "coef": coef_stack,
        "intercept": intercept_stack
    }
    return layers_to_intervene, last_layer_clf


def build_intervene_layers_qwen(model_name, direction_dir, acc_threshold=0.7):
    with open(os.path.join(direction_dir, "vit_token_clf_dict.pkl"), "rb") as f:
        vit_token_clf_dict = pickle.load(f)

    layers_to_intervene = {}
    VIT_layers = MODEL_TO_LAYERS[model_name]["VIT"]

    sample_val = list(vit_token_clf_dict.values())[0]
    D_vit = sample_val["coef"].shape[-1]
    P_vit = max(p[1] for p in vit_token_clf_dict.keys()) + 1

    vit_layers_to_intervene, vit_strong_token_indices = build_intervention_directions(
        vit_token_clf_dict,
        acc_threshold=acc_threshold,
        patch_num=P_vit,
        hidden_dim=D_vit,
        is_proj=False,
        device="cuda",
        model_name=model_name
    )


    for layer_idx, direction_matrix in vit_layers_to_intervene.items():
        if layer_idx >= len(VIT_layers):
            print(f"[Warning] Layer index {layer_idx} exceeds VIT mapping list for {model_name}")
            continue
        layer_name = VIT_layers[layer_idx]
        layers_to_intervene[layer_name] = direction_matrix  


    coef_list = []
    intercept_list = []
    last_layer_clf = {}
    for key, val in vit_token_clf_dict.items():
        layer_idx, patch_idx = key
        if layer_idx == len(VIT_layers)-1:
            last_layer_clf[patch_idx] = val

    for patch_idx in range(P_vit):        
        key = patch_idx
        val = last_layer_clf[key]
        
        coef = torch.tensor(val["coef"], dtype=torch.float16).flatten() 
        intercept = torch.tensor(val["intercept"], dtype=torch.float16).flatten() 
        
        coef_list.append(coef)  
        intercept_list.append(intercept)  

    coef_stack = torch.stack(coef_list, dim=0)  
    intercept_stack = torch.cat(intercept_list, dim=0)  

    print(f"coef_stack.shape is {coef_stack.shape}")
    print(f"intercept_stack.shape is {intercept_stack.shape}")
    last_layer_clf = {
        "coef": coef_stack,
        "intercept": intercept_stack
    }
    return layers_to_intervene, last_layer_clf


def build_intervene_layers(model_name, direction_dir, acc_threshold=0.7):
    if "llava" in model_name:
        return build_intervene_layers_llava(model_name, direction_dir)
    else:
        return build_intervene_layers_qwen(model_name, direction_dir, acc_threshold)

def build_intervention_directions_proj(
    token_clf_dict,
    acc_threshold=0.7,
    patch_num=1024,
    hidden_dim=1664,
    is_proj=False,
    device="cuda",
    model_name="llava-1.5-7b"  
):
    
    layers_to_intervene = {}
    strong_token_indices = {}

    layer_patch_dict = {}

    for key, val in token_clf_dict.items():
        acc = val["acc"]
        coef = val["coef"].flatten()
        std_proj = val.get("std_proj", 1.0)  

        if is_proj:
            layer_idx = "PROJ"
            patch_idx = key
        else:
            layer_idx, patch_idx = key

        if 'llava' in model_name and patch_idx == 0 and layer_idx != "PROJ":
            direction = np.zeros_like(coef)
        elif acc > acc_threshold:
            norm = np.linalg.norm(coef)
            unit_dir = coef / norm if norm > 1e-6 else np.zeros_like(coef)
            direction = unit_dir
        else:
            direction = np.zeros_like(coef)


        if layer_idx not in layer_patch_dict:
            layer_patch_dict[layer_idx] = {}
        layer_patch_dict[layer_idx][patch_idx] = direction

    for layer_idx, patch_dict in layer_patch_dict.items():
        direction_matrix = np.zeros((patch_num, hidden_dim))
        strong_patches = []

        for patch_idx, direction in patch_dict.items():
            direction_matrix[patch_idx] = direction
            if not np.allclose(direction, 0.0):
                strong_patches.append(patch_idx)

        layers_to_intervene[layer_idx] = torch.tensor(direction_matrix, dtype=torch.float16).to(device)
        strong_token_indices[layer_idx] = strong_patches
        print(f"Layer {layer_idx} shape is {layers_to_intervene[layer_idx].shape} has {len(strong_patches)} strong tokens.")
    
    return layers_to_intervene, strong_token_indices


def build_intervene_layers_proj_llava(model_name, direction_dir):
    with open(os.path.join(direction_dir, "vit_token_clf_dict.pkl"), "rb") as f:
        vit_token_clf_dict = pickle.load(f)
    with open(os.path.join(direction_dir, "proj_token_clf_dict.pkl"), "rb") as f:
        proj_token_clf_dict = pickle.load(f)

    layers_to_intervene = {}
    VIT_layers = MODEL_TO_LAYERS[model_name]["VIT"]
    PROJ_layers = MODEL_TO_LAYERS[model_name]["PROJ"]

    sample_val = list(vit_token_clf_dict.values())[0]
    D_vit = sample_val["coef"].shape[-1]
    P_vit = max(p[1] for p in vit_token_clf_dict.keys()) + 1

    vit_layers_to_intervene, vit_strong_token_indices = build_intervention_directions_proj(
        vit_token_clf_dict,
        acc_threshold=0,
        patch_num=P_vit,
        hidden_dim=D_vit,
        is_proj=False,
        device="cuda",
        model_name=model_name
    )

    sample_val = list(proj_token_clf_dict.values())[0]
    D_proj = sample_val["coef"].shape[-1]
    P_proj = max(proj_token_clf_dict.keys()) + 1

    proj_layers_to_intervene, proj_strong_token_indices = build_intervention_directions(
        proj_token_clf_dict,
        acc_threshold=0,
        patch_num=P_proj,
        hidden_dim=D_proj,
        is_proj=True,
        device="cuda",
        model_name=model_name
    )

    for layer_idx, direction_matrix in vit_layers_to_intervene.items():
        if layer_idx >= len(VIT_layers):
            print(f"[Warning] Layer index {layer_idx} exceeds VIT mapping list for {model_name}")
            continue
        layer_name = VIT_layers[layer_idx]
        layers_to_intervene[layer_name] = direction_matrix  
    for proj_name in PROJ_layers:
        layers_to_intervene[proj_name] = proj_layers_to_intervene["PROJ"] 
    return layers_to_intervene


def build_intervene_layers_proj_qwen(model_name, direction_dir):
    with open(os.path.join(direction_dir, "vit_token_clf_dict.pkl"), "rb") as f:
        vit_token_clf_dict = pickle.load(f)

    layers_to_intervene = {}
    VIT_layers = MODEL_TO_LAYERS[model_name]["VIT"]

    sample_val = list(vit_token_clf_dict.values())[0]
    D_vit = sample_val["coef"].shape[-1]
    P_vit = max(p[1] for p in vit_token_clf_dict.keys()) + 1

    vit_layers_to_intervene, vit_strong_token_indices = build_intervention_directions_proj(
        vit_token_clf_dict,
        acc_threshold=0,
        patch_num=P_vit,
        hidden_dim=D_vit,
        is_proj=False,
        device="cuda",
        model_name=model_name
    )

    

    for layer_idx, direction_matrix in vit_layers_to_intervene.items():
        if layer_idx >= len(VIT_layers):
            print(f"[Warning] Layer index {layer_idx} exceeds VIT mapping list for {model_name}")
            continue
        layer_name = VIT_layers[layer_idx]
        layers_to_intervene[layer_name] = direction_matrix  

    return layers_to_intervene


def build_intervene_layers_proj(model_name, direction_dir):
    if "llava" in model_name:
        return build_intervene_layers_proj_llava(model_name, direction_dir)
    else:
        return build_intervene_layers_proj_qwen(model_name, direction_dir)
