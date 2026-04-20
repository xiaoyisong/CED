import yaml

with open('./configs/model_configs.yaml') as file:
    model_configs = yaml.load(file, Loader=yaml.FullLoader)

model_to_variant_mapping = {
    "qwen-chat-7b": "qwen-chat-7b",
    "llava-1.5-7b": "llava-1.5-7b",
    "internvl2-8b": "internvl2-8b",
}

variant_to_a100_batch_size_mapping = {
    "qwen-chat-7b": 8, 
    "llava-1.5-7b": 32,
    "internvl2-8b": 5,
}

