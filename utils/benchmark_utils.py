import os
import torch
import argparse
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from vlms import load_model
from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import DataLoader
from utils.configs import variant_to_a100_batch_size_mapping
from vlms.base import BaseVLM, BasePreprocessor, PreprocessedPromptWithImage


@dataclass
class Prompt:
    query: str
    image: str
    gt_choice: str
    gt_choices: list[str]
    id: int
    filename: str
    occ: str
    occ_sim: str
    gender: str
    image_type: str


MAKE_PROMPT_FROM_ROW_TYPE = Callable[[pd.Series, str], Prompt]
MAKE_PROMPTS_TYPE = Callable[[str, Optional[MAKE_PROMPT_FROM_ROW_TYPE]], list[Prompt]]


class DataCollator:
    def __init__(self, preprocessor: BasePreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(
        self, batch: list[Prompt]
    ) -> tuple[PreprocessedPromptWithImage, list[Prompt]]:
        prompts = [prompt.query for prompt in batch]
        images = [prompt.image for prompt in batch]
        return self.preprocessor.preprocess(prompts=prompts, images=images), batch


def make_dataloader(prompts: list[Prompt], model: BaseVLM, model_name: str, flag1=False) -> DataLoader:
    preprocessor = model.get_preprocessor()

    if ":" in model_name:
        model_name = model_name.split(":")[1]
        
    batch_size = variant_to_a100_batch_size_mapping[model_name]
    if flag1:
        batch_size = 1
    if model_name.startswith("internvl"):
        batch_size = max(1, batch_size // 2)
    num_workers = min(8, mp.cpu_count())

    return DataLoader(
        prompts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(preprocessor),
        num_workers=num_workers,
    )


def encode_option_letter(letter: str, model: BaseVLM, model_name: str) -> int:
    if model_name.startswith("internvl") or "34b" in model_name or "internvl" in model_name:
        return model.tokenizer.convert_tokens_to_ids(letter)
    
    
    if model_name != "qwen" and "qwen" not in model_name:
        
        letter = " " + letter

    encoded_letter = model.tokenizer.encode(letter, add_special_tokens=False)
    encoded_letter = encoded_letter[-1]
    # import pdb; pdb.set_trace()
    return encoded_letter
