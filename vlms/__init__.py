
from .base import BaseVLM
from vlms.qwen import Qwen
from vlms.llava import LLaVA
from vlms.internvl2 import InternVL2
from utils.configs import model_to_variant_mapping

import pdb


def load_model(name: str) -> BaseVLM:

    if name in model_to_variant_mapping:
        variant = model_to_variant_mapping[name]
        if name == "qwen-chat-7b":
            return Qwen(variant=variant)
        elif name.startswith("llava"):
            return LLaVA(variant=variant)
        elif name.startswith("internvl2"):
            return InternVL2(variant=variant)
    else:
        raise ValueError(f"Unknown model {name}")
