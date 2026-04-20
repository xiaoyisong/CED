from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    T5Tokenizer,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
    BertForMaskedLM,
    BartForConditionalGeneration,
    AlbertForMaskedLM,
    LlamaForCausalLM, # Added for llama 2 models
    LlamaTokenizer,
    # PhiForCausalLM, # Added for Phi 2 models
)

from self_debias_generation import (
    SelfDebiasingLogitsProcessor,
    SelfDebiasingLLaVALMHeadModel, # Added for llama 2 models
)
import transformers


def get_top_k_tokens(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: int = 5):
    values, indices = torch.topk(logits, k, dim=-1)
    if len(logits.shape) == 2:
        assert logits.shape[0] == 1
        values, indices = values[0], indices[0]
    return tokenizer.convert_ids_to_tokens(indices), values

class GenerativeLMWrapper(ABC):
    """
    This class represents a wrapper for a pretrained language model that provides some high-level functions, including zero-shot
    classification using cloze questions and the generation of texts with self-debiasing.
    """

    def __init__(self, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = None  # type: Optional[PreTrainedTokenizer]
        self._model = None  # type: Optional[PreTrainedModel]

    def query_model(self, input_text: str) -> torch.FloatTensor:
        """For a given input text, returns the probability distribution over possible next tokens."""
        return self.query_model_batch([input_text])[0]

    @abstractmethod
    def query_model_batch(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass

    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """Generates a continuation for a given input text."""
        pass

    @abstractmethod
    def generate_self_debiasing(
        self,
        input_texts: List[str],
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Generates continuations for the given input texts with self-debiasing.
        :param input_texts: the input texts to generate continuations for
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param kwargs: further arguments are passed on to the original generate function
        :return: the list of generated continuations
        """
        pass

    @abstractmethod
    def compute_loss(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        """Computes cross-entropy loss for the given input ids and corresponding labels."""
        pass

    @abstractmethod
    def compute_loss_self_debiasing(
        self,
        input_ids: torch.Tensor,
        trg_len: int,
        debiasing_prefixes: List[str],
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss for the given input ids with self-debiasing.
        :param input_ids: the input ids
        :param trg_len: only the last trg_len tokens are considered for computing the loss
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :return: the cross entropy loss
        """
        pass

    def get_token_probability_distribution(
        self, input_texts: List[str], output_choices: List[str]
    ) -> List[List[Tuple[str, float]]]:
        """
        For a batch of input texts, returns the probability distribution over possible next tokens considering only the given list of
        output choices.
        :param input_texts: the input texts
        :param output_choices: the allowed output choices (must correspond to single tokens in the model's vocabulary)
        :return: a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
        """
        output_choice_ids = []
        kwargs = {"add_prefix_space": True} if isinstance(self, GPT2Wrapper) else {}
        for word in output_choices:
            tokens = self._tokenizer.tokenize(word, **kwargs)
            assert (
                len(tokens) == 1
            ), f"Word {word} consists of multiple tokens: {tokens}"
            assert (
                tokens[0] not in self._tokenizer.all_special_tokens
            ), f"Word {word} corresponds to a special token: {tokens[0]}"
            token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
            output_choice_ids.append(token_id)

        logits = self.query_model_batch(input_texts)
        result = []

        for idx, _ in enumerate(input_texts):
            output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
            choices_with_probabilities = list(
                zip(output_choices, (prob.item() for prob in output_probabilities))
            )
            result.append(choices_with_probabilities)

        return result



# For llama 2 2 models
# Attempt to add Llama 2 into self-debiasing
# Probably require 1. Query model batch 2. Generate 3. Generate self-debiasing 4. Compute loss 5. Compute loss self-debiasing
class LLaVALLaMA2Wrapper(GenerativeLMWrapper):
    def __init__(self, variant, use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        # import pdb; pdb.set_trace()
        # self._tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
        self.llava = SelfDebiasingLLaVALMHeadModel(variant)
        
        self._tokenizer = self.llava.tokenizer
        self._model = self.llava.model

        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer.batch_encode_plus(
            input_texts, padding=True, return_tensors="pt"
        )
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs["attention_mask"].sum(dim=1) - 1
        output = self._model(**inputs)["logits"]
        return torch.stack(
            [
                output[example_idx, last_word_idx, :]
                for example_idx, last_word_idx in enumerate(output_indices)
            ]
        )

    def generate(self, input_text: str, **kwargs):
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt").to(
            self._device
        )
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(
            self,
            input_texts: List[str],
            debiasing_prefixes: List[str],
            decay_constant: float = 50,
            epsilon: float = 0.01,
            debug: bool = False,
            min_length: int = None,
            max_length: int = None,
            **kwargs,
    ) -> List[str]:

        self._model.init_logits_processor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            debug=debug,
            tokenizer=self._tokenizer,
        )
        inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_text in input_texts:
                inputs += [debiasing_prefix + input_text]

        inputs = self._tokenizer.batch_encode_plus(
            inputs, padding=True, return_tensors="pt"
        )
        inputs["attention_mask"] = torch.flip(inputs["attention_mask"], dims=[1])
        shifts = inputs["attention_mask"].shape[-1] - inputs["attention_mask"].sum(
            dim=-1
        )
        for batch_idx in range(inputs["input_ids"].shape[0]):
            inputs["input_ids"][batch_idx] = inputs["input_ids"][batch_idx].roll(
                shifts[batch_idx].item()
            )

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(
            **inputs, min_length=min_length, max_length=max_length, **kwargs
        )

        batch_size = output_ids.shape[0] // (1 + len(debiasing_prefixes))
        output_ids = output_ids[:batch_size, inputs["input_ids"].shape[1]:]
        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(
            self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> torch.Tensor:
        outputs = self._model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    def compute_loss_self_debiasing(
            self,
            input_ids: torch.Tensor,
            debiasing_prefixes: List[str],
            decay_constant: float = 50,
            epsilon: float = 0.01,
            debug: bool = False,
    ) -> torch.Tensor:
        self._device = "cuda"

        self._model.init_logits_processor(
            num_debiasing_prefixes=len(debiasing_prefixes),
            decay_constant=decay_constant,
            epsilon=epsilon,
            debug=debug,
            tokenizer=self._tokenizer,
        )

        input_prefixes = [""] + debiasing_prefixes
        input_prefixes = self._tokenizer.batch_encode_plus(
            input_prefixes, padding=True, return_tensors="pt"
        )
        input_prefixes["attention_mask"] = torch.flip(
            input_prefixes["attention_mask"], dims=[1]
        )

        shifts = input_prefixes["attention_mask"].shape[-1] - input_prefixes[
            "attention_mask"
        ].sum(dim=-1)
        for batch_idx in range(input_prefixes["input_ids"].shape[0]):
            input_prefixes["input_ids"][batch_idx] = input_prefixes["input_ids"][
                batch_idx
            ].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to(self._device) for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat(
            [input_prefixes["attention_mask"], attention_mask], dim=-1
        )
        input_ids_repeated = torch.cat(
            [input_prefixes["input_ids"], input_ids_repeated], dim=-1
        )

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self._model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        lm_logits = outputs["logits"]

        for idx in range(lm_logits.shape[1]):
            lm_logits[:, idx, :] = self._model.logits_processor(
                input_ids=None, scores=lm_logits[:, idx, :]
            )
        # import pdb; pdb.set_trace()
        return lm_logits, input_ids_repeated
