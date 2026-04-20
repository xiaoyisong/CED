from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel,
    LogitsProcessorList,
    LogitsProcessor,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)
from transformers.generation.utils import (
    GenerationMixin,
    SampleOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
)

from transformers import LlamaForCausalLM, PhiForCausalLM



class SelfDebiasingLogitsProcessor(LogitsProcessor):
    """This class represents a logits processor that applies self-debiasing."""

    def __init__(
        self,
        num_debiasing_prefixes: int = 1,
        decay_constant: float = 50,
        epsilon: float = 0.01,
        debug: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        :param num_debiasing_prefixes: the number of debiasing prefixes used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param tokenizer: a tokenizer used to print debugging output
        """
        assert (
            not debug or tokenizer
        ), "If debug=True, a tokenizer must be passed to SelfDebiasingLogitsProcessor()"
        self.num_debiasing_prefixes = num_debiasing_prefixes
        self.decay_constant = decay_constant
        self.epsilon = epsilon
        self.debug = debug
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = scores.shape[0] // (1 + self.num_debiasing_prefixes)
        regular_sentence_indices = range(batch_size)
        for regular_sentence_idx in regular_sentence_indices:
            # import pdb; pdb.set_trace()
            bias_indices = self._get_bias_indices(regular_sentence_idx, batch_size)
            if bias_indices:
                self._debias_scores(scores, regular_sentence_idx, bias_indices)
        return scores

    def _get_bias_indices(
        self, regular_sentence_idx: int, batch_size: int
    ) -> List[int]:
        """Returns the indices of all self-debiasing inputs for a regular input"""
        return [
            regular_sentence_idx + (prefix_idx + 1) * batch_size
            for prefix_idx in range(self.num_debiasing_prefixes)
        ]

    def _debias_scores(
        self, scores: torch.FloatTensor, regular_sent_idx: int, bias_indices: List[int]
    ) -> None:
        """Partially debiases the given scores considering a single sentence and the corresponding self-debiasing inputs"""
        logits_biased = [scores[bias_idx] for bias_idx in bias_indices]
        # import pdb; pdb.set_trace()
        mask = self._generate_decay_mask(scores[regular_sent_idx], logits_biased)
        scores[regular_sent_idx] = torch.log(
            self._apply_decay_mask(scores[regular_sent_idx], mask)
        )

        for debiasing_sent_idx in bias_indices:
            scores[debiasing_sent_idx] = scores[regular_sent_idx]

    def _apply_decay_mask(
        self, logits: torch.Tensor, decay_mask: torch.Tensor
    ) -> torch.Tensor:
        """Applies exponential decay to a tensor of logits"""
        # import pdb; pdb.set_trace()
        probabilities = logits.softmax(dim=-1)
        decay_mask = torch.exp(-decay_mask * self.decay_constant)
        decay_mask = torch.max(
            decay_mask, torch.tensor([self.epsilon], device=decay_mask.device)
        )
        probabilities = probabilities * decay_mask
        probabilities = probabilities / probabilities.sum(dim=-1)
        return probabilities

    def _generate_decay_mask(
        self,
        logits_regular: torch.FloatTensor,
        logits_biased_list: List[torch.FloatTensor],
    ) -> torch.Tensor:
        """Computes the alpha values (see paper) for each token and stores them in a mask tensor"""
        p_regular = logits_regular.softmax(dim=-1)
        p_biased = None

        for logits_biased in logits_biased_list:
            if p_biased is None:
                p_biased = logits_biased.softmax(dim=-1)
            else:
                p_biased = torch.max(p_biased, logits_biased.softmax(dim=-1))

        if self.debug:
            print(
                f"== Before Debiasing ==\n"
                f"Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}\n"
                f"Top 5 predictions (biased): {self._get_most_likely_tokens(p_biased, k=5)}"
            )

        mask = torch.max(
            p_biased - p_regular, torch.tensor([0.0], device=p_regular.device)
        )

        if self.debug:
            p_regular = self._apply_decay_mask(logits_regular, mask)
            print(
                f"== After Debiasing ==\n"
                f"Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}"
            )

        return mask

    def _get_most_likely_tokens(
        self, probabilities_tensor: torch.Tensor, k: int
    ) -> List[Tuple[str, float]]:
        """Returns the most likely tokens according to a tensor of probabilities"""
        assert len(probabilities_tensor.shape) == 1
        values, indices = torch.topk(probabilities_tensor, k=k, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        return list(zip(tokens, [pv.item() for pv in values]))


