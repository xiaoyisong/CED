import torch

from torch import Tensor
from typing import Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, List
from .self_debias_generation import SelfDebiasingLogitsProcessor


@dataclass
class PreprocessedPrompt:
    input_ids: Tensor
    input_lengths: Tensor


@dataclass
class PreprocessedPromptWithImage(PreprocessedPrompt):
    images: Tensor


class BaseVLM(ABC):
    @abstractmethod
    def get_next_token_probabilities(self, prompt: PreprocessedPrompt) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_next_token_probabilities_from_lm(
        self, prompt: PreprocessedPrompt
    ) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_preprocessor(self) -> "BasePreprocessor":
        raise NotImplementedError
    
    @abstractmethod
    def get_llm_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError
    
    @abstractmethod
    def get_embedding_size(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def get_avg_embedding(self) -> Tensor:
        raise NotImplementedError
    

    
    # @abstractmethod
    def get_image_features(self, prompt: PreprocessedPromptWithImage) -> Tensor:
        raise NotImplementedError
    
    # @abstractmethod
    def get_next_token_logits_with_early_exit(
            self, prompt: PreprocessedPromptWithImage, early_exit_layers: Union[List[int], None],
    ) -> Tensor:    
        raise NotImplementedError
    
    def get_next_token_probabilities_self_debias(
        self, prompt: PreprocessedPromptWithImage, debiasing_prefixes,
        decay_constant=50, epsilon=0.01
    ) -> Tensor:
        raise NotImplementedError
    
    def get_generation(self, prompt: PreprocessedPrompt, **kwargs):
        raise NotImplementedError
    
class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPrompt:
        raise NotImplementedError

    @abstractmethod
    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> PreprocessedPrompt:
        raise NotImplementedError


class LLaVATypeVLM(BaseVLM):
    
    def get_image_features(self, prompt: PreprocessedPromptWithImage) -> Tensor:
        images = prompt.images.to(device=self.model.device, dtype=torch.float16)
        image_features = self.model.encode_images(images).to(self.model.device)
        return image_features
    
    def get_next_token_logits_with_early_exit(
            self, prompt: PreprocessedPromptWithImage, early_exit_layers: Union[List[int], None],
    ) -> Tensor:        
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        image = prompt.images.to(device=self.model.device, dtype=torch.float16)

        # Run inference
        logits_dict, outputs = self.model.forward(input_ids, images=image, early_exit_layers=early_exit_layers)
        # import pdb; pdb.set_trace()
        # Extract next token logits
        outputs = outputs.logits if isinstance(outputs, dict) else outputs
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        for layer in logits_dict:
            logits_dict[layer] = logits_dict[layer][torch.arange(batch_size), last_token_positions]
        return logits, logits_dict
    
    def get_next_token_probabilities(
        self, prompt: PreprocessedPromptWithImage
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        image = prompt.images.to(device=self.model.device, dtype=torch.float16)

        # Run inferece
        outputs = self.model.forward(input_ids, images=image).logits

        # Extract next token probabilities
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)
        return logits

    def get_next_token_probabilities_from_lm(
        self, prompt: PreprocessedPrompt
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)

        # Run inferece
        outputs = self.model.model.forward(input_ids).last_hidden_state
        outputs = self.model.lm_head(outputs)

        # Extract next token probabilities
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)
        return logits
    
    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_embedding_size(self) -> int:
        return self.model.model.embed_tokens.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.model.embed_tokens.weight.mean(dim=0)

    def get_generation(self, prompt: PreprocessedPrompt, **kwargs):
        input_ids = prompt.input_ids.to(device=self.model.device)
        images = prompt.images.to(device=self.model.device, dtype=torch.float16)

        return self.model.generate(input_ids, images, **kwargs)
    
    def get_next_token_probabilities_self_debias(
        self, prompt: PreprocessedPromptWithImage, debiasing_prefixes,
        decay_constant=50, epsilon=0.01
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        image = prompt.images.to(device=self.model.device, dtype=torch.float16)

        
        ### construct debias prefix
        logits_processor = SelfDebiasingLogitsProcessor(num_debiasing_prefixes=len(debiasing_prefixes),
                                                        decay_constant=decay_constant,
                                                        epsilon=epsilon,
                                                        debug=False,
                                                        tokenizer=self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        debiasing_prefix = debiasing_prefixes[0]  
        input_prefixes = self.tokenizer.encode(debiasing_prefix, return_tensors="pt", add_special_tokens=False)

        input_prefixes = input_prefixes.to(self.model.device)
        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes), 1)  
        attention_mask = torch.ones_like(input_ids_repeated)

 
        input_ids_repeated = input_ids_repeated.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)


        input_ids_no_s = input_ids[:, 1:] 
        attention_mask_no_s = attention_mask[:, 1:]
        combined_input_ids = torch.cat([input_prefixes, input_ids_no_s], dim=1)  
        combined_attention_mask = torch.cat([torch.ones_like(input_prefixes), attention_mask_no_s], dim=1) 
        outputs_original = self.model.forward(input_ids=input_ids_repeated, attention_mask=attention_mask, images=image).logits
        outputs_debiased = self.model.forward(input_ids=combined_input_ids, attention_mask=combined_attention_mask, images=image).logits


        last_token_logits_original = outputs_original[:, -1, :]  
        last_token_logits_debiased = outputs_debiased[:, -1, :]  

        outputs = torch.cat([last_token_logits_original, last_token_logits_debiased], dim=0)

        
        outputs = logits_processor(
            input_ids=None, scores=outputs
        )

        logits = torch.softmax(outputs, dim=-1)
        logits = logits[0:1,:]
        return logits