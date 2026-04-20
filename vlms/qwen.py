import torch

from torch import Tensor
from typing import List, Union
from .base import BasePreprocessor
from transformers import AutoTokenizer
from utils.configs import model_configs
from .base import BaseVLM, PreprocessedPrompt, PreprocessedPromptWithImage
from .backbones.qwen.modeling_qwen import QWenLMHeadModel
from .backbones.qwen.qwen_generation_utils import make_context
from .self_debias_generation import SelfDebiasingLogitsProcessor

class QwenPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, generation_config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPrompt:
        # Make sure prompts and images are lists and have the same length
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images
        assert len(prompts) == len(
            images
        ), "The number of prompts and images must be the same."

        # Process prompt-image pairs
        all_input_ids = []
        input_lengths = []

        for prompt, image in zip(prompts, images):
            query = self.tokenizer.from_list_format(
                [
                    {"image": image},
                    {"text": prompt},
                ]
            )

            generation_config = self.generation_config
            history = []

            max_window_size = generation_config.max_window_size
            _, context_tokens = make_context(
                tokenizer=self.tokenizer,
                query=query,
                assistant_prefix=None,
                history=history,
                system="",
                max_window_size=max_window_size,
                chat_format=generation_config.chat_format,
            )

            input_ids = torch.tensor(context_tokens, dtype=torch.long)
            all_input_ids.append(input_ids)
            input_lengths.append(len(context_tokens))

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=0
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        # Return
        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)

    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> Tensor:
        # Make sure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Get tokenizer
        tokenizer = self.tokenizer
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            role_encoded = tokenizer.encode(
                role, add_special_tokens=set(tokenizer.IMAGE_ST)
            )
            content_encoded = tokenizer.encode(
                content, add_special_tokens=set(tokenizer.IMAGE_ST)
            )
            return role_encoded + nl_tokens + content_encoded

        # Process prompts
        all_input_ids = []
        input_lengths = []
        for prompt in prompts:
            system_tokens = (
                im_start_tokens
                + _tokenize_str("system", "")
                + im_end_tokens
                + nl_tokens
            )
            query_tokens = (
                im_start_tokens + _tokenize_str("user", prompt) + im_end_tokens
            )
            response_tokens_part = _tokenize_str("assistant", "")
            response_tokens = im_start_tokens + response_tokens_part
            context_tokens = system_tokens + query_tokens + nl_tokens + response_tokens
            input_ids = torch.tensor(context_tokens, dtype=torch.long)
            all_input_ids.append(input_ids)
            input_lengths.append(len(context_tokens))

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=0
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)


class Qwen(BaseVLM):
    def __init__(self, variant: str) -> None:
        # Select configuaration
        variant_config = model_configs["qwen"][variant]

        self.tokenizer = AutoTokenizer.from_pretrained(
            **variant_config, trust_remote_code=True,
        )
        
        self.model = QWenLMHeadModel.from_pretrained(
            **variant_config, device_map="cuda", trust_remote_code=True,
        ).eval()
        
        self.model.generation_config.top_k = 50


    def get_image_features(self, prompt: PreprocessedPrompt) -> Tensor:
        input_ids = prompt.input_ids.to(self.model.device)
        image_features = self.model.get_image_features(input_ids).to(self.model.device)
        return image_features


    def get_next_token_logits_with_early_exit(
            self, prompt: PreprocessedPrompt, early_exit_layers: Union[List[int], None],
    ) -> Tensor:        
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        input_lengths = prompt.input_lengths
        # Run inference
        logits_dict, outputs = self.model.forward(input_ids, early_exit_layers=early_exit_layers, output_hidden_states=True)
        
        
        # import pdb; pdb.set_trace()
        # Extract next token logits
        logits = outputs.logits if isinstance(outputs, dict) else outputs
        batch_size = logits.shape[0]
        logits = logits[torch.arange(batch_size), input_lengths - 1, :]
        logits = torch.softmax(logits, dim=-1)
        for layer in logits_dict:
            logits_dict[layer] = logits_dict[layer][torch.arange(batch_size), input_lengths - 1]
        return logits, logits_dict

    def get_next_token_probabilities(self, prompt: PreprocessedPrompt) -> Tensor:
        # Get input ids and lengths
        input_ids = prompt.input_ids.to(self.model.device)
        input_lengths = prompt.input_lengths

        # Forward pass
        output = self.model.forward(input_ids).logits

        # Select the last token probabilities
        output = output[torch.arange(output.size(0)), input_lengths - 1, :]
        # Apply softmax
        output = torch.softmax(output, dim=-1)

        return output

    def forward_with_prefix(self, input_ids, prompt_prefix: torch.nn.Parameter) -> Tensor:
        output = self.model.forward_with_prefix(
            input_ids=input_ids, 
            prompt_prefix=prompt_prefix)
        return output
    
    def get_next_token_probabilities_from_lm(self, prompt: PreprocessedPrompt) -> Tensor:
        return self.get_next_token_probabilities(prompt)

    def get_preprocessor(self) -> QwenPreprocessor:
        return QwenPreprocessor(self.tokenizer, self.model.generation_config)

    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.transformer.h
    
    def get_embedding_size(self) -> int:
        return self.model.transformer.wte.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.transformer.wte.weight.mean(dim=0)


    def get_generation(self, prompt: PreprocessedPrompt, **kwargs):
        input_ids = prompt.input_ids.to(self.model.device)
        return self.model.generate(input_ids, **kwargs)
    

    def get_next_token_probabilities_self_debias(
        self, prompt: PreprocessedPrompt, debiasing_prefixes,
        decay_constant=50, epsilon=0.01
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        input_lengths = prompt.input_lengths
        ### construct debias prefix
        logits_processor = SelfDebiasingLogitsProcessor(num_debiasing_prefixes=len(debiasing_prefixes),
                                                        decay_constant=decay_constant,
                                                        epsilon=epsilon,
                                                        debug=False,
                                                        tokenizer=self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        debiasing_prefix = debiasing_prefixes[0]  
        raw_prompt = self.tokenizer.decode(input_ids[0])
        debiasing_prompt = raw_prompt.split("</img>\n")[0] + "</img>\n" + debiasing_prefix + raw_prompt.split("</img>\n")[1]

        debias_input_ids = self.tokenizer.encode(debiasing_prompt, return_tensors="pt", add_special_tokens=False)

        debias_input_ids = debias_input_ids.to(self.model.device)

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes), 1) 
        attention_mask = torch.ones_like(input_ids_repeated)

        input_ids_repeated = input_ids_repeated.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

       
        outputs_original = self.model.forward(input_ids=input_ids_repeated, attention_mask=attention_mask).logits
        outputs_debiased = self.model.forward(input_ids=debias_input_ids).logits

        last_token_logits_original = outputs_original[:, -1, :]  
        last_token_logits_debiased = outputs_debiased[:, -1, :]  
        outputs = torch.cat([last_token_logits_original, last_token_logits_debiased], dim=0)

        
        outputs = logits_processor(
            input_ids=None, scores=outputs
        )

        logits = torch.softmax(outputs, dim=-1)
        logits = logits[0:1,:]
        return logits