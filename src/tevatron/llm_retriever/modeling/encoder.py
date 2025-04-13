"""
The script implements the LLM-enhanced retriever model.

Our model including one retriever and one LLM:
1. an LLM-based retriever
2. an LLM model as enhancer

Data flow:

Three forward-pass:
1. encoder
2. Reference LLM seperately encode the text chunks
3. Reference LLM jointly encode text chunks
"""


from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.llm_retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments, DataArguments

from torch.nn import functional as F

from torch.nn import CrossEntropyLoss

import logging
logger = logging.getLogger(__name__)

import hashlib
import copy


@dataclass
class EncoderOutput(ModelOutput):
    p_reps_llm: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    # TRANSFORMER_CLS = AutoModel
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 reference: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 attn_temperature: float = 1.0,
                 exclude_diagonal: bool = True,
                 top_k: int = 16,
                 dual_loss=False,
                 freeze_reference=False,
                 first_half=False,
                 trainable_temperature=False,
                 disable_v_norm=False,
                 ):
        super().__init__()
        self.config = encoder.config
        # encoder
        self.encoder = encoder
        # reference
        self.reference = reference

        self.pooling = pooling
        self.normalize = normalize
        
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
            self.attn_temperature = nn.Parameter(torch.tensor(attn_temperature))
        else:
            self.temperature = temperature
            self.attn_temperature = attn_temperature
        
        self.exclude_diagonal = exclude_diagonal
        self.group_size = top_k + 1
        self.dual_loss = dual_loss
        self.freeze_reference = freeze_reference
        self.first_half = first_half
        self.disable_v_norm = disable_v_norm
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()


    def calculate_attn(self, cosine_similarity: torch.Tensor, exclude_diagonal: bool = False) -> torch.Tensor:
        """
        Calculate attention weights based on cosine similarity.

        Args:
            cosine_similarity (torch.Tensor): A tensor containing cosine similarity scores.
            exclude_diagonal (bool, optional): If True, excludes self-attention by masking the diagonal. Defaults to False.

        Returns:
            torch.Tensor: A tensor of attention weights after applying softmax.
        """
        if self.attn_temperature <= 0:
            raise ValueError("Temperature must be positive.")
        
        # Scale the cosine similarity by the temperature
        scaled_cosine_similarity = cosine_similarity / self.attn_temperature
        
        if exclude_diagonal:
            # Create a mask to exclude the diagonal elements
            mask = torch.eye(scaled_cosine_similarity.size(0), device=scaled_cosine_similarity.device).bool()
            # Apply the mask by setting diagonal elements to -inf
            scaled_cosine_similarity = scaled_cosine_similarity.masked_fill(mask, float('-inf'))
        
        # Compute attention weights using softmax along the last dimension
        attention_weights = F.softmax(scaled_cosine_similarity, dim=-1)
        
        return attention_weights
    
    def forward(self,
                passage: Dict[str, Tensor],
                passage_llm: Dict[str, Tensor] = None,
                passage_query: Dict[str, Tensor] = None):
        # three-time forward pass
        # Unit test
        # print('self.encoder: ', torch.sum(self.encoder.base_model.model.layers[0].self_attn.q_proj.lora_A.default.weight))
        # print('self.reference: ', torch.sum(self.reference.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight))
        # 1 similarity between passages (encoder) return triangle similarity matrix
        # input: B x L
        # output: B x D - B: per_device_train_batch_size
        p_reps = self.encode_passage(passage)
        normalized_tensor = F.normalize(p_reps, p=2, dim=1)
        # calcate the cosine similarity
        # B x B
        if passage_query is not None:
            p_reps_query = self.encode_passage(passage_query)
            normalized_tensor_query = F.normalize(p_reps_query, p=2, dim=1)
            # use first half to retrieve full docs
            cosine_similarity = torch.matmul(normalized_tensor_query, normalized_tensor.transpose(0, 1))
        else:
            cosine_similarity = torch.matmul(normalized_tensor, normalized_tensor.transpose(0, 1))

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")

        # B x B
        attn_weights = self.calculate_attn(cosine_similarity, exclude_diagonal=self.exclude_diagonal)

        # 2 forward passages to get hidden states (reference)
        # Input: B x L ; Output: Tuple num_layer (key, value) B x num_heads x L x head_dim
        # passage_llm_no_labels = {k: v for k, v in passage_llm.items() if k != "labels"}
        passage_llm_no_mask = {k: v for k, v in passage_llm.items() if k != "first_half_mask"}
        cached_outputs = self.reference(**passage_llm_no_mask, use_cache=True, output_hidden_states=True)
        past_key_values = cached_outputs.past_key_values
        seq_loss = cached_outputs.loss
        
        # print("attn_weights: ", attn_weights)
        # 3 calculate loss by reference - 1. attention + 2. hidden states
        
        loss = self.reference(
            **passage_llm_no_mask,
            inbatch_attn=attn_weights,
            cached_key_values=past_key_values,
            disable_v_norm=self.disable_v_norm).loss
        
        agg_loss = None
        agg_seq_loss = None
        if self.training:
            if self.is_ddp:
                # logits = self._dist_gather_tensor(logits)
                # labels = self._dist_gather_tensor(labels)
                agg_loss = self._dist_gather_tensor(loss)            
                agg_loss = agg_loss.mean()
                if self.dual_loss:
                    agg_seq_loss = self._dist_gather_tensor(seq_loss)
                    agg_seq_loss = agg_seq_loss.mean()
                    agg_loss = agg_loss + agg_seq_loss
            if self.is_ddp:
                agg_loss = agg_loss * self.world_size
        # for eval
        else:
            loss = None
            raise ValueError("LLM-Retriever model is not supported for inference")
        return EncoderOutput(
            loss=agg_loss,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        if t.dim() == 0:
            t = t.unsqueeze(0)

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    def calculate_loss(self, logits, labels):
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss
    
    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            data_args: DataArguments,
            **hf_kwargs,
    ):  
        # LLama default is sdpa - however, only LlamaAttention is modiffed to support eager mode
        retriever_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        
        REFERENCE_BASE_MODEL = model_args.reference_model_name_or_path
        reference_model = cls.TRANSFORMER_CLS.from_pretrained(REFERENCE_BASE_MODEL, **hf_kwargs)

        # Ensure pad_token_id is set
        if retriever_model.config.pad_token_id is None:
            retriever_model.config.pad_token_id = 0
        if reference_model.config.pad_token_id is None:
            reference_model.config.pad_token_id = 0

        if model_args.lora or model_args.retriever_lora_name_or_path:
            if train_args.gradient_checkpointing:
                retriever_model.enable_input_require_grads()
            
            # Apply LoRA to the encoder (retriever)
            if model_args.retriever_lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.retriever_lora_name_or_path, **hf_kwargs)
                retriever_model = PeftModel.from_pretrained(retriever_model.base_model, model_args.retriever_lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                retriever_model = get_peft_model(retriever_model.base_model, lora_config)
        
        if train_args.gradient_checkpointing:
            reference_model.enable_input_require_grads()

        # reference encoder_lora_name_or_path
        if model_args.reference_lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(model_args.reference_lora_name_or_path, **hf_kwargs)
            reference_model = PeftModel.from_pretrained(reference_model, model_args.reference_lora_name_or_path, is_trainable=model_args.reference_training)
        elif model_args.freeze_reference:
            # The key part: if freeze_reference is True, we freeze all parameters in the reference model.
            for param in reference_model.parameters():
                param.requires_grad = False
        else:
            lora_config = LoraConfig(
                base_model_name_or_path=REFERENCE_BASE_MODEL,
                task_type=TaskType.FEATURE_EXTRACTION,
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_target_modules.split(','),
                inference_mode=False
            )
            reference_model = get_peft_model(reference_model, lora_config)
            

        model = cls(
            encoder=retriever_model,
            reference=reference_model,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            temperature=model_args.temperature,
            attn_temperature=model_args.attn_temperature,
            exclude_diagonal=model_args.exclude_diagonal,
            top_k=data_args.top_k,
            dual_loss=model_args.dual_loss,
            freeze_reference=model_args.freeze_reference,
            first_half=data_args.first_half,
            trainable_temperature=model_args.trainable_temperature,
            disable_v_norm=model_args.disable_v_norm,
        )

        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             retriever_lora_name_or_path: str = None,
             reference_lora_name_or_path: str = None,
             **hf_kwargs):
        """
        This funtion is useually used for inference
        """
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        if retriever_lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(retriever_lora_name_or_path, **hf_kwargs)
            encoder_lora_model = PeftModel.from_pretrained(base_model.model, retriever_lora_name_or_path, config=lora_config)
            encoder_lora_model = encoder_lora_model.merge_and_unload()
        else:                    
            encoder_lora_model = base_model.model
        
        if reference_lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(reference_lora_name_or_path, **hf_kwargs)
            reference_lora_model = PeftModel.from_pretrained(base_model, reference_lora_name_or_path, config=lora_config)
            reference_lora_model = reference_lora_model.merge_and_unload()
        else:
            reference_lora_model = base_model
        
        model = cls(
            encoder=encoder_lora_model,
            reference=reference_lora_model,
            pooling=pooling,
            normalize=normalize
        )

        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
