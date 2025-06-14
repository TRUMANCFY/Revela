import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    reference_model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
        default="meta-llama/Llama-3.2-1B",
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    pooling: str = field(
        default='cls',
        metadata={"help": "pooling method for query and passage encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )

    dual_loss: bool = field(
        default=False,
        metadata={"help": "loss combination between single instance and instances across the batch"}
    )

    num_layers: int = field(
        default=-1,
        metadata={"help": "number of layers"}
    )
    
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for softmax"}
    )

    attn_temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for attention"}
    )

    trainable_temperature: bool = field(
        default=False,
        metadata={"help": "trainable temperature"}
    )

    exclude_diagonal: bool = field(
        default=True,
        metadata={"help": "exclude diagonal from self attention"}
    )

    disable_v_norm: bool = field(
        default=False,
        metadata={"help": "disable v norm"}
    )
    
    # for lora
    lora: bool = field(default=False,
        metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )

    retriever_lora_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained lora model or model identifier from huggingface.co/models"}
    )

    reference_lora_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained lora model or model identifier from huggingface.co/models"}
    )

    reference_training: bool = field(
        default=True, metadata={"help": "use reference training"}
    )

    freeze_reference: bool = field(
        default=False, metadata={"help": "freeze reference model"}
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "lora r"}
    )

    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )

    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "lora target modules"}
    )

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    retriever_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default='json', metadata={"help": "We use the dataset (task) in the package of MTEB."}
    )

    dataset_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    dataset_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    dataset_number_of_shards: int = field(
        default=1, metadata={"help": "number of shards to split the dataset into"}
    )

    dataset_shard_index: int = field(
        default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"}
    )

    # add save_predictions
    stopwords: str = field(
        default='en', metadata={"help": "stopwords language"}
    )

    stemmer_language: str = field(
        default='english', metadata={"help": "stemmer language"}
    )
    
    bm25_retrieval_file: str = field(
        default=None, metadata={"help": "where to save the predictions"}
    )
    
    top_k: int = field(
        default=8, metadata={"help": "top k passages to retrieve"}
    )

    first_half: bool = field(
        default=False, metadata={"help": "use the first half of the passages for retrieval and the second half for next token prediction"}
    )

    first_half_ratio: float = field(
        default=0.5, metadata={"help": "ratio for the first half of the passages"}
    )
    
    train_group_size: int = field(
        default=8, metadata={"help": "number of passages used to train for each query"}
    )

    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage for training"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first n negative passages for training"})

    encode_is_query: bool = field(default=False)
    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})

    passage_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_prefix: str = field(
        default='passage: ', metadata={"help": "prefix or instruction for passage"}
    )

    add_passage_prefix: bool = field(
        default=False, metadata={"help": "add passage prefix"}
    )

    query_prefix: str = field(
        default='query: ', metadata={"help": "prefix or instruction for query"}
    )

    add_query_prefix: bool = field(
        default=False, metadata={"help": "add query prefix"}
    )

    append_eos_token: bool = field(
        default=False, metadata={"help": "append eos token to query and passage, this is currently used for repllama"}
    )

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    wandb_run_name: str = field(default=None, metadata={"help": "run name"})
    wandb_project: str = field(default=None, metadata={"help": "wandb project name"})
    wandb_key: str = field(default=None, metadata={"help": "wandb key"})