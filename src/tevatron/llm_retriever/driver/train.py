import logging
import os
import sys
import torch

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)


from transformers.trainer_utils import get_last_checkpoint

from tevatron.llm_retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.llm_retriever.dataset import LLMEnhancedCodeDataset as TrainDataset
from tevatron.llm_retriever.collator import TrainCollator
from tevatron.llm_retriever.collator import LLMEnhancedCollator as TrainCollator
from tevatron.llm_retriever.modeling import DenseModel
from tevatron.llm_retriever.trainer import TevatronTrainer as Trainer
from tevatron.llm_retriever.gc_trainer import GradCacheTrainer as GCTrainer

from transformers import TrainerCallback, TrainerState, TrainerControl

from torch.distributed import is_initialized, get_rank

logger = logging.getLogger(__name__)

import wandb

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        print('model_args: ', model_args)
        print('data_args: ', data_args)
        print('training_args: ', training_args)
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    if (not is_initialized() or get_rank() == 0) and training_args.wandb_key is not None:
        import wandb
        wandb.login(key=training_args.wandb_key)
        assert training_args.wandb_run_name is not None, "wandb_run_name must be set"
        wandb.init(
        # set the wandb project where this run will be logged
            project=training_args.wandb_project if training_args.wandb_project else 'my-awesome-project',
            name=training_args.wandb_run_name,
            id=training_args.wandb_run_name,
            resume='allow',
        )

    set_seed(training_args.seed)

    # we currently set LLama-3.2-1B as the tokenizer as the default tokenizer for reference
    reference_tokenizer = AutoTokenizer.from_pretrained(
        model_args.reference_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    retriever_tokenizer = AutoTokenizer.from_pretrained(
        model_args.retriever_tokenizer_name if model_args.retriever_tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # tokenizer.pad_token_id:  None
    # tokenizer.eos_token_id:  128001
    if reference_tokenizer.pad_token_id is None:
        reference_tokenizer.pad_token_id = reference_tokenizer.eos_token_id
    if retriever_tokenizer.pad_token_id is None:
        retriever_tokenizer.pad_token_id = retriever_tokenizer.eos_token_id

    reference_tokenizer.padding_side = 'right'
    retriever_tokenizer.padding_side = 'right'
    
    model = DenseModel.build(
        model_args,
        training_args,
        data_args,
        cache_dir=model_args.cache_dir,
    )
    
    train_dataset = TrainDataset(data_args)
    
    collator = TrainCollator(data_args, retriever_tokenizer, reference_tokenizer)
    
    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    train_dataset.trainer = trainer
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))  # TODO: resume training
    
    # trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        retriever_tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
