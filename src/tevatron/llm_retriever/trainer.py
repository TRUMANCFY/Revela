import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel
from torch.utils.data import SequentialSampler

import logging
logger = logging.getLogger(__name__)

class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            # Obtain the state_dict if not provided
            if state_dict is None:
                state_dict = self.model.state_dict()

            #
            # If your new EncoderModel has self.model.encoder and self.model.reference,
            # you can split the state dict by prefix. Adjust naming if you need them saved
            # all in one folder or separate.
            #
            encoder_prefix = "encoder."
            reference_prefix = "reference."

            # Make sure we can handle the new structure:
            # we assume that all encoder weights start with "encoder."
            # and all reference weights start with "reference."
            encoder_state_dict = {
                k[len(encoder_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(encoder_prefix)
            }
            reference_state_dict = {
                k[len(reference_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(reference_prefix)
            }

            # Now we can save these separately. If you want to keep
            # them in separate subdirectories, you can do so; otherwise,
            # you can just save in the main `output_dir` with different filenames.

            # Example: subfolders "encoder/" and "reference/" inside output_dir
            encoder_output_dir = os.path.join(output_dir, "encoder")
            reference_output_dir = os.path.join(output_dir, "reference")

            os.makedirs(encoder_output_dir, exist_ok=True)
            os.makedirs(reference_output_dir, exist_ok=True)

            # Save the encoder weights
            self.model.encoder.save_pretrained(
                encoder_output_dir,
                state_dict=encoder_state_dict,
                safe_serialization=self.args.save_safetensors
            )

            # Save the reference weights
            if hasattr(self.model, "reference") and self.model.reference is not None and not self.model.freeze_reference:
                self.model.reference.save_pretrained(
                    reference_output_dir,
                    state_dict=reference_state_dict,
                    safe_serialization=self.args.save_safetensors
                )
            else:
                logger.warning("No reference module found to save.")

        # If a tokenizer is attached, save it
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training args together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        passage_retrieval, passage_llm, passage_retrieval_first_half = inputs
        loss = model(passage=passage_retrieval,
                     passage_llm=passage_llm,
                     passage_query=passage_retrieval_first_half).loss
        
        return loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor
    
    def _get_train_sampler(self):
        # we need the squential sampler to ensure that the same passage is used for all queries
        return super(TevatronTrainer, self)._get_train_sampler()
