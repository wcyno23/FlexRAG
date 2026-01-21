import os
import torch
from typing import Optional
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    Trainer,
    has_length,
    logger,
)


class CompressiveEncoderTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
