"""Minimal Trainer for the Qwen3.5 path.

Upstream's AGUVISTrainer overrides transformers-4.51 internals (accelerator setup, dataloader,
optimizer groups) that changed in transformers 5. The only behaviour GUI-AIMA needs from it is
modality-length grouping, so that is all this subclass adds.
"""
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length


class Qwen35Trainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if getattr(self.args, "group_by_modality_length", False) and ds is not None and has_length(ds):
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        dataset=ds, lengths=ds.modality_lengths)
        return super()._get_train_sampler(train_dataset)
