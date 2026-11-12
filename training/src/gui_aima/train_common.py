"""Shared training pieces from GUI-AIMA's train.py, ported to Qwen3.5 (the training loop itself is train_attn.py)."""
import importlib.util
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import json
import torch
import transformers
from PIL import ImageFile
from transformers import (
    AutoProcessor,
)
import re
from gui_aima.dataset import LazySupervisedDataset
from gui_aima.trainer import rank0_print, EmptyCacheCallback  # noqa: F401
from gui_aima.utils import dump_args_to_json

from gui_aima.constants import (
    IGNORE_INDEX,
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN_0,
    DEFAULT_POINTER_PAD_TOKEN_1,
    DEFAULT_POINTER_PAD_TOKEN_2,
    DEFAULT_POINTER_PAD_TOKEN_3,
    DEFAULT_POINTER_PAD_TOKEN_4,
    DEFAULT_POINTER_PAD_TOKEN_5,
)


torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn_2_enabled: bool = field(default=True)
    model_type: str = field(default="qwen2vl", metadata={"help": "model type: qwen2vl or qwen25vl"})
    weighting: str = field(default=None, metadata={"help": "max or sum or max_soft or max_linear or sum_soft or sum_linear"})
    number_of_points: int = field(default=1, metadata={"help": "number of points to use for the pointer"})

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    min_pixels: Optional[int] = field(default=3136) # 2 * 2 * 28 * 28 = 56 * 56
    max_pixels: Optional[int] = field(default=5720064)
    max_conv_turns: Optional[int] = field(default=10)
    image_patch_size: int = field(default=14, metadata={"help": "ViT patch size: 14 for Qwen2.5-VL, 16 for Qwen3.5"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    group_by_modality_length: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    
    unfreeze_all_parameters: bool = field(default=False)
    unfreeze_pointer_head: bool = field(default=True)
    unfreeze_lm_head: bool = field(default=False)
    unfreeze_base_model: bool = field(default=False)
    unfreeze_last_n_layers: int = field(default=-1)
    unfreeze_new_tokens: bool = field(default=True)
    unfreeze_visual: bool = field(default=False)
    pointer_loss_weight: float = field(default=0.1)
    lm_loss_weight: float = field(default=-1.0)
    empty_cache_every_n_steps: int = field(default=15)
    lora_r: int = field(default=0, metadata={"help": "0 = full fine-tuning; >0 = LoRA rank on the language model"})
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    model_dtype: str = field(default="auto", metadata={
        "help": "dtype the weights are loaded in. auto = bf16 if --bf16 else fp32. Full fine-tuning at small LR needs float32 "
                "(fp32 master weights; --bf16 then only sets autocast), otherwise most bf16 updates round to zero."})


def resolve_model_dtype(training_args) -> torch.dtype:
    choice = getattr(training_args, "model_dtype", "auto")
    if choice == "auto":
        return torch.bfloat16 if training_args.bf16 else torch.float32
    if choice in ("float32", "bfloat16"):
        return getattr(torch, choice)
    raise ValueError(f"--model_dtype must be auto, float32 or bfloat16, got {choice!r}")

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    new_vocab_size = len(tokenizer)    
    # Update base model and current model config
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def update_pointer_token_ids(model_config: transformers.PretrainedConfig, tokenizer: transformers.PreTrainedTokenizer, number_of_points: int):
    model_config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model_config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    if number_of_points==1:
        model_config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
    else:
        pad_tokens = [DEFAULT_POINTER_PAD_TOKEN_0, DEFAULT_POINTER_PAD_TOKEN_1, DEFAULT_POINTER_PAD_TOKEN_2, DEFAULT_POINTER_PAD_TOKEN_3, DEFAULT_POINTER_PAD_TOKEN_4, DEFAULT_POINTER_PAD_TOKEN_5]
        model_config.pointer_pad_token_id=[tokenizer.encode(pad_tokens[i])[0] for i in range(number_of_points)] 
    rank0_print(f"Updated pointer token ids: {model_config.pointer_pad_token_id}, {model_config.pointer_start_token_id}, {model_config.pointer_end_token_id}")

def setup_params_to_update(model: transformers.PreTrainedModel, training_args: TrainingArguments):
    if training_args.unfreeze_all_parameters:
        rank0_print(f"Unfreezing all model parameters...")
        for p in model.parameters():
            p.requires_grad = True
    else:
        rank0_print(f"Freezing all model parameters...")
        for p in model.parameters():
            p.requires_grad = False

        if training_args.unfreeze_pointer_head:
            rank0_print(f"Unfreezing pointer head parameters...")
            for p in model.multi_patch_pointer_head_attention.parameters():
                p.requires_grad = True

        if training_args.unfreeze_lm_head:
            rank0_print(f"Unfreezing lm head parameters...")
            for p in model.lm_head.parameters():
                p.requires_grad = True
        
        if training_args.unfreeze_base_model: # including text tokens
            rank0_print(f"Unfreezing base model parameters...")
            for p in model.model.parameters():
                p.requires_grad = True

        if training_args.unfreeze_last_n_layers > 0:
            rank0_print(f"Unfreezing last {training_args.unfreeze_last_n_layers} layers of base model parameters...")
            for p in model.model.layers[-training_args.unfreeze_last_n_layers:].parameters():
                p.requires_grad = True

        if training_args.unfreeze_new_tokens:
            rank0_print(f"Unfreezing new tokens parameters via embedding hook...")
            model.model.embed_tokens.weight.requires_grad = True
        
        if training_args.unfreeze_visual:
            rank0_print(f"Unfreezing visual parameters...")
            for p in model.visual.parameters():
                p.requires_grad = True

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_id: Optional[int] = None

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {
            "input_ids": input_ids,
            "labels": labels.long() if labels.dtype == torch.int32 else labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.concat([instance["pixel_values"] for instance in instances], dim=0)
            batch["image_grid_thw"] = torch.concat([instance["image_grid_thw"] for instance in instances], dim=0)

        if "coordinates" in instances[0]:
            batch["coordinates"] = [instance["coordinates"] for instance in instances]
            batch["visual_token_indices_of_coordinates"] = [instance["visual_token_indices_of_coordinates"] for instance in instances]

        if "multi_patch_labels" in instances[0]:
            batch["multi_patch_labels"] = [instance["multi_patch_labels"] for instance in instances]

        if self.image_token_id is not None:
            batch["mm_token_type_ids"] = (batch["input_ids"] == self.image_token_id).int()

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                processor: transformers.ProcessorMixin,
                                data_args: DataArguments,
                                training_args: TrainingArguments,
                                number_of_points: int,
                                image_token_id: Optional[int] = None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, processor=processor, data_path=data_args.data_path, data_args=data_args,number_of_points=number_of_points
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, image_token_id=image_token_id)
    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": data_collator}
