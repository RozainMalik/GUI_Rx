"""GUI-AIMA on Qwen3.5: train only the full-attention blocks + pointer-token rows.

Same data, model and losses as GUI-AIMA's train.py; only the parameter setup differs.
"""
import os
import sys
from dataclasses import dataclass, field

import torch
import transformers
from transformers import AutoProcessor

from attn_ft.callbacks import ParamAuditCallback, StopAtStepCallback, latest_checkpoint, write_run_summary
from attn_ft.export import export_bf16
from attn_ft.freeze import cast_trainable_to_fp32, count_trainable, setup_full_attention_trainable
from attn_ft.pointer_rows import PointerRowEmbedding
from gui_aima.qwen35.attention import full_attention_layer_ids
from gui_aima.qwen35.modeling import Qwen3_5ForConditionalGenerationWithPointer
from gui_aima.qwen35.trainer import Qwen35Trainer
from gui_aima.train_common import (ADDITIONAL_SPECIAL_TOKENS, DataArguments, EmptyCacheCallback, ModelArguments,
                                   TrainingArguments, dump_args_to_json, make_supervised_data_module, rank0_print,
                                   smart_tokenizer_and_embedding_resize, update_pointer_token_ids)


@dataclass
class AttnFTArguments:
    stop_at_step: int = field(default=0, metadata={"help": "save and stop at this step (0 = off); resume continues"})
    expect_trainable_range: str = field(default="4.5e8,5.0e8", metadata={"help": "'lo,hi' or 'none'"})
    frozen_dtype: str = field(default="bfloat16", metadata={"help": "dtype of the frozen weights (float32 for CPU tests)"})
    audit_params: bool = field(default=False)
    export_final: bool = field(default=True)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttnFTArguments))
    model_args, data_args, training_args, ft_args = parser.parse_args_into_dataclasses()
    if model_args.model_type != "qwen35":
        raise ValueError("this repo supports --model_type qwen35 only")

    done, resume_ckpt = latest_checkpoint(training_args.output_dir)
    if ft_args.stop_at_step and done >= ft_args.stop_at_step:
        print(f"already at step {done} >= stop_at_step {ft_args.stop_at_step}: nothing to do")
        return

    model = Qwen3_5ForConditionalGenerationWithPointer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="sdpa",
        dtype=getattr(torch, ft_args.frozen_dtype))
    model.config.use_cache = False
    model.reset_loss_weights(pointer_loss_weight=training_args.pointer_loss_weight,
                             lm_loss_weight=training_args.lm_loss_weight)
    if model_args.weighting is not None:
        model.set_attention_args(weighting=model_args.weighting)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length, padding_side="right")
    smart_tokenizer_and_embedding_resize({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}, tokenizer, model)
    update_pointer_token_ids(model.config, tokenizer, model_args.number_of_points)

    setup_full_attention_trainable(model)
    cast_trainable_to_fp32(model)
    PointerRowEmbedding.install(model, tokenizer.convert_tokens_to_ids(ADDITIONAL_SPECIAL_TOKENS))
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    trainable = count_trainable(model)
    rank0_print(f"trainable parameters: {trainable:,} of {sum(p.numel() for p in model.parameters()):,}")
    if ft_args.expect_trainable_range != "none":
        lo, hi = (float(v) for v in ft_args.expect_trainable_range.split(","))
        if not lo <= trainable <= hi:
            raise ValueError(f"trainable parameter count {trainable:,} outside [{lo:,.0f}, {hi:,.0f}]")

    data_args.processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                        min_pixels=data_args.min_pixels, max_pixels=data_args.max_pixels)
    data_args.processor.tokenizer = tokenizer
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.process_index == 0:
        dump_args_to_json(model.config, data_args.processor, model_args, data_args, training_args,
                          training_args.output_dir)
    data_module = make_supervised_data_module(tokenizer=tokenizer, processor=data_args.processor, data_args=data_args,
                                              training_args=training_args, number_of_points=model_args.number_of_points,
                                              image_token_id=model.config.image_token_id)

    callbacks = [EmptyCacheCallback(every_n_steps=training_args.empty_cache_every_n_steps)]
    if ft_args.stop_at_step:
        callbacks.append(StopAtStepCallback(ft_args.stop_at_step))
    if ft_args.audit_params:
        callbacks.append(ParamAuditCallback(os.path.join(training_args.output_dir, "param_audit.json"),
                                            full_attention_layer_ids(model.config.text_config)))
    trainer = Qwen35Trainer(model=model, processing_class=data_args.processor, args=training_args,
                            callbacks=callbacks, **data_module)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()
    write_run_summary(os.path.join(training_args.output_dir, "run_summary.json"), trainer, trainable)

    if trainer.state.global_step < trainer.state.max_steps:
        rank0_print(f"stopped at step {trainer.state.global_step}; resume with the same --output_dir")
        return
    trainer.save_model(training_args.output_dir)
    if trainer.args.should_save:
        tokenizer.save_pretrained(training_args.output_dir)
        data_args.processor.save_pretrained(training_args.output_dir)
        if ft_args.export_final:
            dtype = torch.bfloat16 if ft_args.frozen_dtype == "bfloat16" else torch.float32
            export_bf16(training_args.output_dir, os.path.join(training_args.output_dir, "final_bf16"), dtype=dtype)
    rank0_print(f"done: {training_args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
