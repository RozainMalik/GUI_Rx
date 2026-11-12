"""Trainer callbacks for the attention-only run: stop at a step, audit which weights changed, and a run summary."""
import json
import os
import re

import torch
from transformers import TrainerCallback

from attn_ft.freeze import is_allowed


def latest_checkpoint(output_dir):
    """(step, path) of the newest complete checkpoint, or (0, None). A checkpoint dir without trainer_state.json
    (written last by the Trainer) was cut off mid-save and is ignored."""
    done = [(int(m.group(1)), os.path.join(output_dir, d))
            for d in (os.listdir(output_dir) if os.path.isdir(output_dir) else [])
            if (m := re.fullmatch(r"checkpoint-(\d+)", d))
            and os.path.exists(os.path.join(output_dir, d, "trainer_state.json"))]
    return max(done, default=(0, None))


class StopAtStepCallback(TrainerCallback):
    """Save and stop at `step`. The LR schedule still spans max_steps, so a later resume continues the same cosine."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step and state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


FINGERPRINT_CHUNK = 2 ** 22   # elements per step: <= 16 MiB of fp32, so no parameter is ever copied whole


def _param_sums(p):
    """(sum, abs-sum) in fp32, accumulated in fp64 over chunks: a whole-tensor p.float() of an embedding table
    (248k rows) is 1.9-3.8 GiB and can run a small GPU out of memory."""
    s = a = 0.0
    for c in p.detach().reshape(-1).split(FINGERPRINT_CHUNK):
        c = c.float()
        s += float(c.sum(dtype=torch.float64))
        a += float(c.abs().sum(dtype=torch.float64))
    return [s, a]


def _fingerprint(model):
    return {n: _param_sums(p) for n, p in model.named_parameters()}


class ParamAuditCallback(TrainerCallback):
    """Fingerprint every parameter at train begin and end, and list any that changed outside the allowed set."""

    def __init__(self, out_path: str, full_ids: list[int]):
        self.out_path, self.full_ids, self.begin = out_path, full_ids, None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.begin = _fingerprint(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        end = _fingerprint(model)
        changed = sorted(n for n in end if end[n] != self.begin.get(n))
        unexpected = [n for n in changed if not is_allowed(n, self.full_ids)]
        if args.process_index == 0:
            json.dump({"begin": self.begin, "end": end, "changed": changed, "unexpected": unexpected},
                      open(self.out_path, "w"), indent=0)


def write_run_summary(path, trainer, trainable: int) -> dict:
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    cuda = torch.cuda.is_available()
    s = {"global_step": trainer.state.global_step, "max_steps": trainer.state.max_steps,
         "trainable_params": trainable,
         "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30 if cuda else None,
         "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None,
         "last_loss": losses[-1] if losses else None}
    json.dump(s, open(path, "w"), indent=1)
    return s
