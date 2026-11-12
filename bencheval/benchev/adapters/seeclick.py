"""SeeClick (cckevinn/SeeClick, Qwen-VL 9.6B) via trust_remote_code and model.chat, as in the SeeClick repo and the
MedSPOT paper script: the "(with point)" prompt, answer (x, y) in [0, 1]. model.chat takes an image path, so the
image is written to a temp PNG. Decoding uses the checkpoint's generation config (as upstream). Its own fixed image
size (max_pixels unused). cckevinn/SeeClick ships no tokenizer files: its base model's tokenizer is used, as the SeeClick
repo's README does (Qwen/Qwen-VL-Chat, pinned)."""
import os
import tempfile

from benchev.adapters.base import RemoteRecipeAdapter
from benchev.adapters.recipes import RECIPES


TOKENIZER, TOKENIZER_REV = "Qwen/Qwen-VL-Chat", "f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"


class SeeClick(RemoteRecipeAdapter):
    name = "seeclick"

    def load(self, model_cfg, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        self.recipe = RECIPES[model_cfg.recipe]
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REV, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.ckpt, device_map=device, trust_remote_code=True,
                                                          bf16=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_cfg.ckpt, trust_remote_code=True)

    def _generate(self, image, text):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(path)
            query = self.tok.from_list_format([{"image": path}, {"text": text}])
            response, _ = self.model.chat(self.tok, query=query, history=None)
            return response
        finally:
            os.remove(path)
