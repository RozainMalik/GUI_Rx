"""DeepSeek-VL2 (deepseek-ai/deepseek-vl2-small) via the official deepseek_vl2 package, as in the model card: the
<|ref|> grounding prompt, answer <|det|>[[x1,y1,x2,y2]]<|/det|> in 0-999, decoded with special tokens kept.
The processor's own dynamic tiling sets the image size (max_pixels unused). Needs its own venv (deepseek_vl2 pins
transformers 4.38)."""
from benchev.adapters.base import RemoteRecipeAdapter
from benchev.adapters.recipes import RECIPES


class DeepseekVl2(RemoteRecipeAdapter):
    name = "deepseek_vl2"

    def load(self, model_cfg, device):
        import torch
        from deepseek_vl2.models import DeepseekVLV2Processor
        from transformers import AutoModelForCausalLM
        self.recipe = RECIPES[model_cfg.recipe]
        self.proc = DeepseekVLV2Processor.from_pretrained(model_cfg.ckpt)
        # bf16 at load: transformers 4.38 would otherwise materialise 16B params in fp32 (~64 GB host RAM)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                          low_cpu_mem_usage=True)
        self.model = self.model.to(device).eval()

    def _generate(self, image, text):
        import torch
        tok = self.proc.tokenizer
        conv = [{"role": "<|User|>", "content": text, "images": ["<in memory>"]}, {"role": "<|Assistant|>", "content": ""}]
        x = self.proc(conversations=conv, images=[image], force_batchify=True, system_prompt="").to(self.model.device)
        with torch.no_grad():
            emb = self.model.prepare_inputs_embeds(**x)
            # DeepseekVLV2ForCausalLM names its LM `language`, not `language_model`
            out = self.model.language.generate(inputs_embeds=emb, attention_mask=x.attention_mask,
                                                     pad_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
                                                     eos_token_id=tok.eos_token_id, max_new_tokens=self.recipe.max_new_tokens,
                                                     do_sample=False, use_cache=True)
        return tok.decode(out[0].cpu().tolist(), skip_special_tokens=False)
