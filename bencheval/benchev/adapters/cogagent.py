"""CogAgent-9B-20241220 (zai-org) via trust_remote_code, as in the model card: the query concatenation with no history,
Platform WIN (MedSPOT is desktop software), answer CLICK(box=[[x1,y1,x2,y2]]) in 0-1000. Its own fixed image size
(max_pixels unused). Greedy decoding."""
from benchev.adapters.base import RemoteRecipeAdapter
from benchev.adapters.recipes import RECIPES


class CogAgent(RemoteRecipeAdapter):
    name = "cogagent"

    def load(self, model_cfg, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.recipe = RECIPES[model_cfg.recipe]
        self.tok = AutoTokenizer.from_pretrained(model_cfg.ckpt, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.ckpt, torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True, device_map=device).eval()

    def _generate(self, image, text):
        import torch
        x = self.tok.apply_chat_template([{"role": "user", "image": image, "content": text}], add_generation_prompt=True,
                                         tokenize=True, return_tensors="pt", return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=self.recipe.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
