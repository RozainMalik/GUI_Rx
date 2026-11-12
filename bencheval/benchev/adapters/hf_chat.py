"""Published VLMs that run through transformers' chat templates (Qwen2/2.5/3-VL family, Mllama, Mistral3), driven by a
recipe (benchev.adapters.recipes): the model config's `recipe` picks the prompt, message layout, parser and coordinate
space. Greedy decoding. An unparsable answer is a miss (point None); the raw text is kept.

Image size: Qwen-family processors get the run's max_pixels (the model's own limit from its config, else the protocol's);
the size the model is actually fed (image_grid_thw x patch) fills the {width}/{height} placeholders and maps
abs_input answers back. Other processors keep their own fixed resizing, and the input size is taken as the given image.
"""
from benchev.adapters.base import Adapter
from benchev.adapters.recipes import RECIPES, parse, render, to_norm


class HfChat(Adapter):
    name = "hf_chat"
    crop_multiple = 28      # unused: comparison models run 1-step only

    def load(self, model_cfg, device):
        import torch
        from transformers import AutoModelForImageTextToText
        self.recipe = RECIPES[model_cfg.recipe]
        self.ckpt = model_cfg.ckpt
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.model = AutoModelForImageTextToText.from_pretrained(self.ckpt, torch_dtype=torch.bfloat16,
                                                                 attn_implementation=attn, device_map=device).eval()
        self.model_type = self.model.config.model_type
        self._processors = {}

    def processor(self, max_pixels):
        if max_pixels not in self._processors:
            from transformers import AutoImageProcessor, AutoProcessor
            # Qwen2/2.5/3-VL image processors (the ones with merge_size) take max_pixels, stored as size.longest_edge
            # (transformers 5.x has no max_pixels attribute to test for). Others keep
            # their own resizing.
            ip = AutoImageProcessor.from_pretrained(self.ckpt)
            kw = {"max_pixels": max_pixels} if hasattr(ip, "merge_size") else {}
            if getattr(self, "model_type", None) == "mistral3":
                kw["fix_mistral_regex"] = True   # Ministral-3 ships a wrong pre-tokenizer regex (transformers warns)
            self._processors[max_pixels] = AutoProcessor.from_pretrained(self.ckpt, **kw)
        return self._processors[max_pixels]

    def input_size(self, image, max_pixels):
        """(w, h) of the image the model is fed."""
        proc = self.processor(max_pixels)
        ip = proc.image_processor
        if not hasattr(ip, "merge_size"):
            return image.size
        _, gh, gw = ip(images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
        return gw * ip.patch_size, gh * ip.patch_size

    def messages(self, instruction, input_wh, orig_wh):
        r = self.recipe
        system, user = render(r, instruction, input_wh, orig_wh)
        msgs = [] if system is None else [{"role": "system", "content": [{"type": "text", "text": system}]}]
        text, img = {"type": "text", "text": user}, {"type": "image"}
        if r.layout == "image_text":
            msgs.append({"role": "user", "content": [img, text]})
        elif r.layout == "text_image":
            msgs.append({"role": "user", "content": [text, img]})
        else:   # text_then_image_turn
            msgs += [{"role": "user", "content": [text]}, {"role": "user", "content": [img]}]
        return msgs

    def _generate(self, image, msgs, max_pixels):
        import torch
        r, proc = self.recipe, self.processor(max_pixels)
        kw = {"chat_template": r.chat_template} if r.chat_template else {}
        if r.assistant_prefix:
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw) + r.assistant_prefix
        else:
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        x = proc(text=[text + r.prefill], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=r.max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None)
        raw = r.prefill + proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return raw.split(r.stop)[0] if r.stop else raw

    def predict(self, image, instruction, max_pixels):
        in_wh = self.input_size(image, max_pixels)
        raw = self._generate(image, self.messages(instruction, in_wh, image.size), max_pixels)
        pt = parse(self.recipe.parser, raw)
        return (None if pt is None else to_norm(pt, self.recipe.space, in_wh, image.size)), raw
