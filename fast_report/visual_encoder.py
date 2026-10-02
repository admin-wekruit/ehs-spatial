"""Shared image/text encoders; model execution has no cloud orchestration dependency."""
ENCODERS = {"pe-core-l": "timm/PE-Core-L-14-336", "siglip2-so400m": "google/siglip2-so400m-patch16-384"}
X13_HF = "/v/x13/hf"
TEMPLATES = ("a photo of {}.", "a photo of {} in a workplace.")
BATCH = 256


class Encoder:
    """X13 naming's encoder, trimmed to the two text-image models: image(uint8 NHWC, masks or None) -> unit embeddings (the
    background outside the mask mid grey); text(phrases) -> unit embeddings (TEMPLATES averaged per phrase)."""

    def __init__(self, key):
        import torch
        self.key, self.torch = key, torch
        if key == "pe-core-l":
            import open_clip
            self.model, _, pre = open_clip.create_model_and_transforms("hf-hub:" + ENCODERS[key], cache_dir=X13_HF)
            self.model = self.model.cuda().eval().to(torch.bfloat16)
            self.tok = open_clip.get_tokenizer("hf-hub:" + ENCODERS[key])
            norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
            self.side, self.mean, self.std = 336, tuple(norm.mean), tuple(norm.std)
        else:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(ENCODERS[key], cache_dir=X13_HF, torch_dtype=torch.bfloat16).cuda().eval()
            self.tok = AutoTokenizer.from_pretrained(ENCODERS[key], cache_dir=X13_HF)
            self.side, self.mean, self.std = self.model.config.vision_config.image_size, (.5,) * 3, (.5,) * 3
        self.scale = float(self.model.logit_scale.exp())

    def image(self, arr, masks=None):
        import numpy as np
        torch = self.torch
        F = torch.nn.functional
        mean = torch.tensor(self.mean, device="cuda")[:, None, None]
        std = torch.tensor(self.std, device="cuda")[:, None, None]
        out = []
        for s in range(0, len(arr), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + BATCH])).cuda().permute(0, 3, 1, 2).float() / 255
            if masks is not None:
                x = torch.where(torch.from_numpy(masks[s:s + BATCH]).cuda()[:, None], x, .5)
            x = F.interpolate(x, size=(self.side, self.side), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
            x = ((x - mean) / std).to(torch.bfloat16)
            with torch.inference_mode():
                e = self.model.encode_image(x) if self.key == "pe-core-l" else self.model.get_image_features(pixel_values=x)
                e = getattr(e, "pooler_output", e)
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def text(self, phrases):
        torch = self.torch
        F = torch.nn.functional
        rows = []
        for p in phrases:
            prompts = [t.format(p) for t in TEMPLATES]
            with torch.inference_mode():
                if self.key == "pe-core-l":
                    e = self.model.encode_text(self.tok(prompts).cuda())
                else:
                    t = self.tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
                    e = self.model.get_text_features(**t)
                    e = getattr(e, "pooler_output", e)
            rows.append(F.normalize(F.normalize(e.float(), dim=-1).mean(0), dim=0).cpu())
        return torch.stack(rows).numpy()

