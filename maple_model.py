import torch
import torch.nn as nn
import torch.nn.functional as F
from .models.CLIP.clip import clip
from .models.CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class MAPLEModel(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model = self.load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model

        self.prototype_layer = nn.Parameter(torch.randn(len(classnames), clip_model.visual.output_dim).to(self.device).to(self.dtype))
        self.dtype = torch.float32
        self.image_encoder.class_embedding = nn.Parameter(torch.randn(768, dtype=self.dtype, device=self.device))

        self.to(self.device)
        self.image_encoder.to(self.dtype)

        self.scaler = torch.cuda.amp.GradScaler()

    def load_clip_to_cpu(self, cfg):
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        # model_path = clip._download(url, root="./models")

        model_path = "/kangxueyan/models/ViT-B_32/ViT-B-32.pt"

        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = 768
        image_resolution = 224
        design_details = {
            "embed_dim": vision_width,
            "image_resolution": image_resolution,
            "vision_layers": 12,
            "vision_width": vision_width,
            "vision_patch_size": 32,
            "context_length": 77,
            "vocab_size": 49408,
            "transformer_width": 512,
            "transformer_heads": 8,
            "transformer_layers": 12
        }

        # return clip.build_model(state_dict, design_details)
        return clip.build_model(state_dict)

    def forward(self, image):
        image = image.to(self.dtype)
        x = self.image_encoder.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self.image_encoder.ln_pre(x)
        
        cls_token = self.image_encoder.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        pos_embed = self.image_encoder.positional_embedding.to(x.dtype)
        if pos_embed.size(0) != x.size(1):
            pos_embed = pos_embed[:x.size(1)]
            if pos_embed.size(0) < x.size(1):
                pos_embed = F.pad(pos_embed, (0, 0, 0, x.size(1) - pos_embed.size(0)))
        
        x = x + pos_embed
        
        x = x.permute(1, 0, 2)
        for i, layer in enumerate(self.image_encoder.transformer.resblocks):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN detected in image encoder layer {i}")
                x = torch.nan_to_num(x, nan=0.0)
        
        x = x.permute(1, 0, 2)
        x = self.image_encoder.ln_post(x[:, 0, :])
        
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj.to(x.dtype)
        
        if torch.isnan(x).any():
            print("NaN detected in image features")
            x = torch.nan_to_num(x, nan=0.0)
        
        return x

    def encode_text(self, text):
        return self.text_encoder(text)

    def get_prototypes(self):
        return self.prototype_layer

    def get_similarities(self, features):
        prototypes = self.get_prototypes()
        similarities = F.cosine_similarity(features.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        return similarities

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.device = next(clip_model.parameters()).device

        self.to(self.device)
    
    def forward(self, tokenized_prompts):
        tokenized_prompts = tokenized_prompts.to(self.device)
        seq_len = tokenized_prompts.shape[1]
        
        text_embeds = self.token_embedding(tokenized_prompts).type(self.dtype)
        pos_emd = self.positional_embedding[:seq_len, :].type(self.dtype)
        x = text_embeds + pos_emd

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, layer in enumerate(self.transformer.resblocks):
            x_prev = x.clone()
            x = layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN or Inf detected in transformer layer {i}")
                x = torch.where(torch.isnan(x) | torch.isinf(x), x_prev, x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0], device=self.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in text features")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        return x


def tokenize(texts, context_length=77):
    if texts is None or not texts:
        return None

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def check_and_fix_values(x, layer_name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"Warning: NaN or Inf values detected in {layer_name}")
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        print(f"Fixed {layer_name}: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
    return x
