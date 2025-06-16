import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from .models.CLIP_prefix_caption.predict import ClipCaptionModel, generate2


def init_clip_caption(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClipCaptionModel(prefix_length=10)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()
    model = model.to(device)
    return model


def generate_caption(image, model, clip_model, preprocess, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(image, Image.Image):
        image = preprocess(image).unsqueeze(0).to(device)
    elif isinstance(image, str):
        image = preprocess(Image.open(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, 10, -1)
    return generate2(model, tokenizer, embed=prefix_embed)