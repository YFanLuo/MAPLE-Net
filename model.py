import torch
import torch.nn as nn
import torch.nn.functional as F
from maple_net.maple_model import MAPLEModel, tokenize
from .feature_bank import FeatureBank
import clip
from transformers import GPT2Tokenizer
from PIL import Image
from torchvision import transforms
from .generate_caption import  init_clip_caption, generate_caption


class FusionModel(nn.Module):
    def __init__(self, cfg, classnames, feature_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.classnames = classnames
        self.feature_bank = None

        self.clip_caption_model = init_clip_caption('/kangxueyan/models/COCO_weights/coco_weights.pt')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("/kangxueyan/models/gpt2")

        self.maple_model = MAPLEModel(cfg, classnames).to(self.device)

        self.fc = nn.Linear(feature_dim * 2, len(classnames)).to(self.device).to(self.dtype)
        self.projection = nn.Linear(feature_dim * 2, feature_dim).to(self.device).to(self.dtype)

        self.prototypes = nn.Parameter(torch.randn(len(classnames), feature_dim).to(self.device).to(self.dtype))
        nn.init.xavier_uniform_(self.prototypes)

        self.to(torch.float32)

    def extract_features(self, image, text):
        with torch.cuda.amp.autocast():
            image = image.to(self.device).to(self.dtype)
            image_features = self.maple_model(image)

            image_captions = []
            for img in image:
                img_pil = transforms.ToPILImage()(img.cpu())
                caption = generate_caption(
                    img_pil,
                    self.clip_caption_model,
                    self.clip_model,
                    self.preprocess,
                    self.tokenizer
                )
                image_captions.append(caption)

            tokenized_text = tokenize(text)
            if tokenized_text is None:
                print("Error: tokenize(text) returned None")
                return None

            tokenized_text = tokenized_text.to(self.device)
            text_features = self.maple_model.encode_text(tokenized_text)

            fused_features = torch.cat((image_features, text_features), dim=1)
            logits = self.fc(fused_features)
            projected_features = self.projection(fused_features)
            similarities = F.cosine_similarity(projected_features.unsqueeze(1),
                                               self.prototypes.unsqueeze(0), dim=2)
            similarities = (similarities + 1) / 2

            explanations, modality_weights = self._generate_explanations(logits, similarities,
                                                                         text_features, image_features)

            return logits, self.prototypes, projected_features, similarities, text_features, explanations, modality_weights, image_captions

    def forward(self, image, text):
        outputs = self.extract_features(image, text)
        if outputs is None:
            return None, None, None, None, None, None, None

        logits, prototypes, features, similarities, text_features, explanations, modality_weights, image_captions = outputs

        reference_info = None
        if self.feature_bank is not None and not self.training:
            reference_samples, ref_similarities = self.feature_bank.retrieve_diverse_samples(
                features,
                modality_weights=modality_weights[0] if modality_weights else None
            )
            reference_info = {
                'samples': reference_samples,
                'similarities': ref_similarities
            }

        return logits, prototypes, features, similarities, text_features, explanations, reference_info

    def _generate_explanations(self, logits, similarities, text_features, image_features):
        explanations = []
        modality_weights = []

        for i in range(logits.shape[0]):
            pred_class = torch.argmax(logits[i]).item()
            confidence = torch.softmax(logits[i], dim=0)[pred_class].item()

            text_importance = torch.norm(text_features[i]).item()
            image_importance = torch.norm(image_features[i]).item()
            text_weight = text_importance / (text_importance + image_importance)
            image_weight = image_importance / (text_importance + image_importance)
            modality_weights.append((text_weight, image_weight))

            if len(self.classnames) == 2:
                sentiment_labels = ['negative', 'positive']
            elif len(self.classnames) == 3:
                sentiment_labels = ['negative', 'neutral', 'positive']
            else:
                sentiment_labels = self.classnames

            explanation = f"The model predicts {sentiment_labels[pred_class]} with {confidence:.2f} confidence. "

            if similarities[i][pred_class] > 0.8:
                explanation += f"The prediction is strongly aligned with the {sentiment_labels[pred_class]} prototype."
            elif similarities[i][pred_class] > 0.5:
                explanation += f"The prediction shows moderate alignment with the {sentiment_labels[pred_class]} prototype."
            else:
                explanation += f"The prediction shows weak alignment with the {sentiment_labels[pred_class]} prototype."

            explanations.append(explanation)

        return explanations, modality_weights

    def get_prototypes(self):
        return self.maple_model.get_prototypes()

    def init_feature_bank(self, dataloader):
        if self.feature_bank is None:
            self.feature_bank = FeatureBank(self.device)
            self.feature_bank.build_bank(self, dataloader)
