import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

class FeatureBank:
    def __init__(self, device):
        self.device = device
        self.features = None
        self.samples = None
        self.label_to_indices = {}

    def build_bank(self, model, dataloader):
        model.eval()
        features_list = []
        samples_list = []
        labels_list = []

        print("Building feature bank from training data...")
        progress_bar = tqdm(dataloader, desc="Extracting features")

        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                texts = batch['text']

                outputs = model.extract_features(images, texts)
                if outputs is None:
                    continue

                logits, prototypes, features, similarities, text_features, explanations, modality_weights, image_captions = outputs
                features = features.float().cpu()

                features_list.append(features)
                samples_list.extend([{
                    'text': txt,
                    'image_caption': cap,
                    'label': lbl.item()
                } for txt, cap, lbl in zip(batch['text'], image_captions, batch['label'])])
                labels_list.extend(batch['label'].cpu().numpy())

                progress_bar.set_postfix({
                    'Features': f'{len(features_list) * batch["image"].size(0)}',
                    'Memory': f'{torch.cuda.max_memory_allocated() / 1024 ** 2:.1f}MB'
                })

            if features_list:
                print("\nConcatenating features...")
                self.features = torch.cat(features_list, dim=0)
                self.samples = samples_list
                print("Building label indices...")
                self._build_label_indices(labels_list)
                print(f"Feature bank built successfully with {len(self.samples)} samples")

                # print("\nSamples per class:")
                for label, indices in self.label_to_indices.items():
                    print(f"Class {label}: {len(indices)} samples")

    def _build_label_indices(self, labels):
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def retrieve_diverse_samples(self, query_feature, k_per_class=2, modality_weights=None):

        if self.features is None:
            return [], torch.tensor([], device=self.device)

        with torch.no_grad():
            query_feature = query_feature.to(self.device).float()
            features = self.features.to(self.device).float()

            feature_dim = features.shape[1]
            text_dim = feature_dim // 2

            query_text = query_feature[:, :text_dim]
            query_image = query_feature[:, text_dim:]
            features_text = features[:, :text_dim]
            features_image = features[:, text_dim:]

            text_similarities = F.cosine_similarity(
                query_text.unsqueeze(1),
                features_text.unsqueeze(0),
                dim=2
            )

            image_similarities = F.cosine_similarity(
                query_image.unsqueeze(1),
                features_image.unsqueeze(0),
                dim=2
            )

            if modality_weights is None:
                text_weight, image_weight = 0.5, 0.5
            else:
                text_weight, image_weight = modality_weights

            similarities = text_weight * text_similarities + image_weight * image_similarities

            selected_samples = []
            selected_similarities = []

            for label in sorted(self.label_to_indices.keys()):
                indices = self.label_to_indices[label]
                if not indices:
                    continue

                class_similarities = similarities[:, indices]
                k = min(k_per_class, len(indices))

                top_k_similarities, top_k_indices = class_similarities.topk(k)
                original_indices = [indices[idx.item()] for idx in top_k_indices[0]]

                for idx, sim in zip(original_indices, top_k_similarities[0]):
                    sample = self.samples[idx]
                    selected_samples.append(sample)
                    selected_similarities.append(sim.item())

            if selected_samples:
                combined = list(zip(selected_samples, selected_similarities))
                combined.sort(key=lambda x: x[1], reverse=True)

                final_samples = []
                final_similarities = []
                seen_classes = set()

                for sample, sim in combined:
                    if len(final_samples) >= 3:
                        break
                    if sample['label'] not in seen_classes:
                        formatted_sample = {
                            'text': sample['text'],
                            'image': sample['image_caption'],
                            'label': sample['label']
                        }
                        final_samples.append(formatted_sample)
                        final_similarities.append(sim)
                        seen_classes.add(sample['label'])

                return final_samples, torch.tensor(final_similarities, device=self.device)

            return [], torch.tensor([], device=self.device)

