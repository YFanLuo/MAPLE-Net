import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, device='cpu', file_type='tsv', normalize_images=True):
        self.data = []
        self.label_set = self._get_label_set(csv_file)
        if file_type == 'tsv':
            with open(csv_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if 'tumemo' in csv_file.lower():
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        index, label, image_id, text = parts[:4]
                        self.data.append({
                            'index': index,
                            'label': self._convert_label(label),
                            'image_id': image_id,
                            'text': text,
                            'target': ''
                        })
            elif 'mvsa-s' in csv_file or 'mvsa-m' in csv_file:
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        index, label, image_id, text = parts[:4]
                        self.data.append({
                            'index': index,
                            'label': self._convert_label(label),
                            'image_id': image_id,
                            'text': text,
                            'target': ''
                        })
            else:
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        index, label, image_id, text, target = parts[:5]
                        self.data.append({
                            'index': index,
                            'label': self._convert_label(label),
                            'image_id': image_id,
                            'text': text,
                            'target': target
                        })

        elif file_type == 'txt':
            self.data = self._read_txt_file(csv_file)
        else:
            raise ValueError("Unsupported file type. Use 'tsv' or 'txt'.")
        
        self.img_dir = img_dir
        self.transform = transform
        self.device = device
        self.normalize_images = normalize_images
        
        print(f"Loaded {len(self.data)} samples from {csv_file}")

    def _get_label_set(self, csv_file):
        if 'masad' in csv_file.lower():
            return ['Negative', 'Positive']
        elif 'tumemo' in csv_file.lower():
            return ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
        else:
            return ['Negative', 'Neutral', 'Positive']

    def _read_txt_file(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                text = lines[i].strip()
                target = lines[i+1].strip()
                label = self._convert_label(lines[i+2].strip())
                image_id = lines[i+3].strip()
                data.append({
                    'index': len(data),
                    'label': label,
                    'image_id': image_id,
                    'text': text,
                    'target': target
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        
        # 处理图像
        img_name = str(sample['image_id'])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError, Image.UnidentifiedImageError):
            print(f"Error loading image: {img_path}. Using a blank image instead.")
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)

        # 检查图像是否包含 NaN 或 Inf 值
        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"Warning: Image {img_name} contains NaN or Inf values after transformation")
            print(f"Image stats - Min: {image.min().item():.4f}, Max: {image.max().item():.4f}, Mean: {image.mean().item():.4f}")
            # 替换 NaN 和 Inf 值为 0
            image = torch.where(torch.isnan(image) | torch.isinf(image), torch.zeros_like(image), image)

        # 确保图像值在合理范围内
        image = torch.clamp(image, min=-1.0, max=1.0)

        if self.normalize_images:
            image = (image - image.mean()) / (image.std() + 1e-6)  # 添加小的 epsilon 值以避免除以零

        return {
            'image': image.to(self.device),
            'text': sample['text'],
            'label': torch.tensor(sample['label'], dtype=torch.long).to(self.device),
            'target': sample['target']
        }
    def _convert_label(self, label):
        if isinstance(label, (int, np.integer)):
            return int(label)
        elif isinstance(label, str):
            label = label.lower()
            tumemo_labels = ['angry', 'bored', 'calm', 'fear', 'happy', 'love', 'sad']
            if label in tumemo_labels:
                return tumemo_labels.index(label)
            elif label in ['negative', '-1', '0']:
                return 0
            elif label in ['neutral', '0', '1'] and 'Neutral' in self.label_set:
                return 1
            elif label in ['positive', '1', '2']:
                return len(self.label_set) - 1
        raise ValueError(f"Unexpected label type: {type(label)}, value: {label}")

    def get_labels(self):
        return list(set([sample['label'] for sample in self.data]))
