"""
PyTorch dataset for OCR digit recognition.

Handles loading synthetic digit images and preparing them for CRNN training.
"""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    """
    Dataset for loading synthetic digit images and labels.
    
    Args:
        csv_path: Path to labels CSV file
        img_dir: Directory containing images
        img_h: Target image height
        max_len: Maximum sequence length
        charset: Character set for encoding
        transform: Optional image transforms
    """
    
    def __init__(self, csv_path, img_dir, img_h=32, max_len=12, charset="0123456789", transform=None):
        self.img_dir = img_dir
        self.img_h = img_h
        self.max_len = max_len
        self.charset = charset
        self.transform = transform
        
        # Create character to index mapping
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
        self.char_to_idx['<blank>'] = 0  # CTC blank token
        
        # Load labels
        self.labels_df = pd.read_csv(csv_path)
        
        # Filter out labels that are too long
        self.labels_df = self.labels_df[
            self.labels_df['text'].str.len() <= max_len
        ].reset_index(drop=True)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get image info
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        text = row['text']
        
        # Load image
        img_path = self.img_dir / filename
        img = Image.open(img_path).convert('L')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = self._default_transform(img)
        
        # Encode text to indices
        targets = [self.char_to_idx[char] for char in text]
        target_len = len(targets)
        
        # Pad targets to max_len
        targets += [0] * (self.max_len - target_len)
        
        return {
            'image': img,
            'targets': torch.tensor(targets, dtype=torch.long),
            'target_len': target_len,
            'text': text
        }
    
    def _default_transform(self, img):
        """Default transform for images."""
        # Resize to target height while maintaining aspect ratio
        w, h = img.size
        new_w = int(w * self.img_h / h)
        img = img.resize((new_w, self.img_h), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        
        return transform(img)
    
    def get_char_to_idx(self):
        """Get character to index mapping."""
        return self.char_to_idx
    
    def get_idx_to_char(self):
        """Get index to character mapping."""
        return {idx: char for char, idx in self.char_to_idx.items()}


def get_train_transforms(img_h=32):
    """Get training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((img_h, img_h * 4)),  # Allow wider images
        transforms.RandomAffine(
            degrees=(-5, 5),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=(-2, 2)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_val_transforms(img_h=32):
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_h, img_h * 4)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def collate_fn(batch):
    """
    Custom collate function for variable width images.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    images = [sample['image'] for sample in batch]
    targets = [sample['targets'] for sample in batch]
    target_lens = [sample['target_len'] for sample in batch]
    texts = [sample['text'] for sample in batch]
    
    # Find max width in batch
    max_width = max(img.shape[2] for img in images)
    
    # Pad images to same width
    padded_images = []
    for img in images:
        if img.shape[2] < max_width:
            padding = torch.zeros(img.shape[0], img.shape[1], max_width - img.shape[2])
            img = torch.cat([img, padding], dim=2)
        padded_images.append(img)
    
    return {
        'images': torch.stack(padded_images),
        'targets': torch.stack(targets),
        'target_lens': torch.tensor(target_lens, dtype=torch.long),
        'texts': texts
    }
