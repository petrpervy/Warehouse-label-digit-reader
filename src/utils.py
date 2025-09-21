"""
Utility functions for the OCR project.

Contains helper functions for image processing, model utilities, and common operations.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(file_path: str) -> Dict:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_directories(paths: List[str]):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def resize_image_with_aspect_ratio(image: Image.Image, target_height: int) -> Image.Image:
    """Resize image to target height while maintaining aspect ratio."""
    w, h = image.size
    new_width = int(w * target_height / h)
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def normalize_image(image: np.ndarray, mean: float = 127.5, std: float = 127.5) -> np.ndarray:
    """Normalize image to [-1, 1] range."""
    return (image - mean) / std


def denormalize_image(image: np.ndarray, mean: float = 127.5, std: float = 127.5) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255] range."""
    return image * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Handle different tensor shapes
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]
    if tensor.dim() == 3:  # Channel dimension
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and denormalize
    img_array = tensor.cpu().numpy()
    img_array = denormalize_image(img_array)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    img_array = np.array(image, dtype=np.float32)
    
    if normalize:
        img_array = normalize_image(img_array)
    
    # Add channel dimension if grayscale
    if len(img_array.shape) == 2:
        img_array = img_array[None, ...]
    
    return torch.from_numpy(img_array)


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name}: {format_time(duration)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact match accuracy."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return correct / len(predictions)


def calculate_edit_distance_accuracy(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """Calculate edit distance accuracy (lower is better)."""
    try:
        import editdistance
        distances = [editdistance.eval(pred, target) for pred, target in zip(predictions, targets)]
        return np.mean(distances), np.std(distances)
    except ImportError:
        # Fallback implementation
        distances = [_levenshtein_distance(pred, target) for pred, target in zip(predictions, targets)]
        return np.mean(distances), np.std(distances)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def log_model_info(model: torch.nn.Module, input_shape: Tuple[int, ...]):
    """Log comprehensive model information."""
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model Size: {get_model_size_mb(model):.2f} MB")
    print(f"Input Shape: {input_shape}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    with torch.no_grad():
        with Timer("Forward Pass"):
            output = model(dummy_input)
    
    print(f"Output Shape: {output.shape}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test timer
    with Timer("Test Operation"):
        time.sleep(0.1)
    
    # Test edit distance
    dist = _levenshtein_distance("hello", "world")
    print(f"Edit distance 'hello' -> 'world': {dist}")
    
    # Test accuracy calculation
    preds = ["123", "456", "789"]
    targets = ["123", "456", "999"]
    acc = calculate_accuracy(preds, targets)
    print(f"Accuracy: {acc:.3f}")
    
    print("Utility functions test complete!")
