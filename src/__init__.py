"""
SKU Digit Reader - OCR System for Warehouse Labels

This package provides a complete OCR solution for reading multi-digit SKUs
from warehouse labels using CRNN-CTC architecture.

Modules:
- synth_digits: Synthetic data generation
- dataset: PyTorch dataset and data loading
- model_crnn: CRNN model architecture
- train_crnn: Training script
- eval_crnn: Evaluation and robustness testing
- decode: CTC decoding utilities
- utils: Common utility functions
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main classes for easy access
from .model_crnn import CRNN, create_model
from .dataset import OCRDataset
from .decode import greedy_decode, beam_search_decode
from .utils import set_seed, get_device, count_parameters

__all__ = [
    "CRNN",
    "create_model", 
    "OCRDataset",
    "greedy_decode",
    "beam_search_decode",
    "set_seed",
    "get_device",
    "count_parameters"
]
