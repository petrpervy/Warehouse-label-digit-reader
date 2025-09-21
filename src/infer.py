"""
Inference utility for SKU digit recognition.

Provides simple command-line interface for testing the trained model.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import yaml

from model_crnn import create_model
from decode import greedy_decode, beam_search_decode
from utils import get_device, set_seed


def load_model_for_inference(config_path: str, checkpoint_path: str):
    """Load trained model for inference."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device()
    
    # Create model
    model = create_model(
        charset=config['charset'],
        img_h=config['img_h'],
        cnn_out=config['model']['cnn_out'],
        rnn_hidden=config['model']['rnn_hidden'],
        rnn_layers=config['model']['rnn_layers']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, device


def preprocess_image(image_path: str, img_h: int = 32):
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('L')
    
    # Resize to target height while maintaining aspect ratio
    w, h = image.size
    new_w = int(w * img_h / h)
    image = image.resize((new_w, img_h), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    import numpy as np
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def predict_image(image_path: str, model, config, device, decode_method='greedy'):
    """Predict digits from image."""
    # Preprocess image
    img_tensor = preprocess_image(image_path, config['img_h']).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        
        if decode_method == 'greedy':
            predicted_text, confidence = greedy_decode(logits[0], config['charset'])
        else:
            predicted_text, confidence = beam_search_decode(logits[0], config['charset'])
    
    return predicted_text, confidence


def main():
    parser = argparse.ArgumentParser(description='Inference for SKU digit recognition')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--config', type=str, default='configs/crnn.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, default='artifacts/checkpoints/best.ckpt', help='Model checkpoint path')
    parser.add_argument('--decode', type=str, choices=['greedy', 'beam_search'], default='greedy', help='Decode method')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Please train the model first: python -m src.train_crnn --config configs/crnn.yaml")
        return
    
    try:
        # Load model
        print("Loading model...")
        model, config, device = load_model_for_inference(args.config, args.checkpoint)
        print(f"Model loaded on {device}")
        
        # Predict
        print(f"Predicting digits from: {args.image}")
        predicted_text, confidence = predict_image(args.image, model, config, device, args.decode)
        
        print(f"\nResults:")
        print(f"Predicted text: {predicted_text}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Decode method: {args.decode}")
        
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == '__main__':
    main()
