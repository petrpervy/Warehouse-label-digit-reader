"""
Evaluation script for CRNN model.

Computes accuracy metrics and robustness tests with synthetic noise/blur.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import yaml

from dataset import OCRDataset, get_val_transforms, collate_fn
from model_crnn import create_model
from decode import greedy_decode, beam_search_decode, calculate_edit_distance


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    model = create_model(
        charset=config['charset'],
        img_h=config['img_h'],
        cnn_out=config['model']['cnn_out'],
        rnn_hidden=config['model']['rnn_hidden'],
        rnn_layers=config['model']['rnn_layers']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def apply_noise(img, noise_level):
    """Apply Gaussian noise to image."""
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, noise_level, img_array.shape)
    img_array = img_array + noise
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))


def apply_blur(img, blur_level):
    """Apply Gaussian blur to image."""
    if blur_level > 0:
        return img.filter(ImageFilter.GaussianBlur(radius=blur_level))
    return img


def apply_brightness_contrast(img, brightness, contrast):
    """Apply brightness and contrast adjustments."""
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array * contrast + brightness
    img_array = np.clip(img_array, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))


def evaluate_model(model, dataloader, device, charset, decode_method='greedy'):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        charset: Character set
        decode_method: 'greedy' or 'beam_search'
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_samples = 0
    exact_matches = 0
    total_edit_distance = 0.0
    predictions = []
    targets = []
    
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            texts = batch['texts']
            
            # Measure inference time
            start_time = time.time()
            logits = model(images)
            inference_time = time.time() - start_time
            
            for i in range(images.size(0)):
                # Decode prediction
                if decode_method == 'greedy':
                    predicted_text, confidence = greedy_decode(logits[i], charset)
                else:
                    predicted_text, confidence = beam_search_decode(logits[i], charset)
                
                target_text = texts[i]
                
                # Calculate metrics
                if predicted_text == target_text:
                    exact_matches += 1
                
                edit_dist, _ = calculate_edit_distance(predicted_text, target_text)
                total_edit_distance += edit_dist
                
                predictions.append(predicted_text)
                targets.append(target_text)
                total_samples += 1
            
            inference_times.append(inference_time)
    
    # Calculate final metrics
    exact_accuracy = exact_matches / total_samples
    avg_edit_distance = total_edit_distance / total_samples
    avg_inference_time = np.mean(inference_times)
    
    return {
        'exact_accuracy': exact_accuracy,
        'avg_edit_distance': avg_edit_distance,
        'total_samples': total_samples,
        'avg_inference_time': avg_inference_time,
        'predictions': predictions,
        'targets': targets
    }


def robustness_test(model, dataloader, device, charset, report_dir):
    """
    Test model robustness to noise, blur, and brightness changes.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        charset: Character set
        report_dir: Directory to save plots
    """
    print("Running robustness tests...")
    
    # Test parameters
    noise_levels = [0, 5, 10, 15, 20, 25, 30]
    blur_levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    brightness_levels = [0, -20, -10, 10, 20, 30]
    contrast_levels = [1.0, 0.8, 0.9, 1.1, 1.2, 1.3]
    
    # Get a sample batch for testing
    sample_batch = next(iter(dataloader))
    sample_images = sample_batch['images'][:10]  # Test on 10 samples
    sample_texts = sample_batch['texts'][:10]
    
    # Test noise robustness
    noise_results = []
    for noise_level in noise_levels:
        accuracies = []
        for i, img in enumerate(sample_images):
            # Convert tensor back to PIL Image
            img_array = img.squeeze().numpy()
            img_array = ((img_array + 1) * 127.5).astype(np.uint8)  # Denormalize
            pil_img = Image.fromarray(img_array)
            
            # Apply noise
            noisy_img = apply_noise(pil_img, noise_level)
            
            # Convert back to tensor
            noisy_array = np.array(noisy_img) / 127.5 - 1  # Normalize
            noisy_tensor = torch.from_numpy(noisy_array).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(noisy_tensor)
                predicted_text, _ = greedy_decode(logits[0], charset)
                
                if predicted_text == sample_texts[i]:
                    accuracies.append(1)
                else:
                    accuracies.append(0)
        
        avg_accuracy = np.mean(accuracies)
        noise_results.append(avg_accuracy)
        print(f"Noise level {noise_level}: Accuracy {avg_accuracy:.3f}")
    
    # Test blur robustness
    blur_results = []
    for blur_level in blur_levels:
        accuracies = []
        for i, img in enumerate(sample_images):
            # Convert tensor back to PIL Image
            img_array = img.squeeze().numpy()
            img_array = ((img_array + 1) * 127.5).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            
            # Apply blur
            blurred_img = apply_blur(pil_img, blur_level)
            
            # Convert back to tensor
            blurred_array = np.array(blurred_img) / 127.5 - 1
            blurred_tensor = torch.from_numpy(blurred_array).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(blurred_tensor)
                predicted_text, _ = greedy_decode(logits[0], charset)
                
                if predicted_text == sample_texts[i]:
                    accuracies.append(1)
                else:
                    accuracies.append(0)
        
        avg_accuracy = np.mean(accuracies)
        blur_results.append(avg_accuracy)
        print(f"Blur level {blur_level}: Accuracy {avg_accuracy:.3f}")
    
    # Create robustness plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Noise plot
    axes[0].plot(noise_levels, noise_results, 'b-o')
    axes[0].set_xlabel('Noise Level (σ)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Robustness to Noise')
    axes[0].grid(True)
    axes[0].set_ylim([0, 1])
    
    # Blur plot
    axes[1].plot(blur_levels, blur_results, 'r-o')
    axes[1].set_xlabel('Blur Level (radius)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Robustness to Blur')
    axes[1].grid(True)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(report_dir / 'robustness_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save robustness results
    robustness_data = {
        'noise_levels': noise_levels,
        'noise_results': noise_results,
        'blur_levels': blur_levels,
        'blur_results': blur_results
    }
    
    with open(report_dir / 'robustness_results.json', 'w') as f:
        json.dump(robustness_data, f, indent=2)
    
    print(f"Robustness plots saved to {report_dir}")


def plot_training_metrics(metrics_path, report_dir):
    """Plot training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss plot
    axes[0, 0].plot(epochs, metrics['train_losses'], label='Train', color='blue')
    axes[0, 0].plot(epochs, metrics['val_losses'], label='Validation', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, metrics['val_accuracies'], color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1])
    
    # Edit distance plot
    axes[1, 0].plot(epochs, metrics['val_edit_distances'], color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Edit Distance')
    axes[1, 0].set_title('Validation Edit Distance')
    axes[1, 0].grid(True)
    
    # Summary stats
    axes[1, 1].text(0.1, 0.8, f"Best Validation Accuracy: {metrics['best_val_accuracy']:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Final Validation Accuracy: {metrics['val_accuracies'][-1]:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Final Edit Distance: {metrics['val_edit_distances'][-1]:.3f}", 
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(report_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CRNN model')
    parser.add_argument('--config', type=str, default='configs/crnn.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, default='artifacts/checkpoints/best.ckpt', help='Model checkpoint path')
    parser.add_argument('--decode', type=str, choices=['greedy', 'beam_search'], default='greedy', help='Decode method')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create report directory
    report_dir = Path(config['paths']['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Load dataset
    synth_dir = Path(config['paths']['synth_dir'])
    labels_path = synth_dir / 'labels.csv'
    
    dataset = OCRDataset(
        csv_path=labels_path,
        img_dir=synth_dir,
        img_h=config['img_h'],
        max_len=config['max_len'],
        charset=config['charset'],
        transform=get_val_transforms(config['img_h'])
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate model
    results = evaluate_model(model, dataloader, device, config['charset'], args.decode)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Exact Match Accuracy: {results['exact_accuracy']:.3f}")
    print(f"Average Edit Distance: {results['avg_edit_distance']:.3f}")
    print(f"Average Inference Time: {results['avg_inference_time']*1000:.2f} ms")
    print(f"Total Samples: {results['total_samples']}")
    
    # Run robustness tests
    robustness_test(model, dataloader, device, config['charset'], report_dir)
    
    # Plot training metrics if available
    metrics_path = report_dir / 'training_metrics.json'
    if metrics_path.exists():
        plot_training_metrics(metrics_path, report_dir)
    
    # Save evaluation results
    eval_results = {
        'exact_accuracy': results['exact_accuracy'],
        'avg_edit_distance': results['avg_edit_distance'],
        'avg_inference_time': results['avg_inference_time'],
        'total_samples': results['total_samples'],
        'decode_method': args.decode,
        'checkpoint': args.checkpoint
    }
    
    with open(report_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {report_dir}")


if __name__ == '__main__':
    main()
