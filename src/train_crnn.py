"""
Training script for CRNN model.

Implements training loop with CTC loss, validation, and model checkpointing.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

from dataset import OCRDataset, get_train_transforms, get_val_transforms, collate_fn
from model_crnn import create_model
from decode import greedy_decode, calculate_edit_distance


class CTCLoss(nn.Module):
    """CTC Loss wrapper with proper input formatting."""
    
    def __init__(self, blank=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
    
    def forward(self, logits, targets, target_lens, input_lens):
        """
        Compute CTC loss.
        
        Args:
            logits: Model predictions (batch_size, seq_len, num_classes)
            targets: Target sequences (batch_size, max_len)
            target_lens: Length of each target sequence
            input_lens: Length of each input sequence
        """
        # CTC expects (seq_len, batch_size, num_classes)
        logits = logits.transpose(0, 1)
        
        # Flatten targets and remove padding
        targets_flat = []
        for i, length in enumerate(target_lens):
            targets_flat.extend(targets[i][:length].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long)
        
        return self.ctc_loss(logits, targets_flat, input_lens, target_lens)


def train_epoch(model, dataloader, criterion, optimizer, device, charset):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        target_lens = batch['target_lens'].to(device)
        texts = batch['texts']
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        
        # Calculate input lengths (assuming variable width)
        input_lens = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
        
        # Compute loss
        loss = criterion(logits, targets, target_lens, input_lens)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            for i in range(images.size(0)):
                predicted_text, _ = greedy_decode(logits[i], charset)
                if predicted_text == texts[i]:
                    correct_predictions += 1
                total_predictions += 1
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct_predictions/total_predictions:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, charset):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_edit_distance = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lens = batch['target_lens'].to(device)
            texts = batch['texts']
            
            # Forward pass
            logits = model(images)
            input_lens = torch.full((images.size(0),), logits.size(1), dtype=torch.long, device=device)
            
            # Compute loss
            loss = criterion(logits, targets, target_lens, input_lens)
            total_loss += loss.item()
            
            # Calculate metrics
            for i in range(images.size(0)):
                predicted_text, _ = greedy_decode(logits[i], charset)
                target_text = texts[i]
                
                if predicted_text == target_text:
                    correct_predictions += 1
                
                # Calculate edit distance
                edit_dist, _ = calculate_edit_distance(predicted_text, target_text)
                total_edit_distance += edit_dist
                
                total_predictions += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    avg_edit_distance = total_edit_distance / total_predictions
    
    return avg_loss, accuracy, avg_edit_distance


def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def main():
    parser = argparse.ArgumentParser(description='Train CRNN model')
    parser.add_argument('--config', type=str, default='configs/crnn.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    ckpt_dir = Path(config['paths']['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    synth_dir = Path(config['paths']['synth_dir'])
    labels_path = synth_dir / 'labels.csv'
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        print("Please run data generation first:")
        print("python -m src.synth_digits --count 20000 --len 3-8 --out data/synth")
        return
    
    # Load all data
    full_dataset = OCRDataset(
        csv_path=labels_path,
        img_dir=synth_dir,
        img_h=config['img_h'],
        max_len=config['max_len'],
        charset=config['charset']
    )
    
    # Split into train/val
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=config['train']['val_split'],
        random_state=config['seed']
    )
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create transforms
    train_transforms = get_train_transforms(config['img_h'])
    val_transforms = get_val_transforms(config['img_h'])
    
    # Apply transforms to subsets
    for idx in train_indices:
        full_dataset.transform = train_transforms
    for idx in val_indices:
        full_dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(
        charset=config['charset'],
        img_h=config['img_h'],
        cnn_out=config['model']['cnn_out'],
        rnn_hidden=config['model']['rnn_hidden'],
        rnn_layers=config['model']['rnn_layers']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss and optimizer
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        start_epoch, _, _ = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_edit_distances = []
    
    print("Starting training...")
    
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config['charset'])
        
        # Validate
        val_loss, val_acc, val_edit_dist = validate_epoch(model, val_loader, criterion, device, config['charset'])
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_edit_distances.append(val_edit_dist)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, Val Edit Dist: {val_edit_dist:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = ckpt_dir / 'best.ckpt'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)
            print(f"New best model saved: {save_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = ckpt_dir / f'epoch_{epoch+1}.ckpt'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)
    
    # Save final model and metrics
    save_path = ckpt_dir / 'final.ckpt'
    save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path)
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_edit_distances': val_edit_distances,
        'best_val_accuracy': best_val_acc,
        'config': config
    }
    
    metrics_path = Path(config['paths']['report_dir']) / 'training_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save label encoder
    char_to_idx = full_dataset.get_char_to_idx()
    encoder_path = ckpt_dir / 'label_encoder.json'
    with open(encoder_path, 'w') as f:
        json.dump(char_to_idx, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
