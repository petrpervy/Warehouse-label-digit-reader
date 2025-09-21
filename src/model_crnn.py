"""
CRNN (CNN + RNN) model for sequence recognition.

Implements a CNN feature extractor followed by bidirectional LSTM
and CTC loss for variable-length sequence recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for spatial feature extraction."""
    
    def __init__(self, cnn_out=256):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4 -> 2x2
        
        # Global average pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Keep width dimension
        
        # Project to desired output size
        self.projection = nn.Linear(512, cnn_out)
        
    def forward(self, x):
        # Input: (batch_size, 1, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global average pooling across height
        x = self.adaptive_pool(x)  # (batch_size, 512, 1, width)
        
        # Reshape for RNN: (batch_size, width, 512)
        x = x.squeeze(2).transpose(1, 2)
        
        # Project to desired feature size
        x = self.projection(x)  # (batch_size, width, cnn_out)
        
        return x


class CRNN(nn.Module):
    """
    CRNN model for sequence recognition.
    
    Architecture:
    1. CNN feature extractor
    2. Bidirectional LSTM
    3. Linear layer for CTC output
    """
    
    def __init__(self, charset="0123456789", img_h=32, cnn_out=256, 
                 rnn_hidden=256, rnn_layers=2, dropout=0.1):
        super().__init__()
        
        self.charset = charset
        self.num_classes = len(charset) + 1  # +1 for CTC blank
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(cnn_out)
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=cnn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Output projection
        self.classifier = nn.Linear(rnn_hidden * 2, self.num_classes)  # *2 for bidirectional
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'rnn' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, 1, height, width)
            
        Returns:
            logits: Output logits (batch_size, seq_len, num_classes)
        """
        # Extract CNN features
        cnn_features = self.cnn(x)  # (batch_size, seq_len, cnn_out)
        
        # Apply RNN
        rnn_output, _ = self.rnn(cnn_features)  # (batch_size, seq_len, rnn_hidden*2)
        
        # Apply classifier
        logits = self.classifier(rnn_output)  # (batch_size, seq_len, num_classes)
        
        return logits
    
    def predict(self, x, decode_fn=None):
        """
        Predict text from input image.
        
        Args:
            x: Input image tensor
            decode_fn: Optional custom decode function
            
        Returns:
            predicted_text: Decoded text string
            confidence: Prediction confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if decode_fn:
                return decode_fn(logits)
            else:
                # Simple greedy decoding
                predicted = torch.argmax(logits, dim=-1)
                predicted_text = self._decode_greedy(predicted)
                confidence = self._get_confidence(logits)
                
                return predicted_text, confidence
    
    def _decode_greedy(self, predicted):
        """Simple greedy decoding."""
        # Convert indices to characters
        result = []
        prev_idx = -1
        
        for idx in predicted[0].cpu().numpy():  # Take first batch item
            if idx != prev_idx and idx != 0:  # Skip blanks and duplicates
                if idx <= len(self.charset):
                    result.append(self.charset[idx - 1])
            prev_idx = idx
        
        return ''.join(result)
    
    def _get_confidence(self, logits):
        """Calculate prediction confidence."""
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        confidence = torch.mean(max_probs).item()
        return confidence


def create_model(charset="0123456789", img_h=32, cnn_out=256, 
                rnn_hidden=256, rnn_layers=2):
    """
    Create CRNN model with specified parameters.
    
    Args:
        charset: Character set for recognition
        img_h: Input image height
        cnn_out: CNN output feature size
        rnn_hidden: RNN hidden size
        rnn_layers: Number of RNN layers
        
    Returns:
        CRNN model instance
    """
    model = CRNN(
        charset=charset,
        img_h=img_h,
        cnn_out=cnn_out,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers
    )
    
    return model


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    model = create_model()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 1, 32, 128)  # Batch of 2 images
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
