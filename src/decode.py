"""
CTC decoding utilities for sequence recognition.

Implements greedy decoding and beam search for converting CTC output
logits to text sequences.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def greedy_decode(logits, charset="0123456789", blank_idx=0):
    """
    Greedy CTC decoding.
    
    Args:
        logits: Model output logits (seq_len, num_classes) or (batch_size, seq_len, num_classes)
        charset: Character set
        blank_idx: Index of blank token (usually 0)
        
    Returns:
        decoded_text: Decoded text string
        confidence: Average probability of predicted sequence
    """
    # Handle batch dimension
    if logits.dim() == 3:
        logits = logits[0]  # Take first item in batch
    
    # Get predicted indices
    predicted = torch.argmax(logits, dim=-1)
    
    # Remove blanks and duplicates
    result = []
    prev_idx = -1
    
    for idx in predicted.cpu().numpy():
        if idx != prev_idx and idx != blank_idx:
            if idx <= len(charset):
                result.append(charset[idx - 1])
        prev_idx = idx
    
    # Calculate confidence
    probs = F.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    confidence = torch.mean(max_probs).item()
    
    return ''.join(result), confidence


def beam_search_decode(logits, charset="0123456789", beam_width=10, blank_idx=0):
    """
    Beam search CTC decoding.
    
    Args:
        logits: Model output logits (seq_len, num_classes) or (batch_size, seq_len, num_classes)
        charset: Character set
        beam_width: Beam search width
        blank_idx: Index of blank token
        
    Returns:
        decoded_text: Best decoded text string
        confidence: Probability of best sequence
    """
    # Handle batch dimension
    if logits.dim() == 3:
        logits = logits[0]  # Take first item in batch
    
    seq_len, num_classes = logits.shape
    probs = F.softmax(logits, dim=-1)
    
    # Initialize beam with empty sequence
    beam = [(0.0, [], blank_idx)]  # (score, sequence, last_char)
    
    for t in range(seq_len):
        new_beam = defaultdict(lambda: (float('-inf'), [], -1))
        
        for score, sequence, last_char in beam:
            for c in range(num_classes):
                char_prob = probs[t, c].item()
                new_score = score + np.log(char_prob + 1e-10)
                
                if c == blank_idx:
                    # Blank token - extend existing sequence
                    key = tuple(sequence)
                    if new_score > new_beam[key][0]:
                        new_beam[key] = (new_score, sequence, blank_idx)
                else:
                    # Non-blank token
                    if c == last_char:
                        # Same as last character - extend existing sequence
                        key = tuple(sequence)
                        if new_score > new_beam[key][0]:
                            new_beam[key] = (new_score, sequence, c)
                    else:
                        # Different character - add to sequence
                        new_sequence = sequence + [c]
                        key = tuple(new_sequence)
                        if new_score > new_beam[key][0]:
                            new_beam[key] = (new_score, new_sequence, c)
        
        # Keep top beam_width candidates
        beam = sorted(new_beam.values(), reverse=True)[:beam_width]
    
    # Get best sequence
    if not beam:
        return "", 0.0
    
    best_score, best_sequence, _ = beam[0]
    
    # Convert indices to text
    decoded_text = ''.join(charset[idx - 1] for idx in best_sequence if idx <= len(charset))
    confidence = np.exp(best_score / len(best_sequence)) if best_sequence else 0.0
    
    return decoded_text, confidence


def decode_ctc(logits, charset="0123456789", method='greedy', beam_width=10):
    """
    Main CTC decoding function.
    
    Args:
        logits: Model output logits
        charset: Character set
        method: 'greedy' or 'beam_search'
        beam_width: Beam width for beam search
        
    Returns:
        decoded_text: Decoded text
        confidence: Prediction confidence
    """
    if method == 'greedy':
        return greedy_decode(logits, charset)
    elif method == 'beam_search':
        return beam_search_decode(logits, charset, beam_width)
    else:
        raise ValueError(f"Unknown decode method: {method}")


def batch_decode(logits_batch, charset="0123456789", method='greedy', beam_width=10):
    """
    Decode a batch of logits.
    
    Args:
        logits_batch: Batch of logits (batch_size, seq_len, num_classes)
        charset: Character set
        method: Decode method
        beam_width: Beam width for beam search
        
    Returns:
        List of (text, confidence) tuples
    """
    results = []
    for i in range(logits_batch.shape[0]):
        text, conf = decode_ctc(logits_batch[i], charset, method, beam_width)
        results.append((text, conf))
    
    return results


def calculate_edit_distance(predicted, target):
    """
    Calculate Levenshtein edit distance between predicted and target strings.
    
    Args:
        predicted: Predicted text string
        target: Target text string
        
    Returns:
        edit_distance: Levenshtein distance
        accuracy: 1 - (edit_distance / max_length)
    """
    try:
        import editdistance
        distance = editdistance.eval(predicted, target)
    except ImportError:
        # Fallback implementation
        distance = _levenshtein_distance(predicted, target)
    
    max_len = max(len(predicted), len(target))
    accuracy = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    return distance, accuracy


def _levenshtein_distance(s1, s2):
    """Fallback Levenshtein distance implementation."""
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


if __name__ == '__main__':
    # Test decoding functions
    seq_len, num_classes = 20, 11  # 10 digits + blank
    logits = torch.randn(seq_len, num_classes)
    
    # Test greedy decoding
    text_greedy, conf_greedy = greedy_decode(logits)
    print(f"Greedy decode: '{text_greedy}' (confidence: {conf_greedy:.3f})")
    
    # Test beam search
    text_beam, conf_beam = beam_search_decode(logits, beam_width=5)
    print(f"Beam search: '{text_beam}' (confidence: {conf_beam:.3f})")
    
    # Test edit distance
    dist, acc = calculate_edit_distance("123", "123")
    print(f"Edit distance test: {dist}, accuracy: {acc:.3f}")
