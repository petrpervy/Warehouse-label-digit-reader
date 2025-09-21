"""
Synthetic digit image generator for SKU recognition training.

Generates realistic grayscale images of multi-digit strings with various
transformations to simulate warehouse label conditions.
"""

import argparse
import csv
import os
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def get_random_font():
    """Get a random system font suitable for digit rendering."""
    # Common fonts that work well for digits
    font_paths = [
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc", 
        "/System/Library/Fonts/Times.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Windows/Fonts/arial.ttf",
        "/Windows/Fonts/calibri.ttf"
    ]
    
    available_fonts = [f for f in font_paths if os.path.exists(f)]
    if available_fonts:
        return available_fonts[random.randint(0, len(available_fonts) - 1)]
    
    # Fallback to default font
    try:
        return ImageFont.load_default()
    except:
        return None


def create_digit_image(text, width=None, height=32):
    """
    Create a grayscale image of digit text.
    
    Args:
        text: String of digits to render
        width: Target width (None for auto)
        height: Target height
        
    Returns:
        PIL Image in grayscale
    """
    # Get random font size
    font_size = random.randint(18, 28)
    
    try:
        font = ImageFont.truetype(get_random_font(), font_size)
    except:
        font = ImageFont.load_default()
    
    # Create initial image
    img = Image.new('L', (width or len(text) * 20, height), 255)
    draw = ImageDraw.Draw(img)
    
    # Draw text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2
    
    draw.text((x, y), text, font=font, fill=0)
    
    # Crop to actual text bounds with some padding
    padding = 4
    crop_box = (
        max(0, x - padding),
        max(0, y - padding), 
        min(img.width, x + text_width + padding),
        min(img.height, y + text_height + padding)
    )
    
    img = img.crop(crop_box)
    
    # Resize to target height while maintaining aspect ratio
    if width is None:
        aspect_ratio = img.width / img.height
        new_width = int(height * aspect_ratio)
        img = img.resize((new_width, height), Image.Resampling.LANCZOS)
    
    return img


def apply_perspective_transform(img, max_angle=15):
    """Apply random perspective transformation."""
    if random.random() > 0.3:
        return img
        
    h, w = img.height, img.width
    
    # Define source points (corners of image)
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Add random perspective distortion
    angle = random.uniform(-max_angle, max_angle) * np.pi / 180
    offset = random.uniform(-w * 0.1, w * 0.1)
    
    dst_points = np.float32([
        [offset, 0],
        [w - offset, 0],
        [w, h],
        [0, h]
    ])
    
    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_array = np.array(img)
    transformed = cv2.warpPerspective(img_array, matrix, (w, h))
    
    return Image.fromarray(transformed)


def apply_affine_transform(img):
    """Apply random affine transformation (rotation, scaling)."""
    if random.random() > 0.4:
        return img
        
    h, w = img.height, img.width
    
    # Random rotation
    angle = random.uniform(-10, 10)
    
    # Random scaling
    scale_x = random.uniform(0.9, 1.1)
    scale_y = random.uniform(0.9, 1.1)
    
    # Apply transformation
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 0] *= scale_x
    matrix[1, 1] *= scale_y
    
    img_array = np.array(img)
    transformed = cv2.warpAffine(img_array, matrix, (w, h))
    
    return Image.fromarray(transformed)


def apply_noise_and_blur(img):
    """Apply random noise and blur effects."""
    img_array = np.array(img, dtype=np.float32)
    
    # Add noise
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        img_array = img_array + noise
    
    # Apply blur
    if random.random() > 0.6:
        blur_type = random.choice(['gaussian', 'motion'])
        if blur_type == 'gaussian':
            sigma = random.uniform(0.5, 1.5)
            img_array = cv2.GaussianBlur(img_array, (0, 0), sigma)
        else:  # motion blur
            kernel_size = random.randint(3, 7)
            angle = random.uniform(0, 180)
            kernel = cv2.getMotionKernel((kernel_size, kernel_size), angle)
            img_array = cv2.filter2D(img_array, -1, kernel)
    
    # Adjust brightness/contrast
    if random.random() > 0.4:
        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.8, 1.2)
        img_array = img_array * contrast + brightness
    
    # Clip values
    img_array = np.clip(img_array, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))


def add_background_box(img):
    """Add random background box to simulate label borders."""
    if random.random() > 0.7:
        return img
        
    # Create background
    bg = Image.new('L', (img.width + 20, img.height + 20), 255)
    
    # Paste image with some offset
    offset_x = random.randint(0, 10)
    offset_y = random.randint(0, 10)
    bg.paste(img, (offset_x, offset_y))
    
    # Draw border
    draw = ImageDraw.Draw(bg)
    border_color = random.randint(200, 240)
    draw.rectangle([0, 0, bg.width-1, bg.height-1], outline=border_color, width=1)
    
    return bg


def generate_digit_string(min_len, max_len):
    """Generate random digit string of specified length."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.digits, k=length))


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic digit images')
    parser.add_argument('--count', type=int, default=10000, help='Number of images to generate')
    parser.add_argument('--len', type=str, default='3-8', help='Length range (e.g., "3-8")')
    parser.add_argument('--out', type=str, default='data/synth', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse length range
    min_len, max_len = map(int, args.len.split('-'))
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images and labels
    labels = []
    
    print(f"Generating {args.count} synthetic digit images...")
    
    for i in range(args.count):
        if i % 1000 == 0:
            print(f"Generated {i}/{args.count} images")
            
        # Generate random digit string
        text = generate_digit_string(min_len, max_len)
        
        # Create base image
        img = create_digit_image(text)
        
        # Apply transformations
        if random.random() > 0.3:
            img = apply_perspective_transform(img)
        if random.random() > 0.4:
            img = apply_affine_transform(img)
        if random.random() > 0.2:
            img = apply_noise_and_blur(img)
        if random.random() > 0.5:
            img = add_background_box(img)
        
        # Ensure minimum size
        if img.width < 20 or img.height < 20:
            img = img.resize((max(20, img.width), max(20, img.height)), Image.Resampling.LANCZOS)
        
        # Save image
        filename = f"img_{i:06d}.png"
        img_path = output_dir / filename
        img.save(img_path)
        
        # Store label
        labels.append({
            'filename': filename,
            'text': text
        })
    
    # Save labels CSV
    labels_path = output_dir / 'labels.csv'
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'text'])
        writer.writeheader()
        writer.writerows(labels)
    
    print(f"Generated {args.count} images in {output_dir}")
    print(f"Labels saved to {labels_path}")


if __name__ == '__main__':
    main()
