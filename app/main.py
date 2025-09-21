"""
FastAPI inference server for SKU digit recognition.

Provides REST API endpoint for uploading images and getting digit predictions.
"""

import io
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_crnn import create_model
from decode import greedy_decode, beam_search_decode


# Global variables for model and config
model = None
config = None
device = None


def load_model():
    """Load the trained model and configuration."""
    global model, config, device
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs/crnn.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        charset=config['charset'],
        img_h=config['img_h'],
        cnn_out=config['model']['cnn_out'],
        rnn_hidden=config['model']['rnn_hidden'],
        rnn_layers=config['model']['rnn_layers']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = Path(__file__).parent.parent / 'artifacts/checkpoints/best.ckpt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL image for model inference.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to target height while maintaining aspect ratio
    w, h = image.size
    target_h = config['img_h']
    new_w = int(w * target_h / h)
    image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
    
    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor.to(device)


def predict_digits(image_tensor: torch.Tensor, decode_method: str = 'greedy') -> tuple:
    """
    Predict digits from preprocessed image tensor.
    
    Args:
        image_tensor: Preprocessed image tensor
        decode_method: 'greedy' or 'beam_search'
        
    Returns:
        Tuple of (predicted_text, confidence)
    """
    with torch.no_grad():
        start_time = time.time()
        
        # Forward pass
        logits = model(image_tensor)
        
        # Decode
        if decode_method == 'greedy':
            predicted_text, confidence = greedy_decode(logits[0], config['charset'])
        else:
            predicted_text, confidence = beam_search_decode(logits[0], config['charset'])
        
        inference_time = time.time() - start_time
        
        return predicted_text, confidence, inference_time


# Create FastAPI app
app = FastAPI(
    title="SKU Digit Reader",
    description="OCR API for reading multi-digit SKUs from warehouse labels",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SKU Digit Reader API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST image to get digit recognition",
            "health": "/health - GET API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "charset": config['charset'] if config else None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    decode_method: str = "greedy"
):
    """
    Predict digits from uploaded image.
    
    Args:
        file: Uploaded image file
        decode_method: 'greedy' or 'beam_search'
        
    Returns:
        JSON with predicted text and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if decode_method not in ['greedy', 'beam_search']:
        raise HTTPException(status_code=400, detail="decode_method must be 'greedy' or 'beam_search'")
    
    try:
        # Read and validate image
        contents = await file.read()
        
        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        
        # Validate image dimensions
        if image.width > 5000 or image.height > 5000:
            raise HTTPException(status_code=400, detail="Image dimensions too large")
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Predict
        predicted_text, confidence, inference_time = predict_digits(image_tensor, decode_method)
        
        return {
            "text": predicted_text,
            "confidence": float(confidence),
            "inference_time_ms": float(inference_time * 1000),
            "decode_method": decode_method,
            "image_info": {
                "width": image.width,
                "height": image.height,
                "mode": image.mode
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict digits from multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        JSON with predictions for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files (max 10)")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Preprocess
            image_tensor = preprocess_image(image)
            
            # Predict
            predicted_text, confidence, inference_time = predict_digits(image_tensor)
            
            results.append({
                "index": i,
                "filename": file.filename,
                "text": predicted_text,
                "confidence": float(confidence),
                "inference_time_ms": float(inference_time * 1000)
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "charset": config['charset'],
        "image_height": config['img_h'],
        "max_length": config['max_len'],
        "model_architecture": {
            "cnn_output": config['model']['cnn_out'],
            "rnn_hidden": config['model']['rnn_hidden'],
            "rnn_layers": config['model']['rnn_layers']
        },
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
