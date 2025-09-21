#!/usr/bin/env python3
"""
Demo script showing how to use the Warehouse Label Digit Reader.

This script demonstrates the complete workflow from data generation to inference.
"""

import sys
from pathlib import Path

def print_step(step_num, title, description):
    """Print a formatted step."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    print(description)
    print()

def main():
    """Run the demo workflow."""
    print("🏷️  WAREHOUSE LABEL DIGIT READER - DEMO WORKFLOW")
    print("=" * 60)
    print("This script will guide you through the complete workflow")
    print("from data generation to model deployment.")
    
    print_step(1, "INSTALLATION", """
    First, install the required dependencies:
    
    pip install -r requirements.txt
    
    Then test your installation:
    python test_installation.py
    """)
    
    print_step(2, "DATA GENERATION", """
    Generate synthetic training data:
    
    python -m src.synth_digits --count 10000 --len 3-8 --out data/synth
    
    This creates 10,000 images of digit strings with lengths 3-8.
    The images include realistic transformations like perspective,
    noise, and blur to simulate real warehouse conditions.
    """)
    
    print_step(3, "MODEL TRAINING", """
    Train the CRNN model:
    
    python -m src.train_crnn --config configs/crnn.yaml
    
    Training typically takes 2-4 hours on a GPU or 8-12 hours on CPU.
    The model will be saved to artifacts/checkpoints/best.ckpt
    """)
    
    print_step(4, "MODEL EVALUATION", """
    Evaluate the trained model:
    
    python -m src.eval_crnn --config configs/crnn.yaml
    
    This will:
    - Calculate accuracy metrics
    - Test robustness to noise and blur
    - Generate performance plots
    - Save results to artifacts/reports/
    """)
    
    print_step(5, "API DEPLOYMENT", """
    Start the FastAPI server:
    
    uvicorn app.main:app --reload
    
    The API will be available at http://localhost:8000
    
    Test with curl:
    curl -X POST -F "file=@sample.png" http://localhost:8000/predict
    """)
    
    print_step(6, "INTERACTIVE DEMO", """
    Launch the Gradio interface:
    
    python demo/ui.py
    
    Open your browser to http://localhost:7860
    Upload images and see real-time predictions!
    """)
    
    print_step(7, "COMMAND LINE INFERENCE", """
    Test individual images from command line:
    
    python -m src.infer --image path/to/image.png
    
    This provides a simple way to test the model on specific images.
    """)
    
    print_step(8, "CUSTOMIZATION", """
    Customize the model by editing configs/crnn.yaml:
    
    - Change character set (add letters, symbols)
    - Adjust model architecture (CNN/RNN sizes)
    - Modify training parameters (batch size, learning rate)
    - Update data augmentation settings
    """)
    
    print("\n" + "="*60)
    print("🎉 COMPLETE WORKFLOW DEMONSTRATED!")
    print("="*60)
    print("""
    Your portfolio-ready OCR system includes:
    
    ✅ Synthetic data generation
    ✅ CRNN-CTC model architecture  
    ✅ Training and evaluation scripts
    ✅ FastAPI inference server
    ✅ Interactive Gradio demo
    ✅ Comprehensive documentation
    ✅ Production deployment ready
    
    This demonstrates expertise in:
    - Computer Vision & Deep Learning
    - ML Engineering & MLOps
    - API Development & Deployment
    - Full-stack ML Applications
    
    Perfect for showcasing to potential employers!
    """)
    
    print("\n📚 For detailed documentation, see README.md")
    print("📊 For portfolio summary, see PORTFOLIO_SUMMARY.md")

if __name__ == "__main__":
    main()
