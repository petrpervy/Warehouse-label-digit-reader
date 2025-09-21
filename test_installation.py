#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality.

Run this after installing requirements.txt to ensure everything works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn: {e}")
        return False
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI: {e}")
        return False
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"❌ Gradio: {e}")
        return False
    
    return True


def test_project_structure():
    """Test that project files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "configs/crnn.yaml",
        "src/__init__.py",
        "src/synth_digits.py",
        "src/dataset.py",
        "src/model_crnn.py",
        "src/train_crnn.py",
        "src/eval_crnn.py",
        "src/decode.py",
        "src/utils.py",
        "app/main.py",
        "demo/ui.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    return True


def test_model_creation():
    """Test that model can be created."""
    print("\nTesting model creation...")
    
    try:
        # Add src to path
        sys.path.append('src')
        
        from model_crnn import create_model
        
        # Create model
        model = create_model()
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Model created successfully")
        print(f"✅ Parameters: {param_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    
    try:
        sys.path.append('src')
        from synth_digits import create_digit_image
        
        # Create a test image
        img = create_digit_image("12345")
        
        if img is not None and img.size[0] > 0 and img.size[1] > 0:
            print("✅ Data generation works")
            return True
        else:
            print("❌ Generated image is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Warehouse Label Digit Reader Installation")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Creation", test_model_creation),
        ("Data Generation", test_data_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Generate training data: python -m src.synth_digits --count 1000 --len 3-8 --out data/synth")
        print("2. Train the model: python -m src.train_crnn --config configs/crnn.yaml")
        print("3. Launch demo: python demo/ui.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
