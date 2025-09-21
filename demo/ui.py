"""
Gradio UI for SKU digit recognition demo.

Provides an interactive web interface for uploading images and getting predictions.
"""

import io
import sys
from pathlib import Path

import gradio as gr
import torch
import yaml
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_crnn import create_model
from decode import greedy_decode, beam_search_decode


class DigitRecognizer:
    """Wrapper class for digit recognition model."""
    
    def __init__(self, config_path, checkpoint_path):
        self.load_model(config_path, checkpoint_path)
    
    def load_model(self, config_path, checkpoint_path):
        """Load the trained model."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_model(
            charset=self.config['charset'],
            img_h=self.config['img_h'],
            cnn_out=self.config['model']['cnn_out'],
            rnn_hidden=self.config['model']['rnn_hidden'],
            rnn_layers=self.config['model']['rnn_layers']
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def preprocess_image(self, image):
        """Preprocess PIL image for inference."""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to target height while maintaining aspect ratio
        w, h = image.size
        target_h = self.config['img_h']
        new_w = int(w * target_h / h)
        image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
        
        # Convert to tensor and add batch and channel dimensions
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict(self, image, decode_method='greedy'):
        """Predict digits from image."""
        if image is None:
            return "No image provided", 0.0, None
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Predict
            with torch.no_grad():
                logits = self.model(image_tensor)
                
                if decode_method == 'greedy':
                    predicted_text, confidence = greedy_decode(logits[0], self.config['charset'])
                else:
                    predicted_text, confidence = beam_search_decode(logits[0], self.config['charset'])
            
            # Create visualization
            viz_image = self.create_visualization(image, predicted_text)
            
            return predicted_text, confidence, viz_image
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0, None
    
    def create_visualization(self, original_image, predicted_text):
        """Create visualization with predicted text overlay."""
        # Convert to RGB for display
        if original_image.mode == 'L':
            viz_image = original_image.convert('RGB')
        else:
            viz_image = original_image.copy()
        
        # Resize for better display
        w, h = viz_image.size
        max_width = 400
        if w > max_width:
            new_h = int(h * max_width / w)
            viz_image = viz_image.resize((max_width, new_h), Image.Resampling.LANCZOS)
        
        return viz_image


def create_interface():
    """Create Gradio interface."""
    # Load model
    config_path = Path(__file__).parent.parent / 'configs/crnn.yaml'
    checkpoint_path = Path(__file__).parent.parent / 'artifacts/checkpoints/best.ckpt'
    
    if not checkpoint_path.exists():
        # Create a placeholder interface if model doesn't exist
        def placeholder_predict(image, decode_method):
            return "Model not found. Please train the model first.", 0.0, image
        
        interface = gr.Interface(
            fn=placeholder_predict,
            inputs=[
                gr.Image(type="pil", label="Upload Image"),
                gr.Radio(choices=['greedy', 'beam_search'], value='greedy', label="Decode Method")
            ],
            outputs=[
                gr.Textbox(label="Predicted Text"),
                gr.Number(label="Confidence"),
                gr.Image(label="Visualization")
            ],
            title="SKU Digit Reader",
            description="Upload an image of digits to get recognition results. Model checkpoint not found - please train the model first.",
            examples=[
                # Add some example images if available
            ]
        )
        return interface
    
    # Load the actual model
    try:
        recognizer = DigitRecognizer(config_path, checkpoint_path)
        
        def predict_fn(image, decode_method):
            return recognizer.predict(image, decode_method)
        
        interface = gr.Interface(
            fn=predict_fn,
            inputs=[
                gr.Image(type="pil", label="Upload Image"),
                gr.Radio(choices=['greedy', 'beam_search'], value='greedy', label="Decode Method")
            ],
            outputs=[
                gr.Textbox(label="Predicted Text"),
                gr.Number(label="Confidence"),
                gr.Image(label="Visualization")
            ],
            title="🏷️ SKU Digit Reader",
            description="""
            Upload an image containing digits (like warehouse labels, receipts, or SKU codes) 
            to get OCR recognition results. The model can handle various fonts, sizes, and orientations.
            
            **Instructions:**
            - Upload a clear image of digits
            - Choose decode method (greedy is faster, beam_search is more accurate)
            - View the predicted text and confidence score
            """,
            examples=[
                # Add example paths if available
            ],
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )
        
        return interface
        
    except Exception as e:
        print(f"Error loading model: {e}")
        
        def error_predict(image, decode_method):
            return f"Error loading model: {str(e)}", 0.0, image
        
        interface = gr.Interface(
            fn=error_predict,
            inputs=[
                gr.Image(type="pil", label="Upload Image"),
                gr.Radio(choices=['greedy', 'beam_search'], value='greedy', label="Decode Method")
            ],
            outputs=[
                gr.Textbox(label="Predicted Text"),
                gr.Number(label="Confidence"),
                gr.Image(label="Visualization")
            ],
            title="SKU Digit Reader",
            description="Error loading model. Please check that the model is trained and checkpoint exists.",
            examples=[]
        )
        
        return interface


def main():
    """Main function to launch the Gradio interface."""
    interface = create_interface()
    
    print("Launching Gradio interface...")
    print("Open your browser and go to the URL shown below")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
