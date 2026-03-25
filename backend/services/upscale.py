import cv2
import os
import urllib.request
from cv2 import dnn_superres

MODEL_URL = "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "LapSRN_x2.pb")

def download_model_if_not_exists():
    if not os.path.exists(MODEL_PATH):
        print("Downloading LapSRN_x2 model for upscaling (this only happens once)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def upscale_image_sync(input_path: str, output_path: str) -> str:
    """
    Reads an image from input_path, applies LapSRN x2 AI upscaling purely on CPU,
    and writes the high-res image to output_path.
    """
    download_model_if_not_exists()
    
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    
    # Use LapSRN which is extremely fast and high quality on CPU with very low memory footprint
    sr.setModel("lapsrn", 2)
    
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        # Fallback if image doesn't exist
        raise FileNotFoundError(f"Input image {input_path} not found.")
        
    # Scale up!
    result = sr.upsample(img)
    
    # Save the output
    cv2.imwrite(output_path, result)
    return output_path
