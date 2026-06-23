import sys, os
import random
import pandas as pd
import asyncio
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.services.diffusion import _generate_sync

def calculate_ssim(img1, img2):
    # Convert PIL to numpy, resize if needed
    img1_arr = np.array(img1.resize((384, 384)).convert("L"))
    img2_arr = np.array(img2.resize((384, 384)).convert("L"))
    score, _ = ssim(img1_arr, img2_arr, full=True)
    return score

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "capt_qwen.csv")
    images_dir = os.path.join(base_dir, "allaboutITS")
    
    print(f"Loading captions from {csv_path}...")
    df = pd.read_csv(csv_path, sep='|')
    
    # Filter only Q-series images (allaboutITS) without aug
    q_images = [img for img in df['Image_name'].unique() if img.startswith('Q') and 'aug' not in img]
    
    # Ensure reproducibility for test
    random.seed(42)
    selected_image = random.choice(q_images)
    
    image_df = df[df['Image_name'] == selected_image].sort_values('caption_number')
    combined_caption = " ".join(image_df['caption'].astype(str).tolist())
    
    print(f"Selected Image: {selected_image}")
    print(f"Combined Caption: {combined_caption}")
    
    original_img_path = os.path.join(images_dir, selected_image)
    if not os.path.exists(original_img_path):
        print(f"Error: Original image not found at {original_img_path}")
        return
        
    out_no_controlnet = os.path.join(base_dir, "gen_nocontrol.png")
    out_with_controlnet = os.path.join(base_dir, "gen_withcontrol.png")
    
    # Generate 1: IP-Adapter + LoRA (No ControlNet)
    print("\n--- Generating [IP-Adapter + LoRA] (No ControlNet) ---")
    try:
        _generate_sync(
            prompt=combined_caption,
            output_path=out_no_controlnet,
            image_reference=original_img_path,
            base_model_path=None, 
            lora_path=None,
            use_controlnet=False
        )
    except Exception as e:
        print(f"Generation failed: {e}")
        return

    # Generate 2: IP-Adapter + LoRA + ControlNet
    print("\n--- Generating [IP-Adapter + LoRA + ControlNet Canny] ---")
    try:
        _generate_sync(
            prompt=combined_caption,
            output_path=out_with_controlnet,
            image_reference=original_img_path,
            base_model_path=None, 
            lora_path=None,
            use_controlnet=True
        )
    except Exception as e:
        print(f"Generation failed: {e}")
        return

    print("\n--- Calculating Similarity ---")
    orig_img = Image.open(original_img_path)
    gen_no_c_img = Image.open(out_no_controlnet)
    gen_c_img = Image.open(out_with_controlnet)
    
    ssim_no_c = calculate_ssim(orig_img, gen_no_c_img)
    ssim_c = calculate_ssim(orig_img, gen_c_img)
    
    print(f"SSIM Score (IP-Adapter only): {ssim_no_c:.4f}")
    print(f"SSIM Score (IP-Adapter + ControlNet): {ssim_c:.4f}")
    
    print("\n--- Plotting Results ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original: {selected_image}")
    axes[0].axis('off')
    
    axes[1].imshow(gen_no_c_img)
    axes[1].set_title(f"IP-Adapter Only\nSSIM: {ssim_no_c:.4f}")
    axes[1].axis('off')
    
    axes[2].imshow(gen_c_img)
    axes[2].set_title(f"IP-Adapter + ControlNet\nSSIM: {ssim_c:.4f}")
    axes[2].axis('off')
    
    # Wrap text for title
    import textwrap
    wrapped_caption = textwrap.fill(combined_caption, width=100)
    plt.suptitle(f"Prompt:\n{wrapped_caption}", fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    output_plot = os.path.join(base_dir, "controlnet_comparison.png")
    plt.savefig(output_plot, dpi=150)
    print(f"Comparison plot saved to {output_plot}")

if __name__ == "__main__":
    main()
