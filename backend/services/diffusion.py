import torch
from diffusers import StableDiffusionPipeline
import asyncio
import gc

import os
from dotenv import load_dotenv

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
load_dotenv(config_path)

pipe = None

def _get_pipe():
    global pipe
    if pipe is None:
        hf_token = os.getenv("HF_TOKEN")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16,
            safety_checker=None,
            token=hf_token
        )
        
        # [MODIFIKASI LORA] 1: Load LoRA weights untuk Identitas visual ITS TV
        # Menangani path secara dinamis berdasarkan direktori kerja saat ini
        lora_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "itstv_style.safetensors")
        if os.path.exists(lora_path):
            # Memuat LoRA ke dalam pipeline attention layers
            pipe.load_lora_weights(lora_path)
            print(f"✅ LoRA weigths successfully loaded from: {lora_path}")
        else:
            print(f"⚠️ Warning: LoRA weights not found at {lora_path}. Skipping LoRA.")

        # [MODIFIKASI CONTROLNET/IP-ADAPTER]: Load IP-Adapter weights
        try:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
            print("✅ IP-Adapter successfully loaded.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load IP-Adapter: {e}")

        # Menggunakan CPU offload untuk menghemat VRAM (RTX 3050 friendly)
        pipe.enable_model_cpu_offload()
        # Perhatian: enable_attention_slicing() dihapus karena memiliki bug incompatible
        # dengan IP Adapter di versi Diffusers 0.38+. CPU offload sudah cukup menghemat VRAM.
    return pipe

def _generate_sync(prompt: str, output_path: str, image_reference: str = None) -> str:
    p = _get_pipe()
    
    # [MODIFIKASI LORA] 2: Auto-Trigger Word
    # Otomatis menambahkan trigger word Civit AI agar user tidak perlu mengetik manual
    trigger_word = "itstvstyle"
    if trigger_word not in prompt.lower():
        prompt = f"{trigger_word}, {prompt}"
        
    kwargs = {}
    ref_img = None
    from PIL import Image
    if image_reference and os.path.exists(image_reference):
        try:
            ref_img = Image.open(image_reference).convert("RGB")
            kwargs["ip_adapter_image"] = ref_img
            # Set scale to manage composition preservation (e.g., 0.5 to keep sketch style but follow structure)
            p.set_ip_adapter_scale(0.5)
            print(f"Applying IP-Adapter using reference: {image_reference}")
        except Exception as e:
            print(f"Failed to load image reference {image_reference}: {e}")
            kwargs["ip_adapter_image"] = Image.new("RGB", (512, 512), (0, 0, 0))
            p.set_ip_adapter_scale(0.0)
    else:
        # Diffusers expects an image if IP-Adapter is loaded, so we pass a dummy black image
        # with scale 0.0 to effectively disable it without unloading the weights.
        kwargs["ip_adapter_image"] = Image.new("RGB", (512, 512), (0, 0, 0))
        p.set_ip_adapter_scale(0.0)
            
    image = p(prompt, **kwargs).images[0]
    image.save(output_path)
    
    # [MODIFIKASI MEMORI] 3: Handling VRAM untuk RTX 3050 (6GB)
    # Empty cache otomatis segera setelah LoRA dan difusi selesai agar VRAM segar kembali.
    if ref_img is not None:
        del ref_img
    kwargs.clear()
    
    # Kosongkan cache IP-Adapter/attention
    torch.cuda.empty_cache()
    gc.collect()
    return output_path

async def generate_image(prompt: str, output_path: str, image_reference: str = None) -> str:
    """
    Given an enhanced prompt, generates an image using local Stable Diffusion v1.5 via Diffusers.
    Supports composition preservation via IP-Adapter if image_reference is provided.
    Returns the path to the generated image.
    """
    print(f"Generating image for prompt: {prompt}")
    return await asyncio.to_thread(_generate_sync, prompt, output_path, image_reference)
