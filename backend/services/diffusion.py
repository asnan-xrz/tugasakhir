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
        # Menggunakan CPU offload untuk menghemat VRAM (RTX 3050 friendly)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    return pipe

def _generate_sync(prompt: str, output_path: str) -> str:
    p = _get_pipe()
    image = p(prompt).images[0]
    image.save(output_path)
    
    # Empty cache agresif setelah inferensi agar LLM tidak berebut VRAM
    torch.cuda.empty_cache()
    gc.collect()
    return output_path

async def generate_image(prompt: str, output_path: str) -> str:
    """
    Given an enhanced prompt, generates an image using local Stable Diffusion v1.5 via Diffusers.
    Returns the path to the generated image.
    """
    print(f"Generating image for prompt: {prompt}")
    return await asyncio.to_thread(_generate_sync, prompt, output_path)
