import torch
from diffusers import StableDiffusionPipeline
import os

model_path = "/home/firania/Documents/tugasakhir/models/runwayml/stable-diffusion-v1-5"
lora_path = "/home/firania/Documents/tugasakhir/ai/lora_output_its/pytorch_lora_weights.safetensors"

if not os.path.exists(model_path):
    model_path = "runwayml/stable-diffusion-v1-5"

print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    low_cpu_mem_usage=True
)

pipe.enable_model_cpu_offload()

print("Loading IP Adapter...")
try:
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
except Exception as e:
    print(f"Error loading IP adapter: {e}")

print("Loading LoRA first time...")
try:
    pipe.load_lora_weights(lora_path)
    print("LoRA loaded first time successfully.")
except Exception as e:
    print(f"Error loading LoRA: {e}")

print("Unloading LoRA...")
pipe.unload_lora_weights()

print("Loading LoRA second time...")
try:
    pipe.load_lora_weights(lora_path)
    print("LoRA loaded second time successfully.")
except Exception as e:
    print(f"Error loading LoRA second time: {e}")
