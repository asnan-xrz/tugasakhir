import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def test():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(0.5)
        
        img = Image.new("RGB", (384, 384), (255, 0, 0))
        
        kwargs = {
            "prompt": "test",
            "ip_adapter_image": img,
            "num_inference_steps": 1,
            "cross_attention_kwargs": {"scale": 0.65}
        }
        
        pipe(**kwargs)
        print("SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()

test()
