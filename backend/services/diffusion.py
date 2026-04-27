import torch
from diffusers import StableDiffusionPipeline
import asyncio
import gc
import os
import json
import httpx
import hashlib
from dotenv import load_dotenv

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
load_dotenv(config_path)

civitai_api_key = os.getenv("CIVITAI_API_KEY")

pipe = None
current_base_model = None

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

def _load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def _save_metadata(data):
    with open(METADATA_PATH, "w") as f:
        json.dump(data, f, indent=4)

async def download_model_from_civitai(model_version_id: str, save_path: str, is_lora: bool = False, progress_callback=None):
    """
    Downloads Model from CivitAI asynchronously utilizing httpx.
    Checks SHA256 integrity and stores trainedWords in a local JSON config if LoRA.
    """
    headers = {"Authorization": f"Bearer {civitai_api_key}"} if civitai_api_key else {}
    base_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
    
    # Extended timeout: 30s connect, no read timeout (files can be 2-4GB)
    timeout = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)
    
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(base_url, headers=headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch model metadata from CivitAI. Code: {resp.status_code}, Body: {resp.text}")
        data = resp.json()
        
        files = data.get("files", [])
        best_file = None
        for f in files:
            if f.get("name", "").endswith(".safetensors") and f.get("metadata", {}).get("size") == "pruned":
                best_file = f
                break
        
        if not best_file and files:
            best_file = next((f for f in files if f.get("name", "").endswith(".safetensors")), files[0])
            
        if not best_file:
            raise Exception("No suitable downloadable file found on CivitAI for this version.")
            
        download_url = best_file.get("downloadUrl")
        expected_hash = best_file.get("hashes", {}).get("SHA256")
        total_size_kb = best_file.get("sizeKB", 0)
        
        if civitai_api_key and "token=" not in download_url:
            separator = "&" if "?" in download_url else "?"
            download_url += f"{separator}token={civitai_api_key}"
        
        total_size_mb = round(total_size_kb / 1024, 1) if total_size_kb else "unknown"
        print(f"📥 Downloading model {model_version_id} ({total_size_mb} MB) to {save_path}...")
        sha256_hash = hashlib.sha256()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        downloaded_bytes = 0
        async with client.stream("GET", download_url, headers=headers, follow_redirects=True) as stream_resp:
            if stream_resp.status_code != 200:
                raise Exception(f"Failed to download file from CivitAI. Code: {stream_resp.status_code}")
            with open(save_path, "wb") as f:
                async for chunk in stream_resp.aiter_bytes(chunk_size=1024 * 1024): # 1MB chunks
                    f.write(chunk)
                    sha256_hash.update(chunk)
                    downloaded_bytes += len(chunk)
                    downloaded_mb = round(downloaded_bytes / (1024 * 1024), 1)
                    if progress_callback:
                        progress_callback(f"Downloaded {downloaded_mb} MB / {total_size_mb} MB")
                    if downloaded_mb % 50 < 1.1:  # Log every ~50MB
                        print(f"   ... {downloaded_mb} MB downloaded")
                    
        calculated_hash = sha256_hash.hexdigest().upper()
        if expected_hash and calculated_hash != expected_hash.upper():
            os.remove(save_path)
            raise Exception(f"Hash mismatch! Expected {expected_hash}, got {calculated_hash}")
            
        print(f"✅ Model {model_version_id} downloaded successfully and verified ({round(downloaded_bytes/(1024*1024),1)} MB).")
        
        if is_lora:
            trained_words = data.get("trainedWords", [])
            if trained_words:
                metadata = _load_metadata()
                filename = os.path.basename(save_path)
                metadata[filename] = {"trainedWords": trained_words}
                _save_metadata(metadata)

def _get_pipe(base_model_path=None, lora_path=None, use_ip_adapter=False):
    global pipe
    global current_base_model
    
    # === RTX 3050 6GB SAFETY: Cap CUDA memory to 80% to prevent power surge shutdown ===
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
        # Limit PyTorch memory allocator fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Reload model if base model has changed
    if pipe is None or current_base_model != base_model_path:
        print(f"🔄 Loading Base Model: {base_model_path or 'Default RunwayML 1.5'}")
        
        if pipe is not None:
            del pipe
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            # Allow GPU to cool down briefly after full unload
            import time
            time.sleep(2)
            
        hf_token = os.getenv("HF_TOKEN")
        
        if base_model_path and os.path.exists(base_model_path) and base_model_path.endswith('.safetensors'):
            pipe = StableDiffusionPipeline.from_single_file(
                base_model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                low_cpu_mem_usage=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16,
                safety_checker=None,
                token=hf_token,
                low_cpu_mem_usage=True
            )
            
        current_base_model = base_model_path
        
        # === KRITIS: sequential offload jauh lebih hemat dibanding model_cpu_offload ===
        # model_cpu_offload pindahin satu model penuh ke GPU sekaligus (~3-4GB)
        # sequential offload cuma mindahin 1 layer ke GPU (~200-400MB peak)
        pipe.enable_sequential_cpu_offload()
        
        # Aktifkan VAE slicing agar decode gambar nggak makan >1GB sekaligus
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

        # IP-Adapter HANYA dimuat kalau memang ada reference image beneran
        # Karena IP-Adapter sendiri makan ~1GB VRAM tambahan
        if use_ip_adapter:
            try:
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                print("✅ IP-Adapter loaded (reference image detected).")
            except Exception as e:
                print(f"⚠️ Warning: Could not load IP-Adapter: {e}")
        else:
            print("ℹ️ IP-Adapter SKIPPED — no reference image. Saves ~1GB VRAM.")

    # Remove active LoRAs to prevent conflicts with multiple LoRAs
    try:
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
    except Exception:
        pass
    
    if lora_path is None:
        lora_path = os.path.join(MODELS_DIR, "itstv_style.safetensors")
        
    if os.path.exists(lora_path):
        try:
            pipe.load_lora_weights(lora_path)
            print(f"✅ LoRA weights successfully loaded from: {lora_path}")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights: {e}")
    else:
        print(f"⚠️ Warning: LoRA weights not found at {lora_path}. Skipping LoRA.")

    return pipe

# === RTX 3050 6GB SAFE DEFAULTS ===
# Resolusi 384x384 agar peak VRAM UNet inference ~1.5GB bukan ~3GB (512x512)
# Steps 20 cukup untuk kualitas yang baik tanpa memperpanjang GPU load time
SD_WIDTH = 384
SD_HEIGHT = 384
SD_STEPS = 20
SD_GUIDANCE = 7.5

def _generate_sync(prompt: str, output_path: str, image_reference: str = None, base_model_path: str = None, lora_path: str = None) -> str:
    # Determine apakah kita benar-benar punya reference image
    has_real_reference = bool(image_reference and os.path.exists(image_reference))
    
    p = _get_pipe(base_model_path, lora_path, use_ip_adapter=has_real_reference)
    
    if lora_path is None:
        lora_path = os.path.join(MODELS_DIR, "itstv_style.safetensors")
        
    filename = os.path.basename(lora_path)
    metadata = _load_metadata()
    trained_words = metadata.get(filename, {}).get("trainedWords", [])
    
    if not trained_words and filename == "itstv_style.safetensors":
        trained_words = ["itstvstyle"]
        
    for word in trained_words:
        if word.lower() not in prompt.lower():
            prompt = f"{word}, {prompt}"
            
    kwargs = {
        "width": SD_WIDTH,
        "height": SD_HEIGHT,
        "num_inference_steps": SD_STEPS,
        "guidance_scale": SD_GUIDANCE,
    }
    ref_img = None
    from PIL import Image
    
    if has_real_reference:
        try:
            ref_img = Image.open(image_reference).convert("RGB").resize((SD_WIDTH, SD_HEIGHT))
            kwargs["ip_adapter_image"] = ref_img
            p.set_ip_adapter_scale(0.5)
            print(f"🖼️ IP-Adapter active with reference: {image_reference}")
        except Exception as e:
            print(f"Failed to load IP-Adapter image {image_reference}: {e}")
            # IP-Adapter was loaded, must provide dummy
            kwargs["ip_adapter_image"] = Image.new("RGB", (SD_WIDTH, SD_HEIGHT), (0, 0, 0))
            p.set_ip_adapter_scale(0.0)
    # Kalau IP-Adapter tidak dimuat, JANGAN kirim ip_adapter_image sama sekali
    
    print(f"⚡ Generating at {SD_WIDTH}x{SD_HEIGHT}, {SD_STEPS} steps (RTX 3050 Safe Mode)")
    image = p(prompt, **kwargs).images[0]
    image.save(output_path)
    
    if ref_img is not None:
        del ref_img
    kwargs.clear()
    
    # Aggressive cleanup setelah setiap generasi
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU VRAM flushed post-generation.")
    return output_path

async def generate_image(prompt: str, output_path: str, image_reference: str = None, base_model_path: str = None, lora_path: str = None) -> str:
    """
    Generates an image safely wrapping PyTorch rendering on a background thread.
    Optimized for RTX 3050 6GB to prevent system shutdown.
    """
    print(f"Generating image for prompt: {prompt}")
    return await asyncio.to_thread(_generate_sync, prompt, output_path, image_reference, base_model_path, lora_path)
