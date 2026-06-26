import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np
import asyncio
import gc
import os
import json
import httpx
import hashlib
import random
from dotenv import load_dotenv

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
load_dotenv(config_path)

civitai_api_key = os.getenv("CIVITAI_API_KEY")

pipe = None
current_base_model = None
current_lora_path = None
current_pipeline_type = None  # 'base' or 'controlnet'

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

def _get_pipe(base_model_path=None, lora_path=None, use_ip_adapter=False, use_controlnet=False):
    global pipe
    global current_base_model
    global current_lora_path
    global current_pipeline_type
    
    # === RTX 3050 6GB SAFETY: Cap CUDA memory to 80% to prevent power surge shutdown ===
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
        # Limit PyTorch memory allocator fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    desired_pipeline_type = "controlnet" if use_controlnet else "base"
    
    # Cek tipe pipe aktual (bukan hanya state tracking) — ini penting untuk RTX 3050
    # dimana reload bisa gagal OOM dan meninggalkan state tidak konsisten
    actual_pipe_type = None
    if pipe is not None:
        actual_pipe_type = "controlnet" if isinstance(pipe, StableDiffusionControlNetPipeline) else "base"
    
    # Reload jika: pipe tidak ada, base model berbeda, ATAU tipe pipeline tidak sesuai (cek aktual!)
    needs_reload = (
        pipe is None
        or current_base_model != base_model_path
        or current_pipeline_type != desired_pipeline_type
        or actual_pipe_type != desired_pipeline_type  # Safety: cek tipe nyata, bukan hanya cached state
    )
    
    if needs_reload:
        print(f"🔄 Loading Base Model: {base_model_path or 'Default RunwayML 1.5'} [Type: {desired_pipeline_type}]")
        
        if pipe is not None:
            del pipe
            pipe = None
            current_lora_path = None
            gc.collect()
            torch.cuda.empty_cache()
            # Allow GPU to cool down briefly after full unload
            import time
            time.sleep(2)
            
        hf_token = os.getenv("HF_TOKEN")
        
        controlnet = None
        if use_controlnet:
            print("🔄 Loading ControlNet Canny Model...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", 
                torch_dtype=torch.float16,
                token=hf_token
            )
            pipeline_class = StableDiffusionControlNetPipeline
            kwargs_pipe = {"controlnet": controlnet}
        else:
            pipeline_class = StableDiffusionPipeline
            kwargs_pipe = {}
        
        try:
            if base_model_path and os.path.exists(base_model_path) and base_model_path.endswith('.safetensors'):
                pipe = pipeline_class.from_single_file(
                    base_model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    low_cpu_mem_usage=False,
                    **kwargs_pipe
                )
            else:
                pipe = pipeline_class.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", 
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    token=hf_token,
                    low_cpu_mem_usage=False,
                    **kwargs_pipe
                )
                
            # Hanya update state jika load BERHASIL — mencegah state korup pada RTX 3050
            current_base_model = base_model_path
            current_pipeline_type = desired_pipeline_type
            
        except torch.cuda.OutOfMemoryError as oom_err:
            # RTX 3050: jika OOM saat ganti pipeline type, coba tanpa ControlNet sebagai fallback
            print(f"🔴 OOM saat load pipeline [{desired_pipeline_type}]: {oom_err}")
            print("⚠️ Fallback: mencoba load base pipeline saja (tanpa ControlNet) untuk hemat VRAM...")
            gc.collect()
            torch.cuda.empty_cache()
            import time
            time.sleep(3)
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                token=hf_token,
                low_cpu_mem_usage=False,
            )
            current_base_model = base_model_path
            current_pipeline_type = "base"  # Paksa ke base karena ControlNet OOM
        
        # IP-Adapter HANYA dimuat kalau memang ada reference image beneran
        if use_ip_adapter:
            try:
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                print("✅ IP-Adapter loaded (reference image detected).")
            except Exception as e:
                print(f"⚠️ Warning: Could not load IP-Adapter: {e}")
        else:
            print("ℹ️ IP-Adapter SKIPPED — no reference image. Saves ~1GB VRAM.")
            
        # === FIX: Use model_cpu_offload instead of sequential_cpu_offload ===
        # sequential_cpu_offload causes "Cannot copy out of meta tensor" when loading LoRAs dynamically.
        # model_cpu_offload is still extremely memory efficient (well within 6GB limits) but safe for LoRAs.
        # MUST BE CALLED AFTER IP-ADAPTER IS LOADED!
        pipe.enable_model_cpu_offload()
        
        # Aktifkan VAE slicing agar decode gambar nggak makan >1GB sekaligus
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    if lora_path is None:
        # Prioritas: lora_output_its (hasil training terbaru), fallback ke models/
        default_lora = os.path.join(os.path.dirname(MODELS_DIR), "ai", "lora_output_its", "pytorch_lora_weights.safetensors")
        if not os.path.exists(default_lora):
            default_lora = os.path.join(MODELS_DIR, "its_new_lora.safetensors")
        lora_path = default_lora
        
    if current_lora_path != lora_path:
        print(f"🔄 Swapping/Loading LoRA: {lora_path}")
        # Remove active LoRAs to prevent conflicts
        try:
            if hasattr(pipe, 'unload_lora_weights'):
                pipe.unload_lora_weights()
        except Exception:
            pass
            
        if os.path.exists(lora_path):
            try:
                pipe.load_lora_weights(lora_path)
                current_lora_path = lora_path
                print(f"✅ LoRA weights successfully loaded from: {lora_path}")
            except Exception as e:
                print(f"⚠️ Failed to load LoRA weights: {e}")
                current_lora_path = None
        else:
            print(f"⚠️ Warning: LoRA weights not found at {lora_path}. Skipping LoRA.")
            current_lora_path = None

    return pipe

# === RTX 3050 6GB SAFE DEFAULTS ===
# Resolusi 384x384 agar peak VRAM UNet inference ~1.5GB bukan ~3GB (512x512)
# Steps 20 cukup untuk kualitas yang baik tanpa memperpanjang GPU load time
SD_WIDTH = 384
SD_HEIGHT = 384
SD_STEPS = 20
SD_GUIDANCE = 7.5

def _generate_sync(prompt: str, output_path: str, image_reference: str = None, base_model_path: str = None, lora_path: str = None, use_controlnet: bool = False) -> str:
    from PIL import Image
    
    # 1. Validasi reference image di awal untuk mencegah penggunaan dummy blank image
    ref_img = None
    has_real_reference = False
    
    if image_reference and os.path.exists(image_reference):
        try:
            ref_img = Image.open(image_reference).convert("RGB").resize((SD_WIDTH, SD_HEIGHT))
            has_real_reference = True
        except Exception as e:
            print(f"⚠️ Failed to load image reference {image_reference}: {e}")
            ref_img = None
            has_real_reference = False
            
    # === BRANCHING LOGIC ===
    # Jika tidak ada gambar valid (RAG mati atau gambar korup), paksa ke Jalur B (Pure Text-to-Image)
    if not has_real_reference:
        use_controlnet = False

    p = _get_pipe(base_model_path, lora_path, use_ip_adapter=has_real_reference, use_controlnet=use_controlnet)
    
    # === SAFETY NET untuk RTX 3050 ===
    # Cek tipe pipeline AKTUAL yang dikembalikan. Jika masih ControlNet padahal
    # seharusnya base (karena OOM fallback), override use_controlnet agar konsisten.
    is_controlnet_pipe = isinstance(p, StableDiffusionControlNetPipeline)
    blank_controlnet_image = None  # Inisialisasi selalu None

    if is_controlnet_pipe and not has_real_reference:
        # Pipeline ControlNet tapi tidak ada reference image → AKAN CRASH jika image=None!
        # Buat blank black image sebagai ControlNet conditioning SAJA (bukan IP-Adapter)
        # agar pipeline tidak crash, output tetap dikendalikan 100% oleh teks prompt.
        print("⚠️ [RTX 3050 Guard] ControlNet pipeline aktif tapi tidak ada reference image!")
        print("⚠️ Menyiapkan blank black image sebagai ControlNet conditioning fallback...")
        from PIL import Image as PILImage
        blank_arr = np.zeros((SD_HEIGHT, SD_WIDTH, 3), dtype=np.uint8)
        blank_controlnet_image = PILImage.fromarray(blank_arr)  # Hitam = tidak ada edge = pure text-driven
        use_controlnet = True   # Tetap pakai ControlNet pipe tapi dengan blank image
        ref_img = None          # IP-Adapter tetap OFF
        has_real_reference = False  # IP-Adapter scale tetap tidak aktif
    elif not is_controlnet_pipe:
        use_controlnet = False  # Pipeline base → tidak perlu ControlNet kwargs
    
    if lora_path is None:
        # Prioritas: lora_output_its (hasil training terbaru), fallback ke models/
        default_lora = os.path.join(os.path.dirname(MODELS_DIR), "ai", "lora_output_its", "pytorch_lora_weights.safetensors")
        if not os.path.exists(default_lora):
            default_lora = os.path.join(MODELS_DIR, "its_new_lora.safetensors")
        lora_path = default_lora
        
    filename = os.path.basename(lora_path)
    metadata = _load_metadata()
    # Guard: trainedWords bisa null di JSON, paksa jadi list
    trained_words = metadata.get(filename, {}).get("trainedWords", []) or []
    
    if not trained_words and filename == "itstv_style.safetensors":
        trained_words = ["itstvstyle"]
    
    # Trigger word khusus untuk LoRA hasil training dataset_combined
    if not trained_words and "pytorch_lora_weights" in filename:
        trained_words = ["itstvstyle"]
        
    for word in trained_words:
        if word and word.lower() not in prompt.lower():
            prompt = f"{word}, {prompt}"
            
    # === FIX KRITIS: Setiap frame WAJIB pakai seed acak yang BERBEDA ===
    # Tanpa ini, pipeline yang di-cache (pipe global) akan menghasilkan gambar
    # yang identik di setiap frame karena PRNG PyTorch tidak di-reset.
    frame_seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(frame_seed)
    print(f"🎲 Frame seed: {frame_seed} (unik per frame, untuk reproduksi simpan nilai ini)")

    kwargs = {
        "width": SD_WIDTH,
        "height": SD_HEIGHT,
        "num_inference_steps": SD_STEPS,
        "guidance_scale": SD_GUIDANCE,
        "generator": generator,  # KUNCI: seed berbeda = gambar berbeda
    }
    
    # Balance LoRA intensity when combining with IP-Adapter
    if lora_path and "its_new_lora.safetensors" in lora_path:
        kwargs["cross_attention_kwargs"] = {"scale": 0.65}
        
    if has_real_reference:
        # JALUR A: Mode RAG (Butuh Gambar)
        if use_controlnet:
            # Convert reference image to Canny Edge Map
            image_arr = np.array(ref_img)
            low_threshold = 50  # Diturunkan agar menangkap lebih banyak detail halus (teks/logo)
            high_threshold = 150
            edges = cv2.Canny(image_arr, low_threshold, high_threshold)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            canny_img = Image.fromarray(edges)
            
            # Simpan Canny Map untuk keperluan debugging agar pengguna bisa melihat
            # outline apa yang sebenarnya ditangkap oleh ControlNet
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            debug_path = output_path.replace(".png", "_canny_debug.png")
            canny_img.save(debug_path)
            
            kwargs["image"] = canny_img
            kwargs["controlnet_conditioning_scale"] = 1.0
            print(f"🧠 ControlNet Canny Edge Map generated and saved to {debug_path}")
            
        kwargs["ip_adapter_image"] = ref_img
        p.set_ip_adapter_scale(0.35)  # 0.35: referensi "menginspirasi" bukan "mendominasi" — lebih kreatif
        print(f"🖼️ IP-Adapter active with reference: {image_reference}")
    else:
        # JALUR B: Mode NO RAG (Pure Text-to-Image)
        print("ℹ️ Mode NO RAG: Pure Text-to-Image pipeline active (No visual references).")
        
        # === FIX KRITIS: Inject blank_controlnet_image jika pipeline ControlNet
        # ter-cache dari sesi sebelumnya (bisa terjadi pada RTX 3050 karena pipe global).
        # Tanpa ini, ControlNet pipeline akan menerima image=None → NoneType crash!
        if use_controlnet and blank_controlnet_image is not None:
            kwargs["image"] = blank_controlnet_image
            kwargs["controlnet_conditioning_scale"] = 0.0  # Scale=0 → ControlNet buta, murni text-driven
            print("⚠️ [RTX 3050 Guard] Blank ControlNet image injected dengan scale=0.0 (pure text mode).")
    
    # === FINAL GUARD: Pastikan ControlNet pipeline TIDAK dipanggil tanpa 'image' ===
    # Ini adalah safety net terakhir untuk menangkap edge case apapun yang lolos di atas.
    if is_controlnet_pipe and "image" not in kwargs:
        print("🛑 [Final Guard] ControlNet pipe terdeteksi tanpa kwargs['image']! Injecting emergency blank image.")
        from PIL import Image as _EmergPIL
        _blank = _EmergPIL.fromarray(np.zeros((SD_HEIGHT, SD_WIDTH, 3), dtype=np.uint8))
        kwargs["image"] = _blank
        kwargs["controlnet_conditioning_scale"] = 0.0
    
    print(f"⚡ Generating at {SD_WIDTH}x{SD_HEIGHT}, {SD_STEPS} steps, seed={frame_seed} (RTX 3050 Safe Mode)")
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

async def generate_image(prompt: str, output_path: str, image_reference: str = None, base_model_path: str = None, lora_path: str = None, use_controlnet: bool = False) -> str:
    """
    Generates an image safely wrapping PyTorch rendering on a background thread.
    Optimized for RTX 3050 6GB to prevent system shutdown.
    """
    print(f"Generating image for prompt: {prompt} (ControlNet: {use_controlnet})")
    return await asyncio.to_thread(_generate_sync, prompt, output_path, image_reference, base_model_path, lora_path, use_controlnet)
