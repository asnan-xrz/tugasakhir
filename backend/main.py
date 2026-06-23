import os
from dotenv import load_dotenv

# Load config.env before initializing the app so HuggingFace gets HF_TOKEN
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.env")
load_dotenv(config_path)

from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import uuid
import fitz
import torch
import gc

from services.llm import analyze_script_to_scenes, enhance_prompt
from services.rag import get_visual_context
from services.diffusion import generate_image

app = FastAPI(title="ITS TV Storyboard API")

# Ensure output directory exists before mounting
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for task statuses (use Redis or DB for production)
tasks_db: Dict[str, Dict[str, Any]] = {}

def calculate_nlp_scores(reference_text: str, candidate_text: str) -> dict:
    import re
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    
    def normalize_tokens(text: str) -> list:
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text_clean.split() if w]

    # Reference preparation: split reference_text into sentences
    sentences_list = re.split(r'[.\n?!]+', reference_text)
    reference = [normalize_tokens(s) for s in sentences_list if s.strip()]
    reference = [r for r in reference if r]
    if not reference:
        reference = [normalize_tokens(reference_text)]
        
    candidate = normalize_tokens(candidate_text)
    
    scores = {
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "rougeL": 0.0,
        "cosine": 0.0
    }
    
    if not candidate or not reference or not reference[0]:
        return scores
        
    smoothie = SmoothingFunction().method4
    
    try:
        scores["bleu1"] = sentence_bleu(reference, candidate, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
        scores["bleu2"] = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        scores["bleu3"] = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        scores["bleu4"] = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
        
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        candidate_clean = " ".join(candidate)
        
        # Max ROUGE-L over split sentences to avoid document length mismatch penalty
        best_rouge_l = 0.0
        for s in sentences_list:
            s_clean = " ".join(normalize_tokens(s))
            if not s_clean:
                continue
            r_score = scorer.score(s_clean, candidate_clean)['rougeL'].fmeasure
            if r_score > best_rouge_l:
                best_rouge_l = r_score
        
        # Fallback to direct overall if sentences are empty or score is 0
        if best_rouge_l == 0.0:
            ref_clean = " ".join(normalize_tokens(reference_text))
            best_rouge_l = scorer.score(ref_clean, candidate_clean)['rougeL'].fmeasure
            
        scores["rougeL"] = best_rouge_l
    except Exception as e:
        print(f"Error calculating ROUGE score: {e}")
        
    try:
        from collections import Counter
        import math
        
        ref_counts = Counter(normalize_tokens(reference_text))
        cand_counts = Counter(candidate)
        
        intersection = set(ref_counts.keys()) & set(cand_counts.keys())
        numerator = sum([ref_counts[x] * cand_counts[x] for x in intersection])
        
        sum1 = sum([ref_counts[x]**2 for x in ref_counts.keys()])
        sum2 = sum([cand_counts[x]**2 for x in cand_counts.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            scores["cosine"] = 0.0
        else:
            scores["cosine"] = float(numerator) / denominator
    except Exception as e:
        print(f"Error calculating Cosine score: {e}")
        
    return scores

def calculate_rag_similarity(visual_contexts: list) -> float:
    """
    Converts ChromaDB L2 distances into a 0-100% similarity score.
    Lower distance = higher similarity. Each frame will get a unique score
    based on whichever images the RAG retriever actually found.
    Returns 0.0 if no RAG context (ablation mode).
    """
    if not visual_contexts:
        return 0.0
    import math
    scores = []
    for ctx in visual_contexts:
        dist = ctx.get('distance', 1.0)
        # Convert L2 distance to similarity: sigmoid-based scale
        # dist=0 -> 100%, dist=0.5 -> ~78%, dist=1.0 -> ~60%, dist=2.0 -> ~37%
        sim = 100.0 * math.exp(-dist * 0.8)
        scores.append(round(sim, 2))
    return round(sum(scores) / len(scores), 2) if scores else 0.0

class GenerationRequest(BaseModel):
    prompt: str
    visual_description: Optional[str] = None
    script_dialogue: Optional[str] = None
    use_rag: bool = True
    base_model_path: Optional[str] = None
    lora_path: Optional[str] = None
    bleu_score: Optional[float] = None
    nlp_scores: Optional[Dict[str, float]] = None
    prompt_technique: str = "zero-shot"

class FullGenerationRequest(BaseModel):
    concept: str
    use_rag: bool = True
    base_model_path: Optional[str] = None
    lora_path: Optional[str] = None
    prompt_technique: str = "zero-shot"

class ModelSyncRequest(BaseModel):
    base_model_id: Optional[str] = "128713" # Placeholder ID for Storyboard Sketch
    lora_id: Optional[str] = "87153" # Placeholder ID for ITS TV Style

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# Semaphore untuk membatasi proses GPU agar tidak concurrent memory OOM pada RTX 3050 (6GB)
gpu_semaphore = asyncio.Semaphore(1)

import re as _re
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def _pick_q_priority_image(visual_contexts: list) -> str:
    """
    Memilih gambar referensi untuk IP-Adapter dengan PRIORITAS Q-series (allaboutITS).
    Dataset allaboutITS (Q*.jpg) lebih berkualitas — foto kampus ITS dengan resolusi tinggi
    dan estetika yang lebih konsisten dibanding P/F/TC-series.
    
    Strategi:
    1. Cari Q-series dari context RAG yang ada
    2. Fallback ke seri lain jika tidak ada Q
    3. Selalu strip suffix _aug{N} sebelum resolve ke filesystem
    """
    if not visual_contexts:
        return None
    
    # Pass 1: cari Q-series dulu (allaboutITS — lebih berkualitas)
    for ctx in visual_contexts:
        src = ctx.get('source', '')
        if not src or src == 'Unknown':
            continue
        src_clean = _re.sub(r'_aug\d+(?=\.)', '', src)
        if src_clean.upper().startswith('Q'):
            candidate = os.path.join(_BASE_DIR, 'ai', 'allaboutITS', src_clean)
            if os.path.exists(candidate):
                print(f"[IP-Adapter] ✅ Q-priority hit: {candidate} (dari src={src})")
                return candidate
    
    # Pass 2: fallback ke seri lain jika tidak ada Q yang cocok
    for ctx in visual_contexts:
        src = ctx.get('source', '')
        if not src or src == 'Unknown':
            continue
        src_clean = _re.sub(r'_aug\d+(?=\.)', '', src)
        for img_dir in ['ai/allaboutITS', 'ai/images_its']:
            candidate = os.path.join(_BASE_DIR, img_dir, src_clean)
            if os.path.exists(candidate):
                print(f"[IP-Adapter] ⚠️ Non-Q fallback: {candidate} (dari src={src})")
                return candidate
    
    print("[IP-Adapter] ❌ Tidak ada referensi gambar yang valid ditemukan.")
    return None

def _rerank_q_first(visual_contexts: list) -> list:
    """
    Re-rank RAG contexts agar Q-series (allaboutITS) naik ke atas.
    Ini memastikan LLM Director mendapat konteks teks dari dataset ITS terbaik.
    """
    if not visual_contexts:
        return visual_contexts
    q_items = [c for c in visual_contexts if c.get('source', '').upper().startswith('Q')]
    non_q_items = [c for c in visual_contexts if not c.get('source', '').upper().startswith('Q')]
    reranked = q_items + non_q_items
    if q_items:
        print(f"[RAG Re-rank] {len(q_items)} Q-series naik ke atas dari total {len(visual_contexts)} konteks.")
    return reranked

async def async_generate_storyboard(task_id: str, prompt: str, visual_description: str = None, use_rag: bool = True, script_dialogue: str = None, base_model_path: str = None, lora_path: str = None, bleu_score: float = None, nlp_scores: dict = None, prompt_technique: str = "zero-shot"):
    """
    Background Task to handle text-to-image generation asynchronously inside a GPU queue.
    """
    async with gpu_semaphore: # Masuk antrean GPU
        try:
            tasks_db[task_id]["status"] = "processing"
            
            rag_sources = []
            image_reference = None
            if use_rag:
                print(f"[Task {task_id}] Fetching visual context via RAG...")
                tasks_db[task_id]["status"] = "rag_search"
                visual_contexts = await get_visual_context(prompt)
            else:
                print(f"[Task {task_id}] Ablation Mode: Skipping RAG...")
                tasks_db[task_id]["status"] = "skip_rag"
                visual_contexts = None
            
            print(f"[Task {task_id}] Enhancing prompt via LLM...")
            
            if visual_contexts:
                # Re-rank: Q-series (allaboutITS) naik ke atas untuk konteks RAG dan IP-Adapter
                visual_contexts = _rerank_q_first(visual_contexts)
                image_reference = _pick_q_priority_image(visual_contexts)
                context_parts = []
                for ctx in visual_contexts:
                    # Membatasi panjang teks konteks RAG maksimal 100 karakter agar tidak melebihi limit token CLIP
                    truncated_text = ctx['text'][:100] + ("..." if len(ctx['text']) > 100 else "")
                    
                    if ctx['source'] != 'Unknown':
                        context_parts.append(f"(File {ctx['source']}: {truncated_text})")
                        rag_sources.append(ctx['source'])
                    else:
                        context_parts.append(f"({truncated_text})")
                
                context_str = ", ".join(context_parts)
            else:
                context_str = ""
                
            enhanced_prompt = await enhance_prompt(
                base_prompt=prompt,
                visual_description=visual_description,
                context_str=context_str,
                technique=prompt_technique
            )
            
            print(f"[Task {task_id}] Queueing Stable Diffusion Generation...")
            tasks_db[task_id]["status"] = "diffusion"
            output_filename = f"outputs/task_{task_id}.png"
            image_path = await generate_image(
                prompt=enhanced_prompt, 
                output_path=output_filename, 
                image_reference=image_reference,
                base_model_path=base_model_path,
                lora_path=lora_path
            )
            
            print(f"[Task {task_id}] Generation completed successfully!")
            
            # Recalculate metrics based on the dynamically generated enhanced_prompt
            reference_text = f"{prompt} {visual_description or ''} {script_dialogue or ''}".strip()
            new_nlp_scores = calculate_nlp_scores(reference_text, enhanced_prompt)
            rag_similarity = calculate_rag_similarity(visual_contexts)
            
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = {
                "image_url": f"/{output_filename}",
                "enhanced_prompt": enhanced_prompt,
                "visual_description": visual_description,
                "rag_context": visual_contexts,
                "rag_sources": rag_sources,
                "mode_ablasi": not use_rag,
                "script_dialogue": script_dialogue,
                "prompt_technique": prompt_technique,
                "bleu_score": new_nlp_scores["bleu4"],
                "nlp_scores": new_nlp_scores,
                "rag_similarity": rag_similarity
            }
            
        except Exception as e:
            print(f"[Task {task_id}] Error in generation queue: {e}")
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["result"] = {"error": str(e)}
        finally:
            # PENTING: Membersihkan VRAM pada RTX 3050 (6GB)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"[Task {task_id}] GPU cache cleared to avoid OOM.")

class UpscaleRequest(BaseModel):
    task_id: str

async def async_upscale_image(task_id: str):
    try:
        if task_id not in tasks_db:
            raise ValueError("Original task not found")
        
        orig_url = tasks_db[task_id]["result"]["image_url"]
        local_input_path = orig_url.lstrip("/")
        local_output_path = f"outputs/upscaled_{task_id}.png"
        
        from services.upscale import upscale_image_sync
        await asyncio.to_thread(upscale_image_sync, local_input_path, local_output_path)
        
        tasks_db[task_id]["status"] = "upscaled"
        tasks_db[task_id]["result"]["upscaled_image_url"] = f"/{local_output_path}"
        
    except Exception as e:
        print(f"Upscale error: {e}")
        tasks_db[task_id]["status"] = "upscale_failed"
        tasks_db[task_id]["result"]["error"] = str(e)

@app.post("/api/upscale", response_model=TaskStatusResponse)
async def start_upscale(req: UpscaleRequest, background_tasks: BackgroundTasks):
    if req.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    tasks_db[req.task_id]["status"] = "upscaling"
    background_tasks.add_task(async_upscale_image, req.task_id)
    return TaskStatusResponse(task_id=req.task_id, status="upscaling")

@app.get("/")
def read_root():
    return {"message": "ITS TV Storyboard API is running. Hardware Profile: RTX 3050 Async Mode."}

async def async_generate_full_storyboard(task_id: str, concept: str, use_rag: bool = True, base_model_path: str = None, lora_path: str = None, prompt_technique: str = "zero-shot"):
    """
    Background Task to handle full automation: Concept -> Script -> Array of Storyboards
    This loops sequentially through each scene, maintaining aggressive VRAM clearing between cycles.
    Tracks state incrementally via log_stream and result_scenes arrays so frontend can print progressively.
    """
    try:
        from services.llm import generate_scenes_from_concept
        tasks_db[task_id]["status"] = "generating_script"
        tasks_db[task_id]["log_stream"].append("LLM: Interpreting concept and expanding into a multi-scene storyboard...")
        
        scenes = await generate_scenes_from_concept(concept)
        if not scenes:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["log_stream"].append("Failed to extract logical scenes from LLM output. Stopping.")
            tasks_db[task_id]["result"] = {"error": "Failed to generate scenes."}
            return
            
        tasks_db[task_id]["total_frames"] = len(scenes)
        tasks_db[task_id]["log_stream"].append(f"Scripts successfully generated! Total Scenes: {len(scenes)}")
        
        # PRE-POPULATE Scenes so UI can render the standard Storyboard Table before image synthesis begins!
        for idx, scene in enumerate(scenes):
            candidate_text = f"{scene.get('deskripsi_adegan', '')} {scene.get('script', '')}"
            nlp_scores = calculate_nlp_scores(concept, candidate_text)
            tasks_db[task_id]["result_scenes"].append({
                "image_url": None,
                "enhanced_prompt": scene.get('prompt_gambar', ""),
                "original_prompt": f"Adegan {scene.get('scene', idx+1)} di {scene.get('keterangan', '')}: {scene.get('deskripsi_adegan', '')}. Shot: {scene.get('shot', '')}",
                "visual_description": scene.get('deskripsi_visual', ""),
                "script_dialogue": scene.get('script', ""),
                "scene_no": scene.get('scene', idx + 1),
                "durasi": scene.get('durasi', "3s"),
                "transisi": scene.get('transisi', "cut to cut"),
                "audio": scene.get('audio', ""),
                "shot_letter": scene.get('shot', ""),
                "keterangan": scene.get('keterangan', ""),
                "id": f"{task_id}_{idx+1}",
                "is_generating": True,
                "bleu_score": nlp_scores["bleu4"],
                "nlp_scores": nlp_scores
            })
        
        for idx, scene in enumerate(scenes):
            tasks_db[task_id]["status"] = "processing_frame"
            tasks_db[task_id]["current_frame"] = idx + 1
            tasks_db[task_id]["log_stream"].append(f"--- INIT FRAME {idx + 1}/{len(scenes)}: Adegan {scene.get('scene', idx+1)} ---")
            
            prompt = f"Adegan {scene.get('scene', idx+1)} di {scene.get('keterangan', '')}: {scene.get('deskripsi_adegan', '')}. Shot: {scene.get('shot', '')}"
            visual_description = scene.get('deskripsi_visual', "")
            script_dialogue = scene.get('script', "")
            
            rag_sources = []
            image_reference = None
            if use_rag:
                tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: Searching Context for Background Assets (RAG)...")
                visual_contexts = await get_visual_context(prompt)
            else:
                tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: Ablation Mode: Skipping RAG...")
                visual_contexts = None
                
            tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: Director AI generating SD Prompt...")
            if visual_contexts:
                # Re-rank: Q-series (allaboutITS) naik ke atas untuk konteks RAG dan IP-Adapter
                visual_contexts = _rerank_q_first(visual_contexts)
                image_reference = _pick_q_priority_image(visual_contexts)
                tasks_db[task_id]["log_stream"].append(
                    f"Frame {idx + 1}: IP-Adapter ref → {os.path.basename(image_reference) if image_reference else 'None (no Q-series found)'}"
                )
                context_parts = []
                for ctx in visual_contexts:
                    ctx_text = ctx.get('text', ctx.get('caption', ''))
                    context_parts.append(f"{ctx.get('source', '')} ({ctx_text})")
                    rag_sources.append({"source": ctx.get('source', ''), "caption": ctx_text})
                context_str = ", ".join(context_parts)
            else:
                context_str = ""
                
            enhanced_prompt = await enhance_prompt(prompt, visual_description, context_str, technique=prompt_technique)
            
            # Wait 3 seconds to let Ollama physically drop its model from VRAM 
            # before initializing PyTorch operations for Stable Diffusion!
            tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: Pausing (3s) to allow Ollama VRAM unload...")
            await asyncio.sleep(3.0)
            
            tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: Synthesizing Visuals (Diffusion SD 1.5)...")
            async with gpu_semaphore:
                from services.diffusion import generate_image
                output_filename = f"task_{task_id}_frame_{idx+1}.png"
                output_path = os.path.join("outputs", output_filename)
                
                final_image_path = await generate_image(
                    prompt=enhanced_prompt,
                    output_path=output_path,
                    image_reference=image_reference,
                    base_model_path=base_model_path,
                    lora_path=lora_path
                )
            
            reference_text = f"{prompt} {visual_description or ''} {script_dialogue or ''}".strip()
            new_nlp_scores = calculate_nlp_scores(reference_text, enhanced_prompt)
            rag_similarity = calculate_rag_similarity(visual_contexts)
            frame_result = {
                "id": f"{task_id}_{idx+1}",
                "image_url": f"/outputs/{output_filename}",
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "script_dialogue": script_dialogue,
                "visual_description": visual_description,
                "scene_no": scene.get('scene', idx + 1),
                "durasi": scene.get('durasi', "3s"),
                "transisi": scene.get('transisi', "cut to cut"),
                "audio": scene.get('audio', ""),
                "shot_letter": scene.get('shot', ""),
                "keterangan": scene.get('keterangan', ""),
                "rag_context": visual_contexts,
                "rag_sources": rag_sources,
                "mode_ablasi": not use_rag,
                "prompt_technique": prompt_technique,
                "is_generating": False,
                "bleu_score": new_nlp_scores["bleu4"],
                "nlp_scores": new_nlp_scores,
                "rag_similarity": rag_similarity
            }
            
            # Since this takes 15 mins, we push incremental results!
            tasks_db[task_id]["result_scenes"][idx] = frame_result
            tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1} completed and saved!")
            
            import gc
            gc.collect()
            
            import ctypes
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
                tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: POSIX System RAM Flushed natively.")
            except Exception:
                pass
                
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                tasks_db[task_id]["log_stream"].append(f"Frame {idx + 1}: GPU Hardware Memory Flushed.")

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["log_stream"].append("Full Storyboard Generation Cycle Completed Successfully!")
        
    except Exception as e:
        print(f"[Task {task_id}] Full Processing failed: {e}")
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["result"] = {"error": str(e)}
        tasks_db[task_id]["log_stream"].append(f"CRITICAL ERROR: {str(e)}")

@app.post("/api/generate", response_model=TaskStatusResponse)
async def start_generation(req: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to receive generation prompt. Starts a background task and returns a task_id immediately.
    """
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "pending", "result": None}
    
    # Offload heavy AI processing to background task
    background_tasks.add_task(async_generate_storyboard, task_id, req.prompt, req.visual_description, req.use_rag, req.script_dialogue, req.base_model_path, req.lora_path, req.bleu_score, req.nlp_scores, req.prompt_technique)
    
    return TaskStatusResponse(task_id=task_id, status="pending")

@app.post("/api/generate-full")
async def generate_full_storyboard(req: FullGenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    tasks_db[task_id] = {
        "status": "pending", 
        "result_scenes": [], 
        "log_stream": ["Task Initiated. Awaiting Worker..."],
        "current_frame": 0,
        "total_frames": 0,
        "result": None
    }
    
    background_tasks.add_task(async_generate_full_storyboard, task_id, req.concept, req.use_rag, req.base_model_path, req.lora_path, req.prompt_technique)
    
    return TaskStatusResponse(task_id=task_id, status="pending")

async def async_sync_models(task_id: str, base_id: Optional[str], lora_id: Optional[str]):
    from services.diffusion import download_model_from_civitai
    import traceback
    
    def make_progress_cb(label):
        def cb(msg):
            # Update the last log line with download progress
            progress_line = f"[{label}] {msg}"
            if tasks_db[task_id]["log_stream"] and tasks_db[task_id]["log_stream"][-1].startswith(f"[{label}]"):
                tasks_db[task_id]["log_stream"][-1] = progress_line
            else:
                tasks_db[task_id]["log_stream"].append(progress_line)
        return cb
    
    try:
        tasks_db[task_id]["status"] = "syncing_base_model"
        if base_id:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"base_{base_id}.safetensors")
            if os.path.exists(base_path):
                tasks_db[task_id]["log_stream"].append(f"Base Model {base_id} already exists. Skipping download.")
            else:
                tasks_db[task_id]["log_stream"].append(f"Downloading Base Model {base_id} (~2GB)...")
                await download_model_from_civitai(base_id, base_path, is_lora=False, progress_callback=make_progress_cb("BASE"))
            tasks_db[task_id]["result"]["base_model_path"] = base_path
            
        tasks_db[task_id]["status"] = "syncing_lora"
        if lora_id:
            lora_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"lora_{lora_id}.safetensors")
            if os.path.exists(lora_path):
                tasks_db[task_id]["log_stream"].append(f"LoRA {lora_id} already exists. Skipping download.")
            else:
                tasks_db[task_id]["log_stream"].append(f"Downloading LoRA {lora_id}...")
                await download_model_from_civitai(lora_id, lora_path, is_lora=True, progress_callback=make_progress_cb("LORA"))
            tasks_db[task_id]["result"]["lora_path"] = lora_path
            
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["log_stream"].append("✅ All models synced successfully!")
    except Exception as e:
        full_error = traceback.format_exc()
        print(f"[Sync Error] {full_error}")
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["result"]["error"] = str(e) or full_error
        tasks_db[task_id]["log_stream"].append(f"CRITICAL ERROR: {str(e) or full_error}")

@app.post("/api/models/sync", response_model=TaskStatusResponse)
async def sync_models(req: ModelSyncRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "status": "pending",
        "result": {},
        "log_stream": ["Model sync task initiated."]
    }
    background_tasks.add_task(async_sync_models, task_id, req.base_model_id, req.lora_id)
    return TaskStatusResponse(task_id=task_id, status="pending")

@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    """
    Endpoint to check the status of a generation task. 
    Extended to return Live Log Stream and Array of completed Scenes.
    """
    if task_id not in tasks_db:
        return TaskStatusResponse(task_id=task_id, status="not_found")
    
    task_data = tasks_db[task_id]
    
    # We dynamically return different structures based on if it's a full-auto or single-shot
    return {
        "task_id": task_id,
        "status": task_data["status"],
        "result": task_data["result"],
        "log_stream": task_data.get("log_stream"),
        "result_scenes": task_data.get("result_scenes"),
        "current_frame": task_data.get("current_frame"),
        "total_frames": task_data.get("total_frames")
    }

@app.post("/api/ingest-csv")
async def api_ingest_csv(background_tasks: BackgroundTasks):
    from services.rag import ingest_csv
    import os
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai", "caption.csv")
    background_tasks.add_task(ingest_csv, csv_path)
    return {"message": "CSV ingestion started in background"}

@app.post("/api/upload-script")
async def upload_script(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        contents = await file.read()
        pdf_document = fitz.open(stream=contents, filetype="pdf")
        
        extracted_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            extracted_text += page.get_text()
            
        pdf_document.close()
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
            
        from services.llm import analyze_script_to_scenes
        scenes = await analyze_script_to_scenes(extracted_text)
        
        if not scenes:
            raise HTTPException(status_code=400, detail="Gagal mengekstrak scene dari naskah. Silakan coba lagi atau cek format naskah.")
        
        try:
            for s in scenes:
                candidate_text = f"{s.get('deskripsi_adegan', '')} {s.get('script', '')}"
                nlp_scores = calculate_nlp_scores(extracted_text, candidate_text)
                s["nlp_scores"] = nlp_scores
                # Maintain compatibility by setting bleu_score to bleu4
                s["bleu_score"] = nlp_scores["bleu4"]
                    
            generated_text = " ".join([f"{s.get('deskripsi_adegan', '')} {s.get('script', '')}" for s in scenes])
            overall_scores = calculate_nlp_scores(extracted_text, generated_text)
            print(f"Calculated Optimized Overall NLP Scores: {overall_scores}")
        except Exception as bleu_err:
            print(f"Failed to calculate optimized NLP scores: {bleu_err}")
            for s in scenes:
                s["nlp_scores"] = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rougeL": 0.0, "cosine": 0.0}
                s["bleu_score"] = 0.0
            overall_scores = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rougeL": 0.0, "cosine": 0.0}

        return {"filename": file.filename, "scenes": scenes, "bleu_score": overall_scores["bleu4"], "nlp_scores": overall_scores}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
