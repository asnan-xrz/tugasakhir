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

class GenerationRequest(BaseModel):
    prompt: str
    visual_description: Optional[str] = None
    script_dialogue: Optional[str] = None
    use_rag: bool = True

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# Semaphore untuk membatasi proses GPU agar tidak concurrent memory OOM pada RTX 3050 (6GB)
gpu_semaphore = asyncio.Semaphore(1)

async def async_generate_storyboard(task_id: str, prompt: str, visual_description: str = None, use_rag: bool = True, script_dialogue: str = None):
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
                context_parts = []
                for ctx in visual_contexts:
                    # Membatasi panjang teks konteks RAG maksimal 100 karakter agar tidak melebihi limit token CLIP
                    truncated_text = ctx['text'][:100] + ("..." if len(ctx['text']) > 100 else "")
                    
                    if ctx['source'] != 'Unknown':
                        context_parts.append(f"(File {ctx['source']}: {truncated_text})")
                        rag_sources.append(ctx['source'])
                        
                        # Set image reference for ControlNet/IP-Adapter (ambil yang pertama valid)
                        if image_reference is None:
                            base_dir = os.path.dirname(os.path.dirname(__file__))
                            img_name = ctx['source']
                            if not (img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png') or img_name.lower().endswith('.jpeg')):
                                image_candidates = [
                                    os.path.join(base_dir, "ai", "images", f"{img_name}.jpg"),
                                    os.path.join(base_dir, "ai", "images", f"{img_name}.png"),
                                    os.path.join(base_dir, "ai", "images", f"{img_name}.jpeg")
                                ]
                            else:
                                image_candidates = [os.path.join(base_dir, "ai", "images", img_name)]
                                
                            for c_path in image_candidates:
                                if os.path.exists(c_path):
                                    image_reference = c_path
                                    break
                    else:
                        context_parts.append(f"({truncated_text})")
                
                context_str = ", ".join(context_parts)
            else:
                context_str = ""
                
            enhanced_prompt = await enhance_prompt(
                base_prompt=prompt,
                visual_description=visual_description,
                context_str=context_str
            )
            
            print(f"[Task {task_id}] Queueing Stable Diffusion Generation...")
            tasks_db[task_id]["status"] = "diffusion"
            output_filename = f"outputs/task_{task_id}.png"
            image_path = await generate_image(enhanced_prompt, output_filename, image_reference)
            
            print(f"[Task {task_id}] Generation completed successfully!")
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = {
                "image_url": f"/{output_filename}",
                "enhanced_prompt": enhanced_prompt,
                "visual_description": visual_description,
                "rag_context": visual_contexts,
                "rag_sources": rag_sources, # Menyimpan metadata visual
                "mode_ablasi": not use_rag,
                "script_dialogue": script_dialogue
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
        tasks_db[task_id]["status"] = "upscale_failed"
        tasks_db[task_id]["result"] = {"error": str(e)}

@app.post("/api/upscale", response_model=TaskStatusResponse)
async def start_upscale(req: UpscaleRequest, background_tasks: BackgroundTasks):
    if req.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    tasks_db[req.task_id]["status"] = "upscaling"
    background_tasks.add_task(async_upscale_image, req.task_id)
    return TaskStatusResponse(task_id=req.task_id, status="upscaling")

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
        tasks_db[task_id]["status"] = "upscale_failed"
        tasks_db[task_id]["result"] = {"error": str(e)}

@app.post("/api/upscale", response_model=TaskStatusResponse)
async def start_upscale(req: UpscaleRequest, background_tasks: BackgroundTasks):
    if req.task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    tasks_db[req.task_id]["status"] = "upscaling"
    background_tasks.add_task(async_upscale_image, req.task_id)
    return TaskStatusResponse(task_id=req.task_id, status="upscaling")

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

@app.post("/api/generate", response_model=TaskStatusResponse)
async def start_generation(req: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to receive generation prompt. Starts a background task and returns a task_id immediately.
    """
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "pending", "result": None}
    
    # Offload heavy AI processing to background task
    background_tasks.add_task(async_generate_storyboard, task_id, req.prompt, req.visual_description, req.use_rag, req.script_dialogue)
    
    return TaskStatusResponse(task_id=task_id, status="pending")

@app.get("/api/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Endpoint to check the status of a generation task.
    """
    if task_id not in tasks_db:
        return TaskStatusResponse(task_id=task_id, status="not_found")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=tasks_db[task_id]["status"],
        result=tasks_db[task_id]["result"]
    )

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
        
        return {"filename": file.filename, "scenes": scenes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
