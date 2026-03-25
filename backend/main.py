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

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# Semaphore untuk membatasi proses GPU agar tidak concurrent memory OOM pada RTX 3050 (6GB)
gpu_semaphore = asyncio.Semaphore(1)

async def async_generate_storyboard(task_id: str, prompt: str):
    """
    Background Task to handle text-to-image generation asynchronously inside a GPU queue.
    """
    async with gpu_semaphore: # Masuk antrean GPU
        try:
            tasks_db[task_id]["status"] = "processing"
            
            print(f"[Task {task_id}] Fetching visual context via RAG...")
            visual_context = await get_visual_context(prompt)
            
            print(f"[Task {task_id}] Enhancing prompt via LLM...")
            contextual_prompt = f"{prompt}. {visual_context}" if visual_context else prompt
            enhanced_prompt = await enhance_prompt(contextual_prompt)
            
            print(f"[Task {task_id}] Queueing Stable Diffusion Generation...")
            output_filename = f"outputs/task_{task_id}.png"
            image_path = await generate_image(enhanced_prompt, output_filename)
            
            print(f"[Task {task_id}] Generation completed successfully!")
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = {
                "image_url": f"/{output_filename}",
                "enhanced_prompt": enhanced_prompt,
                "rag_context": visual_context
            }
            
        except Exception as e:
            print(f"[Task {task_id}] Error in generation queue: {e}")
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["result"] = {"error": str(e)}

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
    background_tasks.add_task(async_generate_storyboard, task_id, req.prompt)
    
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
        
        return {"filename": file.filename, "scenes": scenes}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
