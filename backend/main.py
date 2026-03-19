from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uuid
import fitz

app = FastAPI(title="ITS TV Storyboard API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=False, # Must be False if allow_origins is ["*"]
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

async def async_generate_storyboard(task_id: str, prompt: str):
    """
    Background Task to handle text-to-image generation asynchronously.
    This prevents the main event loop from blocking while the RTX 3050 processes the request.
    """
    try:
        tasks_db[task_id]["status"] = "processing"
        
        # TODO: integrate Ollama here to enhance prompt
        # from services.llm import enhance_prompt
        # enhanced_prompt = await enhance_prompt(prompt)
        
        # Simulate LLM processing time
        await asyncio.sleep(2)
        
        # TODO: integrate Diffusers here to generate image
        # from services.diffusion import generate_image
        # image_url = await generate_image(enhanced_prompt)
        
        # Simulate rendering time
        await asyncio.sleep(3)
        
        # Fake successful response
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = {
            "image_url": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=2564&auto=format&fit=crop",
            "enhanced_prompt": prompt + " (enhanced)"
        }
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["result"] = {"error": str(e)}

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
