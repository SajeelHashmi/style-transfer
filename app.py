import uvicorn
import uuid
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from celery_app import generate_image_task, generate_style_transfer_task
from starlette.responses import JSONResponse
from typing import Optional
import traceback

app = FastAPI(title="AI Image Generation API")

# Add CORS middleware to allow cross-origin requests if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active tasks for cleanup if needed
active_tasks = {}

@app.post("/generate")
async def generate_image(prompt: str, background_tasks: BackgroundTasks):
    try:
        task_id = str(uuid.uuid4())
        task = generate_image_task.apply_async(args=[prompt], task_id=task_id)
        active_tasks[task_id] = task
        
        # Add task to cleanup after a timeout period if desired
        # background_tasks.add_task(cleanup_task, task_id, timeout=3600)
        
        print(f"Returning task id {task.id} for image generation")
        return {"task_id": task.id, "status": "Processing"}
    except Exception as e:
        print(f"Error submitting image generation task: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

@app.post("/style_transfer")
async def style_transfer(
    style_image: UploadFile = File(...),
    source_image: UploadFile = File(...),
    prompt: str = Form(""),
    scale: float = Form(1.0),
    control_scale: float = Form(0.5),
    guidance_scale: float = Form(5.0),
    layout_enabled: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    try:
        task_id = str(uuid.uuid4())
        
        # Read files into memory
        style_bytes = await style_image.read()
        source_bytes = await source_image.read()
        
        task = generate_style_transfer_task.apply_async(
            args=[prompt, style_bytes, source_bytes, scale, control_scale, guidance_scale, layout_enabled],
            task_id=task_id
        )
        active_tasks[task_id] = task
        
        print(f"Returning Task ID for Style Transfer {task.id}")
        return {"task_id": task.id, "status": "Processing"}
    except Exception as e:
        print(f"Error submitting style transfer task: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

@app.get("/result/{task_id}")
async def get_task_result(task_id: str, return_format: Optional[str] = "binary"):
    try:
        task_result = AsyncResult(task_id)
        
        if task_result.state == "PENDING":
            return JSONResponse(content={"task_id": task_id, "status": "Processing"}, status_code=202)
        elif task_result.state == "SUCCESS":
            result = task_result.result
            
            # Option to return as base64 for direct embedding in frontend
            if return_format == "base64":
                if isinstance(result, bytes):
                    base64_image = base64.b64encode(result).decode('utf-8')
                    return JSONResponse(content={
                        "task_id": task_id, 
                        "status": "Completed", 
                        "image_data": f"data:image/png;base64,{base64_image}"
                    })
            
            # Default binary response
            return JSONResponse(content={
                "task_id": task_id, 
                "status": "Completed", 
                "image": result
            })
        elif task_result.state == "FAILURE":
            error_msg = str(task_result.info) if task_result.info else "Unknown error"
            return JSONResponse(content={
                "task_id": task_id, 
                "status": "Failed", 
                "error": error_msg
            }, status_code=500)
        
        # Any other state (REVOKED, RETRY, etc.)
        return JSONResponse(content={
            "task_id": task_id, 
            "status": task_result.state
        }, status_code=200)
    except Exception as e:
        print(f"Error retrieving task result: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={
            "task_id": task_id, 
            "status": "Error", 
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

async def cleanup_task(task_id: str, timeout: int = 3600):
    """Background task to clean up old tasks"""
    import asyncio
    await asyncio.sleep(timeout)
    if task_id in active_tasks:
        del active_tasks[task_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)