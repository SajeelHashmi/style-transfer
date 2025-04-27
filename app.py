import uvicorn
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from celery.result import AsyncResult
from celery_app import generate_image_task, generate_style_transfer_task
from starlette.responses import JSONResponse

app =FastAPI()
@app.post("/generate")
async def generate_image(prompt:str):
    task_id = str(uuid.uuid4())
    task = generate_image_task.apply_async(args=[prompt],task_id=task_id)
    print(f"Returning task id {task.id} for image generation")
    return {"task_id":task.id,"status":"Processing"}


@app.post("/style_transfer")
async def style_transfer(
    style_image: UploadFile = File(...),
    source_image: UploadFile = File(...),
    prompt: str = Form(""),
    scale: float = Form(1.0),
    control_scale: float = Form(0.5),
    guidance_scale: float = Form(5.0),
):
    task_id = str(uuid.uuid4())  # Generate a unique task ID
    task = generate_style_transfer_task.apply_async(
        args=[prompt, await style_image.read(), await source_image.read(), scale, control_scale, guidance_scale],
        task_id=task_id
    )
    print(f"Returning Tak ID for Style Transfer {task.id}")
    return {"task_id": task.id, "status": "Processing"}



@app.get("/result/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id)

    if task_result.state == "PENDING":
        return JSONResponse(content={"task_id": task_id, "status": "Processing"}, status_code=202)
    elif task_result.state == "SUCCESS":
        return JSONResponse(content={"task_id": task_id, "status": "Completed", "image": task_result.result})
    elif task_result.state == "FAILURE":
        return JSONResponse(content={"task_id": task_id, "status": "Failed", "error": str(task_result.info)}, status_code=500)

    return JSONResponse(content={"task_id": task_id, "status": task_result.state}, status_code=200)
                                                                            
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
                                                   