from celery import Celery
import torch
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from model_singleton import ModelSingleton

# Initialize Celery
celery = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
model_singleton = ModelSingleton()
model_singleton.get_ip_adapter()  # Preload the model to avoid delays in task execution
# Configure Celery
celery.conf.update(
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10,  # Restart worker after processing this many tasks (helps with memory leaks)
    task_acks_late=True,  # Tasks are acknowledged after the task is executed
    task_reject_on_worker_lost=True  # Ensure tasks are re-queued if worker dies
)

def resize_img(input_image, max_side=1024):
    input_image = input_image.convert("RGB")
    input_image = input_image.resize((max_side, max_side), Image.BILINEAR)
    return input_image

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

@celery.task
def generate_image_task(prompt):
    try:
        # Get the singleton instance with initialized models
        model_singleton = ModelSingleton()
        pipe = model_singleton.get_sd3_pipeline()
        
        # Generate the image
        image = pipe(prompt).images[0]
        
        # Convert to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        return img_bytes.getvalue()
    except Exception as e:
        # Capture and log the error
        import traceback
        error_message = f"Error in generate_image_task: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise Exception(error_message)

@celery.task
def generate_style_transfer_task(prompt, style_image_bytes, source_image_bytes, scale, control_scale, guidance_scale, layout_enabled=False):
    try:
        # Get the singleton instance with initialized models
        model_singleton = ModelSingleton()
        pipe_controlnet = model_singleton.get_controlnet_pipeline()
        ip_model = model_singleton.get_ip_adapter(layout_enabled=layout_enabled)
        
        # Process input images
        style_pil = Image.open(BytesIO(style_image_bytes))
        style_pil = resize_img(style_pil, max_side=1024)
        
        source_pil = Image.open(BytesIO(source_image_bytes))
        source_pil = resize_img(source_pil, max_side=1024)
        
        # Create canny map for controlnet
        cv_input_image = pil_to_cv2(source_pil)
        detected_map = cv2.Canny(cv_input_image, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
        
        # Generate stylized image
        images = ip_model.generate(
            pil_image=style_pil,
            prompt=prompt,
            negative_prompt="text, watermark, lowres, deformed",
            scale=scale,
            guidance_scale=guidance_scale,
            num_samples=1,
            num_inference_steps=20,
            seed=42,
            image=canny_map,
            controlnet_conditioning_scale=float(control_scale),
        )
        
        # Convert output to bytes
        img_bytes = BytesIO()
        images[0].save(img_bytes, format="PNG")
        return img_bytes.getvalue()
    except Exception as e:
        # Capture and log the error
        import traceback
        error_message = f"Error in generate_style_transfer_task: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise Exception(error_message)