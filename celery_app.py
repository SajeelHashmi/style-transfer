from celery import Celery
import torch
from PIL import Image
import numpy as np
import cv2
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline,StableDiffusion3Pipeline
from InstantStyle.ip_adapter import IPAdapterXL
from io import BytesIO


celery = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

base_model_path = "./stable-diffusion-3.5-large"
base_model_path_ = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
image_encoder_path = "./InstantStyle/sdxl_models/image_encoder"
ip_ckpt = "./InstantStyle/sdxl_models/ip-adapter_sdxl.bin"




controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=dtype).to(device,memory_format=torch.channels_last)

pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path_,
        controlnet=controlnet,
        torch_dtype=dtype,
        add_watermarker=False,
    ).to(device,memory_format=torch.channels_last)

pipe_stable_diffusion = StableDiffusion3Pipeline.from_pretrained(
            base_model_path,    
        torch_dtype=torch.float16,
            low_cpu_mem_usage=True)
pipe_stable_diffusion = pipe_stable_diffusion.to(device,memory_format=torch.channels_last)


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
    image = pipe_stable_diffusion(prompt).images[0]
    
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    return img_bytes.getvalue()  # Returning binary image data

@celery.task
def generate_style_transfer_task(prompt, style_image_bytes, source_image_bytes, scale, control_scale, guidance_scale,layout_enabled=False):

    blocks = ["up_blocks.0.attentions.1"]    
    if layout_enabled:
        blocks.append("down_blocks.2.attentions.1")
    
    ip_model = IPAdapterXL(pipe_controlnet, image_encoder_path, ip_ckpt, device, target_blocks=blocks)
    
    style_pil = Image.open(BytesIO(style_image_bytes))
    style_pil = resize_img(style_pil, max_side=1024)
    
    source_pil = Image.open(BytesIO(source_image_bytes))
    source_pil = resize_img(source_pil, max_side=1024)
    
    
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
    return img_bytes.getvalue()  # Returning binary image data
