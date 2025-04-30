import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusion3Pipeline
from InstantStyle.ip_adapter import IPAdapterXL

class ModelSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Model paths
        self.base_model_path = "./stable-diffusion-3.5-large"
        self.base_model_path_xl = "stabilityai/stable-diffusion-xl-base-1.0"
        self.controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
        self.image_encoder_path = "./InstantStyle/sdxl_models/image_encoder"
        self.ip_ckpt = "./InstantStyle/sdxl_models/ip-adapter_sdxl.bin"
        
        # Initialize models as None
        self.pipe_stable_diffusion = None
        self.pipe_controlnet = None
        self.ip_model = None
        
        self._initialized = True
    
    def get_sd3_pipeline(self):
        """Lazy loading of SD3 pipeline"""
        if self.pipe_stable_diffusion is None:
            print("Initializing SD3 pipeline...")
            self.pipe_stable_diffusion = StableDiffusion3Pipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            )
            self.pipe_stable_diffusion = self.pipe_stable_diffusion.to(
                self.device, 
                memory_format=torch.channels_last
            )
        return self.pipe_stable_diffusion
    
    def get_controlnet_pipeline(self):
        """Lazy loading of ControlNet pipeline"""
        if self.pipe_controlnet is None:
            print("Initializing ControlNet pipeline...")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_path, 
                use_safetensors=False, 
                torch_dtype=self.dtype
            ).to(self.device, memory_format=torch.channels_last)
            
            self.pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.base_model_path_xl,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                add_watermarker=False,
            ).to(self.device, memory_format=torch.channels_last)
        return self.pipe_controlnet
    
    def get_ip_adapter(self, layout_enabled=False):
        """Lazy loading of IP-Adapter"""
        # Make sure controlnet is initialized first
        self.get_controlnet_pipeline()
        
        if self.ip_model is None:
            print("Initializing IP-Adapter...")
            blocks = ["up_blocks.0.attentions.1"]
            if layout_enabled:
                blocks.append("down_blocks.2.attentions.1")
                
            self.ip_model = IPAdapterXL(
                self.pipe_controlnet, 
                self.image_encoder_path, 
                self.ip_ckpt, 
                self.device, 
                target_blocks=blocks
            )
        return self.ip_model