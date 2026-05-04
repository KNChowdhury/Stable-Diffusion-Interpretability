from diffusers import StableDiffusionPipeline
import torch

# 1. Load Model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# 2. Pro-Level Prompts (Simple English keywords for better texture)
PROMPT = "masterpiece, best quality, 8k, a high-resolution robot made of liquid mercury, highly reflective chrome surface, molten metal, cinematic lighting, ray-traced reflections, futuristic, solid silhouette"
NEG_PROMPT = "matte, plastic, skin, human, blurry, grainy, deformed, low quality, bad anatomy"

# 3. Settings for High Quality
image = pipe(
    PROMPT, 
    negative_prompt=NEG_PROMPT, 
    num_inference_steps=50, 
    guidance_scale=10.5, # High guidance for better prompt accuracy
    width=768, 
    height=768
).images[0]

image.save("pro_mercury_output.png")