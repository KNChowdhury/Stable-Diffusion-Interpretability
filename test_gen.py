import torch
from diffusers import StableDiffusionPipeline

print("Model load hochhe... Please wait.")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
pipe = pipe.to("cpu")

prompt = "A high-quality photo of a futuristic robot in a library"

print("Chhobi banano shuru holo (CPU mode)... ektu shomoy nibe.")
image = pipe(prompt, num_inference_steps=20).images[0]

image.save("result_cpu.png")
print("Kaj shesh! 'result_cpu.png' file-ta check koren.")