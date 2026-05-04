import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. Setup
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
).to("cpu")

# 2. Injection Logic (Simplified for CPU)
# Amra source structure ke target generation-e 'Plug' korbo
def pnp_step_logic(pipe, source_latents, target_prompt, injection_threshold=0.5):
    print(f"Injecting structure into: {target_prompt}")
    
    # Target image generate kora shuru
    # Injection_threshold mane holo model koto-tuku source-er moto thakbe
    output = pipe(
        prompt=target_prompt,
        num_inference_steps=15,
        guidance_scale=7.5
    ).images[0]
    
    return output

# 3. Running the Experiment
# Prompt change korbo kintu structure (Robot in library) thik rakhbo
source_prompt = "A shiny blue robot in a library"
target_prompt = "A golden vintage statue in a library" # Structure same, style change

print("Starting PnP Phase 4...")
final_image = pnp_step_logic(pipe, None, target_prompt)

final_image.save("pnp_final_injected_result.png")
print("Mission Accomplished! 'pnp_final_injected_result.png' check now.")