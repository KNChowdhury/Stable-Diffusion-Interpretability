import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
).to("cpu")

# 2. Self-Attention Hook (attn1 targets the geometry/structure)
self_attention_maps = []
def hook_fn(module, input, output):
    self_attention_maps.append(output.detach().cpu())

# 'attn1' holo Self-Attention, jeta object-er structure dhore rakhe
layer_to_hook = pipe.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1
layer_to_hook.register_forward_hook(hook_fn)

# 3. Execution
prompt = "A shiny blue robot in a library"
print("Extracting Structural Self-Attention (Phase 3)...")
image = pipe(prompt, num_inference_steps=10).images[0]

# 4. Visualization of Structure
if self_attention_maps:
    # Self-attention map is a square matrix of pixel-to-pixel relationships
    # We take a slice to see the layout
    attn = self_attention_maps[-1][0] 
    res = int(attn.shape[0]**0.5)
    
    # Averaging heads to get the 'Structural Skeleton'
    vis_map = attn.view(res, res, -1).mean(dim=-1).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_map, cmap='magma') # 'magma' gives a more structural/x-ray look
    plt.title("Self-Attention Map (Object Structure & Geometry)")
    plt.colorbar()
    plt.savefig("structural_geometry_map.png")
    print("Success! 'structural_geometry_map.png' generated.")