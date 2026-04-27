import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

# 1. Model Load (Safety Checker OFF kora hoyeche jate black image na ashe)
print("Loading model for visualization...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32,
    safety_checker=None, 
    requires_safety_checker=False
).to("cpu")

# 2. Attention Hook - Model-er layer theke data capture korar jonno
attention_maps = []
def hook_fn(module, input, output):
    # Output[0] holo attention weights
    attention_maps.append(output.detach().cpu())

# 'attn2' layer-e hook lagano (Eita Cross-Attention layer)
# Ai layer-tei model text ar image-er moddhe somporko gore tole
layer_to_hook = pipe.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2
layer_to_hook.register_forward_hook(hook_fn)

# 3. Prompt & Generation
# Lead-ke dekhate paren kivabe 'blue' ba 'robot' shobdo gulo impact korche
prompt = "A shiny blue robot in a library" 
print("Generating and capturing attention (CPU mode)...")

# Steps 5-10 dile heatmap-ta arektu porishkar hoy
image = pipe(prompt, num_inference_steps=10).images[0]

# 4. Save Image & Heatmap
image.save("pnp_test_result.png")

if attention_maps:
    # Prothom attention head-ta nibo visualization-er jonno
    attn = attention_maps[-1][0] # Sobcheye shesh step-er attention
    res = int(attn.shape[0]**0.5)
    vis_map = attn.view(res, res, -1).mean(dim=-1).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_map, cmap='jet')
    plt.title(f"Cross-Attention Heatmap\nPrompt: {prompt}")
    plt.colorbar(label='Attention Intensity')
    plt.savefig("attention_heatmap.png")
    print("\nWell done,Kawsar! 'pnp_test_result.png' and 'attention_heatmap.png' check Now.")