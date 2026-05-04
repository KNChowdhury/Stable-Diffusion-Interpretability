import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Model & Setup
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
).to("cpu")

# 2. Token Analysis Function
def get_tokens(prompt):
    return pipe.tokenizer.tokenize(prompt)

# 3. Attention Hook
attention_maps = []
def hook_fn(module, input, output):
    # Cross-attention weights capture
    attention_maps.append(output.detach().cpu())

# Register hook to the cross-attention layer
layer = pipe.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn2
layer.register_forward_hook(hook_fn)

# 4. Run Generation
prompt = "A shiny blue robot in a library"
tokens = get_tokens(prompt)
print(f"Tokens detected: {tokens}")

image = pipe(prompt, num_inference_steps=10).images[0]
image.save("original_image.png")

# 5. Pro Visualization (Overlay)
if attention_maps:
    # Get last step attention
    attn = attention_maps[-1][0] # Shape: (Heads * Pixels, Tokens)
    res = int(attn.shape[0]**0.5)
    
    # Analyze first 5-8 tokens (ignoring start/end tokens)
    num_to_show = min(len(tokens), 6)
    fig, axes = plt.subplots(1, num_to_show, figsize=(20, 5))
    
    for i in range(num_to_show):
        # Extract map for specific token
        token_map = attn[:, i+1].view(res, res).numpy() # i+1 to skip <start>
        
        # Resize map to match image
        token_map_rescaled = np.array(Image.fromarray(token_map).resize((512, 512), resample=Image.BICUBIC))
        
        # Plotting Overlay
        axes[i].imshow(image)
        axes[i].imshow(token_map_rescaled, cmap='jet', alpha=0.5) # Alpha blending for transparency
        axes[i].set_title(f"Token: {tokens[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("pro_token_visualization.png")
    print("Success! 'pro_token_visualization.png' check koren.")