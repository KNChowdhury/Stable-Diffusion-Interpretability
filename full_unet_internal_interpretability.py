import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline

# 1. SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 2. LOAD MODEL
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE).to(DEVICE)

# 3. GLOBAL STORAGE & HOOKS
internal_data = [] # List to keep order
def get_hook(name):
    def hook(m, i, o):
        data = o[0] if isinstance(o, tuple) else o
        internal_data.append((name, data.detach().cpu().numpy()))
    return hook

# 4. HOOKING EVERYTHING (Lead's Requirement)
# Encoder Internal Patterns
for i in range(3):
    pipe.unet.down_blocks[i].register_forward_hook(get_hook(f"Down_{i}"))

# Bottleneck
pipe.unet.mid_block.register_forward_hook(get_hook("Mid_Bottleneck"))

# Decoder Internal Patterns (Deepest details eikhane)
for i in range(3):
    pipe.unet.up_blocks[i].register_forward_hook(get_hook(f"Up_{i}"))

# 5. RUN INFERENCE
prompt = "(masterpiece, 8k, portrait of a human face made of liquid mercury, flowing molten chrome, high reflection:1.4)"
print("🧬 Extracting ALL available internal patterns...")
pipe(prompt, num_inference_steps=30)

# 6. PCA & MATRIX GRID (Visualizing the Internal Flow)
def get_pca(feat):
    f = feat[0] if feat.ndim == 4 else feat
    c, h, w = f.shape
    pca = PCA(n_components=3)
    res = pca.fit_transform(f.reshape(c, -1).T)
    res = (res - res.min()) / (res.max() - res.min())
    return res.reshape(h, w, 3)

# Creating a grid for all captured layers
num_layers = len(internal_data)
fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

for i, (name, feat) in enumerate(internal_data):
    pca_img = get_pca(feat)
    axes[i].imshow(pca_img)
    axes[i].set_title(f"{name}\n{feat.shape[1]} Channels", fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle(f"Comprehensive Internal Pattern Research (Encoder to Decoder)", fontsize=20, y=1.1)
plt.savefig("Full_Internal_Pattern_Research.png", bbox_inches='tight', dpi=300)
print("✅ Done! Check 'Full_Internal_Pattern_Research.png'")