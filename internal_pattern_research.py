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

# 3. SELECTIVE HOOKING (Pattern tracking logic)
storage = {}
def get_hook(name):
    def hook(m, i, o):
        data = o[0] if isinstance(o, tuple) else o
        storage[name] = data.detach().cpu().numpy()
    return hook

# Encoder stages (Pattern learning)
pipe.unet.down_blocks[1].register_forward_hook(get_hook('Encoder_Patterns'))
# Bottleneck
pipe.unet.mid_block.register_forward_hook(get_hook('Mid_Latent_Space'))
# Decoder stages (Texture reconstruction)
pipe.unet.up_blocks[1].register_forward_hook(get_hook('Decoder_Patterns'))

# 4. RUN INFERENCE
prompt = "(masterpiece, 8k, portrait of a human face made of liquid mercury, flowing molten chrome, high reflection:1.4)"
print("🧬 Extracting Hierarchical Patterns...")
pipe(prompt, num_inference_steps=30)

# 5. PCA & VERTICAL GRID (Niche niche pattern)
def get_pca(feat):
    f = feat[0] if feat.ndim == 4 else feat
    c, h, w = f.shape
    pca = PCA(n_components=3)
    res = pca.fit_transform(f.reshape(c, -1).T)
    res = (res - res.min()) / (res.max() - res.min())
    return res.reshape(h, w, 3)

# 3 rows (Encoder, Mid, Decoder)
fig, axes = plt.subplots(3, 1, figsize=(10, 25))
rows = ['Encoder_Patterns', 'Mid_Latent_Space', 'Decoder_Patterns']
titles = ['Stage 1: Input Pattern (Encoder)', 'Stage 2: Core Concept (Bottleneck)', 'Stage 3: Reconstruction (Decoder)']

for i, key in enumerate(rows):
    pca_img = get_pca(storage[key])
    axes[i].imshow(pca_img)
    axes[i].set_title(titles[i], fontsize=16, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Internal_Pattern_Learning.png", bbox_inches='tight', dpi=300)
print("✅ Done! Chobi 'Internal_Pattern_Learning.png' nam-e save hoyeche।")