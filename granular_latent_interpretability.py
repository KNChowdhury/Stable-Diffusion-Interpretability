import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline
import os

# 1. SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 2. LOAD MODEL
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE).to(DEVICE)

# 3. INTERNAL HOOKING (L6 ar L7 er bhetorer shob layer)
internal_features = {}
def get_hook(name):
    def hook(m, i, o):
        data = o[0] if isinstance(o, tuple) else o
        internal_features[name] = data.detach().cpu().numpy()
    return hook

# Level 6 Internal (৪ ta pic)
pipe.unet.up_blocks[1].resnets[0].register_forward_hook(get_hook('L6_Stage_1'))
pipe.unet.up_blocks[1].attentions[0].register_forward_hook(get_hook('L6_Stage_2'))
pipe.unet.up_blocks[1].resnets[1].register_forward_hook(get_hook('L6_Stage_3'))
pipe.unet.up_blocks[1].attentions[1].register_forward_hook(get_hook('L6_Stage_4'))

# Level 7 Internal (৪ ta pic)
pipe.unet.up_blocks[2].resnets[0].register_forward_hook(get_hook('L7_Stage_1'))
pipe.unet.up_blocks[2].attentions[0].register_forward_hook(get_hook('L7_Stage_2'))
pipe.unet.up_blocks[2].resnets[1].register_forward_hook(get_hook('L7_Stage_3'))
pipe.unet.up_blocks[2].attentions[1].register_forward_hook(get_hook('L7_Stage_4'))

# 4. RUN INFERENCE
prompt = "(masterpiece, 8k, portrait of a human face made of liquid mercury, flowing molten chrome, high reflection:1.4)"
image = pipe(prompt, num_inference_steps=30).images[0]

# 5. PCA & GRID (Total 8 internal sub-layers)
def get_pca(feat):
    f = feat[0] if feat.ndim == 4 else feat
    c, h, w = f.shape
    pca = PCA(n_components=3)
    res = pca.fit_transform(f.reshape(c, -1).T)
    res = (res - res.min()) / (res.max() - res.min())
    return res.reshape(h, w, 3)

keys = list(internal_features.keys())
fig, axes = plt.subplots(2, 4, figsize=(20, 10)) # 2 Rows (L6 & L7), 4 Columns

for i, key in enumerate(keys):
    row = i // 4
    col = i % 4
    pca_img = get_pca(internal_features[key])
    axes[row, col].imshow(pca_img)
    axes[row, col].set_title(key, fontsize=12, fontweight='bold')
    axes[row, col].axis('off')

plt.suptitle("Internal Layer Progression (L6 vs L7)", fontsize=20)
plt.savefig("Total_Internal_Grid.png", bbox_inches='tight', dpi=300)
print("✅ Saved: Total_Internal_Grid.png")