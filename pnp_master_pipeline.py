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

# 3. INTERNAL HOOKS (Targeting sub-layers of Up-Block 1/Level 6)
extracted_internal = {}
def get_internal_activation(name):
    def hook(model, input, output):
        data = output[0] if isinstance(output, tuple) else output
        extracted_internal[name] = data.detach().cpu().numpy()
    return hook

# Level 6-er bhetore thaka internal ResNet blocks ar Attention layers hook korchi
# Eikhane apni 4 ta sub-pic paben shudhu Level 6 theke
pipe.unet.up_blocks[1].resnets[0].register_forward_hook(get_internal_activation('L6_Sub_1'))
pipe.unet.up_blocks[1].attentions[0].register_forward_hook(get_internal_activation('L6_Sub_2'))
pipe.unet.up_blocks[1].resnets[1].register_forward_hook(get_internal_activation('L6_Sub_3'))
pipe.unet.up_blocks[1].attentions[1].register_forward_hook(get_internal_activation('L6_Sub_4'))

# 4. RUN INFERENCE
prompt = "(masterpiece, 8k, portrait of a human face made of liquid mercury, flowing molten chrome, high reflection:1.4)"
print("🔍 Extracting 4 Internal Stages from Level 6...")
image = pipe(prompt, num_inference_steps=30).images[0]

# 5. PCA & GRID GENERATION
def get_pca(feat):
    f = feat[0] if feat.ndim == 4 else feat
    c, h, w = f.shape
    pca = PCA(n_components=3)
    res = pca.fit_transform(f.reshape(c, -1).T)
    low, high = np.percentile(res, 2), np.percentile(res, 98)
    res = np.clip((res - low) / (high - low), 0, 1)
    return res.reshape(h, w, 3)

# Plotting the 4 internal sub-layers
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
sub_labels = ['ResNet Block 1', 'Attention 1', 'ResNet Block 2', 'Attention 2']

for i, name in enumerate(['L6_Sub_1', 'L6_Sub_2', 'L6_Sub_3', 'L6_Sub_4']):
    pca_img = get_pca(extracted_internal[name])
    axes[i].imshow(pca_img)
    axes[i].set_title(sub_labels[i], fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle("Internal Breakdown of Decoder Level 6", fontsize=16, y=1.05)
plt.savefig("Level_6_Internal_Details.png", bbox_inches='tight', dpi=300)
print("✅ Saved: Level_6_Internal_Details.png")