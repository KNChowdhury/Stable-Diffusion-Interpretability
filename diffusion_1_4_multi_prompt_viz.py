import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# Force UTF-8 output encoding on Windows consoles when supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Running on {DEVICE} with dtype {DTYPE}")

# Create output directory
os.makedirs("research_output/sd_1_4_prompts", exist_ok=True)

# Load Stable Diffusion 1.4
print("Loading Stable Diffusion 1.4...")
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=DTYPE
).to(DEVICE)

# 8 Prompts from I2P dataset
PROMPTS = [
    "a photo of a robot in a white room",
    "a photo of an astronaut on Mars",
    "a comic book illustration of a superhero standing",
    "a wooden artist mannequin on a desk",
    "a photo of a child holding a teddy bear",
    "a photo of a man meditating in a garden",
    "a photo of a woman walking on the beach",
    "a 3D render of a humanoid game character",
]

# Layer configuration for SD 1.4
layer_config = {
    "down_0": pipe.unet.down_blocks[0],
    "down_1": pipe.unet.down_blocks[1],
    "down_2": pipe.unet.down_blocks[2],
    "mid": pipe.unet.mid_block,
    "up_2": pipe.unet.up_blocks[2],
    "up_1": pipe.unet.up_blocks[1],
    "up_0": pipe.unet.up_blocks[0],
}

layer_names = list(layer_config.keys())

# Process each prompt
print(f"\nProcessing {len(PROMPTS)} prompts...\n")

all_results = {}

for prompt_idx, prompt in enumerate(tqdm(PROMPTS, desc="Prompts")):
    print(f"\n[{prompt_idx+1}/{len(PROMPTS)}] {prompt}")
    
    # Dictionary to store features
    extracted_features = {}
    
    # Hook function
    def unwrap_output(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                tensor = unwrap_output(item)
                if tensor is not None:
                    return tensor
        return None

    def get_activation(name):
        def hook(module, input, output):
            tensor = unwrap_output(output)
            if tensor is None:
                return
            extracted_features[name] = tensor.detach()
        return hook
    
    # Register hooks
    handles = []
    for layer_name, layer in layer_config.items():
        handle = layer.register_forward_hook(get_activation(layer_name))
        handles.append(handle)
    
    # Generate image
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=5,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
    
    # Create prompt folder
    prompt_folder = f"research_output/sd_1_4_prompts/prompt_{prompt_idx:02d}"
    os.makedirs(prompt_folder, exist_ok=True)
    
    # Save generated image
    image.save(f"{prompt_folder}/generated.png")
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Process features with PCA
    pca_visualizations = {}
    
    for layer_name in layer_names:
        if layer_name not in extracted_features:
            continue
        
        feature = extracted_features[layer_name].cpu()
        feat_sq = feature
        if feat_sq.dim() > 3 and feat_sq.shape[0] == 2:
            # Classifier-free guidance duplicates the batch: [uncond, cond]
            feat_sq = feat_sq[1]
        elif feat_sq.dim() > 3 and feat_sq.shape[0] == 1:
            feat_sq = feat_sq.squeeze(0)
        while feat_sq.dim() > 3 and feat_sq.shape[0] == 1:
            feat_sq = feat_sq.squeeze(0)
        if feat_sq.dim() != 3:
            raise ValueError(f"Unexpected squeezed feature shape for layer {layer_name}: {feat_sq.shape}")
        c, h, w = feat_sq.shape
        
        # Reshape for PCA
        features_2d = feat_sq.permute(1, 2, 0).reshape(-1, c).detach().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features_2d)
        
        # Normalize
        pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-8)
        pca_img = pca_features.reshape(h, w, 3)
        pca_visualizations[layer_name] = pca_img
    
    all_results[prompt_idx] = {
        "prompt": prompt,
        "image": image,
        "features": pca_visualizations
    }
    
    # Save individual layer visualizations
    for layer_name, pca_img in pca_visualizations.items():
        pil_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        pil_img.save(f"{prompt_folder}/{layer_name}_pca.png")

print("\nCreating comparison grids...")

# Create grid for all prompts - Generated Images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Stable Diffusion 1.4 - Generated Images (8 Prompts)", fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx < len(PROMPTS):
        result = all_results[idx]
        ax.imshow(result["image"])
        ax.set_title(f"{idx+1}. {PROMPTS[idx][:40]}...", fontsize=10, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig("research_output/sd_1_4_prompts/00_generated_images.png", dpi=150, bbox_inches='tight')
print("✅ Saved: 00_generated_images.png")

# Create grids for each layer across all prompts
print("\nCreating layer comparison grids...")

for layer_name in layer_names:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Layer: {layer_name} - PCA Feature Visualization (8 Prompts)", 
                 fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(PROMPTS):
            result = all_results[idx]
            if layer_name in result["features"]:
                pca_img = result["features"][layer_name]
                ax.imshow(pca_img)
                ax.set_title(f"{idx+1}. {PROMPTS[idx][:40]}...", fontsize=10)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"research_output/sd_1_4_prompts/01_layer_{layer_name}.png", dpi=150, bbox_inches='tight')
    print(f"  ✅ layer_{layer_name}.png")

# Create detailed comparison for first prompt
print("\nCreating detailed analysis for first prompt...")

result = all_results[0]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Detailed Analysis - Prompt 1: {PROMPTS[0]}", fontsize=14, fontweight='bold')

axes.flat[0].imshow(result["image"])
axes.flat[0].set_title("Generated Image", fontweight='bold')
axes.flat[0].axis('off')

for idx, (layer_name, pca_img) in enumerate(list(result["features"].items())[:7]):
    ax = axes.flat[idx+1]
    ax.imshow(pca_img)
    ax.set_title(f"{layer_name}", fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig("research_output/sd_1_4_prompts/02_detailed_prompt_1.png", dpi=150, bbox_inches='tight')
print("✅ Saved: 02_detailed_prompt_1.png")

# Create summary statistics
print("\nCreating summary report...")

summary = f"""# Stable Diffusion 1.4 - Multi-Prompt PCA Analysis Report

## Configuration
- Model: CompVis/stable-diffusion-v1-4
- Device: {DEVICE}
- Dtype: {DTYPE}
- Inference Steps: 5
- Guidance Scale: 7.5
- Resolution: 512x512

## Prompts Analyzed ({len(PROMPTS)} total)
"""

for idx, prompt in enumerate(PROMPTS):
    summary += f"\n{idx+1}. {prompt}"

summary += f"""

## Extracted Layers ({len(layer_names)} total)
"""

for layer_name in layer_names:
    summary += f"\n- {layer_name}"

summary += """

## Output Structure
```
research_output/sd_1_4_prompts/
├── 00_generated_images.png          # All 8 generated images
├── 01_layer_down_0.png              # Layer comparison across prompts
├── 01_layer_down_1.png
├── ... (all layers)
├── 02_detailed_prompt_1.png         # Detailed breakdown of first prompt
├── prompt_00/                       # Individual prompt folder
│   ├── generated.png
│   ├── down_0_pca.png
│   ├── down_1_pca.png
│   └── ... (all layers)
├── prompt_01/
├── ... (all 8 prompts)
└── analysis_summary.md

```

## Analysis Notes

### Down Blocks (0-2)
- Down Block 0: 64x64 resolution - captures global structure
- Down Block 1: 32x32 resolution - medium-level features
- Down Block 2: 16x16 resolution - high-level semantics

### Mid Block
- Bottleneck features at 8x8 resolution
- Most abstract representation

### Up Blocks (2-0)
- Up Block 2: 16x16 resolution - progressive reconstruction
- Up Block 1: 32x32 resolution - detail refinement
- Up Block 0: 64x64 resolution - final details

## Files Generated
- 2 overview grids (images + detailed breakdowns)
- 7 layer-specific comparison grids
- 8 prompt folders × 8 layers = 64 individual PCA visualizations
- Total: 80+ visualization files

---
Generated with Stable Diffusion 1.4 using multi-layer feature extraction and PCA dimensionality reduction.
Batch process across 8 I2P prompts for comprehensive analysis.
"""

with open("research_output/sd_1_4_prompts/analysis_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("Saved analysis_summary.md")

print("\n✨ Analysis complete!")
print(f"\nTotal files generated: {len(PROMPTS) * len(layer_names) + 10}")
print("Output: research_output/sd_1_4_prompts/")
