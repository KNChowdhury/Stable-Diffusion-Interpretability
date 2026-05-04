import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from matplotlib.gridspec import GridSpec

# Create folder
os.makedirs("research_output/diffusion_2_1_advanced", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"🚀 Running on {DEVICE}")

# Load Stable Diffusion 2.1
print("📦 Loading Stable Diffusion 2.1...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=DTYPE,
).to(DEVICE)

# Store timestep-wise features
timestep_features = {}

# Enhanced hook that captures at each timestep
class FeatureCapture:
    def __init__(self):
        self.features_by_step = {}
    
    def get_hook(self, layer_name):
        def hook(module, input, output):
            step = getattr(pipe.scheduler, 'step_index', 0)
            if step not in self.features_by_step:
                self.features_by_step[step] = {}
            self.features_by_step[step][layer_name] = output.detach()
        return hook

feature_capture = FeatureCapture()

# Register hooks on multiple blocks for hierarchical analysis
print("🪝 Registering multi-level hooks...")
layer_config = {
    "down_block_0": pipe.unet.down_blocks[0],  # 64x64 features
    "down_block_1": pipe.unet.down_blocks[1],  # 32x32 features
    "down_block_2": pipe.unet.down_blocks[2],  # 16x16 features
    "mid_block": pipe.unet.mid_block,          # 8x8 bottleneck features
    "up_block_2": pipe.unet.up_blocks[2],      # 16x16 features (decoder)
    "up_block_1": pipe.unet.up_blocks[1],      # 32x32 features (decoder)
    "up_block_0": pipe.unet.up_blocks[0],      # 64x64 features (decoder)
}

handles = []
for layer_name, layer in layer_config.items():
    handle = layer.register_forward_hook(feature_capture.get_hook(layer_name))
    handles.append(handle)

# Generation with detailed parameters
prompt = "a majestic dragon flying through cosmic nebula, highly detailed, photorealistic, 8k"
negative_prompt = "low quality, blurry, distorted"

print(f"\n🎨 Generating image...")
print(f"   Prompt: {prompt}")

with torch.no_grad():
    output = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=768,
        width=768,
    )
    image = output.images[0]

image.save("research_output/diffusion_2_1_advanced/generated.png")
print("✅ Image saved")

# Cleanup
for handle in handles:
    handle.remove()

# Process captured features
print("\n📊 Processing hierarchical features...")

feature_visualizations = {}

for layer_name, layer_module in layer_config.items():
    print(f"\n  Processing {layer_name}...")
    
    # Collect features from first step (or average across steps)
    if 0 in feature_capture.features_by_step and layer_name in feature_capture.features_by_step[0]:
        feature = feature_capture.features_by_step[0][layer_name].cpu()
    else:
        print(f"    ⚠️  No features captured")
        continue
    
    b, c, h, w = feature.shape
    print(f"    Shape: {feature.shape}")
    
    # Reshape: (h*w, c)
    feat_flat = feature.squeeze(0).permute(1, 2, 0).reshape(-1, c).numpy()
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feat_flat)
    
    # Normalize and reshape
    pca_result = (pca_result - pca_result.min(axis=0)) / (pca_result.max(axis=0) - pca_result.min(axis=0) + 1e-8)
    pca_result = pca_result.reshape(h, w, 3)
    
    feature_visualizations[layer_name] = pca_result
    print(f"    Variance explained: {pca.explained_variance_ratio_}")

# Create publication-quality grid visualization
print("\n🎨 Creating advanced grid visualization...")

# Figure layout: 2 rows x 4 columns
fig = plt.figure(figsize=(20, 10))
fig.suptitle("Stable Diffusion 2.1 - Hierarchical PCA Feature Analysis", 
             fontsize=18, fontweight='bold', y=0.98)

layers_to_plot = [
    "down_block_0",
    "down_block_1",
    "down_block_2",
    "mid_block",
    "up_block_2",
    "up_block_1",
    "up_block_0",
]

# Top row: down sampling blocks + mid
ax_main = plt.subplot(2, 2, 1)
ax_main.imshow(image)
ax_main.set_title("Generated Image", fontsize=14, fontweight='bold')
ax_main.axis('off')

# Create subgrid for features
gs = GridSpec(2, 3, left=0.52, right=0.98, hspace=0.3, wspace=0.3)

plot_idx = 0
for idx, layer in enumerate(layers_to_plot):
    if layer in feature_visualizations and plot_idx < 6:
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.imshow(feature_visualizations[layer])
        ax.set_title(f"{layer}", fontsize=12, fontweight='bold')
        ax.axis('off')
        plot_idx += 1

plt.tight_layout()
plt.savefig("research_output/diffusion_2_1_advanced/hierarchical_analysis.png", 
            dpi=150, bbox_inches='tight')
print("✅ Saved: hierarchical_analysis.png")

# Create detailed technical report
print("\n📝 Creating technical analysis...")

report = f"""
# Stable Diffusion 2.1 - PCA Feature Analysis Report

## Generation Parameters
- Model: stabilityai/stable-diffusion-2-1
- Prompt: {prompt}
- Negative Prompt: {negative_prompt}
- Steps: 50
- Guidance Scale: 7.5
- Resolution: 768x768
- Device: {DEVICE}
- Dtype: {DTYPE}

## Extracted Layers and Dimensions
"""

for layer_name, viz in feature_visualizations.items():
    report += f"\n- **{layer_name}**: PCA from high-dimensional feature space to RGB"

report += f"""

## Analysis Notes
- Down blocks (0-2): Progressive feature extraction with downsampling
- Mid block: Bottleneck features (lowest resolution, highest abstraction)
- Up blocks (2-0): Progressive reconstruction with upsampling

## Output Files
- generated.png: The generated image
- hierarchical_analysis.png: Grid visualization
- individual_layers/: Individual PCA visualizations for each layer

---
Generated with Stable Diffusion 2.1 using feature extraction and PCA dimensionality reduction.
"""

with open("research_output/diffusion_2_1_advanced/analysis_report.md", "w") as f:
    f.write(report)

print("✅ Saved: analysis_report.md")

# Save individual high-res visualizations
print("\n💾 Saving individual layer visualizations...")
os.makedirs("research_output/diffusion_2_1_advanced/individual_layers", exist_ok=True)

for layer_name, viz in feature_visualizations.items():
    # Upscale to 512x512 for better visibility
    pil_img = Image.fromarray((viz * 255).astype(np.uint8))
    pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
    pil_img.save(f"research_output/diffusion_2_1_advanced/individual_layers/{layer_name}_pca.png")
    print(f"  ✅ {layer_name}_pca.png")

print("\n✨ Analysis complete!")
print("\nOutput directory: research_output/diffusion_2_1_advanced/")
print("Files generated:")
print("  - generated.png")
print("  - hierarchical_analysis.png")
print("  - analysis_report.md")
print("  - individual_layers/ (7 files)")
