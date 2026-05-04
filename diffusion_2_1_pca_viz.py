import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# Create folder to save results
os.makedirs("research_output/diffusion_2_1_viz", exist_ok=True)

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"🚀 Running on {DEVICE} with dtype {DTYPE}")

# Load Stable Diffusion 2.1
print("📦 Loading Stable Diffusion 2.1...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=DTYPE
).to(DEVICE)

# Dictionary to store features from different layers
extracted_features = {}
layer_names = []

# Hook function to capture features
def get_activation(name):
    def hook(module, input, output):
        # output shape is typically [batch, channels, height, width]
        extracted_features[name] = output.detach()
        print(f"  ✓ Captured {name}: {output.shape}")
    return hook

# Register hooks on decoder blocks (up_blocks)
# These are responsible for upsampling and detail refinement
print("🪝 Registering feature hooks...")
hook_layers = [
    ("down_0", pipe.unet.down_blocks[0]),
    ("down_1", pipe.unet.down_blocks[1]),
    ("down_2", pipe.unet.down_blocks[2]),
    ("up_0", pipe.unet.up_blocks[0]),
    ("up_1", pipe.unet.up_blocks[1]),
    ("up_2", pipe.unet.up_blocks[2]),
]

handles = []
for name, layer in hook_layers:
    handle = layer.register_forward_hook(get_activation(name))
    handles.append(handle)
    layer_names.append(name)

# Generation parameters
prompt = "masterpiece, 8k, a beautiful landscape with mountains and aurora borealis, highly detailed"
negative_prompt = "blurry, low quality"
num_inference_steps = 50
height, width = 512, 512

print(f"\n🎨 Generating image with prompt: '{prompt}'")
print(f"   Steps: {num_inference_steps}, Resolution: {height}x{width}")

# Generate image
with torch.no_grad():
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=7.5
    ).images[0]

image.save("research_output/diffusion_2_1_viz/generated_image.png")
print("✅ Image saved: research_output/diffusion_2_1_viz/generated_image.png")

# Remove hooks
for handle in handles:
    handle.remove()

# Process features with PCA
print("\n📊 Processing features with PCA...")

pca_visualizations = {}

for layer_name in layer_names:
    if layer_name not in extracted_features:
        print(f"⚠️  No features for {layer_name}")
        continue
    
    feature = extracted_features[layer_name].cpu()
    
    # Shape: [batch, channels, height, width]
    b, c, h, w = feature.shape
    print(f"\n  Processing {layer_name}:")
    print(f"    Shape: {feature.shape}")
    
    # Reshape for PCA: (height*width, channels)
    features_2d = feature.squeeze(0).permute(1, 2, 0)  # (h, w, c)
    features_2d = features_2d.reshape(-1, c).numpy()  # (h*w, c)
    
    # Apply PCA to get 3 components (for RGB visualization)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_2d)
    
    # Normalize to [0, 1]
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-8)
    
    # Reshape back to image
    pca_img = pca_features.reshape(h, w, 3)
    pca_visualizations[layer_name] = pca_img
    
    print(f"    Explained variance: {pca.explained_variance_ratio_}")
    print(f"    PCA shape: {pca_img.shape}")

# Create individual visualizations
print("\n💾 Saving individual layer visualizations...")
for layer_name, pca_img in pca_visualizations.items():
    # Save as high-res
    pca_pil = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_pil = pca_pil.resize((512, 512), Image.Resampling.LANCZOS)
    pca_pil.save(f"research_output/diffusion_2_1_viz/{layer_name}_pca.png")
    print(f"  ✅ {layer_name}_pca.png")

# Create grid visualization (like your reference picture)
print("\n🎨 Creating grid visualization...")

fig, axes = plt.subplots(len(layer_names), 1, figsize=(14, 4*len(layer_names)))
if len(layer_names) == 1:
    axes = [axes]

for idx, (ax, layer_name) in enumerate(zip(axes, layer_names)):
    if layer_name in pca_visualizations:
        pca_img = pca_visualizations[layer_name]
        ax.imshow(pca_img)
        ax.set_title(f"{layer_name} - PCA Feature Visualization", fontsize=14, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig("research_output/diffusion_2_1_viz/feature_grid.png", dpi=150, bbox_inches='tight')
print("✅ Saved: research_output/diffusion_2_1_viz/feature_grid.png")

# Create a side-by-side comparison with generated image
fig, axes = plt.subplots(1, len(layer_names) + 1, figsize=(5*(len(layer_names)+1), 5))

axes[0].imshow(image)
axes[0].set_title("Generated Image", fontsize=12, fontweight='bold')
axes[0].axis('off')

for idx, (layer_name, ax) in enumerate(zip(layer_names, axes[1:])):
    if layer_name in pca_visualizations:
        ax.imshow(pca_visualizations[layer_name])
        ax.set_title(f"{layer_name}", fontsize=12, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig("research_output/diffusion_2_1_viz/comparison_grid.png", dpi=150, bbox_inches='tight')
print("✅ Saved: research_output/diffusion_2_1_viz/comparison_grid.png")

print("\n✨ Done! Results saved to research_output/diffusion_2_1_viz/")
print("\nGenerated files:")
print("  - generated_image.png")
print("  - feature_grid.png")
print("  - comparison_grid.png")
print("  - [layer]_pca.png (individual layers)")
