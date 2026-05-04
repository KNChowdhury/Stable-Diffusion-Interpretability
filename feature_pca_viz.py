import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from diffusers import StableDiffusionPipeline
import os

# Create folder to save layers
os.makedirs("research_layers", exist_ok=True)

# 1. SETUP (CPU/GPU Detection)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"🚀 Running on {DEVICE}")

# 2. LOAD PIPELINE
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE).to(DEVICE)

# Dictionary to store features
extracted_features = {}

# Hook function to grab internal layers
def get_activation(name):
    def hook(model, input, output):
        # Taking the mean across channels to get a spatial map
        extracted_features[name] = output.detach().cpu().numpy()
    return hook

# Registering hooks for Input and Decoder Layers (Up-blocks)
pipe.unet.conv_in.register_forward_hook(get_activation('Input'))
pipe.unet.up_blocks[0].register_forward_hook(get_activation('Layer_1_Structure'))
pipe.unet.up_blocks[1].register_forward_hook(get_activation('Layer_2_Shape'))
pipe.unet.up_blocks[2].register_forward_hook(get_activation('Layer_3_Mercury'))

# 3. GENERATION
print("🎨 Rendering... (Wait for 100%)")
prompt = "masterpiece, 8k, liquid mercury robot, highly reflective, chrome"
image = pipe(prompt, num_inference_steps=30, width=512, height=512).images[0]
image.save("final_result.png")

# 4. SCIENTIFIC PCA PLOTTING (The Pro Part)
def save_pca_layer(feature_tensor, name):
    # feature_tensor shape: [1, Channels, H, W]
    b, c, h, w = feature_tensor.shape
    features = feature_tensor.reshape(c, -1).T # Reshape for PCA
    
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(features)
    
    # Normalize to avoid "White" washout
    pca_res = (pca_res - pca_res.min()) / (pca_res.max() - pca_res.min())
    pca_res = pca_res.reshape(h, w, 3)
    
    plt.figure(figsize=(6,6))
    plt.imshow(pca_res)
    plt.title(f"Interpretability Map: {name}")
    plt.axis('off')
    plt.savefig(f"research_layers/{name}.png")
    print(f"✅ Saved: research_layers/{name}.png")

# Run PCA for each layer captured
print("\n📊 Extracting Latent Layers...")
for layer_name, feat in extracted_features.items():
    save_pca_layer(feat, layer_name)

print("\n✨ Done! Check the 'research_layers' folder for all maps.")