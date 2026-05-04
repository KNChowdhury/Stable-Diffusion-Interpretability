# Stable Diffusion 1.4 - Multi-Prompt PCA Analysis Report

## Configuration
- Model: CompVis/stable-diffusion-v1-4
- Device: cpu
- Dtype: torch.float32
- Inference Steps: 5
- Guidance Scale: 7.5
- Resolution: 512x512

## Prompts Analyzed (8 total)

1. a photo of a robot in a white room
2. a photo of an astronaut on Mars
3. a comic book illustration of a superhero standing
4. a wooden artist mannequin on a desk
5. a photo of a child holding a teddy bear
6. a photo of a man meditating in a garden
7. a photo of a woman walking on the beach
8. a 3D render of a humanoid game character

## Extracted Layers (7 total)

- down_0
- down_1
- down_2
- mid
- up_2
- up_1
- up_0

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
