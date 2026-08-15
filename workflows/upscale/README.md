# Upscale Workflow

This workflow adds the official LTX Spatial Upscaler after detail refinement.

## Target Pipeline

LTX 13B Q4

↓

IC-LoRA Detailer

↓

LTX Spatial Upscaler

↓

Final video

## Model

The spatial upscaler used by this project is:

ltxv-spatial-upscaler-0.9.8.safetensors

## Purpose

The spatial upscaler is the preferred first quality enhancement method for this project.

The goal is to improve:

- Spatial detail
- Perceived sharpness
- High-resolution visual quality

while maintaining better temporal consistency than a frame-by-frame external image upscaler.

## Testing Strategy

The workflow will be tested in three stages:

1. LTX 13B Q4
2. LTX 13B Q4 + IC-LoRA Detailer
3. LTX 13B Q4 + Detailer + Spatial Upscaler

Each stage should be compared using the same source prompt and similar generation settings.

## Note

Real-ESRGAN or other frame-by-frame upscalers are not part of the primary pipeline because independent frame processing can introduce temporal flicker or shimmer.

They may be evaluated later as optional post-processing.
