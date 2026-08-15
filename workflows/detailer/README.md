# Detailer Workflow

This workflow extends the verified LTX-13B baseline with the official LTX 0.9.8 IC-LoRA Detailer.

## Target Pipeline

LTX 13B Q4 baseline

↓

IC-LoRA Detailer

↓

Video output

## Model

The detail enhancement model used by this project is:

ltxv-098-ic-lora-detailer-comfyui.safetensors

## Purpose

The purpose of this workflow is to improve:

- Visual detail
- Texture definition
- Fine subject detail
- Overall image refinement

## Testing Strategy

The workflow will be compared directly against the baseline.

Test sequence:

1. Baseline LTX 13B Q4
2. LTX 13B Q4 + IC-LoRA Detailer

The same prompt, seed, resolution, frame count, and generation settings should be used whenever possible.

## Important

The Detailer belongs to the LTX 0.9.8 ecosystem.

It should be tested carefully with the Q4 Distilled pipeline because the exact behavior may differ from a full-precision development model configuration.
