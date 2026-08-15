# Baseline Workflow

This directory contains the verified baseline LTX-13B video generation workflow.

## Target Pipeline

- LTXV 13B 0.9.8 Distilled
- GGUF Q4_K_M model
- T5 Q4 text encoder
- LTX VAE
- 24 FPS
- H.264 MP4 output

## Purpose

The baseline workflow is the reference configuration for all future quality tests.

The workflow must be stable before adding:

- IC-LoRA Detailer
- LTX Spatial Upscaler
- Multi-GPU experimentation
- Long-form shot automation

## Important

Do not manually reconstruct a large ComfyUI API workflow.

The production workflow should be created and verified in ComfyUI, then saved or exported using the appropriate ComfyUI workflow format.

## Current Target

The practical generation baseline is approximately:

- Resolution: 1280 × 704
- Frame rate: 24 FPS
- Model: LTXV 13B 0.9.8 Distilled Q4_K_M
