# Production Workflow

This directory represents the final long-form AI video production pipeline.

## Architecture

Story Planner

↓

Story and Character Data

↓

Shot Generation

↓

LTX 13B Q4 Distilled

↓

IC-LoRA Detailer

↓

LTX Spatial Upscaler

↓

Final Shot Output

↓

Shot Library

↓

FFmpeg Assembly

↓

Final Cinematic Video

## Long-Form Strategy

The project will not attempt to generate a complete multi-minute video in a single diffusion job.

Instead, the final video will be divided into individual shots.

Example:

- Shot 001
- Shot 002
- Shot 003
- Shot 004

Each shot can contain metadata describing:

- Character information
- Prompt
- Negative prompt
- Seed
- Resolution
- Frame count
- FPS
- Reference image
- Previous shot relationship
- Continuity notes

## Final Goal

Produce cinematic story videos approximately 3–5 minutes long by generating, refining, storing, and assembling individual shots.
