# finalyearproject-diffusion_model-

**Objective:**

Minimizing the Procesing Time (Inference Time + Time taken till complete generation) for faster processing in diffusion models while maintaing high quality image generation

---

## Update 0.2.3: UI fix

Fix:

- For DDIM-DSS sampler: ETA becoming negative in UI display
- Application crashing with ZeroDivisionError after running few inference steps (UI problem)

## Update 0.2.2: DDIM with Dynamic Step Skipping(DSS)

Features:

- Integrates DDIM with a step skipping method
- skips number of steps based on L2 normalization of Latents
- reduces DDIM sampler Processing Time by ~5% (statistical data)
- UI: Dropdown for changing sampler

Fixes:

- Processing error: step must be greater than zero

## Update 0.2.1: DDIM Integration

Features:

- DDIM Integration
- 75% reduction in Processing Time (compared to DDPM sampler)
- Quality Improved

Fixes:

- Missing Checkpoint file for weights file(.ckpt)

## Update 0.2: UI Overhaul

Features:

- Restructuring UI components, setup and modules into proper abstraction

Fixes:

- Removed Slider: It was making range limitation
- UI Simplification

## Update 0.1: UI Overhaul

Features:

- Separate Buttons for individual Tasks:
  1. Text-to-Image
  2. Image-to-Image

## Update 0.0.3: UI fix

Fixes:

- UI Update: UI Window Maximization problem solved

## Update 0.0.2: UI integration with pipeline

## Update 0.0.1: Basic Models update
