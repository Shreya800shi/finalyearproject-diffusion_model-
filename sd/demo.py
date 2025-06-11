import os
import sd.model_loader as model_loader
import sd.pipeline as pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_built() and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Tokenizer (assuming files in sd/data/)
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
# Model file path
project_root = Path(__file__).resolve().parents[2]  # From sd/ to code/
model_file = project_root / "data" / "v1-5-pruned-emaonly.ckpt"

# Cache models to avoid reloading
_models = None

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A cat playing with wool, ultra sharp, photorealistic."
uncond_prompt = ""  # Also known as negative prompt
# do_cfg = True
# cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

# input_image = None
# Comment to disable image to image
# image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
# strength = 0.9

## SAMPLER

# sampler = "ddpm"
# num_inference_steps = 50
# seed = 42

# output_image = pipeline.generate(
#     prompt=prompt,
#     uncond_prompt=uncond_prompt,
#     #input_image=input_image,
#     strength=strength,
#     do_cfg=do_cfg,
#     cfg_scale=cfg_scale,
#     sampler_name=sampler,
#     n_inference_steps=num_inference_steps,
#     seed=seed,
#     models=models,
#     device=DEVICE,
#     idle_device="cpu",
#     tokenizer=tokenizer,
# )

# Combine the input image and the output image into a single image.
# Image.fromarray(output_image)

def generate_image(
    input_image=None,  # Allow None for text-to-image
    prompt=str,
    uncond_prompt="",
    strength=0.9,
    do_cfg=True,
    cfg_scale=8,
    sampler="ddpm",
    num_inference_steps=50,
    seed=42,
    progress_callback=None
):
    """
    Generate an image using the diffusion pipeline.
    
    Args:
        input_image (PIL.Image.Image, optional): Input image for the pipeline. None for text-to-image.
        prompt (str): Text prompt for image generation.
        uncond_prompt (str): Unconditional prompt (default: "").
        strength (float): Strength of the diffusion process (default: 0.9).
        do_cfg (bool): Whether to use classifier-free guidance (default: True).
        cfg_scale (float): Classifier-free guidance scale (default: 8).
        sampler (str): Sampler name (default: "ddpm").
        num_inference_steps (int): Number of inference steps (default: 50).
        seed (int): Random seed for reproducibility (default: 42).
        progress_callback (callable): Callback for progress updates (default: None).
    
    Returns:
        numpy.ndarray: Generated image as a NumPy array (RGB).
    
    Raises:
        FileNotFoundError: If model_file is missing.
    """
    global _models
    if _models is None:
        if not model_file.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {model_file}")
        _models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
    
    # Use existing pipeline and parameters
    kwargs = {
        "prompt": prompt,
        "uncond_prompt": uncond_prompt,
        "strength": strength,
        "do_cfg": do_cfg,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler,
        "n_inference_steps": num_inference_steps,
        "seed": seed,
        "models": _models,
        "device": DEVICE,
        "idle_device": "cpu",
        "tokenizer": tokenizer,
        "progress_callback": progress_callback
    }
    if input_image is not None:
        kwargs["input_image"] = input_image
    return pipeline.generate(**kwargs)