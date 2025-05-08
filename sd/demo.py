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
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

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

def generate_image(input_image, prompt, uncond_prompt="", strength=0.9, do_cfg=True, cfg_scale=8, sampler="ddpm", num_inference_steps=50, seed=42, progress_callback=None):
    """
    Generate an image using the diffusion pipeline.
    
    Args:
        input_image (PIL.Image.Image): Input image for the pipeline.
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
    """
    # Use existing pipeline and parameters from your demo.py
    return pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
        progress_callback=progress_callback
    )