from typing import Callable
from PIL import Image
import numpy as np
from sd.demo import generate_image

class DiffusionModel:
    @staticmethod
    def process(
        image_path: str,
        sentence: str,
        uncond_prompt: str,
        strength: float,
        do_cfg: bool,
        cfg_scale: float,
        sampler: str,
        num_inference_steps: int,
        seed: int,
        progress_callback: Callable[[int, int, float], None] = None
    ) -> np.ndarray:
        """
        Process an image with the diffusion model.
        """
        try:
            input_image = Image.open(image_path).convert("RGB") if image_path else None
            output_image = generate_image(
                input_image=input_image,
                prompt=sentence,
                uncond_prompt=uncond_prompt,
                strength=strength,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler=sampler,
                num_inference_steps=num_inference_steps,
                seed=seed,
                progress_callback=progress_callback
            )
            return output_image
        except Exception as e:
            raise RuntimeError(f"Diffusion model failed: {str(e)}")