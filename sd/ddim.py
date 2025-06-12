import torch
import numpy as np

class DDIMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120, eta=0.0):
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.eta = eta  # noise factor (0 = deterministic)

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps) ** 2
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.set_inference_timesteps(50)

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        self.step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = (torch.arange(num_inference_steps) * self.step_ratio).flip(0)

    def set_strength(self, strength=1.0):
        if not (0.0 < strength <= 1.0):
            raise ValueError("strength must be in (0, 1]")
        start_step = int(self.num_inference_steps * (1 - strength))
        start_step = min(start_step, self.num_inference_steps - 1)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        step_size = self.step_ratio
        prev_t = t - step_size

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, dtype=alpha_t.dtype)

        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_alpha_prev = alpha_prev.sqrt()

        pred_x0 = (latents - (1 - alpha_t).sqrt() * model_output) / sqrt_alpha_t

        # DDIM update rule
        sigma_t = self.eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=self.generator) if t > 0 else 0.0

        dir_xt = (1 - alpha_prev - sigma_t ** 2).sqrt() * model_output
        x_prev = sqrt_alpha_prev * pred_x0 + dir_xt + sigma_t * noise

        return x_prev
