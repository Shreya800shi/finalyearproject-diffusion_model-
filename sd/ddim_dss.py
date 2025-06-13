import torch
import numpy as np

class DDIMDSSSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120, eta=0.0, skip_threshold=0.01, max_skip_steps=2, min_steps=5):
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.eta = eta  # Noise factor (0 = deterministic)
        self.skip_threshold = skip_threshold  # L2 norm threshold for skipping
        self.max_skip_steps = max_skip_steps  # Max timesteps to skip at once
        self.min_steps = min_steps  # Minimum number of steps to ensure quality

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps) ** 2
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Initialize inference timesteps
        self.set_inference_timesteps(50)

    def set_inference_timesteps(self, num_inference_steps=50):
        """Set the initial timestep sequence for DDIM sampling."""
        self.num_inference_steps = num_inference_steps
        self.step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = (torch.arange(num_inference_steps) * self.step_ratio).flip(0).to(self.alphas_cumprod.device)
        self.current_step_idx = 0  # Track current position in timesteps

    def set_strength(self, strength=1.0):
        """Adjust timesteps based on strength for image-to-image tasks."""
        if not (0.0 < strength <= 1.0):
            raise ValueError("strength must be in (0, 1]")
        start_step = int(self.num_inference_steps * (1 - strength))
        start_step = min(start_step, self.num_inference_steps - 1)
        self.timesteps = self.timesteps[start_step:]
        self.current_step_idx = 0

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor, prev_latents: torch.Tensor = None):
        """
        Perform one DDIM step with dynamic step skipping based on L2 norm.
        Args:
            timestep: Current timestep (t).
            latents: Current latent (x_t).
            model_output: Predicted noise from UNet (epsilon_theta).
            prev_latents: Previous latent (x_{t-1}) for L2 norm calculation.
        Returns:
            Tuple: (next_latents, next_timestep, skip_count).
        """
        t = timestep
        step_size = self.step_ratio
        prev_t = t - step_size

        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, dtype=alpha_t.dtype, device=latents.device)

        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_alpha_prev = alpha_prev.sqrt()

        pred_x0 = (latents - (1 - alpha_t).sqrt() * model_output) / sqrt_alpha_t

        # DDIM update rule
        sigma_t = self.eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=self.generator) if t > 0 else 0.0
        dir_xt = (1 - alpha_prev - sigma_t ** 2).sqrt() * model_output
        x_prev = sqrt_alpha_prev * pred_x0 + dir_xt + sigma_t * noise

        # Dynamic step skipping
        skip_count = 1  # Default: move to next timestep
        if prev_latents is not None and self.current_step_idx < len(self.timesteps) - self.min_steps:
            # Compute normalized L2 norm of latent change
            l2_norm = torch.norm(x_prev - prev_latents, p=2) / torch.norm(x_prev, p=2)
            if l2_norm < self.skip_threshold:
                # Skip to a later timestep (up to max_skip_steps)
                skip_count = min(self.max_skip_steps, len(self.timesteps) - self.current_step_idx - self.min_steps)
                next_t = self.timesteps[self.current_step_idx + skip_count] if self.current_step_idx + skip_count < len(self.timesteps) else self.timesteps[-self.min_steps]
            else:
                next_t = t - step_size if t - step_size >= 0 else 0
        else:
            next_t = t - step_size if t - step_size >= 0 else 0

        # Update current step index
        if next_t in self.timesteps:
            self.current_step_idx = torch.where(self.timesteps == next_t)[0].item()
        else:
            self.current_step_idx += skip_count

        return x_prev, next_t, skip_count