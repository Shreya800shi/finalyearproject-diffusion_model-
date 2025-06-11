import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from sd.demo import generate_image
from PIL import Image

class Worker(QObject):
    progress = pyqtSignal(int, int, float)  # step, total_steps, step_time
    finished = pyqtSignal(np.ndarray)  # output_image
    error = pyqtSignal(str)  # error_message

    def __init__(self, image_path, sentence, uncond_prompt, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed):
        super().__init__()
        self.image_path = image_path
        self.sentence = sentence
        self.uncond_prompt = uncond_prompt
        self.strength = strength
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.sampler = sampler
        self.num_inference_steps = num_inference_steps
        self.seed = seed

    def progress_callback(self, step, total_steps, step_time):
        self.progress.emit(step, total_steps, step_time)

    def run(self):
        try:
            input_image = None
            if self.image_path:
                input_image = Image.open(self.image_path).convert("RGB")
            
            output_image = generate_image(
                input_image=input_image,
                prompt=self.sentence,
                uncond_prompt=self.uncond_prompt,
                strength=self.strength,
                do_cfg=self.do_cfg,
                cfg_scale=self.cfg_scale,
                sampler=self.sampler,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed,
                progress_callback=self.progress_callback
            )
            self.finished.emit(output_image)
        except FileNotFoundError as e:
            self.error.emit(f"Missing checkpoint file: {str(e)}. Please run setup.")
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")