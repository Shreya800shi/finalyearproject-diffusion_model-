import os
import subprocess
import requests
import time
from PyQt5.QtCore import pyqtSignal, QObject
import sys
from pathlib import Path

class SetupWorker(QObject):
    progress_updated = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, data_dir, ckpt_file, requirements_file, ckpt_url):
        super().__init__()
        self.data_dir = data_dir
        self.ckpt_file = ckpt_file
        self.requirements_file = requirements_file
        self.ckpt_url = ckpt_url

    def install_requirements(self):
        self.progress_updated.emit("Starting installation of missing requirements.")
        if not self.data_dir.exists():
            self.progress_updated.emit(f"Creating directory: {self.data_dir}")
            try:
                self.data_dir.mkdir(parents=True)
                self.progress_updated.emit("Directory created successfully.")
            except Exception as e:
                self.progress_updated.emit(f"Error creating directory: {str(e)}")
                self.finished.emit()
                return

        if not self.ckpt_file.exists():
            self.progress_updated.emit(f"Downloading checkpoint: {self.ckpt_file}")
            try:
                response = requests.get(self.ckpt_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 8192
                start_time = time.time()
                with open(self.ckpt_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                elapsed = time.time() - start_time
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                eta = (total_size - downloaded) / speed if speed > 0 else 0
                                self.progress_updated.emit(
                                    f"Progress: {progress:.1f}% | "
                                    f"ETA: {eta:.1f}s | "
                                    f"Speed: {speed / 1024:.1f} KB/s"
                                )
                self.progress_updated.emit("Checkpoint downloaded successfully.")
            except Exception as e:
                self.progress_updated.emit(f"Error downloading checkpoint: {str(e)}")
                self.finished.emit()
                return

        if self.requirements_file.exists():
            self.progress_updated.emit(f"Installing packages from: {self.requirements_file}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.progress_updated.emit("Packages installed successfully.")
                self.progress_updated.emit(result.stdout)
            except subprocess.CalledProcessError as e:
                self.progress_updated.emit(f"Error installing packages: {e.stderr}")
                self.finished.emit()
                return
            except Exception as e:
                self.progress_updated.emit(f"Unexpected error: {str(e)}")
                self.finished.emit()
                return
        else:
            self.progress_updated.emit(f"requirements.txt not found: {self.requirements_file}")
        self.finished.emit()

class SetupManager(QObject):
    progress_updated = pyqtSignal(str)

    def __init__(self, project_root):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.data_dir = self.project_root / "data"
        self.ckpt_file = self.data_dir / "v1-5-pruned-emaonly.ckpt"
        self.requirements_file = self.project_root / "requirements.txt"
        self.ckpt_url = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"

    def check_requirements(self):
        self.progress_updated.emit(f"Checking requirements at project root: {self.project_root}")
        missing = []
        if not self.data_dir.exists():
            missing.append(f"Directory missing: {self.data_dir}")
            self.progress_updated.emit(f"Missing: {self.data_dir}")
        else:
            self.progress_updated.emit(f"Found: {self.data_dir}")
        if not self.ckpt_file.exists():
            missing.append(f"Checkpoint file missing: {self.ckpt_file}")
            self.progress_updated.emit(f"Missing: {self.ckpt_file}")
        else:
            self.progress_updated.emit(f"Found: {self.ckpt_file}")
        if self.requirements_file.exists():
            self.progress_updated.emit(f"Checking packages from: {self.requirements_file}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                installed = {line.split("==")[0].lower() for line in result.stdout.splitlines()}
                with open(self.requirements_file, "r") as f:
                    required = {line.strip().split("==")[0].lower() for line in f if line.strip() and not line.startswith("#")}
                missing_pkgs = required - installed
                if missing_pkgs:
                    missing.append(f"Missing packages: {', '.join(missing_pkgs)}")
                    self.progress_updated.emit(f"Missing packages: {', '.join(missing_pkgs)}")
                else:
                    self.progress_updated.emit("All required packages found.")
            except Exception as e:
                missing.append(f"Error checking pip requirements: {str(e)}")
                self.progress_updated.emit(f"Error checking pip requirements: {str(e)}")
        else:
            missing.append(f"requirements.txt missing: {self.requirements_file}")
            self.progress_updated.emit(f"Missing: {self.requirements_file}")
        return missing

    def create_worker(self):
        return SetupWorker(self.data_dir, self.ckpt_file, self.requirements_file, self.ckpt_url)