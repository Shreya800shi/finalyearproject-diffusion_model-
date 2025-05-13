import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QPushButton, QLineEdit, QProgressBar, QLabel, QFileDialog, QMessageBox, QDialog,
                             QSizePolicy, QCheckBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PIL import Image
import os
from sd.demo import generate_image  # Import the function from demo.py

class DiffusionModel:
    @staticmethod
    def process(image_path, sentence, uncond_prompt, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed, progress_callback=None):
        try:
            # Load the input image as a PIL Image
            input_image = Image.open(image_path).convert("RGB")
            # Call the generate_image function from demo.py
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
            # Return the NumPy array directly
            return output_image
        except Exception as e:
            raise RuntimeError(f"Diffusion model failed: {str(e)}")

class Worker(QObject):
    progress = pyqtSignal(int, int, float)  # step, total_steps, step_time
    finished = pyqtSignal(np.ndarray)  # output_image
    error = pyqtSignal(str)  # error message

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

    def run(self):
        try:
            def progress_callback(step, total_steps, step_time):
                self.progress.emit(step, total_steps, step_time)

            output_image = DiffusionModel.process(
                self.image_path, self.sentence, self.uncond_prompt, self.strength, self.do_cfg,
                self.cfg_scale, self.sampler, self.num_inference_steps, self.seed, progress_callback
            )
            self.finished.emit(output_image)
        except Exception as e:
            self.error.emit(str(e))

class FullScreenImageDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")

        # Set dialog to full-screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        # Main widget to hold image and button
        main_widget = QWidget(self)
        main_widget.setGeometry(screen)

        # Layout for image with no padding
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No padding
        layout.setSpacing(0)
        main_widget.setLayout(layout)

        # Image label with margins
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: transparent; margin-top: 40px; margin-bottom: 70px;")
        # Scale pixmap to fit screen (accounting for 110px total margin) while maintaining aspect ratio
        scaled_size = screen.size() - QSize(0, 110)  # Subtract 40px top + 70px bottom margins
        scaled_pixmap = pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        layout.addWidget(self.image_label)

        # Close button with absolute positioning
        self.close_button = QPushButton("✕", self)
        self.close_button.setFixedSize(40, 40)  # 20px radius
        self.close_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
                border: 2px solid white;
                border-radius: 20px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        # Position button 20px from top-right corner
        self.close_button.setGeometry(screen.width() - 60, 20, 40, 40)  # 60 = 40px size + 20px margin
        self.close_button.clicked.connect(self.close)
        self.close_button.raise_()  # Ensure button is on top

    def mousePressEvent(self, event):
        # Close dialog when clicking outside the image, but not on the close button
        if not self.image_label.geometry().contains(event.pos()) and not self.close_button.geometry().contains(event.pos()):
            self.close()

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image and Sentence Processor")
        self.image_path = None
        self.output_image = None  # Store output image for download
        self._gradient_color = QColor("#f0f0f0")  # Initial color
        # Set initial size and remove minimum size constraints
        self.resize(1280, 720)
        self.setMinimumSize(0, 0)
        self.init_ui()
        # Maximize window after UI setup
        self.setWindowState(Qt.WindowMaximized)
        # Initialize random noise for input image placeholder
        self.set_random_noise()

    def init_ui(self):
        # Main widget and splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_widget.setLayout(QVBoxLayout())
        main_widget.layout().addWidget(splitter)

        # Left half: Inputs
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # Add top spacer for vertical centering
        left_layout.addStretch()

        # Set larger font for all widgets
        large_font = QFont("Arial", 14)

        # Image load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setFont(large_font)
        self.load_btn.setStyleSheet("padding: 15px; margin: 10px;")
        self.load_btn.setMinimumHeight(60)
        self.load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_btn)

        # Prompt input
        prompt_layout = QHBoxLayout()
        self.prompt_label = QLabel("Enter Prompt:")
        self.prompt_label.setFont(large_font)
        self.prompt_label.setStyleSheet("margin: 10px;")
        prompt_layout.addWidget(self.prompt_label)
        self.sentence_input = QLineEdit()
        self.sentence_input.setFont(large_font)
        self.sentence_input.setStyleSheet("padding: 15px; margin: 10px;")
        self.sentence_input.setMinimumHeight(60)
        self.sentence_input.setPlaceholderText("Enter a sentence")
        prompt_layout.addWidget(self.sentence_input)
        left_layout.addLayout(prompt_layout)

        # Unconditional prompt input
        uncond_prompt_layout = QHBoxLayout()
        self.uncond_prompt_label = QLabel("Unconditional Prompt(Leave Blank if None):")
        self.uncond_prompt_label.setFont(large_font)
        self.uncond_prompt_label.setStyleSheet("margin: 10px;")
        uncond_prompt_layout.addWidget(self.uncond_prompt_label)
        self.uncond_prompt_input = QLineEdit()
        self.uncond_prompt_input.setFont(large_font)
        self.uncond_prompt_input.setStyleSheet("padding: 15px; margin: 10px;")
        self.uncond_prompt_input.setMinimumHeight(60)
        self.uncond_prompt_input.setPlaceholderText("Leave blank if none")
        uncond_prompt_layout.addWidget(self.uncond_prompt_input)
        left_layout.addLayout(uncond_prompt_layout)

        # Use Default Parameters checkbox
        self.default_params_checkbox = QCheckBox("Use Default Parameters")
        self.default_params_checkbox.setFont(large_font)
        self.default_params_checkbox.setStyleSheet("transform: scale(1.5); margin: 15px;")
        self.default_params_checkbox.setChecked(True)
        self.default_params_checkbox.stateChanged.connect(self.toggle_default_params)
        left_layout.addWidget(self.default_params_checkbox)

        # Custom parameters widget
        self.params_widget = QWidget()
        self.params_widget.setStyleSheet("""
            background-color: #8B8A7B;
            border-radius: 15px;
            margin: 20px;
            padding: 20px;
        """)
        params_layout = QVBoxLayout()
        params_layout.setSpacing(0)  # No gaps between elements
        self.params_widget.setLayout(params_layout)

        # Strength input
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("Strength:")
        self.strength_label.setFont(large_font)
        self.strength_label.setStyleSheet("margin: 10px;")
        strength_layout.addWidget(self.strength_label)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(1)  # 0.01 * 100
        self.strength_slider.setMaximum(100)  # 1.0 * 100
        self.strength_slider.setValue(90)  # 0.9 * 100
        self.strength_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #2776EA;
                border: 1px solid #2776EA;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d3d3d3;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #2776EA;
                border-radius: 2px;
            }
            margin: 10px;
        """)
        strength_layout.addWidget(self.strength_slider)
        self.strength_input = QLineEdit("0.9")
        self.strength_input.setFont(large_font)
        self.strength_input.setStyleSheet("""
            background-color: #f0f0f0;
            border: none;
            border-radius: 8px;
            padding: 5px;
            margin: 10px;
        """)
        self.strength_input.setFixedWidth(100)
        strength_layout.addWidget(self.strength_input)
        params_layout.addLayout(strength_layout)

        # CFG Scale input
        cfg_scale_layout = QHBoxLayout()
        self.cfg_scale_label = QLabel("CFG Scale:")
        self.cfg_scale_label.setFont(large_font)
        self.cfg_scale_label.setStyleSheet("margin: 10px;")
        cfg_scale_layout.addWidget(self.cfg_scale_label)
        self.cfg_scale_slider = QSlider(Qt.Horizontal)
        self.cfg_scale_slider.setMinimum(100)  # 1 * 100
        self.cfg_scale_slider.setMaximum(1400)  # 14 * 100
        self.cfg_scale_slider.setValue(800)  # 8 * 100
        self.cfg_scale_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #2776EA;
                border: 1px solid #2776EA;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d3d3d3;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #2776EA;
                border-radius: 2px;
            }
            margin: 10px;
        """)
        cfg_scale_layout.addWidget(self.cfg_scale_slider)
        self.cfg_scale_input = QLineEdit("8")
        self.cfg_scale_input.setFont(large_font)
        self.cfg_scale_input.setStyleSheet("""
            background-color: #f0f0f0;
            border: none;
            border-radius: 8px;
            padding: 5px;
            margin: 10px;
        """)
        self.cfg_scale_input.setFixedWidth(100)
        cfg_scale_layout.addWidget(self.cfg_scale_input)
        params_layout.addLayout(cfg_scale_layout)

        # Inference Steps input
        steps_layout = QHBoxLayout()
        self.steps_label = QLabel("Inference Steps:")
        self.steps_label.setFont(large_font)
        self.steps_label.setStyleSheet("margin: 10px;")
        steps_layout.addWidget(self.steps_label)
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(1)
        self.steps_slider.setMaximum(200)
        self.steps_slider.setValue(50)
        self.steps_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #2776EA;
                border: 1px solid #2776EA;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d3d3d3;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #2776EA;
                border-radius: 2px;
            }
            margin: 10px;
        """)
        steps_layout.addWidget(self.steps_slider)
        self.steps_input = QLineEdit("50")
        self.steps_input.setFont(large_font)
        self.steps_input.setStyleSheet("""
            background-color: #f0f0f0;
            border: none;
            border-radius: 8px;
            padding: 5px;
            margin: 10px;
        """)
        self.steps_input.setFixedWidth(100)
        steps_layout.addWidget(self.steps_input)
        params_layout.addLayout(steps_layout)

        # Note for manual entry
        self.note_label = QLabel("*Use text box for manual data entry")
        self.note_label.setFont(large_font)
        self.note_label.setStyleSheet("margin: 10px;")
        params_layout.addWidget(self.note_label)

        left_layout.addWidget(self.params_widget)

        # Connect sliders and text inputs
        self.strength_slider.valueChanged.connect(self.update_strength_input)
        self.strength_input.textChanged.connect(self.update_strength_slider)
        self.cfg_scale_slider.valueChanged.connect(self.update_cfg_scale_input)
        self.cfg_scale_input.textChanged.connect(self.update_cfg_scale_slider)
        self.steps_slider.valueChanged.connect(self.update_steps_input)
        self.steps_input.textChanged.connect(self.update_steps_slider)

        # Initialize default parameters state
        self.toggle_default_params()

        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setFont(large_font)
        self.generate_btn.setStyleSheet("""
            background-color: #01d449;
            color: white;
            border-radius: 30px;
            padding: 15px;
            margin: 10px;
        """)
        self.generate_btn.setMinimumHeight(60)
        self.generate_btn.clicked.connect(self.start_processing)
        left_layout.addWidget(self.generate_btn)

        # ETA timer label
        self.eta_label = QLabel("ETA: -- s")
        self.eta_label.setFont(large_font)
        self.eta_label.setStyleSheet("margin: 10px;")
        self.eta_label.setVisible(False)
        left_layout.addWidget(self.eta_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(large_font)
        self.progress_bar.setStyleSheet("padding: 10px; margin: 10px; height: 40px;")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Time elapsed label
        self.elapsed_label = QLabel("Elapsed: 0.0 s")
        self.elapsed_label.setFont(large_font)
        self.elapsed_label.setStyleSheet("margin: 10px;")
        self.elapsed_label.setVisible(False)
        left_layout.addWidget(self.elapsed_label)

        # Add bottom spacer for vertical centering
        left_layout.addStretch()

        # Separator
        separator = QWidget()
        separator.setFixedWidth(3)
        separator.setStyleSheet("""
            background-color: #2776EA;
            border-radius: 3px;
            margin-top: 20px;
            margin-bottom: 20px;
        """)

        # Right half: Input and Output Images
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        # Add top spacer for vertical centering
        right_layout.addStretch()

        # Input Image Section
        self.input_label = QLabel("Input Image:")
        self.input_label.setFont(large_font)
        self.input_label.setAlignment(Qt.AlignLeft)
        self.input_label.setStyleSheet("margin: 10px 10px 0px 10px; height: 20px;")
        right_layout.addWidget(self.input_label)

        self.input_image_label = QLabel()
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet("""
            border: 1px solid black;
            border-radius: 15px;
            padding: 15px;
            margin: 10px;
            background-color: #f0f0f0;
        """)
        self.input_image_label.setFixedHeight(350)
        self.input_image_label.setMouseTracking(True)
        self.input_image_label.mousePressEvent = self.show_full_screen_input
        right_layout.addWidget(self.input_image_label)

        # Output Image Section
        self.output_label = QLabel("Output Image:")
        self.output_label.setFont(large_font)
        self.output_label.setAlignment(Qt.AlignLeft)
        self.output_label.setStyleSheet("margin: 10px 10px 0px 10px; height: 20px;")
        right_layout.addWidget(self.output_label)

        self.output_image_label = QLabel()
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setStyleSheet("""
            border: 1px dotted black;
            border-radius: 15px;
            padding: 15px;
            margin: 10px;
            background-color: #f0f0f0;
        """)
        self.output_image_label.setFixedHeight(350)
        self.output_image_label.setMouseTracking(True)
        self.output_image_label.mousePressEvent = self.show_full_screen_output
        right_layout.addWidget(self.output_image_label)

        # Download button
        self.download_btn = QPushButton("Download")
        self.download_btn.setFont(large_font)
        self.download_btn.setStyleSheet("""
            background-color: #2776EA;
            color: white;
            border-radius: 15px;
            padding: 15px;
            margin: 10px;
        """)
        self.download_btn.setMinimumHeight(60)
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self.download_image)
        right_layout.addWidget(self.download_btn)

        # Add bottom spacer for vertical centering
        right_layout.addStretch()

        # Add widgets to splitter with separator
        splitter.addWidget(left_widget)
        splitter.addWidget(separator)
        splitter.addWidget(right_widget)
        splitter.setSizes([self.width() // 2, 3, self.width() // 2 - 3])
        left_widget.setMinimumSize(0, 0)
        right_widget.setMinimumSize(0, 0)
        separator.setMinimumSize(3, 0)

    def toggle_default_params(self):
        # Enable/disable custom parameter inputs, reset to defaults, and update background
        is_checked = self.default_params_checkbox.isChecked()
        self.params_widget.setEnabled(not is_checked)
        bg_color = "#4d495b" if is_checked else "#8B8A7B"
        self.params_widget.setStyleSheet(f"""
            background-color: {bg_color};
            border-radius: 15px;
            margin: 20px;
            padding: 20px;
        """)
        if is_checked:
            # Block signals to prevent recursive updates
            self.strength_slider.blockSignals(True)
            self.strength_input.blockSignals(True)
            self.cfg_scale_slider.blockSignals(True)
            self.cfg_scale_input.blockSignals(True)
            self.steps_slider.blockSignals(True)
            self.steps_input.blockSignals(True)

            # Reset to default values
            self.strength_slider.setValue(90)  # 0.9 * 100
            self.strength_input.setText("0.9")
            self.cfg_scale_slider.setValue(800)  # 8 * 100
            self.cfg_scale_input.setText("8")
            self.steps_slider.setValue(50)
            self.steps_input.setText("50")

            # Re-enable signals
            self.strength_slider.blockSignals(False)
            self.strength_input.blockSignals(False)
            self.cfg_scale_slider.blockSignals(False)
            self.cfg_scale_input.blockSignals(False)
            self.steps_slider.blockSignals(False)
            self.steps_input.blockSignals(False)

    def update_strength_input(self):
        value = self.strength_slider.value() / 100.0
        self.strength_input.setText(f"{value:.2f}")

    def update_strength_slider(self):
        try:
            value = float(self.strength_input.text())
            if 0.01 <= value <= 1.0:
                self.strength_slider.setValue(int(value * 100))
            else:
                self.strength_input.setText(f"{self.strength_slider.value() / 100.0:.2f}")
        except ValueError:
            self.strength_input.setText(f"{self.strength_slider.value() / 100.0:.2f}")

    def update_cfg_scale_input(self):
        value = self.cfg_scale_slider.value() / 100.0
        self.cfg_scale_input.setText(f"{value:.1f}")

    def update_cfg_scale_slider(self):
        try:
            value = float(self.cfg_scale_input.text())
            if 1.0 <= value <= 14.0:
                self.cfg_scale_slider.setValue(int(value * 100))
            else:
                self.cfg_scale_input.setText(f"{self.cfg_scale_slider.value() / 100.0:.1f}")
        except ValueError:
            self.cfg_scale_input.setText(f"{self.cfg_scale_slider.value() / 100.0:.1f}")

    def update_steps_input(self):
        value = self.steps_slider.value()
        self.steps_input.setText(str(value))

    def update_steps_slider(self):
        try:
            value = int(self.steps_input.text())
            if 1 <= value <= 200:
                self.steps_slider.setValue(value)
            else:
                self.steps_input.setText(str(self.steps_slider.value()))
        except ValueError:
            self.steps_input.setText(str(self.steps_slider.value()))

    def set_random_noise(self):
        # Generate random noise (512x512 RGB)
        noise = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        height, width, channels = noise.shape
        bytes_per_line = channels * width
        image = QImage(noise.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        # Scale to fit within 350px height, accounting for 15px padding
        scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.input_image_label.setPixmap(scaled_pixmap)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            self.image_path = file_dialog.selectedFiles()[0]
            self.load_btn.setText(f"Image: {os.path.basename(self.image_path)}")
            # Display input image
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Scale to fit within 350px height, accounting for 15px padding
                scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.input_image_label.setPixmap(scaled_pixmap)

    def show_full_screen_input(self, event):
        if self.image_path and self.input_image_label.pixmap() and not self.input_image_label.pixmap().isNull():
            pixmap = QPixmap(self.image_path)  # Use original image
            dialog = FullScreenImageDialog(pixmap, self)
            dialog.showFullScreen()
            dialog.exec_()

    def show_full_screen_output(self, event):
        if self.output_image_label.pixmap() and not self.output_image_label.pixmap().isNull():
            pixmap = self.output_image_label.pixmap()  # Use current pixmap
            dialog = FullScreenImageDialog(pixmap, self)
            dialog.showFullScreen()
            dialog.exec_()

    def download_image(self):
        if self.output_image is None:
            QMessageBox.warning(self, "Error", "No output image available to download.")
            return
        # Open save file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            try:
                # Convert NumPy array to PIL Image and save
                Image.fromarray(self.output_image).save(file_path)
                QMessageBox.information(self, "Success", "Image saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def start_processing(self):
        # Validate inputs
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image.")
            return
        if not self.sentence_input.text().strip():
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        # Disable UI elements
        self.load_btn.setEnabled(False)
        self.sentence_input.setEnabled(False)
        self.uncond_prompt_input.setEnabled(False)
        self.default_params_checkbox.setEnabled(False)
        self.params_widget.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.download_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.eta_label.setVisible(True)
        self.elapsed_label.setVisible(True)
        self.progress_bar.setValue(0)

        # Clear output image and start gradient animation
        self.output_image_label.setPixmap(QPixmap())
        self.output_image = None
        self.start_gradient_animation()

        # Initialize progress tracking
        self.start_time = time.time()
        self.step_times = []
        self.total_steps = None

        # Get parameters
        sentence = self.sentence_input.text().strip()
        uncond_prompt = self.uncond_prompt_input.text().strip()
        if self.default_params_checkbox.isChecked():
            strength = 0.9
            do_cfg = True
            cfg_scale = 8
            sampler = "ddpm"
            num_inference_steps = 50
            seed = 42
        else:
            try:
                strength = float(self.strength_input.text())
                cfg_scale = float(self.cfg_scale_input.text())
                num_inference_steps = int(self.steps_input.text())
            except ValueError:
                QMessageBox.warning(self, "Error", "Invalid parameter values.")
                self.cleanup()
                return
            do_cfg = True
            sampler = "ddpm"
            seed = 42

        # Start worker thread
        self.thread = QThread()
        self.worker = Worker(self.image_path, sentence, uncond_prompt, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        # Update elapsed time in real-time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.elapsed_timer.start(100)  # Update every 100ms

    def start_gradient_animation(self):
        # Apply gradient animation to output_image_label
        self.animation = QPropertyAnimation(self, b"gradient")
        self.animation.setDuration(2000)  # 2 seconds per cycle
        self.animation.setLoopCount(-1)  # Loop indefinitely
        self.animation.setEasingCurve(QEasingCurve.InOutSine)
        self.animation.setKeyValueAt(0.0, QColor("#ff6f61"))  # Coral
        self.animation.setKeyValueAt(0.5, QColor("#6b7280"))  # Gray
        self.animation.setKeyValueAt(1.0, QColor("#ff6f61"))  # Coral
        self.animation.start()

    def update_progress(self, step, total_steps, step_time):
        self.total_steps = total_steps
        self.step_times.append(step_time)
        # Calculate progress percentage
        self.progress_value = int((step + 1) / total_steps * 100)
        self.progress_bar.setValue(self.progress_value)
        # Calculate ETA based on average step time
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = total_steps - (step + 1)
            eta = avg_step_time * remaining_steps
            self.eta_label.setText(f"ETA: {eta:.1f} s")
        else:
            self.eta_label.setText("ETA: -- s")

    def update_elapsed_time(self):
        elapsed_time = time.time() - self.start_time
        self.elapsed_label.setText(f"Elapsed: {elapsed_time:.1f} s")

    def on_processing_finished(self, output_image):
        # Stop gradient animation and elapsed timer
        if hasattr(self, 'animation'):
            self.animation.stop()
            self.output_image_label.setStyleSheet("""
                border: 1px dotted black;
                border-radius: 15px;
                padding: 15px;
                margin: 10px;
                background-color: #f0f0f0;
            """)
        self.elapsed_timer.stop()
        # Store output image for download
        self.output_image = output_image
        # Convert NumPy array to QPixmap
        try:
            pil_image = Image.fromarray(output_image)
            pil_image = pil_image.convert("RGB")
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            if not pixmap.isNull():
                # Scale to fit within 350px height, accounting for 15px padding
                scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.output_image_label.setPixmap(scaled_pixmap)
                # Show download button
                self.download_btn.setVisible(True)
            else:
                raise ValueError("Failed to convert image to pixmap")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load output image: {str(e)}")
        # Cleanup
        self.cleanup()

    def on_processing_error(self, error_message):
        # Stop gradient animation and elapsed timer
        if hasattr(self, 'animation'):
            self.animation.stop()
            self.output_image_label.setStyleSheet("""
                border: 1px dotted black;
                border-radius: 15px;
                padding: 15px;
                margin: 10px;
                background-color: #f0f0f0;
            """)
        self.elapsed_timer.stop()
        # Show error
        QMessageBox.critical(self, "Error", f"Processing failed: {error_message}")
        self.output_image_label.setText("Error displaying output")
        # Cleanup
        self.cleanup()

    def cleanup(self):
        # Re-enable UI elements and hide progress indicators
        self.load_btn.setEnabled(True)
        self.sentence_input.setEnabled(True)
        self.uncond_prompt_input.setEnabled(True)
        self.default_params_checkbox.setEnabled(True)
        self.toggle_default_params()  # Update params_widget state
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.eta_label.setVisible(False)
        self.elapsed_label.setVisible(False)
        # Quit thread
        self.thread.quit()
        self.thread.wait()

    # Custom property for gradient animation
    @pyqtProperty(QColor)
    def gradient(self):
        return self._gradient_color

    @gradient.setter
    def gradient(self, color):
        self._gradient_color = color
        self.output_image_label.setStyleSheet(f"""
            border: 1px dotted black;
            border-radius: 15px;
            padding: 15px;
            margin: 10px;
            background-color: {color.name()};
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()  # Explicitly call show to ensure proper initialization
    sys.exit(app.exec_())