import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QPushButton, QLineEdit, QProgressBar, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPixmap, QColor, QFont
from PIL import Image
import os
import numpy as np
from sd.demo import generate_image  # Import the function from demo.py

class DiffusionModel:
    @staticmethod
    def process(image_path, sentence, progress_callback=None):
        try:
            # Load the input image as a PIL Image
            input_image = Image.open(image_path).convert("RGB")
            # Call the generate_image function from demo.py
            output_image = generate_image(input_image, sentence, progress_callback)
            # Save the output image to a temporary file
            output_path = "temp_output.png"
            Image.fromarray(output_image).save(output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Diffusion model failed: {str(e)}")

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image and Sentence Processor")
        self.image_path = None
        self._gradient_color = QColor("#f0f0f0")  # Initial color
        # Set initial size and remove minimum size constraints
        self.resize(1280, 720)
        self.setMinimumSize(0, 0)
        self.init_ui()
        # Maximize window after UI setup
        self.setWindowState(Qt.WindowMaximized)

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

        # Set larger font for all widgets
        large_font = QFont("Arial", 14)

        # Image load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setFont(large_font)
        self.load_btn.setStyleSheet("padding: 15px; margin: 10px;")
        self.load_btn.setMinimumHeight(60)
        self.load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_btn)

        # Sentence input
        self.sentence_input = QLineEdit()
        self.sentence_input.setFont(large_font)
        self.sentence_input.setStyleSheet("padding: 15px; margin: 10px;")
        self.sentence_input.setMinimumHeight(60)
        self.sentence_input.setPlaceholderText("Enter a sentence")
        left_layout.addWidget(self.sentence_input)

        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setFont(large_font)
        self.generate_btn.setStyleSheet("padding: 15px; margin: 10px;")
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

        # Spacer to push items up
        left_layout.addStretch()

        # Right half: Output
        self.output_label = QLabel()
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setFont(large_font)
        self.output_label.setText("Output will appear here")
        self.output_label.setStyleSheet("background-color: #f0f0f0; margin: 10px;")

        # Add widgets to splitter with relaxed size constraints
        splitter.addWidget(left_widget)
        splitter.addWidget(self.output_label)
        splitter.setSizes([self.width() // 2, self.width() // 2])
        left_widget.setMinimumSize(0, 0)
        self.output_label.setMinimumSize(0, 0)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            self.image_path = file_dialog.selectedFiles()[0]
            self.load_btn.setText(f"Image: {os.path.basename(self.image_path)}")

    def start_processing(self):
        # Validate inputs
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please select an image.")
            return
        if not self.sentence_input.text().strip():
            QMessageBox.warning(self, "Error", "Please enter a sentence.")
            return

        # Disable UI elements
        self.load_btn.setEnabled(False)
        self.sentence_input.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.eta_label.setVisible(True)
        self.elapsed_label.setVisible(True)
        self.progress_bar.setValue(0)

        # Start gradient animation
        self.start_gradient_animation()

        # Initialize progress tracking
        self.progress_value = 0
        self.start_time = time.time()
        self.step_times = []
        self.total_steps = None

        # Define progress callback
        def progress_callback(step, total_steps, step_time):
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
            # Update elapsed time
            elapsed_time = time.time() - self.start_time
            self.elapsed_label.setText(f"Elapsed: {elapsed_time:.1f} s")
            # Process UI events to keep it responsive
            QApplication.processEvents()

        # Start processing in a timer to keep UI responsive
        self.process_timer = QTimer()
        self.process_timer.setSingleShot(True)
        self.process_timer.timeout.connect(lambda: self.process_image(progress_callback))
        self.process_timer.start(0)

    def start_gradient_animation(self):
        # Clear any existing pixmap or text
        self.output_label.setPixmap(QPixmap())
        self.output_label.setText("")

        # Gradient animation using QPropertyAnimation
        self.animation = QPropertyAnimation(self, b"gradient")
        self.animation.setDuration(2000)  # 2 seconds per cycle
        self.animation.setLoopCount(-1)  # Loop indefinitely
        self.animation.setEasingCurve(QEasingCurve.InOutSine)

        # Define gradient keyframes
        self.animation.setKeyValueAt(0.0, QColor("#ff6f61"))  # Coral
        self.animation.setKeyValueAt(0.5, QColor("#6b7280"))  # Gray
        self.animation.setKeyValueAt(1.0, QColor("#ff6f61"))  # Coral

        self.animation.start()

    def process_image(self, progress_callback):
        try:
            # Process with diffusion model
            sentence = self.sentence_input.text().strip()
            output_path = DiffusionModel.process(self.image_path, sentence, progress_callback)

            # Stop gradient animation
            if hasattr(self, 'animation'):
                self.animation.stop()
                self.output_label.setStyleSheet("background-color: #f0f0f0; margin: 10px;")

            # Display output image
            pixmap = QPixmap(output_path)
            if pixmap.isNull():
                raise ValueError("Failed to load output image.")
            
            # Scale pixmap to fit right half while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.output_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.output_label.setPixmap(scaled_pixmap)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            self.output_label.setText("Error displaying output")
        finally:
            # Re-enable UI elements
            self.load_btn.setEnabled(True)
            self.sentence_input.setEnabled(True)
            self.generate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.eta_label.setVisible(False)
            self.elapsed_label.setVisible(False)

    # Custom property for gradient animation
    @pyqtProperty(QColor)
    def gradient(self):
        return self._gradient_color

    @gradient.setter
    def gradient(self, color):
        self._gradient_color = color
        self.output_label.setStyleSheet(f"background-color: {color.name()}; margin: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()  # Explicitly call show to ensure proper initialization
    sys.exit(app.exec_())

