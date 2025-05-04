import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
                             QPushButton, QLineEdit, QProgressBar, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PIL import Image
import os

class PlaceholderModel:
    @staticmethod
    def process(image_path, sentence):
        # Simulate processing (e.g., delay)
        time.sleep(2)
        # Placeholder: return input image path as output
        return image_path

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image and Sentence Processor")
        self.showFullScreen()
        self.image_path = None
        self.init_ui()

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

        # Image load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_btn)

        # Sentence input
        self.sentence_input = QLineEdit()
        self.sentence_input.setPlaceholderText("Enter a sentence")
        left_layout.addWidget(self.sentence_input)

        # Generate button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.start_processing)
        left_layout.addWidget(self.generate_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Spacer to push items up
        left_layout.addStretch()

        # Right half: Output
        self.output_label = QLabel()
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setText("Output will appear here")
        self.output_label.setStyleSheet("background-color: #f0f0f0;")

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(self.output_label)
        splitter.setSizes([self.width() // 2, self.width() // 2])

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
        self.progress_bar.setValue(0)

        # Start progress simulation
        self.progress_value = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # Update every 50ms

    def update_progress(self):
        self.progress_value += 2
        self.progress_bar.setValue(self.progress_value)

        if self.progress_value >= 100:
            self.timer.stop()
            self.process_image()
            # Re-enable UI elements
            self.load_btn.setEnabled(True)
            self.sentence_input.setEnabled(True)
            self.generate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def process_image(self):
        try:
            # Process with placeholder model
            sentence = self.sentence_input.text().strip()
            output_path = PlaceholderModel.process(self.image_path, sentence)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    sys.exit(app.exec_())
