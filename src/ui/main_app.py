import time
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLineEdit, QProgressBar, QLabel, QFileDialog,
    QMessageBox, QSizePolicy, QCheckBox, QDialog, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QThread, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize
from PyQt5.QtGui import QPixmap, QColor
import numpy as np
from PIL import Image
from ui.dialogs import FullScreenImageDialog
from workers.processing import Worker
from utils.image_utils import numpy_to_pixmap, validate_image_path
from utils.setup import SetupManager
from config import *

class SetupDialog(QDialog):
    def __init__(self, setup_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup Requirements")
        self.setFixedSize(600, 400)
        self.setup_manager = setup_manager
        self.thread = None  # Initialize thread as None
        self.init_ui()
        self.setup_manager.progress_updated.connect(self.update_log)

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(LARGE_FONT)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.log_area)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.recheck_btn = QPushButton("Recheck")
        self.recheck_btn.setFont(LARGE_FONT)
        self.recheck_btn.setStyleSheet("""
            background-color: #2776EA;
            color: white;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
        """)
        self.recheck_btn.clicked.connect(self.recheck_requirements)
        layout.addWidget(self.recheck_btn)

        self.install_requirements()

    def update_log(self, message):
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def install_requirements(self):
        self.recheck_btn.setEnabled(False)
        self.thread = QThread()
        self.worker = self.setup_manager.create_worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.install_requirements)
        self.worker.progress_updated.connect(self.update_log)
        self.worker.finished.connect(self.on_install_finished)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def on_install_finished(self):
        self.recheck_btn.setEnabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None  # Clear thread reference

    def on_thread_finished(self):
        self.thread = None  # Clear thread reference after it finishes

    def recheck_requirements(self):
        self.recheck_btn.setEnabled(False)
        missing = self.setup_manager.check_requirements()
        if not missing:
            self.log_area.append("All Set up Done!")
            self.recheck_btn.setText("Done")
            self.recheck_btn.setStyleSheet("""
                background-color: #01D449;
                color: white;
                border-radius: 15px;
                padding: 10px;
                margin: 5px;
            """)
            self.recheck_btn.clicked.disconnect()
            self.recheck_btn.clicked.connect(self.close_and_hide_setup)
        else:
            self.log_area.append("Requirements still missing:")
            for item in missing:
                self.log_area.append(f"- {item}")
            self.install_requirements()

    def close_and_hide_setup(self):
        parent = self.parent()
        if parent and hasattr(parent, 'setup_requirements_btn'):
            parent.setup_requirements_btn.setVisible(False)
        self.accept()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        self.thread = None  # Clear thread reference
        super().closeEvent(event)

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image and Sentence Processor")
        self.image_path = None
        self.output_image = None
        self._gradient_color = QColor("#222222")
        self.current_mode = DEFAULT_MODE  # Set to Text-to-Image
        self.resize(1280, 720)
        self.setMinimumSize(0, 0)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.setup_manager = SetupManager(project_root)
        self.init_ui()
        self.setWindowState(Qt.WindowMaximized)
        self.set_random_noise()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_widget.setLayout(QVBoxLayout())
        main_widget.layout().addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        left_layout.addStretch()

        self.setup_requirements_btn = QPushButton("CLICK HERE! Set up Requirements")
        self.setup_requirements_btn.setFont(LARGE_FONT)
        self.setup_requirements_btn.setStyleSheet(SETUP_BUTTON_STYLE)
        self.setup_requirements_btn.setMinimumHeight(60)
        self.setup_requirements_btn.clicked.connect(self.show_setup_dialog)
        missing_requirements = self.setup_manager.check_requirements()
        self.setup_requirements_btn.setVisible(bool(missing_requirements))
        if missing_requirements:
            left_layout.addWidget(self.setup_requirements_btn)

        mode_layout = QHBoxLayout()
        self.button_style_active = f"""
            QPushButton {{
                background-color: {BUTTON_ACTIVE_COLOR};
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px;
                margin: 5px;
            }}
            QPushButton:hover:enabled {{
                background-color: {BUTTON_ACTIVE_COLOR};
            }}
        """
        self.button_style_inactive = f"""
            QPushButton {{
                background-color: {BUTTON_INACTIVE_COLOR};
                color: white;
                border: none;
                border-radius: 25px;
                padding: 15px;
                margin: 5px;
            }}
            QPushButton:hover:enabled {{
                background-color: {BUTTON_INACTIVE_COLOR};
            }}
        """

        self.text_to_image_btn = QPushButton("Text-to-Image")
        self.text_to_image_btn.setFont(LARGE_FONT)
        self.text_to_image_btn.setStyleSheet(self.button_style_active)
        self.text_to_image_btn.setMinimumHeight(60)
        self.text_to_image_btn.clicked.connect(lambda: self.switch_mode("Text-to-Image"))
        mode_layout.addWidget(self.text_to_image_btn)

        self.image_to_image_btn = QPushButton("Image-to-Image")
        self.image_to_image_btn.setFont(LARGE_FONT)
        self.image_to_image_btn.setStyleSheet(self.button_style_inactive)
        self.image_to_image_btn.setMinimumHeight(60)
        self.image_to_image_btn.clicked.connect(lambda: self.switch_mode("Image-to-Image"))
        mode_layout.addWidget(self.image_to_image_btn)

        self.inpainting_btn = QPushButton("Image-InPainting")
        self.inpainting_btn.setFont(LARGE_FONT)
        self.inpainting_btn.setStyleSheet(self.button_style_inactive)
        self.inpainting_btn.setMinimumHeight(60)
        self.inpainting_btn.setEnabled(False)
        mode_layout.addWidget(self.inpainting_btn)

        left_layout.addLayout(mode_layout)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.setFont(LARGE_FONT)
        self.load_btn.setStyleSheet("padding: 15px; margin: 10px;")
        self.load_btn.setMinimumHeight(60)
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setVisible(False)
        left_layout.addWidget(self.load_btn)

        prompt_layout = QHBoxLayout()
        self.prompt_label = QLabel("Enter Prompt:")
        self.prompt_label.setFont(LARGE_FONT)
        self.prompt_label.setStyleSheet("margin: 10px;")
        prompt_layout.addWidget(self.prompt_label)
        self.sentence_input = QLineEdit()
        self.sentence_input.setFont(LARGE_FONT)
        self.sentence_input.setStyleSheet("padding: 15px; margin: 10px;")
        self.sentence_input.setMinimumHeight(60)
        self.sentence_input.setPlaceholderText("Enter a sentence")
        prompt_layout.addWidget(self.sentence_input)
        left_layout.addLayout(prompt_layout)

        uncond_prompt_layout = QHBoxLayout()
        self.uncond_prompt_label = QLabel("Unconditional Prompt (Leave Blank if None):")
        self.uncond_prompt_label.setFont(LARGE_FONT)
        self.uncond_prompt_label.setStyleSheet("margin: 10px;")
        uncond_prompt_layout.addWidget(self.uncond_prompt_label)
        self.uncond_prompt = QLineEdit()
        self.uncond_prompt.setFont(LARGE_FONT)
        self.uncond_prompt.setStyleSheet("padding: 15px; margin: 10px;")
        self.uncond_prompt.setMinimumHeight(40)
        self.uncond_prompt.setPlaceholderText("Leave blank if none")
        uncond_prompt_layout.addWidget(self.uncond_prompt)
        left_layout.addLayout(uncond_prompt_layout)

        self.update_default_params_label()
        self.default_params_checkbox = QCheckBox(self.default_params_text)
        self.default_params_checkbox.setFont(LARGE_FONT)
        self.default_params_checkbox.setStyleSheet("transform: scale(1.5); margin: 15px;")
        self.default_params_checkbox.setChecked(True)
        self.default_params_checkbox.stateChanged.connect(self.toggle_default_params)
        left_layout.addWidget(self.default_params_checkbox)

        self.params_widget = QWidget()
        self.params_widget.setStyleSheet(f"""
            background-color: {PARAMS_CHECKED_COLOR};
            border-radius: 15px;
            margin: 20px;
            padding: 20px;
        """)
        params_layout = QHBoxLayout()
        params_layout.setSpacing(5)
        self.params_widget.setLayout(params_layout)

        strength_widget = QWidget()
        strength_widget.setStyleSheet(LAYOUT_STYLE)
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("Strength:")
        self.strength_label.setFont(LARGE_FONT)
        self.strength_label.setStyleSheet(SIMPLE_LABEL_STYLE)
        strength_layout.addWidget(self.strength_label)
        self.strength_input = QLineEdit(str(DEFAULT_PARAMETERS[self.current_mode]["STRENGTH"]))
        self.strength_input.setFont(LARGE_FONT)
        self.strength_input.setStyleSheet(SIMPLE_INPUT_STYLE)
        self.strength_input.setFixedWidth(100)
        strength_layout.addWidget(self.strength_input)
        strength_layout.addStretch()
        strength_widget.setLayout(strength_layout)
        params_layout.addWidget(strength_widget)

        cfg_scale_widget = QWidget()
        cfg_scale_widget.setStyleSheet(LAYOUT_STYLE)
        cfg_scale_layout = QHBoxLayout()
        self.cfg_scale_label = QLabel("CFG Scale:")
        self.cfg_scale_label.setFont(LARGE_FONT)
        self.cfg_scale_label.setStyleSheet(SIMPLE_LABEL_STYLE)
        cfg_scale_layout.addWidget(self.cfg_scale_label)
        self.cfg_scale_input = QLineEdit(str(DEFAULT_PARAMETERS[self.current_mode]["CFG_SCALE"]))
        self.cfg_scale_input.setFont(LARGE_FONT)
        self.cfg_scale_input.setStyleSheet(SIMPLE_INPUT_STYLE)
        self.cfg_scale_input.setFixedWidth(100)
        cfg_scale_layout.addWidget(self.cfg_scale_input)
        cfg_scale_layout.addStretch()
        cfg_scale_widget.setLayout(cfg_scale_layout)
        params_layout.addWidget(cfg_scale_widget)

        steps_widget = QWidget()
        steps_widget.setStyleSheet(LAYOUT_STYLE)
        steps_layout = QHBoxLayout()
        self.steps_label = QLabel("Inference Steps:")
        self.steps_label.setFont(LARGE_FONT)
        self.steps_label.setStyleSheet(SIMPLE_LABEL_STYLE)
        steps_layout.addWidget(self.steps_label)
        self.steps_input = QLineEdit(str(DEFAULT_PARAMETERS[self.current_mode]["INFERENCE_STEPS"]))
        self.steps_input.setFont(LARGE_FONT)
        self.steps_input.setStyleSheet(SIMPLE_INPUT_STYLE)
        self.steps_input.setFixedWidth(100)
        steps_layout.addWidget(self.steps_input)
        steps_layout.addStretch()
        steps_widget.setLayout(steps_layout)
        params_layout.addWidget(steps_widget)

        left_layout.addWidget(self.params_widget)
        self.toggle_default_params()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setFont(LARGE_FONT)
        self.generate_btn.setStyleSheet(f"""
            background-color: {GENERATE_BUTTON_COLOR};
            color: white;
            border-radius: 30px;
            padding: 15px;
            margin: 10px;
        """)
        self.generate_btn.setMinimumHeight(60)
        self.generate_btn.clicked.connect(self.start_processing)
        left_layout.addWidget(self.generate_btn)

        self.eta_label = QLabel("ETA: -- s")
        self.eta_label.setFont(LARGE_FONT)
        self.eta_label.setStyleSheet("margin: 10px;")
        self.eta_label.setVisible(False)
        left_layout.addWidget(self.eta_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFont(LARGE_FONT)
        self.progress_bar.setStyleSheet("padding: 10px; margin: 10px; height: 40px;")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.elapsed_label = QLabel("Elapsed: 0.0 s")
        self.elapsed_label.setFont(LARGE_FONT)
        self.elapsed_label.setStyleSheet("margin: 10px;")
        self.elapsed_label.setVisible(False)
        left_layout.addWidget(self.elapsed_label)

        left_layout.addStretch()

        separator = QWidget()
        separator.setFixedWidth(3)
        separator.setStyleSheet("""
            background-color: #2776EA;
            border-radius: 3px;
            margin-top: 20px;
            margin-bottom: 20px;
        """)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        right_layout.addStretch()

        self.input_label = QLabel("Input Image:")
        self.input_label.setFont(LARGE_FONT)
        self.input_label.setStyleSheet("margin: 10px 10px 0px 10px; height: 20px;")
        self.input_label.setVisible(False)
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
        self.input_image_label.setVisible(False)
        right_layout.addWidget(self.input_image_label)

        self.output_label = QLabel("Output Image:")
        self.output_label.setFont(LARGE_FONT)
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

        self.download_btn = QPushButton("Download")
        self.download_btn.setFont(LARGE_FONT)
        self.download_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DOWNLOAD_BUTTON_COLOR};
                color: white;
                border: none;
                border-radius: 15px;
                padding: 15px;
                margin: 10px;
            }}
        """)
        self.download_btn.setMinimumHeight(60)
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self.download_image)
        right_layout.addWidget(self.download_btn)

        right_layout.addStretch()

        splitter.addWidget(left_widget)
        splitter.addWidget(separator)
        splitter.addWidget(right_widget)
        splitter.setSizes([self.width() // 2, 3, self.width() // 2 - 3])
        left_widget.setMinimumSize(0, 0)
        right_widget.setMinimumSize(0, 0)

    def show_setup_dialog(self):
        dialog = SetupDialog(self.setup_manager, self)
        dialog.exec_()
        if not self.setup_manager.check_requirements():
            self.setup_requirements_btn.setVisible(True)
        else:
            self.setup_requirements_btn.setVisible(False)

    def update_default_params_label(self):
        params = DEFAULT_PARAMETERS[self.current_mode]
        self.default_params_text = (
            f"Use Default Parameters: Strength({params['STRENGTH']}), "
            f"CFG Scale({params['CFG_SCALE']}), "
            f"Inference Steps({params['INFERENCE_STEPS']})"
        )
        if hasattr(self, 'default_params_checkbox'):
            self.default_params_checkbox.setText(self.default_params_text)

    def toggle_default_params(self):
        is_checked = self.default_params_checkbox.isChecked()
        self.params_widget.setVisible(not is_checked)
        if is_checked:
            params = DEFAULT_PARAMETERS[self.current_mode]
            self.strength_input.setText(str(params["STRENGTH"]))
            self.cfg_scale_input.setText(str(params["CFG_SCALE"]))
            self.steps_input.setText(str(params["INFERENCE_STEPS"]))

    def switch_mode(self, mode):
        if mode == self.current_mode:
            return

        self.current_mode = mode

        self.text_to_image_btn.setStyleSheet(
            self.button_style_active if mode == "Text-to-Image" else self.button_style_inactive
        )
        self.image_to_image_btn.setStyleSheet(
            self.button_style_active if mode == "Image-to-Image" else self.button_style_inactive
        )
        self.inpainting_btn.setStyleSheet(self.button_style_inactive)

        self.text_to_image_btn.setEnabled(True)
        self.image_to_image_btn.setEnabled(True)
        self.inpainting_btn.setEnabled(False)

        if mode == "Text-to-Image":
            self.load_btn.setVisible(False)
            self.input_label.setVisible(False)
            self.input_image_label.setVisible(False)
            self.image_path = None
            self.input_image_label.setPixmap(QPixmap())
        elif mode == "Image-to-Image":
            self.load_btn.setVisible(True)
            self.input_label.setVisible(True)
            self.input_image_label.setVisible(True)
            self.set_random_noise()
        elif mode == "Image-InPainting":
            pass

        self.update_default_params_label()
        self.toggle_default_params()

    def set_random_noise(self):
        noise = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        pixmap = numpy_to_pixmap(noise)
        scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.input_image_label.setPixmap(scaled_pixmap)

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            self.image_path = file_dialog.selectedFiles()[0]
            if validate_image_path(self.image_path):
                self.load_btn.setText(f"Image: {os.path.basename(self.image_path)}")
                pixmap = QPixmap(self.image_path)
                scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.input_image_label.setPixmap(scaled_pixmap)

    def show_full_screen_input(self, event):
        if self.image_path and self.input_image_label.pixmap() and not self.input_image_label.pixmap().isNull():
            pixmap = QPixmap(self.image_path)
            dialog = FullScreenImageDialog(pixmap, self)
            dialog.showFullScreen()
            dialog.exec_()

    def show_full_screen_output(self, event):
        if self.output_image_label.pixmap() and not self.output_image_label.pixmap().isNull():
            pixmap = self.output_image_label.pixmap()
            dialog = FullScreenImageDialog(pixmap, self)
            dialog.showFullScreen()
            dialog.exec_()

    def download_image(self):
        if self.output_image is None:
            QMessageBox.warning(self, "Error", "No output image available to download.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            try:
                pil_image = Image.fromarray(self.output_image.astype(np.uint8))
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path += '.png'
                pil_image.save(file_path)
                QMessageBox.information(self, "Success", "Image saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def start_processing(self):
        if not self.sentence_input.text().strip():
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        if self.current_mode in ["Image-to-Image", "Image-InPainting"]:
            if not self.image_path or not validate_image_path(self.image_path):
                QMessageBox.warning(self, "Error", "Please select a valid image.")
                return
        elif self.current_mode == "Image-InPainting":
            QMessageBox.information(self, "Info", "Image-InPainting mode is not yet implemented.")
            return

        self.load_btn.setEnabled(False)
        self.sentence_input.setEnabled(False)
        self.uncond_prompt.setEnabled(False)
        self.default_params_checkbox.setEnabled(False)
        self.params_widget.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.download_btn.setVisible(False)
        self.progress_bar.setVisible(True)
        self.eta_label.setVisible(True)
        self.elapsed_label.setVisible(True)
        self.progress_bar.setValue(0)

        self.output_image_label.setPixmap(QPixmap())
        self.output_image = None
        self.start_gradient_animation()

        self.start_time = time.time()
        self.step_times = []
        self.total_steps = None

        sentence = self.sentence_input.text().strip()
        uncond_prompt = self.uncond_prompt.text().strip()
        if self.default_params_checkbox.isChecked():
            params = DEFAULT_PARAMETERS[self.current_mode]
            strength = params["STRENGTH"]
            cfg_scale = params["CFG_SCALE"]
            num_inference_steps = params["INFERENCE_STEPS"]
        else:
            try:
                strength = float(self.strength_input.text())
                if not 0.01 <= strength <= 1.0:
                    raise ValueError("Strength must be between 0.01 and 1.0")
                cfg_scale = float(self.cfg_scale_input.text())
                if not 1.0 <= cfg_scale <= 14.0:
                    raise ValueError("CFG Scale must be between 1.0 and 14.0")
                num_inference_steps = int(self.steps_input.text())
                if not 1 <= num_inference_steps <= 200:
                    raise ValueError("Inference Steps must be between 1 and 200")
            except ValueError as e:
                QMessageBox.warning(self, "Error", f"Invalid parameter values: {str(e)}")
                self.cleanup()
                return
        do_cfg = True
        sampler = "ddpm"
        seed = DEFAULT_SEED

        image_path = self.image_path if self.current_mode in ["Image-to-Image", "Image-InPainting"] else None

        self.thread = QThread()
        self.worker = Worker(image_path, sentence, uncond_prompt, strength, do_cfg, cfg_scale, sampler, num_inference_steps, seed)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.elapsed_timer.start(100)

    def start_gradient_animation(self):
        self.animation = QPropertyAnimation(self, b"gradient")
        self.animation.setDuration(2000)
        self.animation.setLoopCount(-1)
        self.animation.setEasingCurve(QEasingCurve.InOutSine)
        self.animation.setKeyValueAt(0.0, QColor("#ff6f61"))
        self.animation.setKeyValueAt(0.5, QColor("#6b7280"))
        self.animation.setKeyValueAt(1.0, QColor("#ff6f61"))
        self.animation.start()

    def update_progress(self, step, total_steps, step_time):
        self.total_steps = total_steps
        self.step_times.append(step_time)
        self.progress_value = int((step + 1) / total_steps * 100)
        self.progress_bar.setValue(self.progress_value)
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
        self.output_image = output_image
        pixmap = numpy_to_pixmap(output_image)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(512, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.output_image_label.setPixmap(scaled_pixmap)
            self.download_btn.setVisible(True)
        else:
            QMessageBox.critical(self, "Error", "Failed to load output image.")
        self.cleanup()

    def on_processing_error(self, error_message):
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
        QMessageBox.critical(self, "Error", error_message)
        self.output_image_label.setText("Error displaying output")
        self.setup_requirements_btn.setVisible(True)
        self.cleanup()

    def cleanup(self):
        self.load_btn.setEnabled(True)
        self.sentence_input.setEnabled(True)
        self.uncond_prompt.setEnabled(True)
        self.default_params_checkbox.setEnabled(True)
        self.params_widget.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.eta_label.setVisible(False)
        self.elapsed_label.setVisible(False)
        self.thread.quit()
        self.thread.wait()

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