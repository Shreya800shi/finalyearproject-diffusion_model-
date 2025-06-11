from PyQt5.QtWidgets import QDialog, QWidget, QVBoxLayout, QLabel, QPushButton, QApplication, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

class FullScreenImageDialog(QDialog):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0.7);")

        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)

        main_widget = QWidget(self)
        main_widget.setGeometry(screen)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: transparent; margin-top: 40px; margin-bottom: 70px;")
        scaled_size = screen.size() - QSize(0, 110)  # 40px top + 70px bottom
        scaled_pixmap = pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        layout.addWidget(self.image_label)

        main_widget.setLayout(layout)

        self.close_button = QPushButton("✕", self)
        self.close_button.setFixedSize(40, 40)
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
        self.close_button.setGeometry(screen.width() - 60, 20, 40, 40)
        self.close_button.clicked.connect(self.close)
        self.close_button.raise_()

    def mousePressEvent(self, event):
        if not self.image_label.geometry().contains(event.pos()) and not self.close_button.geometry().contains(event.pos()):
            self.close()