from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import os

def numpy_to_pixmap(pixels: np.ndarray) -> QPixmap:
    """
    Convert a NumPy array to a QPixmap.
    """
    try:
        pil_image = Image.fromarray(pixels).convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    except Exception as e:
        raise RuntimeError(f"Failed to convert NumPy array to QPixmap: {str(e)}")

def validate_image_path(image_path: str) -> bool:
    """
    Validate if an image file exists and is readable.
    """
    if not os.path.exists(image_path):
        return False
    try:
        Image.open(image_path).close()
        return True
    except Exception:
        return False