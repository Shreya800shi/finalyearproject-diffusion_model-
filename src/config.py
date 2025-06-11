from PyQt5.QtGui import QFont

# Colors
BUTTON_ACTIVE_COLOR = "#2776EA"
BUTTON_INACTIVE_COLOR = "#4D495B"
GENERATE_BUTTON_COLOR = "#01D449"
DOWNLOAD_BUTTON_COLOR = "#2776EA"
PARAMS_CHECKED_COLOR = "#4D495B"

# Styles
LAYOUT_STYLE = """
    margin: 10px;
    border-radius: 10px;
    padding: 5px;
"""
SIMPLE_INPUT_STYLE = """
    background-color: #f0f0f0;
    border-radius: 10px;
    padding: 5px;
"""
SIMPLE_LABEL_STYLE = """
    color: white;
"""
SETUP_BUTTON_STYLE = """
    QPushButton {
        background-color: #FF0000;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 15px;
        margin: 5px;
    }
    QPushButton:hover:enabled {
        background-color: #CC0000;
    }
"""

# Font
LARGE_FONT = QFont("Arial", 14)

# Default Parameters
DEFAULT_MODE = "Text-to-Image"
DEFAULT_PARAMETERS = {
    "Text-to-Image": {
        "STRENGTH": 0.9,
        "CFG_SCALE": 8.0,
        "INFERENCE_STEPS": 50
    },
    "Image-to-Image": {
        "STRENGTH": 0.4,
        "CFG_SCALE": 9.0,
        "INFERENCE_STEPS": 100
    },
    "Image-InPainting": {
        "STRENGTH": 0.5,
        "CFG_SCALE": 7.0,
        "INFERENCE_STEPS": 75
    }
}
DEFAULT_SEED = 42