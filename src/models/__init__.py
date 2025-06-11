import sys
import os

# Updates global path rather than single module path update like __path__
# NOTE:(In future) If collides with other modules imports in src, change this to __path__ in the necessary module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))