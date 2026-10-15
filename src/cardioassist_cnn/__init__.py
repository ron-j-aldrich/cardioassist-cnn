"""CardioAssist echocardiography ejection-fraction inference.

This package wraps the official EchoNet-Dynamic implementation. It does not
redefine the R(2+1)D-18 architecture.
"""

from cardioassist_cnn.inference import EFPredictor, InferenceResult
from cardioassist_cnn.model import MODEL_ID

__all__ = ["EFPredictor", "InferenceResult", "MODEL_ID"]
__version__ = "0.1.0"
