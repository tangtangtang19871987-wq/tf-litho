"""
TensorFlow implementation of computational lithography simulation engine.
Ported from TorchLitho-Lite (https://github.com/OpenOPC/TorchLitho-Lite)
"""

from .abbe import abbe_simulate
from .hopkins import HopkinsSimulator, hopkins_simulate
from .gradient import (
    AbbeGradientSimulator, 
    HopkinsGradientSimulator,
    compute_spectral_loss,
    optimize_mask
)
from .utils import get_mask_fft, interpolate_aerial_image
from .source import get_source_points

__version__ = "0.1.0"
__author__ = "OpenClaw AI Assistant"
__license__ = "MIT"