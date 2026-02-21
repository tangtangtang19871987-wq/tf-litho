"""
Error handling and input validation for tf-litho.
"""
import tensorflow as tf
import numpy as np


def validate_mask_input(mask):
    """Validate mask input parameters."""
    if mask is None:
        raise ValueError("Mask cannot be None")
    
    # Convert to tensor if needed
    if not isinstance(mask, (tf.Tensor, np.ndarray)):
        raise TypeError(f"Mask must be tf.Tensor or np.ndarray, got {type(mask)}")
    
    # Check shape
    if len(mask.shape) not in [2, 3]:
        raise ValueError(f"Mask must be 2D or 3D, got shape {mask.shape}")
    
    # Check dtype
    if mask.dtype not in [tf.float32, tf.float64, np.float32, np.float64]:
        raise TypeError(f"Mask must be float type, got {mask.dtype}")
    
    # Check value range
    mask_min = tf.reduce_min(mask) if isinstance(mask, tf.Tensor) else np.min(mask)
    mask_max = tf.reduce_max(mask) if isinstance(mask, tf.Tensor) else np.max(mask)
    
    if mask_min < 0 or mask_max > 1:
        raise ValueError(f"Mask values must be in [0, 1], got range [{mask_min}, {mask_max}]")
    
    return True


def validate_simulation_parameters(pixel, canvas, na, wavelength):
    """Validate simulation parameters."""
    if pixel <= 0:
        raise ValueError(f"Pixel size must be positive, got {pixel}")
    
    if canvas <= 0:
        raise ValueError(f"Canvas size must be positive, got {canvas}")
    
    if canvas % pixel != 0:
        raise ValueError(f"Canvas size must be divisible by pixel size ({canvas} % {pixel} != 0)")
    
    if na <= 0 or na > 1.5:
        raise ValueError(f"Numerical aperture must be in (0, 1.5], got {na}")
    
    if wavelength <= 0:
        raise ValueError(f"Wavelength must be positive, got {wavelength}")
    
    return True


def safe_abbe_simulate(mask, **kwargs):
    """Abbe simulation with comprehensive error handling."""
    try:
        # Validate inputs
        validate_mask_input(mask)
        if 'pixel' in kwargs and 'canvas' in kwargs:
            validate_simulation_parameters(
                kwargs.get('pixel', 14),
                kwargs.get('canvas', mask.shape[0] * kwargs.get('pixel', 14)),
                kwargs.get('na', 1.35),
                kwargs.get('wavelength', 193)
            )
        
        # Run simulation
        from .abbe import abbe_simulate
        return abbe_simulate(mask, **kwargs)
        
    except Exception as e:
        error_msg = f"Abbe simulation failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e


def safe_hopkins_simulate(mask, **kwargs):
    """Hopkins simulation with comprehensive error handling."""
    try:
        # Validate inputs  
        validate_mask_input(mask)
        
        # Run simulation
        from .hopkins import hopkins_simulate
        return hopkins_simulate(mask, **kwargs)
        
    except Exception as e:
        error_msg = f"Hopkins simulation failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e