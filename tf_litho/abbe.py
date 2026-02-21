"""
Abbe model implementation in TensorFlow.
Ported from TorchLitho-Lite's Abbe simulation.
"""
import tensorflow as tf
import numpy as np
from .utils import get_mask_fft, BBox, Point
from .source import get_source_points, get_freq_support, get_freq_cut, get_delta_freq, get_defocus


def abbe_simulate(mask, pixel=14, sigma=0.05, na=1.35, wavelength=193, 
                  defocus=0, batch=False, parallel=False):
    """
    Core Abbe simulation function in TensorFlow with partial coherent illumination.
    
    Args:
        mask: Input mask tensor
        pixel: Pixel size
        sigma: Source coherence parameter  
        na: Numerical aperture
        wavelength: Wavelength in nm
        defocus: Defocus value in nm
        batch: Whether input has batch dimension
        parallel: Use parallel computation (not implemented yet)
    
    Returns:
        Aerial image tensor
    """
    # Convert inputs to tensors
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    if batch:
        mask_fft = tf.map_fn(get_mask_fft, mask)
        canvas_size = mask.shape[-2]
    else:
        mask_fft = get_mask_fft(mask)
        canvas_size = mask.shape[-2]
    
    # Get simulation parameters
    bbox = BBox(Point(0, 0), Point(canvas_size * pixel, canvas_size * pixel))
    
    # Compute frequency grid
    freq_grid = get_freq_support(canvas_size, pixel)
    
    # Compute source points for partial coherent illumination
    freq_cut = get_freq_cut(sigma, na, wavelength)
    source_points = get_source_points(freq_grid, freq_cut)
    
    # Initialize aerial image
    if batch:
        aerial_image = tf.zeros_like(mask_fft, dtype=tf.float32)
    else:
        aerial_image = tf.zeros_like(mask_fft, dtype=tf.float32)
    
    # Compute contribution from each source point
    if parallel:
        # Parallel computation not implemented yet
        # Fall back to sequential processing
        pass
    
    # Sequential processing for each source point
    for i in range(source_points.shape[0]):
        freq_src = source_points[i]
        
        # Shift frequency support relative to current source point
        if batch:
            freq_shifted = freq_grid[None, ...] - freq_src
        else:
            freq_shifted = freq_grid - freq_src
        
        # Create shifted pupil function
        delta_freq = get_delta_freq(na, wavelength)
        if batch:
            pupil_shifted = tf.cast(freq_shifted < delta_freq, tf.complex64)
        else:
            pupil_shifted = tf.cast(freq_shifted < delta_freq, tf.complex64)
        
        # Apply defocus if specified
        if defocus != 0:
            pupil_shifted = get_defocus(pupil_shifted, freq_shifted, wavelength, defocus)
        
        # Apply pupil filter to mask spectrum
        if batch:
            mask_filtered = mask_fft * pupil_shifted
            aerial_complex = tf.signal.ifft2d(tf.signal.ifftshift(mask_filtered))
        else:
            mask_filtered = mask_fft * pupil_shifted
            aerial_complex = tf.signal.ifft2d(tf.signal.ifftshift(mask_filtered))
        
        # Compute intensity and accumulate
        if batch:
            aerial_contribution = tf.abs(aerial_complex)**2
            aerial_image += aerial_contribution
        else:
            aerial_contribution = tf.abs(aerial_complex)**2
            aerial_image += aerial_contribution
    
    # Normalize by number of source points
    aerial_image = aerial_image / tf.cast(source_points.shape[0], tf.float32)
    
    return aerial_image