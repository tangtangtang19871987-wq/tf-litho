"""
Source point generation for partial coherent illumination.
Implements circular source distribution for Abbe model.
"""
import tensorflow as tf
import numpy as np


def generate_circular_source_points(num_points, sigma, na, wavelength):
    """
    Generate source points distributed in a circular pattern.
    
    Args:
        num_points: Number of source points to generate
        sigma: Source coherence parameter (0.0 = coherent, 1.0 = fully incoherent)
        na: Numerical aperture
        wavelength: Wavelength in nm
        
    Returns:
        Tensor of shape [num_points, 2] with (fx, fy) coordinates
    """
    if num_points == 1 or sigma == 0.0:
        # Coherent case: single source point at origin
        return tf.constant([[0.0, 0.0]], dtype=tf.float32)
    
    # Calculate maximum frequency cutoff
    freq_cutoff = sigma * na / wavelength
    
    if num_points == 2:
        # Simple dipole source
        points = [[-freq_cutoff/2, 0.0], [freq_cutoff/2, 0.0]]
        return tf.constant(points, dtype=tf.float32)
    
    # Generate points on concentric circles
    points = []
    
    # Center point
    if num_points > 1:
        points.append([0.0, 0.0])
        remaining_points = num_points - 1
    else:
        remaining_points = num_points
    
    if remaining_points > 0:
        # Distribute remaining points on circle
        angles = np.linspace(0, 2*np.pi, remaining_points, endpoint=False)
        radius = freq_cutoff * 0.7  # Keep within cutoff
        
        for angle in angles:
            fx = radius * np.cos(angle)
            fy = radius * np.sin(angle)
            points.append([fx, fy])
    
    return tf.constant(points, dtype=tf.float32)


def get_source_points_from_grid(freq_grid, freq_cutoff):
    """
    Extract source points from frequency grid (alternative approach).
    This mimics the original TorchLitho-Lite behavior more closely.
    
    Args:
        freq_grid: Frequency grid tensor of shape [H, W, 2]
        freq_cutoff: Maximum frequency for source points
        
    Returns:
        Flattened tensor of source points within cutoff
    """
    # Calculate distance from origin for each point
    distances = tf.norm(freq_grid, axis=-1)
    
    # Create mask for points within cutoff
    mask = distances <= freq_cutoff
    
    # Extract coordinates of valid points
    valid_coords = tf.where(mask)
    
    # Get frequency values at valid coordinates
    source_points = tf.gather_nd(freq_grid, valid_coords)
    
    # If no points found, return origin
    if tf.shape(source_points)[0] == 0:
        return tf.constant([[0.0, 0.0]], dtype=tf.float32)
    
    return source_points


def create_frequency_grid(size, pixel):
    """
    Create frequency grid for source point generation.
    
    Args:
        size: Grid size (assumes square grid)
        pixel: Pixel size in nm
        
    Returns:
        Frequency grid tensor of shape [size, size, 2]
    """
    # Create frequency vectors
    freqs = tf.signal.fftshift(tf.signal.fftfreq(size, d=pixel))
    
    # Create meshgrid
    freq_x, freq_y = tf.meshgrid(freqs, freqs, indexing='ij')
    
    # Stack into [size, size, 2] tensor
    freq_grid = tf.stack([freq_x, freq_y], axis=-1)
    
    return freq_grid