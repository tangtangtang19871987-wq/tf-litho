"""
Hopkins model implementation in TensorFlow.
Ported from TorchLitho-Lite's Hopkins simulation.
"""
import tensorflow as tf
import numpy as np
from .tcc import gen_tcc, read_tcc_from_disc
from skimage.transform import resize


def hopkins_simulate(mask, tcc=None, defocus=False, device=None, filename=None):
    """
    Core Hopkins simulation function in TensorFlow.
    
    Args:
        mask: Input mask tensor
        tcc: Pre-computed TCC parameters (phis, weights)
        defocus: Whether to include defocus effects
        device: TensorFlow device (CPU/GPU)
        filename: Path to saved TCC file
        
    Returns:
        Aerial image tensor
    """
    # Convert mask to tensor
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    # Load or use provided TCC
    if tcc is None:
        if filename is not None:
            tcc = read_tcc_from_disc(filename)
        else:
            raise ValueError("Either tcc or filename must be provided")
    
    phis, weights = tcc
    
    # Convert to TensorFlow tensors
    if device is None:
        device = '/cpu:0'
    
    with tf.device(device):
        # Compute FFT of mask
        if len(mask.shape) == 2:
            mask_fft = tf.signal.fft2d(tf.cast(mask, tf.complex64))
        elif len(mask.shape) == 3:
            mask_fft = tf.map_fn(
                lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64)), 
                mask
            )
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        
        # Initialize aerial image
        aerial = tf.zeros_like(mask_fft, dtype=tf.float32)
        
        # Compute contribution from each TCC component
        for idx, phi in enumerate(phis):
            weight = weights[idx]
            
            # Convert phi to tensor and resize if needed
            phi_tensor = tf.constant(phi, dtype=tf.complex64)
            
            # Resize phi to match mask size if necessary
            if phi_tensor.shape != mask_fft.shape[-2:]:
                phi_np = phi_tensor.numpy()
                phi_resized = resize(
                    phi_np, 
                    mask_fft.shape[-2:], 
                    order=1, 
                    anti_aliasing=True, 
                    preserve_range=True
                )
                phi_tensor = tf.constant(phi_resized, dtype=tf.complex64)
            
            # Compute FFT of phi
            phi_fft = tf.signal.fft2d(phi_tensor)
            
            # Handle batch dimension
            if len(mask_fft.shape) == 3:
                phi_fft = tf.expand_dims(phi_fft, 0)
                phi_fft = tf.tile(phi_fft, [mask_fft.shape[0], 1, 1])
            
            # Convolution in frequency domain
            convolved = tf.signal.ifft2d(mask_fft * phi_fft)
            convolved = tf.signal.fftshift(convolved, axes=[-2, -1])
            
            # Normalize and compute intensity
            normalization = tf.cast(tf.reduce_prod(tf.shape(convolved)[-2:]), tf.complex64)
            convolved = convolved / normalization
            
            # Add weighted contribution
            intensity = tf.cast(weight, tf.float32) * tf.abs(convolved)**2
            aerial += intensity
    
    return aerial


class HopkinsSimulator:
    """Hopkins simulator class with pre-computed TCC."""
    
    def __init__(self, pixel=14, canvas=512, na=1.35, wavelength=193, 
                 defocus=None, thresh=1e-6, device=None):
        """
        Initialize Hopkins simulator with TCC generation.
        
        Args:
            pixel: Pixel size
            canvas: Canvas size  
            na: Numerical aperture
            wavelength: Wavelength in nm
            defocus: Defocus values (list or None)
            thresh: SVD threshold for TCC compression
            device: TensorFlow device
        """
        self.pixel = pixel
        self.canvas = canvas
        self.na = na
        self.wavelength = wavelength
        self.defocus = defocus
        self.thresh = thresh
        self.device = device or '/cpu:0'
        
        # Generate TCC
        self.tcc = gen_tcc(
            pixel=pixel,
            canvas=canvas, 
            na=na,
            wavelength=wavelength,
            defocus=defocus,
            thresh=thresh
        )
    
    def __call__(self, mask):
        """Simulate aerial image using Hopkins model."""
        return hopkins_simulate(
            mask=mask,
            tcc=self.tcc,
            defocus=self.defocus is not None,
            device=self.device
        )
    
    def save_tcc(self, filename):
        """Save TCC parameters to disk."""
        from .tcc import write_tcc_to_disc
        write_tcc_to_disc(self.tcc[0], self.tcc[1], filename)
    
    @classmethod
    def from_file(cls, filename, **kwargs):
        """Create simulator from saved TCC file."""
        instance = cls.__new__(cls)
        instance.tcc = read_tcc_from_disc(filename)
        # Set other attributes from kwargs or defaults
        for key, value in kwargs.items():
            setattr(instance, key, value)
        return instance