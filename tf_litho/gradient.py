"""
Custom gradient implementations for Abbe and Hopkins models in TensorFlow.
This module provides gradient functions for lithography simulation optimization.
"""
import tensorflow as tf
import numpy as np
from .abbe import abbe_simulate
from .hopkins import hopkins_simulate


@tf.custom_gradient
def abbe_with_gradient(mask, pixel=14, sigma=0.05, na=1.35, wavelength=193, 
                      defocus=0, batch=False, parallel=False):
    """
    Abbe simulation with custom gradient for optimization.
    
    Args:
        mask: Input mask tensor (requires gradient)
        pixel: Pixel size
        sigma: Source coherence parameter
        na: Numerical aperture  
        wavelength: Wavelength in nm
        defocus: Defocus value in nm
        batch: Whether input has batch dimension
        parallel: Use parallel computation
        
    Returns:
        aerial_image: Simulated aerial image
        grad_fn: Custom gradient function
    """
    # Forward pass
    aerial_image = abbe_simulate(
        mask, pixel, sigma, na, wavelength, defocus, batch, parallel
    )
    
    def grad_fn(grad_output):
        """
        Custom gradient function for Abbe model.
        
        The gradient is computed using the adjoint method:
        ∇M L = 2 * Re{ FFT^{-1}[ H^*(f) * FFT[ aerial_grad * aerial_complex ] ] }
        
        Where:
        - H(f) is the optical transfer function
        - aerial_grad is the gradient from loss function
        - aerial_complex is the complex aerial image before intensity calculation
        """
        # For simplicity, we use numerical gradient approximation
        # In practice, this should be replaced with analytical gradient
        
        with tf.GradientTape() as tape:
            tape.watch(mask)
            aerial_fwd = abbe_simulate(
                mask, pixel, sigma, na, wavelength, defocus, batch, parallel
            )
        
        # Compute gradient using TensorFlow's automatic differentiation
        # This is a fallback implementation - analytical gradient would be more efficient
        grad_mask = tape.gradient(aerial_fwd, mask, output_gradients=grad_output)
        
        return grad_mask, None, None, None, None, None, None, None
    
    return aerial_image, grad_fn


@tf.custom_gradient  
def hopkins_with_gradient(mask, tcc=None, defocus=False, device=None, filename=None):
    """
    Hopkins simulation with custom gradient for optimization.
    
    Args:
        mask: Input mask tensor (requires gradient)
        tcc: Pre-computed TCC parameters
        defocus: Whether to include defocus effects  
        device: TensorFlow device
        filename: Path to saved TCC file
        
    Returns:
        aerial_image: Simulated aerial image
        grad_fn: Custom gradient function
    """
    # Forward pass
    aerial_image = hopkins_simulate(mask, tcc, defocus, device, filename)
    
    def grad_fn(grad_output):
        """
        Custom gradient function for Hopkins model.
        
        The gradient uses the TCC adjoint property:
        ∇M L = Σ_i w_i * TCC_i^† [ aerial_grad * aerial_i ]
        
        Where:
        - w_i are TCC weights
        - TCC_i^† is the adjoint of TCC operator i
        - aerial_i is the aerial image component from TCC_i
        """
        # Use automatic differentiation as fallback
        with tf.GradientTape() as tape:
            tape.watch(mask)
            aerial_fwd = hopkins_simulate(mask, tcc, defocus, device, filename)
        
        grad_mask = tape.gradient(aerial_fwd, mask, output_gradients=grad_output)
        return grad_mask, None, None, None, None
    
    return aerial_image, grad_fn


class AbbeGradientSimulator:
    """Abbe simulator with built-in gradient support."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, mask):
        return abbe_with_gradient(mask, **self.kwargs)


class HopkinsGradientSimulator:
    """Hopkins simulator with built-in gradient support."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, mask):
        return hopkins_with_gradient(mask, **self.kwargs)


# Utility functions for gradient-based optimization
def compute_spectral_loss(mask_pred, mask_target, simulator, **sim_kwargs):
    """
    Compute spectral loss between predicted and target masks.
    
    Args:
        mask_pred: Predicted mask (with gradient)
        mask_target: Target mask
        simulator: Simulation function (abbe_with_gradient or hopkins_with_gradient)
        **sim_kwargs: Simulator parameters
        
    Returns:
        loss: Spectral loss value
    """
    aerial_pred = simulator(mask_pred, **sim_kwargs)
    aerial_target = simulator(mask_target, **sim_kwargs)
    
    # Spectral loss in frequency domain
    fft_pred = tf.signal.fft2d(tf.cast(aerial_pred, tf.complex64))
    fft_target = tf.signal.fft2d(tf.cast(aerial_target, tf.complex64))
    
    loss = tf.reduce_mean(tf.abs(fft_pred - fft_target)**2)
    return loss


def optimize_mask(initial_mask, target_aerial, simulator, num_iterations=100, 
                  learning_rate=0.01, **sim_kwargs):
    """
    Optimize mask using gradient descent.
    
    Args:
        initial_mask: Initial mask guess
        target_aerial: Target aerial image
        simulator: Simulation function with gradient
        num_iterations: Number of optimization steps
        learning_rate: Learning rate for gradient descent
        **sim_kwargs: Simulator parameters
        
    Returns:
        optimized_mask: Optimized mask
        losses: Loss history
    """
    mask_var = tf.Variable(initial_mask, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    losses = []
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            aerial_pred = simulator(mask_var, **sim_kwargs)
            loss = tf.reduce_mean((aerial_pred - target_aerial)**2)
        
        gradients = tape.gradient(loss, mask_var)
        optimizer.apply_gradients([(gradients, mask_var)])
        losses.append(loss.numpy())
        
        if i % 20 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy():.6f}")
    
    return mask_var.numpy(), losses