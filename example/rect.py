#!/usr/bin/env python3
"""
TensorFlow implementation of rectangular pattern example.
This demonstrates Abbe and Hopkins simulation with gradient computation.
Corresponds to TorchLitho-Lite's rect.py example.
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tf_litho.gradient import abbe_with_gradient, hopkins_with_gradient


def main():
    """Main example function."""
    print("TensorFlow Rectangular Pattern Example")
    print("=" * 40)
    
    # Simulation parameters
    pixelsize = 8
    canvas = 512
    size = round(canvas / pixelsize)
    
    # Create rectangular mask pattern
    nb = np.zeros([1, 1, size, size], dtype=np.float32)
    nb[0, 0, 16:48, 16:48] = 1.0
    
    # Convert to TensorFlow tensor with gradient tracking
    mask_tensor = tf.Variable(nb, dtype=tf.float32)
    
    print(f"Mask shape: {mask_tensor.shape}")
    print(f"Mask range: [{tf.reduce_min(mask_tensor):.3f}, {tf.reduce_max(mask_tensor):.3f}]")
    
    # Hopkins simulation with gradient
    with tf.GradientTape() as tape_h:
        tape_h.watch(mask_tensor)
        image1 = hopkins_with_gradient(
            mask_tensor,
            canvas=canvas,
            pixel=pixelsize
        )
    
    # Compute gradients for Hopkins
    hopkins_grad = tape_h.gradient(image1, mask_tensor)
    
    # Abbe simulation with gradient  
    with tf.GradientTape() as tape_a:
        tape_a.watch(mask_tensor)
        image2 = abbe_with_gradient(
            mask_tensor,
            canvas=canvas,
            pixel=pixelsize
        )
    
    # Compute gradients for Abbe
    abbe_grad = tape_a.gradient(image2, mask_tensor)
    
    print(f"Hopkins image shape: {image1.shape}")
    print(f"Abbe image shape: {image2.shape}")
    print(f"Hopkins gradient shape: {hopkins_grad.shape if hopkins_grad is not None else 'None'}")
    print(f"Abbe gradient shape: {abbe_grad.shape if abbe_grad is not None else 'None'}")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image1[0, 0].numpy(), cmap='gray')
    plt.title('Hopkins Simulation')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(image2[0, 0].numpy(), cmap='gray')
    plt.title('Abbe Simulation')
    plt.colorbar()
    
    if hopkins_grad is not None:
        plt.subplot(2, 2, 3)
        plt.imshow(hopkins_grad[0, 0].numpy(), cmap='RdBu_r')
        plt.title('Hopkins Gradient Map')
        plt.colorbar()
    else:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, 'No Gradient', ha='center', va='center')
        plt.title('Hopkins Gradient Map')
    
    if abbe_grad is not None:
        plt.subplot(2, 2, 4)
        plt.imshow(abbe_grad[0, 0].numpy(), cmap='RdBu_r')
        plt.title('Abbe Gradient Map')
        plt.colorbar()
    else:
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, 'No Gradient', ha='center', va='center')
        plt.title('Abbe Gradient Map')
    
    plt.tight_layout()
    plt.savefig("example_rect.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nExample completed successfully!")
    print("Gradient maps saved to example_rect.png")


if __name__ == "__main__":
    # Enable memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    
    main()