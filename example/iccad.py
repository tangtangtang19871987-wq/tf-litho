"""
TensorFlow equivalent of TorchLitho-Lite's ICCAD example.
Demonstrates gradient computation for lithography simulation.
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from tf_litho.gradient import AbbeGradientSimulator


def load_iccad_design(filepath):
    """
    Load ICCAD design file (placeholder implementation).
    In practice, this would parse the .glp file format.
    """
    # Placeholder: create a simple test pattern
    canvas = 2048
    mask = np.zeros((1, 1, canvas, canvas), dtype=np.float32)
    
    # Create some simple patterns to simulate real mask data
    mask[0, 0, 500:1500, 500:1500] = 0.5  # Large rectangle
    mask[0, 0, 800:1200, 800:1200] = 1.0   # Smaller bright rectangle
    
    return mask


def main():
    """Main example function demonstrating gradient computation."""
    print("TensorFlow ICCAD Example - Gradient Map Generation")
    print("=" * 50)
    
    # Simulation parameters
    pixelsize = 1
    canvas = 2048
    
    # Initialize Abbe simulator with gradient support
    sim = AbbeGradientSimulator(
        canvas=canvas,
        pixel=pixelsize,
        sigma=0.05,
        na=1.35,
        wavelength=193
    )
    
    # Load design (using placeholder for now)
    try:
        # Try to load actual ICCAD file if available
        design_data = load_iccad_design("benchmark/ICCAD2013/M1_test1.glp")
    except:
        print("Using placeholder design pattern...")
        design_data = load_iccad_design("")
    
    # Convert to TensorFlow tensor with gradient tracking
    mask_tensor = tf.Variable(design_data, dtype=tf.float32)
    
    # Perform Abbe simulation
    with tf.GradientTape() as tape:
        printed = sim(mask_tensor)
    
    # Compute gradients
    gradients = tape.gradient(printed, mask_tensor)
    
    print(f"Mask shape: {mask_tensor.shape}")
    print(f"Printed image shape: {printed.shape}")
    print(f"Gradient shape: {gradients.shape}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask_tensor[0, 0].numpy(), cmap='gray')
    plt.title('Original Mask')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(printed[0, 0].numpy(), cmap='gray')
    plt.title('Printed Image (Abbe)')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(gradients[0, 0].numpy(), cmap='RdBu')
    plt.title('Gradient Map')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("example_iccad.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Example completed! Check example_iccad.png for results.")
    print(f"Gradient statistics: min={gradients.numpy().min():.6f}, max={gradients.numpy().max():.6f}")


if __name__ == "__main__":
    main()