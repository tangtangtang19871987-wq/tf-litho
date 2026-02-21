"""
Lightweight test script for Abbe model.
This file is ready to run on your high-performance machine.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_litho.abbe import abbe_simulate


def test_simple_mask():
    """Test with a simple rectangular mask."""
    # Create a simple 64x64 mask with a rectangle
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:44, 20:44] = 1.0
    
    print("Testing Abbe simulation...")
    aerial_image = abbe_simulate(
        mask, 
        pixel=10, 
        sigma=0.05, 
        na=1.35, 
        wavelength=193,
        defocus=0
    )
    
    print(f"Input mask shape: {mask.shape}")
    print(f"Output aerial image shape: {aerial_image.shape}")
    print(f"Output range: [{tf.reduce_min(aerial_image):.6f}, {tf.reduce_max(aerial_image):.6f}]")
    
    # Save test results for visualization
    np.save('test_mask.npy', mask)
    np.save('test_aerial_image.npy', aerial_image.numpy())
    
    # Create simple plot (lightweight)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Input Mask')
    plt.subplot(1, 2, 2)
    plt.imshow(aerial_image.numpy(), cmap='hot')
    plt.title('Aerial Image')
    plt.savefig('abbe_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Test completed! Results saved as:")
    print("- test_mask.npy")
    print("- test_aerial_image.npy") 
    print("- abbe_test_results.png")


if __name__ == "__main__":
    # Only run lightweight validation
    try:
        test_simple_mask()
        print("✅ Abbe model basic functionality verified!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise