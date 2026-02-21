"""
Test script for partial coherent illumination in Abbe model.
This demonstrates the difference between coherent and partial coherent simulation.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_litho.abbe import abbe_simulate


def test_partial_coherent():
    """Compare coherent vs partial coherent illumination."""
    # Create a simple mask with fine features
    mask = np.zeros((128, 128), dtype=np.float32)
    # Add some lines to show coherence effects
    mask[60:64, 20:108] = 1.0  # horizontal line
    mask[20:108, 60:64] = 1.0  # vertical line
    
    print("Testing coherent illumination (sigma=0.0)...")
    aerial_coherent = abbe_simulate(
        mask, 
        pixel=5, 
        sigma=0.0,  # Coherent illumination
        na=1.35, 
        wavelength=193,
        defocus=0
    )
    
    print("Testing partial coherent illumination (sigma=0.5)...")
    aerial_partial = abbe_simulate(
        mask, 
        pixel=5, 
        sigma=0.5,  # Partial coherent illumination  
        na=1.35, 
        wavelength=193,
        defocus=0
    )
    
    print(f"Coherent result shape: {aerial_coherent.shape}")
    print(f"Partial coherent result shape: {aerial_partial.shape}")
    
    # Save results
    np.save('mask_test.npy', mask)
    np.save('aerial_coherent.npy', aerial_coherent.numpy())
    np.save('aerial_partial_coherent.npy', aerial_partial.numpy())
    
    # Create comparison plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Input Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(aerial_coherent.numpy(), cmap='hot')
    plt.title('Coherent Illumination\n(sigma=0.0)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(aerial_partial.numpy(), cmap='hot')
    plt.title('Partial Coherent Illumination\n(sigma=0.5)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('partial_coherent_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Test completed! Results saved as:")
    print("- mask_test.npy")
    print("- aerial_coherent.npy")
    print("- aerial_partial_coherent.npy")
    print("- partial_coherent_comparison.png")


if __name__ == "__main__":
    try:
        test_partial_coherent()
        print("✅ Partial coherent illumination test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise