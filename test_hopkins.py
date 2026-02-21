"""
Test script for Hopkins model implementation.
This demonstrates TCC generation and Hopkins simulation.
Note: This is a lightweight test - full validation should be done on high-performance machine.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_litho.hopkins import HopkinsSimulator


def test_hopkins_basic():
    """Test basic Hopkins simulation with small parameters."""
    print("Testing Hopkins model with small parameters...")
    
    # Create a simple mask
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[12:20, 12:20] = 1.0  # Simple square
    
    # Initialize Hopkins simulator with small canvas for quick testing
    try:
        simulator = HopkinsSimulator(
            pixel=8,      # Larger pixel size for smaller effective grid
            canvas=64,    # Small canvas for quick computation  
            na=1.35,
            wavelength=193,
            defocus=None,
            thresh=1e-3   # Higher threshold for fewer TCC components
        )
        
        print(f"TCC generated with {len(simulator.tcc[0])} components")
        
        # Run simulation
        aerial_image = simulator(mask)
        
        print(f"Input mask shape: {mask.shape}")
        print(f"Output aerial image shape: {aerial_image.shape}")
        print(f"Output range: [{tf.reduce_min(aerial_image):.6f}, {tf.reduce_max(aerial_image):.6f}]")
        
        # Save results
        np.save('hopkins_test_mask.npy', mask)
        np.save('hopkins_test_aerial.npy', aerial_image.numpy())
        
        # Create simple plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Input Mask')
        plt.subplot(1, 2, 2)
        plt.imshow(aerial_image.numpy(), cmap='hot')
        plt.title('Hopkins Aerial Image')
        plt.savefig('hopkins_test_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Hopkins basic test completed!")
        print("Results saved as:")
        print("- hopkins_test_mask.npy")
        print("- hopkins_test_aerial.npy") 
        print("- hopkins_test_results.png")
        
    except Exception as e:
        print(f"❌ Hopkins test failed: {e}")
        print("Note: Full Hopkins model requires significant computational resources.")
        print("Consider running on high-performance machine with larger parameters.")
        raise


def test_tcc_generation():
    """Test TCC generation separately."""
    print("\nTesting TCC generation...")
    
    from tf_litho.tcc import gen_tcc
    
    try:
        # Generate TCC with very small parameters
        phis, weights = gen_tcc(
            pixel=16,     # Very large pixel
            canvas=32,    # Very small canvas
            na=1.35,
            wavelength=193,
            defocus=None,
            thresh=1e-2   # High threshold
        )
        
        print(f"TCC generation successful!")
        print(f"Number of components: {len(phis)}")
        print(f"Weight range: [{min(weights):.6f}, {max(weights):.6f}]")
        
        return True
        
    except Exception as e:
        print(f"TCC generation failed: {e}")
        return False


if __name__ == "__main__":
    # Run lightweight tests only
    success = test_tcc_generation()
    if success:
        test_hopkins_basic()
    else:
        print("Skipping Hopkins simulation due to TCC generation failure.")