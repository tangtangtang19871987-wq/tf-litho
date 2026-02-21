"""
Test script for custom gradient implementation.
This demonstrates gradient computation and basic optimization.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_litho.gradient import AbbeGradientSimulator, compute_spectral_loss


def test_abbe_gradient():
    """Test Abbe model with custom gradient."""
    print("Testing Abbe model with custom gradient...")
    
    # Create simple mask
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[12:20, 12:20] = 1.0
    
    # Convert to TensorFlow variable (requires gradient)
    mask_tf = tf.Variable(mask, dtype=tf.float32)
    
    # Initialize gradient simulator
    simulator = AbbeGradientSimulator(
        pixel=8,
        sigma=0.3,
        na=1.35,
        wavelength=193,
        defocus=0
    )
    
    # Test forward pass
    with tf.GradientTape() as tape:
        aerial_image = simulator(mask_tf)
        loss = tf.reduce_mean(aerial_image**2)  # Simple loss
    
    # Compute gradient
    grad_mask = tape.gradient(loss, mask_tf)
    
    print(f"Input mask shape: {mask.shape}")
    print(f"Aerial image shape: {aerial_image.shape}")
    print(f"Gradient shape: {grad_mask.shape}")
    print(f"Gradient range: [{tf.reduce_min(grad_mask):.6f}, {tf.reduce_max(grad_mask):.6f}]")
    
    # Save results
    np.save('gradient_test_mask.npy', mask)
    np.save('gradient_test_aerial.npy', aerial_image.numpy())
    np.save('gradient_test_grad.npy', grad_mask.numpy())
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Input Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(aerial_image.numpy(), cmap='hot')
    plt.title('Aerial Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(grad_mask.numpy(), cmap='coolwarm')
    plt.title('Gradient')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradient_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Abbe gradient test completed!")
    print("Results saved as:")
    print("- gradient_test_mask.npy")
    print("- gradient_test_aerial.npy")
    print("- gradient_test_grad.npy")
    print("- gradient_test_results.png")


def test_spectral_loss():
    """Test spectral loss computation."""
    print("\nTesting spectral loss computation...")
    
    # Create two similar masks
    mask1 = np.zeros((32, 32), dtype=np.float32)
    mask1[12:20, 12:20] = 1.0
    
    mask2 = np.zeros((32, 32), dtype=np.float32) 
    mask2[14:18, 14:18] = 1.0
    
    mask1_tf = tf.constant(mask1, dtype=tf.float32)
    mask2_tf = tf.constant(mask2, dtype=tf.float32)
    
    # Compute spectral loss
    loss = compute_spectral_loss(
        mask1_tf, mask2_tf,
        AbbeGradientSimulator(pixel=8, sigma=0.3, na=1.35, wavelength=193),
        pixel=8, sigma=0.3, na=1.35, wavelength=193
    )
    
    print(f"Spectral loss between similar masks: {loss:.6f}")
    
    # Test with identical masks
    loss_same = compute_spectral_loss(
        mask1_tf, mask1_tf,
        AbbeGradientSimulator(pixel=8, sigma=0.3, na=1.35, wavelength=193),
        pixel=8, sigma=0.3, na=1.35, wavelength=193
    )
    
    print(f"Spectral loss between identical masks: {loss_same:.6f}")
    
    return float(loss), float(loss_same)


if __name__ == "__main__":
    try:
        test_abbe_gradient()
        test_spectral_loss()
        print("\n✅ All gradient tests completed successfully!")
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        raise