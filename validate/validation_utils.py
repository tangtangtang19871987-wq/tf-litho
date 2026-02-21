"""
Validation utilities for tf-litho implementation.
Provides functions to compare with original TorchLitho-Lite outputs.
"""
import numpy as np
import tensorflow as tf


def compare_outputs(tf_output, torch_output, tolerance=1e-5):
    """
    Compare TensorFlow and PyTorch outputs for numerical equivalence.
    
    Args:
        tf_output: TensorFlow output array
        torch_output: PyTorch output array  
        tolerance: Numerical tolerance for comparison
        
    Returns:
        dict: Comparison results with metrics
    """
    # Convert to numpy if needed
    if hasattr(tf_output, 'numpy'):
        tf_output = tf_output.numpy()
    if hasattr(torch_output, 'numpy'):
        torch_output = torch_output.numpy()
    
    # Ensure same shape
    assert tf_output.shape == torch_output.shape, f"Shape mismatch: {tf_output.shape} vs {torch_output.shape}"
    
    # Compute differences
    abs_diff = np.abs(tf_output - torch_output)
    rel_diff = abs_diff / (np.abs(torch_output) + 1e-12)
    
    results = {
        'max_absolute_error': float(np.max(abs_diff)),
        'mean_absolute_error': float(np.mean(abs_diff)),
        'max_relative_error': float(np.max(rel_diff)),
        'mean_relative_error': float(np.mean(rel_diff)),
        'within_tolerance': bool(np.all(abs_diff <= tolerance)),
        'output_shape': tf_output.shape,
        'tf_output_range': (float(np.min(tf_output)), float(np.max(tf_output))),
        'torch_output_range': (float(np.min(torch_output)), float(np.max(torch_output)))
    }
    
    return results


def validate_gradient_analytical_vs_numerical(
    func, input_tensor, epsilon=1e-6, tolerance=1e-4
):
    """
    Validate analytical gradient against numerical gradient.
    
    Args:
        func: Function to differentiate
        input_tensor: Input tensor (requires gradient)
        epsilon: Step size for numerical gradient
        tolerance: Tolerance for gradient comparison
        
    Returns:
        dict: Gradient validation results
    """
    # Analytical gradient (using TensorFlow's automatic differentiation)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        output = func(input_tensor)
        loss = tf.reduce_sum(output)  # Simple loss for gradient computation
    
    analytical_grad = tape.gradient(loss, input_tensor)
    
    # Numerical gradient
    input_np = input_tensor.numpy()
    numerical_grad = np.zeros_like(input_np)
    
    for i in range(input_np.size):
        # Flatten index
        idx = np.unravel_index(i, input_np.shape)
        
        # Positive perturbation
        input_plus = input_np.copy()
        input_plus[idx] += epsilon
        output_plus = func(tf.constant(input_plus, dtype=input_tensor.dtype))
        loss_plus = tf.reduce_sum(output_plus)
        
        # Negative perturbation  
        input_minus = input_np.copy()
        input_minus[idx] -= epsilon
        output_minus = func(tf.constant(input_minus, dtype=input_tensor.dtype))
        loss_minus = tf.reduce_sum(output_minus)
        
        # Central difference
        numerical_grad[idx] = (loss_plus.numpy() - loss_minus.numpy()) / (2 * epsilon)
    
    # Compare gradients
    abs_diff = np.abs(analytical_grad.numpy() - numerical_grad)
    results = {
        'max_gradient_error': float(np.max(abs_diff)),
        'mean_gradient_error': float(np.mean(abs_diff)),
        'gradient_within_tolerance': bool(np.all(abs_diff <= tolerance)),
        'input_shape': input_np.shape
    }
    
    return results


def create_test_masks():
    """Create standard test masks for validation."""
    masks = {}
    
    # Simple rectangle
    rect = np.zeros((64, 64), dtype=np.float32)
    rect[20:44, 20:44] = 1.0
    masks['rectangle'] = rect
    
    # Cross pattern
    cross = np.zeros((64, 64), dtype=np.float32)
    cross[30:34, :] = 1.0
    cross[:, 30:34] = 1.0
    masks['cross'] = cross
    
    # Checkerboard
    checker = np.zeros((64, 64), dtype=np.float32)
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i//8 + j//8) % 2 == 0:
                checker[i:i+8, j:j+8] = 1.0
    masks['checkerboard'] = checker
    
    # Random mask
    np.random.seed(42)  # For reproducibility
    random_mask = np.random.rand(64, 64).astype(np.float32)
    random_mask = (random_mask > 0.5).astype(np.float32)
    masks['random'] = random_mask
    
    return masks


def save_validation_results(results, filename):
    """Save validation results to JSON file."""
    import json
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_types(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_torch_reference_outputs():
    """
    Load reference outputs from original TorchLitho-Lite.
    This function should be implemented when TorchLitho-Lite outputs are available.
    """
    raise NotImplementedError(
        "This function requires pre-computed TorchLitho-Lite outputs. "
        "Run the original implementation and save outputs for comparison."
    )