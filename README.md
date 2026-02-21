# TF-Litho: TensorFlow-based Lithography Simulation Engine

TensorFlow 2.12移植版本 of TorchLitho-Lite, with OpenCV dependencies removed.

## Key Features
- **Abbe Model**: Fast partially coherent lithography simulation
- **Hopkins Model**: Accurate TCC-based simulation  
- **Gradient Support**: Built-in custom gradients for mask optimization
- **No OpenCV**: Uses scikit-image and TensorFlow native functions
- **Lightweight Testing**: Ready for high-performance validation
- **Example Directory**: Complete examples mirroring TorchLitho-Lite functionality

## Gradient Computation

Unlike PyTorch's automatic `tensor.grad` attribute, TensorFlow uses:
- **tf.GradientTape()**: Records operations for automatic differentiation
- **Custom Gradients**: Implemented via `@tf.custom_gradient` decorator
- **Variable Tracking**: Use `tf.Variable` for trainable parameters

The `gradient.py` module provides gradient-enabled simulators that can be used directly in optimization loops.

## Examples

The `example/` directory contains complete implementations mirroring TorchLitho-Lite:

- `iccad.py`: ICCAD2013 benchmark example with gradient map visualization
- `rect.py`: Rectangular pattern example comparing Abbe and Hopkins models

Both examples demonstrate how to compute and visualize gradient maps for mask optimization.

## Dependencies
- tensorflow>=2.12.0
- numpy>=1.19.0
- scipy>=1.7.0
- scikit-image>=0.18.0
- matplotlib>=3.3.0