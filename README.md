# TF-Litho: TensorFlow-based Lithography Simulation Engine

TensorFlow 2.12移植版本 of TorchLitho-Lite, with OpenCV dependencies removed.

## Key Features
- **Abbe Model**: Fast partially coherent lithography simulation
- **Hopkins Model**: Accurate TCC-based simulation  
- **No OpenCV**: Uses scikit-image and TensorFlow native functions
- **Lightweight Testing**: Ready for high-performance validation

## Dependencies
- tensorflow>=2.12.0
- numpy>=1.19.0
- scipy>=1.7.0
- scikit-image>=0.18.0
- matplotlib>=3.3.0