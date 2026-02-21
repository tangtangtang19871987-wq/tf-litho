# TensorFlow Lithography Implementation Notes

## Design Decisions

### 1. OpenCV Replacement Strategy
- **Original**: `cv2.resize()` for image interpolation
- **Replacement**: `skimage.transform.resize()` with anti-aliasing
- **Rationale**: scikit-image is pure Python/C and doesn't require external dependencies like OpenCV

### 2. TensorFlow vs PyTorch Differences
- **FFT**: `tf.signal.fft2d` vs `torch.fft.fft2`
- **Complex numbers**: TensorFlow has better native complex support
- **Batch operations**: `tf.map_fn` for batch processing instead of PyTorch's native batching

### 3. Performance Considerations
- **Memory**: TensorFlow eager execution uses more memory than PyTorch
- **Compilation**: Consider using `@tf.function` decorator for performance-critical sections
- **GPU**: Code is GPU-ready but tested only on CPU in this environment

## Current Implementation Status

### ✅ Completed
- Project structure and dependencies
- Core utility functions (FFT, interpolation, geometry)
- Basic Abbe model simulation
- Lightweight test script

### ⏳ Pending (for your high-performance machine)
- Full partial coherent illumination (multiple source points)
- Defocus phase calculation validation
- Batch processing optimization
- Hopkins model implementation
- Comprehensive benchmarking against original TorchLitho-Lite

## Usage Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run lightweight test**:
   ```bash
   python test_abbe.py
   ```

3. **For full validation on high-performance machine**:
   - Compare outputs with original TorchLitho-Lite
   - Test with ICCAD13 benchmark data
   - Validate gradient computation for optimization tasks

## Known Limitations

- **Single source point only**: Current implementation uses coherent illumination
- **Basic defocus**: Phase calculation needs validation with optical physics
- **No parallel processing**: `parallel=True` parameter not implemented yet
- **Memory intensive**: Large masks (>2048x2048) may require GPU memory

## Next Steps for Full Implementation

1. Implement multiple source points for partial coherent illumination
2. Add proper TCC generation for Hopkins model  
3. Implement custom gradients using `tf.GradientTape`
4. Add comprehensive test suite with reference outputs
5. Optimize for large-scale mask processing