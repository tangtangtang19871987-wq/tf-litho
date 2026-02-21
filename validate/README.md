# Validation Framework for tf-litho

This directory contains validation scripts to verify the correctness of the TensorFlow implementation against the original TorchLitho-Lite.

## Validation Strategy

### 1. Numerical Precision Validation
- Compare outputs from identical inputs
- Verify numerical precision (absolute and relative error)
- Test edge cases and boundary conditions

### 2. Functional Equivalence  
- Same API interface
- Identical parameter behavior
- Consistent output shapes and ranges

### 3. Gradient Correctness
- Compare analytical gradients with numerical gradients
- Verify gradient descent convergence
- Test optimization stability

### 4. Performance Benchmarking
- Execution time comparison
- Memory usage analysis
- Scalability testing

## Usage Instructions

1. **Install both implementations**:
   ```bash
   # Original TorchLitho-Lite
   git clone https://github.com/OpenOPC/TorchLitho-Lite.git
   cd TorchLitho-Lite && pip install -e .
   
   # TensorFlow implementation  
   cd tf-litho && pip install -e .
   ```

2. **Run validation tests**:
   ```bash
   python validate_abbe.py
   python validate_hopkins.py  
   python validate_gradients.py
   ```

3. **Analyze results**:
   - Check error thresholds
   - Review performance metrics
   - Validate optimization results

## Expected Results

- **Numerical error**: < 1e-6 for most cases
- **Functional equivalence**: Identical behavior within floating-point precision
- **Gradient accuracy**: Relative error < 1e-4
- **Performance**: Comparable execution times (TensorFlow may be slower on CPU)

## Notes

- Validation requires significant computational resources
- Run on high-performance machine with GPU support
- Some differences expected due to framework-specific optimizations
- Focus on functional correctness rather than exact numerical equality