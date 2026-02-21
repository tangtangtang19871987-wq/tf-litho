"""
Performance benchmarking for tf-litho implementation.
This script measures execution time and memory usage.
"""
import time
import tensorflow as tf
import numpy as np
from tf_litho.abbe import abbe_simulate
from tf_litho.hopkins import HopkinsSimulator


def benchmark_abbe_simulation(mask_sizes, pixel_sizes, num_runs=3):
    """Benchmark Abbe simulation performance."""
    results = {}
    
    for mask_size in mask_sizes:
        for pixel_size in pixel_sizes:
            if mask_size % pixel_size != 0:
                continue
                
            print(f"Benchmarking Abbe: {mask_size}x{mask_size}, pixel={pixel_size}")
            
            # Create test mask
            mask = np.random.rand(mask_size, mask_size).astype(np.float32)
            mask = (mask > 0.5).astype(np.float32)
            
            times = []
            for run in range(num_runs):
                start_time = time.time()
                _ = abbe_simulate(
                    mask, 
                    pixel=pixel_size, 
                    sigma=0.3, 
                    na=1.35, 
                    wavelength=193
                )
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            key = f"abbe_{mask_size}x{mask_size}_p{pixel_size}"
            results[key] = {
                'avg_time': float(avg_time),
                'std_time': float(std_time),
                'mask_size': mask_size,
                'pixel_size': pixel_size,
                'effective_grid': mask_size // pixel_size
            }
            
            print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    
    return results


def benchmark_hopkins_simulation(small_params, num_runs=3):
    """Benchmark Hopkins simulation performance with small parameters."""
    results = {}
    
    for params in small_params:
        canvas, pixel = params['canvas'], params['pixel']
        print(f"Benchmarking Hopkins: canvas={canvas}, pixel={pixel}")
        
        # Create test mask
        mask_size = canvas // pixel
        mask = np.random.rand(mask_size, mask_size).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        
        # Initialize simulator (TCC generation is expensive)
        simulator = HopkinsSimulator(
            pixel=pixel,
            canvas=canvas,
            na=1.35,
            wavelength=193,
            thresh=1e-2  # Higher threshold for faster TCC
        )
        
        times = []
        for run in range(num_runs):
            start_time = time.time()
            _ = simulator(mask)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        key = f"hopkins_canvas{canvas}_p{pixel}"
        results[key] = {
            'avg_time': float(avg_time),
            'std_time': float(std_time),
            'canvas': canvas,
            'pixel': pixel,
            'effective_grid': canvas // pixel
        }
        
        print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    
    return results


def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 Starting performance benchmark...\n")
    
    # Benchmark Abbe model
    abbe_results = benchmark_abbe_simulation(
        mask_sizes=[64, 128, 256],
        pixel_sizes=[4, 8, 16],
        num_runs=2
    )
    
    # Benchmark Hopkins model (with small parameters only)
    hopkins_params = [
        {'canvas': 64, 'pixel': 16},
        {'canvas': 128, 'pixel': 16},
        {'canvas': 64, 'pixel': 8}
    ]
    hopkins_results = benchmark_hopkins_simulation(hopkins_params, num_runs=2)
    
    # Combine results
    all_results = {
        'abbe_benchmark': abbe_results,
        'hopkins_benchmark': hopkins_results,
        'timestamp': time.time()
    }
    
    # Save results
    import json
    with open('performance_benchmark.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Performance benchmark completed!")
    print("Results saved to: performance_benchmark.json")
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_performance_benchmark()
        print("\n📊 Benchmark Summary:")
        for key, value in results.items():
            if isinstance(value, dict) and 'avg_time' in value:
                print(f"  {key}: {value['avg_time']:.4f}s")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise