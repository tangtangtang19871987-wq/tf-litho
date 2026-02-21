#!/usr/bin/env python3
"""
Validation Results Analysis for TF-Litho
======================================

This script provides comprehensive validation and comparison tools to:
1. Verify correctness of TensorFlow implementation against PyTorch reference
2. Generate diagnostic plots and metrics
3. Compare performance characteristics
4. Identify numerical discrepancies

Usage:
    python validate/validation_results.py --reference-dir /path/to/torchlitho/results
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import tf-litho modules
from tf_litho.abbe import abbe_simulate
from tf_litho.hopkins import hopkins_simulate


class ValidationResults:
    """Comprehensive validation and comparison class for TF-Litho."""
    
    def __init__(self, output_dir="validation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        
    def create_test_masks(self):
        """Create standardized test masks for validation."""
        test_masks = {}
        
        # Simple rectangle mask
        rect_mask = np.zeros((64, 64), dtype=np.float32)
        rect_mask[16:48, 16:48] = 1.0
        test_masks['rectangle'] = rect_mask
        
        # Checkerboard pattern
        checker = np.zeros((64, 64), dtype=np.float32)
        checker[::2, ::2] = 1.0
        checker[1::2, 1::2] = 1.0
        test_masks['checkerboard'] = checker
        
        # Gaussian blob
        x = np.linspace(-3, 3, 64)
        y = np.linspace(-3, 3, 64)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-(X**2 + Y**2) / 2.0).astype(np.float32)
        test_masks['gaussian'] = gaussian
        
        return test_masks
    
    def run_tf_simulation(self, mask, model_type='abbe', **kwargs):
        """Run TensorFlow simulation with specified parameters."""
        if model_type == 'abbe':
            result = abbe_simulate(
                mask[None, None, ...],  # Add batch and channel dims
                pixel=kwargs.get('pixel', 8),
                sigma=kwargs.get('sigma', 0.05),
                na=kwargs.get('na', 0.75),
                wavelength=kwargs.get('wavelength', 193),
                defocus=kwargs.get('defocus', 0),
                batch=True
            )
        elif model_type == 'hopkins':
            result = hopkins_simulate(
                mask[None, None, ...],
                filename=kwargs.get('tcc_file', None)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return result.numpy()[0, 0]  # Remove batch and channel dims
    
    def compute_gradients(self, mask, model_type='abbe', **kwargs):
        """Compute gradients using TensorFlow's GradientTape."""
        mask_tf = tf.Variable(mask[None, None, ...], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            if model_type == 'abbe':
                aerial = abbe_simulate(
                    mask_tf,
                    pixel=kwargs.get('pixel', 8),
                    sigma=kwargs.get('sigma', 0.05),
                    na=kwargs.get('na', 0.75),
                    wavelength=kwargs.get('wavelength', 193),
                    defocus=kwargs.get('defocus', 0),
                    batch=True
                )
            elif model_type == 'hopkins':
                aerial = hopkins_simulate(
                    mask_tf,
                    filename=kwargs.get('tcc_file', None)
                )
        
        # Compute gradient w.r.t. input mask
        grad_mask = tape.gradient(aerial, mask_tf)
        return grad_mask.numpy()[0, 0]
    
    def compare_with_reference(self, tf_result, ref_result, name="comparison"):
        """Compare TensorFlow result with reference (PyTorch) result."""
        metrics = {}
        
        # Basic statistics
        metrics['tf_mean'] = float(np.mean(tf_result))
        metrics['tf_std'] = float(np.std(tf_result))
        metrics['ref_mean'] = float(np.mean(ref_result))
        metrics['ref_std'] = float(np.std(ref_result))
        
        # Difference metrics
        diff = tf_result - ref_result
        metrics['mse'] = float(np.mean(diff**2))
        metrics['mae'] = float(np.mean(np.abs(diff)))
        metrics['max_abs_diff'] = float(np.max(np.abs(diff)))
        metrics['relative_error'] = float(np.mean(np.abs(diff) / (np.abs(ref_result) + 1e-12)))
        
        # Correlation
        metrics['correlation'] = float(np.corrcoef(tf_result.flatten(), ref_result.flatten())[0, 1])
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'tf_result': tf_result,
            'ref_result': ref_result,
            'difference': diff
        }
        
        return metrics
    
    def plot_comparison(self, name, save_plots=True):
        """Generate comprehensive comparison plots."""
        if name not in self.results:
            print(f"No results found for {name}")
            return
            
        data = self.results[name]
        tf_result = data['tf_result']
        ref_result = data['ref_result']
        diff = data['difference']
        metrics = data['metrics']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Validation Results: {name}\n'
                    f'MSE: {metrics["mse"]:.2e}, MAE: {metrics["mae"]:.2e}, '
                    f'Correlation: {metrics["correlation"]:.4f}', fontsize=14)
        
        # TF Result
        im1 = axes[0, 0].imshow(tf_result, cmap='viridis')
        axes[0, 0].set_title('TensorFlow Result')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Reference Result
        im2 = axes[0, 1].imshow(ref_result, cmap='viridis')
        axes[0, 1].set_title('Reference Result')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference
        im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', norm=Normalize(vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff))))
        axes[0, 2].set_title('Difference (TF - Ref)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Histogram comparison
        axes[1, 0].hist(tf_result.flatten(), bins=50, alpha=0.7, label='TF', density=True)
        axes[1, 0].hist(ref_result.flatten(), bins=50, alpha=0.7, label='Ref', density=True)
        axes[1, 0].set_title('Value Distribution')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Density')
        
        # Scatter plot
        axes[1, 1].scatter(ref_result.flatten(), tf_result.flatten(), alpha=0.5, s=1)
        axes[1, 1].plot([ref_result.min(), ref_result.max()], [ref_result.min(), ref_result.max()], 'r--')
        axes[1, 1].set_title('Scatter Plot (Ref vs TF)')
        axes[1, 1].set_xlabel('Reference Value')
        axes[1, 1].set_ylabel('TensorFlow Value')
        
        # Error histogram
        axes[1, 2].hist(diff.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].set_xlabel('Error (TF - Ref)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, f'{name}_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to: {plot_path}")
        
        plt.show()
    
    def run_comprehensive_validation(self, reference_dir=None):
        """Run comprehensive validation suite."""
        print("Running comprehensive validation...")
        
        # Create test masks
        test_masks = self.create_test_masks()
        
        # Test different configurations
        configs = [
            {'name': 'abbe_default', 'model_type': 'abbe', 'params': {}},
            {'name': 'abbe_high_na', 'model_type': 'abbe', 'params': {'na': 1.35}},
            {'name': 'abbe_defocus', 'model_type': 'abbe', 'params': {'defocus': 100}},
            {'name': 'abbe_fine_pixel', 'model_type': 'abbe', 'params': {'pixel': 2}},
        ]
        
        for config in configs:
            print(f"\nTesting {config['name']}...")
            
            for mask_name, mask in test_masks.items():
                full_name = f"{config['name']}_{mask_name}"
                
                # Run TF simulation
                tf_result = self.run_tf_simulation(
                    mask, 
                    model_type=config['model_type'], 
                    **config['params']
                )
                
                # If reference directory provided, load reference results
                if reference_dir:
                    ref_path = os.path.join(reference_dir, f"{full_name}_result.npy")
                    if os.path.exists(ref_path):
                        ref_result = np.load(ref_path)
                        metrics = self.compare_with_reference(tf_result, ref_result, full_name)
                        print(f"  {full_name}: MSE={metrics['mse']:.2e}, Corr={metrics['correlation']:.4f}")
                        
                        # Generate plots
                        self.plot_comparison(full_name)
                    else:
                        print(f"  Warning: Reference file not found: {ref_path}")
                        # Store TF result only
                        self.results[full_name] = {'tf_result': tf_result}
                else:
                    # Store TF result only
                    self.results[full_name] = {'tf_result': tf_result}
                    print(f"  {full_name}: TF simulation completed")
        
        # Save all results
        self.save_results()
        print(f"\nValidation completed! Results saved to {self.output_dir}")
    
    def save_results(self):
        """Save all validation results to disk."""
        # Save numerical results
        for name, data in self.results.items():
            if 'tf_result' in data:
                np.save(os.path.join(self.output_dir, f"{name}_tf_result.npy"), data['tf_result'])
            if 'ref_result' in data:
                np.save(os.path.join(self.output_dir, f"{name}_ref_result.npy"), data['ref_result'])
            if 'difference' in data:
                np.save(os.path.join(self.output_dir, f"{name}_difference.npy"), data['difference'])
        
        # Save metrics summary
        metrics_summary = {}
        for name, data in self.results.items():
            if 'metrics' in data:
                metrics_summary[name] = data['metrics']
        
        if metrics_summary:
            summary_path = os.path.join(self.output_dir, 'validation_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            print(f"Metrics summary saved to: {summary_path}")
    
    def generate_diagnostic_report(self):
        """Generate a comprehensive diagnostic report."""
        report_lines = []
        report_lines.append("TF-Litho Validation Diagnostic Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # TensorFlow version info
        report_lines.append("Environment Information:")
        report_lines.append(f"  TensorFlow version: {tf.__version__}")
        report_lines.append(f"  NumPy version: {np.__version__}")
        report_lines.append(f"  Python version: {os.sys.version.split()[0]}")
        report_lines.append("")
        
        # Results summary
        if self.results:
            report_lines.append("Validation Results Summary:")
            report_lines.append("-" * 30)
            
            for name, data in self.results.items():
                report_lines.append(f"\n{name}:")
                if 'metrics' in data:
                    metrics = data['metrics']
                    report_lines.append(f"  MSE: {metrics['mse']:.2e}")
                    report_lines.append(f"  MAE: {metrics['mae']:.2e}")
                    report_lines.append(f"  Max Abs Diff: {metrics['max_abs_diff']:.2e}")
                    report_lines.append(f"  Correlation: {metrics['correlation']:.4f}")
                    report_lines.append(f"  Relative Error: {metrics['relative_error']:.2%}")
                else:
                    report_lines.append("  No reference comparison available")
            
            # Overall assessment
            report_lines.append("\nOverall Assessment:")
            report_lines.append("-" * 20)
            
            if any('metrics' in data for data in self.results.values()):
                avg_mse = np.mean([data['metrics']['mse'] for data in self.results.values() if 'metrics' in data])
                avg_corr = np.mean([data['metrics']['correlation'] for data in self.results.values() if 'metrics' in data])
                
                if avg_mse < 1e-10 and avg_corr > 0.9999:
                    report_lines.append("  ✅ EXCELLENT: Implementation matches reference perfectly")
                elif avg_mse < 1e-8 and avg_corr > 0.999:
                    report_lines.append("  ✅ GOOD: Implementation matches reference closely")
                elif avg_mse < 1e-6 and avg_corr > 0.99:
                    report_lines.append("  ⚠️  ACCEPTABLE: Minor differences detected")
                else:
                    report_lines.append("  ❌ POOR: Significant differences detected")
            else:
                report_lines.append("  ℹ️  No reference comparison performed")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'diagnostic_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Diagnostic report saved to: {report_path}")
        return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Validate TF-Litho implementation')
    parser.add_argument('--reference-dir', type=str, 
                       help='Directory containing reference (PyTorch) results')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--plot-only', type=str, 
                       help='Plot specific comparison (name from results)')
    
    args = parser.parse_args()
    
    validator = ValidationResults(args.output_dir)
    
    if args.plot_only:
        # Just plot existing results
        validator.plot_comparison(args.plot_only)
    else:
        # Run full validation
        validator.run_comprehensive_validation(args.reference_dir)
        validator.generate_diagnostic_report()


if __name__ == "__main__":
    main()