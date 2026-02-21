#!/usr/bin/env python3
"""
Quick validation script to run all validation tests and generate comparison reports.
"""
import os
import sys
import argparse
from validate.validation_results import ValidationResults

def main():
    parser = argparse.ArgumentParser(description='Run TF-Litho validation suite')
    parser.add_argument('--output-dir', '-o', default='validation_output',
                       help='Output directory for validation results')
    parser.add_argument('--test-iccad', action='store_true',
                       help='Run ICCAD benchmark validation')
    parser.add_argument('--test-rect', action='store_true', 
                       help='Run rectangle pattern validation')
    parser.add_argument('--all', action='store_true',
                       help='Run all validation tests')
    
    args = parser.parse_args()
    
    if not any([args.test_iccad, args.test_rect, args.all]):
        print("Please specify at least one test to run (--test-iccad, --test-rect, or --all)")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    validator = ValidationResults(output_dir=args.output_dir)
    
    if args.all or args.test_iccad:
        print("Running ICCAD validation...")
        validator.validate_iccad_benchmark()
        
    if args.all or args.test_rect:
        print("Running rectangle pattern validation...")
        validator.validate_rect_pattern()
    
    print(f"Validation completed! Results saved to {args.output_dir}/")
    print("Check validation_report.html for detailed comparison.")

if __name__ == "__main__":
    main()