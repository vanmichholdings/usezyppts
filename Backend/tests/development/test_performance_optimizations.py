#!/usr/bin/env python3
"""
Performance testing script for the optimized logo processor.
Tests various scenarios and measures performance improvements.
"""

import os
import sys
import time
import json
import logging
import multiprocessing
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyppts.utils.logo_processor import LogoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing for the logo processor."""
    
    def __init__(self):
        self.results = {}
        self.test_images = []
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Setup test environment and create test images."""
        logger.info("Setting up test environment...")
        
        # Create test directory
        self.test_dir = Path("test_performance")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test images of different sizes and complexities
        self.create_test_images()
        
        logger.info(f"Created {len(self.test_images)} test images")
    
    def create_test_images(self):
        """Create test images with varying complexity."""
        sizes = [(500, 500), (1000, 1000), (2000, 2000)]
        complexities = ['simple', 'medium', 'complex']
        
        for size in sizes:
            for complexity in complexities:
                image_path = self.test_dir / f"test_{size[0]}x{size[1]}_{complexity}.png"
                self.create_test_image(image_path, size, complexity)
                self.test_images.append(str(image_path))
    
    def create_test_image(self, path, size, complexity):
        """Create a test image with specified characteristics."""
        width, height = size
        
        if complexity == 'simple':
            # Simple logo with 2 colors
            img = Image.new('RGB', size, (255, 255, 255))
            # Add simple shapes
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], fill=(0, 0, 255))
            draw.ellipse([width//3, height//3, 2*width//3, 2*height//3], fill=(255, 0, 0))
            
        elif complexity == 'medium':
            # Medium complexity with 4-5 colors
            img = Image.new('RGB', size, (240, 240, 240))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Add multiple shapes with different colors
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, color in enumerate(colors):
                x1 = (i * width) // len(colors)
                x2 = ((i + 1) * width) // len(colors)
                draw.rectangle([x1, height//4, x2, 3*height//4], fill=color)
                
        else:  # complex
            # Complex logo with gradients and many colors
            img = Image.new('RGB', size, (255, 255, 255))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Create complex pattern
            for i in range(0, width, 50):
                for j in range(0, height, 50):
                    color = ((i * 255) // width, (j * 255) // height, 128)
                    draw.rectangle([i, j, i+40, j+40], fill=color)
        
        img.save(path, 'PNG')
    
    def test_basic_processing(self):
        """Test basic logo processing performance."""
        logger.info("Testing basic processing performance...")
        
        processor = LogoProcessor()
        results = []
        
        for image_path in self.test_images[:3]:  # Test with first 3 images
            start_time = time.time()
            
            options = {
                'transparent_png': True,
                'black_version': True,
                'distressed_effect': True
            }
            
            result = processor.process_logo(image_path, options)
            processing_time = time.time() - start_time
            
            results.append({
                'image': os.path.basename(image_path),
                'processing_time': processing_time,
                'success': result.get('success', False),
                'outputs_count': len(result.get('outputs', {}))
            })
            
            logger.info(f"Processed {os.path.basename(image_path)} in {processing_time:.2f}s")
        
        self.results['basic_processing'] = results
        return results
    
    def test_vector_trace_performance(self):
        """Test vector trace performance."""
        logger.info("Testing vector trace performance...")
        
        processor = LogoProcessor()
        results = []
        
        for image_path in self.test_images[:2]:  # Test with first 2 images
            start_time = time.time()
            
            options = {
                'vector_trace': True,
                'max_dimension': 1500
            }
            
            result = processor.generate_vector_trace(image_path, options)
            processing_time = time.time() - start_time
            
            results.append({
                'image': os.path.basename(image_path),
                'processing_time': processing_time,
                'success': result.get('status') == 'success',
                'colors_used': result.get('colors_used', 0)
            })
            
            logger.info(f"Vector traced {os.path.basename(image_path)} in {processing_time:.2f}s")
        
        self.results['vector_trace'] = results
        return results
    
    def test_parallel_processing(self):
        """Test parallel processing performance."""
        logger.info("Testing parallel processing performance...")
        
        processor = LogoProcessor()
        
        # Test with multiple images simultaneously
        test_images = self.test_images[:4]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for image_path in test_images:
                options = {
                    'transparent_png': True,
                    'black_version': True
                }
                future = executor.submit(processor.process_logo, image_path, options)
                futures.append((future, image_path))
            
            results = []
            for future, image_path in futures:
                try:
                    result = future.result(timeout=60)
                    results.append({
                        'image': os.path.basename(image_path),
                        'success': result.get('success', False),
                        'outputs_count': len(result.get('outputs', {}))
                    })
                except Exception as e:
                    results.append({
                        'image': os.path.basename(image_path),
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        self.results['parallel_processing'] = {
            'total_time': total_time,
            'images_processed': len(test_images),
            'avg_time_per_image': total_time / len(test_images),
            'results': results
        }
        
        logger.info(f"Parallel processed {len(test_images)} images in {total_time:.2f}s")
        return self.results['parallel_processing']
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        logger.info("Testing memory usage...")
        
        processor = LogoProcessor()
        memory_stats = []
        
        for image_path in self.test_images[:3]:
            # Get initial memory usage
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            start_time = time.time()
            options = {
                'transparent_png': True,
                'black_version': True,
                'vector_trace': True
            }
            
            result = processor.process_logo(image_path, options)
            processing_time = time.time() - start_time
            
            # Get final memory usage
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            memory_stats.append({
                'image': os.path.basename(image_path),
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'processing_time': processing_time
            })
            
            logger.info(f"Memory test for {os.path.basename(image_path)}: "
                       f"Initial: {initial_memory:.1f}MB, "
                       f"Final: {final_memory:.1f}MB, "
                       f"Increase: {memory_increase:.1f}MB")
        
        self.results['memory_usage'] = memory_stats
        return memory_stats
    
    def test_configuration_optimization(self):
        """Test different configuration optimizations."""
        logger.info("Testing configuration optimizations...")
        
        processor = LogoProcessor()
        config_results = {}
        
        # Test different performance configurations
        for config_type in ['fast', 'balanced', 'quality']:
            logger.info(f"Testing {config_type} configuration...")
            
            # Optimize configuration
            optimization_result = processor.optimize_configuration(config_type)
            
            # Test with one image
            image_path = self.test_images[0]
            start_time = time.time()
            
            options = {
                'transparent_png': True,
                'black_version': True,
                'vector_trace': True
            }
            
            result = processor.process_logo(image_path, options)
            processing_time = time.time() - start_time
            
            config_results[config_type] = {
                'processing_time': processing_time,
                'success': result.get('success', False),
                'optimization_config': optimization_result.get('configuration', {}),
                'estimated_improvement': optimization_result.get('estimated_improvement', '')
            }
            
            logger.info(f"{config_type} config: {processing_time:.2f}s")
        
        self.results['configuration_optimization'] = config_results
        return config_results
    
    def test_system_resources(self):
        """Test system resource monitoring."""
        logger.info("Testing system resource monitoring...")
        
        processor = LogoProcessor()
        
        # Get performance stats
        stats = processor.get_performance_stats()
        
        # Get optimization recommendations
        recommendations = processor.get_optimization_recommendations()
        
        # Get vector trace stats
        vector_stats = processor.get_vector_trace_stats()
        
        system_info = {
            'performance_stats': stats,
            'recommendations': recommendations,
            'vector_trace_stats': vector_stats,
            'system_resources': {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'current_memory_percent': psutil.virtual_memory().percent,
                'current_cpu_percent': psutil.cpu_percent(interval=1)
            }
        }
        
        self.results['system_resources'] = system_info
        return system_info
    
    def run_all_tests(self):
        """Run all performance tests."""
        logger.info("Starting comprehensive performance testing...")
        
        try:
            # Run all tests
            self.test_basic_processing()
            self.test_vector_trace_performance()
            self.test_parallel_processing()
            self.test_memory_usage()
            self.test_configuration_optimization()
            self.test_system_resources()
            
            # Generate summary report
            self.generate_report()
            
            logger.info("All performance tests completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during performance testing: {e}", exc_info=True)
    
    def generate_report(self):
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_summary': {},
            'recommendations': [],
            'performance_metrics': {}
        }
        
        # Basic processing summary
        if 'basic_processing' in self.results:
            basic_results = self.results['basic_processing']
            avg_time = sum(r['processing_time'] for r in basic_results) / len(basic_results)
            success_rate = sum(1 for r in basic_results if r['success']) / len(basic_results)
            
            report['test_summary']['basic_processing'] = {
                'avg_processing_time': avg_time,
                'success_rate': success_rate,
                'total_tests': len(basic_results)
            }
        
        # Vector trace summary
        if 'vector_trace' in self.results:
            vector_results = self.results['vector_trace']
            avg_time = sum(r['processing_time'] for r in vector_results) / len(vector_results)
            success_rate = sum(1 for r in vector_results if r['success']) / len(vector_results)
            
            report['test_summary']['vector_trace'] = {
                'avg_processing_time': avg_time,
                'success_rate': success_rate,
                'total_tests': len(vector_results)
            }
        
        # Parallel processing summary
        if 'parallel_processing' in self.results:
            parallel_results = self.results['parallel_processing']
            report['test_summary']['parallel_processing'] = {
                'total_time': parallel_results['total_time'],
                'avg_time_per_image': parallel_results['avg_time_per_image'],
                'images_processed': parallel_results['images_processed']
            }
        
        # Memory usage summary
        if 'memory_usage' in self.results:
            memory_results = self.results['memory_usage']
            avg_memory_increase = sum(r['memory_increase_mb'] for r in memory_results) / len(memory_results)
            max_memory_increase = max(r['memory_increase_mb'] for r in memory_results)
            
            report['test_summary']['memory_usage'] = {
                'avg_memory_increase_mb': avg_memory_increase,
                'max_memory_increase_mb': max_memory_increase,
                'total_tests': len(memory_results)
            }
        
        # Configuration optimization summary
        if 'configuration_optimization' in self.results:
            config_results = self.results['configuration_optimization']
            fastest_config = min(config_results.keys(), 
                               key=lambda k: config_results[k]['processing_time'])
            
            report['test_summary']['configuration_optimization'] = {
                'fastest_config': fastest_config,
                'fastest_time': config_results[fastest_config]['processing_time']
            }
        
        # Generate recommendations
        if 'system_resources' in self.results:
            recommendations = self.results['system_resources'].get('recommendations', {})
            report['recommendations'] = recommendations.get('recommendations', [])
        
        # Save detailed results
        report_path = self.test_dir / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        summary_path = self.test_dir / 'performance_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        if 'basic_processing' in report['test_summary']:
            bp = report['test_summary']['basic_processing']
            print(f"Basic Processing: {bp['avg_processing_time']:.2f}s avg, {bp['success_rate']*100:.1f}% success")
        
        if 'vector_trace' in report['test_summary']:
            vt = report['test_summary']['vector_trace']
            print(f"Vector Trace: {vt['avg_processing_time']:.2f}s avg, {vt['success_rate']*100:.1f}% success")
        
        if 'parallel_processing' in report['test_summary']:
            pp = report['test_summary']['parallel_processing']
            print(f"Parallel Processing: {pp['total_time']:.2f}s total, {pp['avg_time_per_image']:.2f}s per image")
        
        if 'memory_usage' in report['test_summary']:
            mu = report['test_summary']['memory_usage']
            print(f"Memory Usage: {mu['avg_memory_increase_mb']:.1f}MB avg increase")
        
        if 'configuration_optimization' in report['test_summary']:
            co = report['test_summary']['configuration_optimization']
            print(f"Fastest Config: {co['fastest_config']} ({co['fastest_time']:.2f}s)")
        
        print()
        print(f"Detailed results saved to: {report_path}")
        print(f"Summary report saved to: {summary_path}")
        print("="*60)

def main():
    """Main function to run performance tests."""
    print("Logo Processor Performance Testing")
    print("="*40)
    
    tester = PerformanceTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 