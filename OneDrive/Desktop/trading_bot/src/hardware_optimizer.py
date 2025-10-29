"""
Hardware optimization manager for CPU/GPU deployment scenarios
"""
import os
import logging
import platform
import psutil
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """Automatically detects and optimizes for available hardware"""
    
    def __init__(self):
        self.hardware_info = self.detect_hardware()
        self.optimization_config = self.generate_optimization_config()
        
        logger.info(f"Hardware detected: {self.hardware_info}")
        logger.info(f"Optimization strategy: {self.optimization_config['strategy']}")
    
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware capabilities"""
        hardware_info = {
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory_gb': [],
            'cuda_version': None,
            'has_intel_mkl': self._check_intel_mkl(),
            'has_avx2': self._check_avx2_support()
        }
        
        # GPU Detection
        if torch.cuda.is_available():
            hardware_info['gpu_count'] = torch.cuda.device_count()
            hardware_info['cuda_version'] = torch.version.cuda
            
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                hardware_info['gpu_names'].append(gpu_props.name)
                hardware_info['gpu_memory_gb'].append(
                    round(gpu_props.total_memory / (1024**3), 2)
                )
        
        return hardware_info
    
    def _check_intel_mkl(self) -> bool:
        """Check if Intel MKL is available for NumPy optimization"""
        try:
            import numpy as np
            return 'mkl' in np.__config__.show().lower()
        except:
            return False
    
    def _check_avx2_support(self) -> bool:
        """Check if CPU supports AVX2 instructions"""
        try:
            import cpuinfo
            cpu_flags = cpuinfo.get_cpu_info().get('flags', [])
            return 'avx2' in cpu_flags
        except:
            # Fallback check for Windows
            if platform.system() == 'Windows':
                try:
                    import subprocess
                    result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                         capture_output=True, text=True)
                    # Most modern CPUs support AVX2
                    return 'Intel' in result.stdout or 'AMD' in result.stdout
                except:
                    return True  # Assume modern CPU
            return True
    
    def generate_optimization_config(self) -> Dict[str, Any]:
        """Generate optimization configuration based on detected hardware"""
        config = {
            'strategy': 'cpu_optimized',
            'model_config': {},
            'processing_config': {},
            'memory_config': {},
            'inference_config': {}
        }
        
        # Determine primary strategy
        if self.hardware_info['gpu_available'] and self.hardware_info['gpu_count'] > 0:
            if max(self.hardware_info['gpu_memory_gb']) >= 8:
                config['strategy'] = 'gpu_accelerated'
            elif max(self.hardware_info['gpu_memory_gb']) >= 4:
                config['strategy'] = 'gpu_light'
            else:
                config['strategy'] = 'cpu_optimized'
        else:
            config['strategy'] = 'cpu_optimized'
        
        # Configure based on strategy
        if config['strategy'] == 'gpu_accelerated':
            config = self._configure_gpu_accelerated(config)
        elif config['strategy'] == 'gpu_light':
            config = self._configure_gpu_light(config)
        else:
            config = self._configure_cpu_optimized(config)
        
        return config
    
    def _configure_gpu_accelerated(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration for high-end GPU deployment"""
        config.update({
            'model_config': {
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'use_transformer_cache': True,
                'batch_size': 32,
                'max_length': 512,
                'use_fp16': True,  # Half precision for memory efficiency
                'device': 'cuda',
                'model_parallel': self.hardware_info['gpu_count'] > 1
            },
            'processing_config': {
                'parallel_workers': min(8, self.hardware_info['cpu_count']),
                'async_processing': True,
                'batch_processing': True,
                'prefetch_factor': 4
            },
            'memory_config': {
                'pin_memory': True,
                'memory_fraction': 0.8,  # Use 80% of GPU memory
                'gradient_checkpointing': True
            },
            'inference_config': {
                'torch_compile': True,  # PyTorch 2.0 optimization
                'dynamic_batching': True,
                'model_caching': True
            }
        })
        return config
    
    def _configure_gpu_light(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration for low-memory GPU deployment"""
        config.update({
            'model_config': {
                'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'use_transformer_cache': False,
                'batch_size': 16,
                'max_length': 256,
                'use_fp16': True,
                'device': 'cuda',
                'model_parallel': False
            },
            'processing_config': {
                'parallel_workers': min(4, self.hardware_info['cpu_count']),
                'async_processing': True,
                'batch_processing': True,
                'prefetch_factor': 2
            },
            'memory_config': {
                'pin_memory': False,
                'memory_fraction': 0.6,
                'gradient_checkpointing': True
            },
            'inference_config': {
                'torch_compile': False,
                'dynamic_batching': False,
                'model_caching': True
            }
        })
        return config
    
    def _configure_cpu_optimized(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration for CPU-only deployment"""
        config.update({
            'model_config': {
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'use_transformer_cache': True,
                'batch_size': 8,
                'max_length': 256,
                'use_fp16': False,
                'device': 'cpu',
                'model_parallel': False,
                'num_threads': self.hardware_info['cpu_count']
            },
            'processing_config': {
                'parallel_workers': max(2, self.hardware_info['cpu_count'] // 2),
                'async_processing': True,
                'batch_processing': False,
                'prefetch_factor': 1,
                'use_multiprocessing': True
            },
            'memory_config': {
                'pin_memory': False,
                'memory_fraction': 0.4,
                'gradient_checkpointing': False
            },
            'inference_config': {
                'torch_compile': False,
                'dynamic_batching': False,
                'model_caching': True,
                'onnx_optimization': True  # Use ONNX for CPU optimization
            }
        })
        
        # Intel MKL optimizations
        if self.hardware_info['has_intel_mkl']:
            config['processing_config']['mkl_threads'] = self.hardware_info['cpu_count']
            config['processing_config']['use_mkl_blas'] = True
        
        return config
    
    def optimize_torch_settings(self):
        """Apply PyTorch optimizations based on hardware"""
        if self.optimization_config['strategy'] == 'cpu_optimized':
            # CPU optimizations
            torch.set_num_threads(self.hardware_info['cpu_count'])
            torch.set_num_interop_threads(self.hardware_info['cpu_count'])
            
            # Intel MKL optimizations
            if self.hardware_info['has_intel_mkl']:
                os.environ['MKL_NUM_THREADS'] = str(self.hardware_info['cpu_count'])
                os.environ['OMP_NUM_THREADS'] = str(self.hardware_info['cpu_count'])
        
        elif 'gpu' in self.optimization_config['strategy']:
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            if self.optimization_config['model_config']['use_fp16']:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
    
    def get_optimal_worker_count(self, task_type: str = 'general') -> int:
        """Get optimal worker count for different task types"""
        base_workers = self.optimization_config['processing_config']['parallel_workers']
        
        if task_type == 'io_bound':  # For API calls, WebSocket connections
            return min(base_workers * 2, 16)
        elif task_type == 'cpu_bound':  # For calculations, data processing
            return base_workers
        elif task_type == 'ml_inference':  # For model inference
            if 'gpu' in self.optimization_config['strategy']:
                return min(base_workers, 4)  # Fewer workers for GPU
            else:
                return base_workers
        
        return base_workers
    
    def get_memory_limits(self) -> Dict[str, float]:
        """Get memory limits for different components"""
        total_memory_gb = self.hardware_info['memory_gb']
        
        return {
            'market_data_cache_mb': min(512, total_memory_gb * 50),  # 50MB per GB
            'sentiment_cache_mb': min(256, total_memory_gb * 25),
            'model_cache_mb': min(1024, total_memory_gb * 100),
            'feature_store_mb': min(256, total_memory_gb * 25)
        }
    
    def should_use_lightweight_models(self) -> bool:
        """Determine if lightweight models should be used"""
        return (
            self.hardware_info['memory_gb'] < 8 or
            (not self.hardware_info['gpu_available'] and self.hardware_info['cpu_count'] < 8)
        )
    
    def get_deployment_recommendations(self) -> Dict[str, Any]:
        """Get deployment recommendations based on hardware"""
        recommendations = {
            'deployment_type': 'single_node',
            'scaling_strategy': 'vertical',
            'resource_allocation': {},
            'performance_expectations': {},
            'optimization_tips': []
        }
        
        if self.hardware_info['gpu_available']:
            recommendations['deployment_type'] = 'gpu_accelerated'
            recommendations['performance_expectations'] = {
                'sentiment_analysis_speed': 'High (GPU accelerated)',
                'model_inference_speed': 'High',
                'memory_efficiency': 'Good',
                'power_consumption': 'High'
            }
            recommendations['optimization_tips'] = [
                'Use batch processing for maximum GPU utilization',
                'Enable FP16 precision for memory efficiency',
                'Consider model parallelism for multiple GPUs'
            ]
        else:
            recommendations['deployment_type'] = 'cpu_optimized'
            recommendations['performance_expectations'] = {
                'sentiment_analysis_speed': 'Medium (CPU optimized)',
                'model_inference_speed': 'Medium',
                'memory_efficiency': 'Excellent',
                'power_consumption': 'Low'
            }
            recommendations['optimization_tips'] = [
                'Use ONNX runtime for faster CPU inference',
                'Enable Intel MKL for mathematical operations',
                'Consider horizontal scaling for high throughput'
            ]
        
        # Resource allocation
        cpu_allocation = max(2, self.hardware_info['cpu_count'] // 2)
        memory_allocation = max(2, self.hardware_info['memory_gb'] // 2)
        
        recommendations['resource_allocation'] = {
            'cpu_cores': cpu_allocation,
            'memory_gb': memory_allocation,
            'gpu_memory_gb': max(self.hardware_info['gpu_memory_gb']) if self.hardware_info['gpu_available'] else 0
        }
        
        return recommendations

# Global hardware optimizer instance
hardware_optimizer = HardwareOptimizer()