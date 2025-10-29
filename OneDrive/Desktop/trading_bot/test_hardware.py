"""
Test script to demonstrate hardware optimization capabilities
"""
import sys
import os
sys.path.append(os.getcwd())

try:
    from src.hardware_optimizer import hardware_optimizer
    import json
    
    print("🤖 Trading Bot Hardware Optimization Analysis")
    print("=" * 60)
    
    # Display hardware information
    print("\n💻 DETECTED HARDWARE:")
    print(f"Platform: {hardware_optimizer.hardware_info['platform']}")
    print(f"CPU Cores: {hardware_optimizer.hardware_info['cpu_count']} ({hardware_optimizer.hardware_info['cpu_count_logical']} logical)")
    print(f"Memory: {hardware_optimizer.hardware_info['memory_gb']} GB")
    print(f"GPU Available: {'Yes' if hardware_optimizer.hardware_info['gpu_available'] else 'No'}")
    
    if hardware_optimizer.hardware_info['gpu_available']:
        print(f"GPU Count: {hardware_optimizer.hardware_info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(hardware_optimizer.hardware_info['gpu_names'], 
                                              hardware_optimizer.hardware_info['gpu_memory_gb'])):
            print(f"  GPU {i}: {name} ({memory} GB)")
        print(f"CUDA Version: {hardware_optimizer.hardware_info['cuda_version']}")
    
    print(f"Intel MKL: {'Available' if hardware_optimizer.hardware_info['has_intel_mkl'] else 'Not Available'}")
    print(f"AVX2 Support: {'Yes' if hardware_optimizer.hardware_info['has_avx2'] else 'No'}")
    
    # Display optimization strategy
    print(f"\n⚡ OPTIMIZATION STRATEGY: {hardware_optimizer.optimization_config['strategy'].upper()}")
    
    # Display deployment recommendations
    recommendations = hardware_optimizer.get_deployment_recommendations()
    print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    print(f"Type: {recommendations['deployment_type']}")
    print(f"Scaling: {recommendations['scaling_strategy']}")
    
    print(f"\n📊 PERFORMANCE EXPECTATIONS:")
    for key, value in recommendations['performance_expectations'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n💡 OPTIMIZATION TIPS:")
    for tip in recommendations['optimization_tips']:
        print(f"  • {tip}")
    
    # Display resource allocation
    print(f"\n🔧 RESOURCE ALLOCATION:")
    allocation = recommendations['resource_allocation']
    print(f"  CPU Cores: {allocation['cpu_cores']}")
    print(f"  Memory: {allocation['memory_gb']} GB")
    if allocation['gpu_memory_gb'] > 0:
        print(f"  GPU Memory: {allocation['gpu_memory_gb']} GB")
    
    # Display optimal worker counts
    print(f"\n👥 OPTIMAL WORKER COUNTS:")
    print(f"  I/O Bound Tasks: {hardware_optimizer.get_optimal_worker_count('io_bound')} workers")
    print(f"  CPU Bound Tasks: {hardware_optimizer.get_optimal_worker_count('cpu_bound')} workers")
    print(f"  ML Inference: {hardware_optimizer.get_optimal_worker_count('ml_inference')} workers")
    
    # Display memory limits
    print(f"\n💾 MEMORY LIMITS:")
    memory_limits = hardware_optimizer.get_memory_limits()
    for component, limit in memory_limits.items():
        print(f"  {component.replace('_', ' ').title()}: {limit:.0f} MB")
    
    # Display model configuration
    print(f"\n🧠 MODEL CONFIGURATION:")
    model_config = hardware_optimizer.optimization_config['model_config']
    print(f"  Device: {model_config.get('device', 'cpu').upper()}")
    print(f"  Batch Size: {model_config.get('batch_size', 8)}")
    print(f"  Max Length: {model_config.get('max_length', 256)}")
    print(f"  Use FP16: {'Yes' if model_config.get('use_fp16', False) else 'No'}")
    print(f"  Model Parallel: {'Yes' if model_config.get('model_parallel', False) else 'No'}")
    
    # Show lightweight model recommendation
    lightweight = hardware_optimizer.should_use_lightweight_models()
    print(f"\n🏃 USE LIGHTWEIGHT MODELS: {'Yes' if lightweight else 'No'}")
    if lightweight:
        print("  Recommendation: Use DistilBERT or similar lightweight models for better performance")
    else:
        print("  Recommendation: Can use full-size models like RoBERTa for maximum accuracy")

except Exception as e:
    print(f"Error: {e}")
    print("Make sure you're running this from the trading_bot directory")