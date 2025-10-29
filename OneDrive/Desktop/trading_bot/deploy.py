#!/usr/bin/env python3
"""
Smart deployment script that automatically detects hardware and deploys optimal configuration
"""
import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

def detect_gpu():
    """Detect if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def detect_docker():
    """Detect if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def detect_docker_compose():
    """Detect Docker Compose version"""
    try:
        result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'compose'
        
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'docker-compose'
    except FileNotFoundError:
        pass
    return None

def get_system_info():
    """Get system information for optimal configuration"""
    import psutil
    
    return {
        'platform': platform.system(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'has_gpu': detect_gpu(),
        'has_docker': detect_docker(),
        'docker_compose': detect_docker_compose()
    }

def create_env_file(deployment_type, system_info):
    """Create optimized .env file based on deployment type"""
    cpu_cores = system_info['cpu_count']
    memory_gb = system_info['memory_gb']
    
    if deployment_type == 'gpu':
        env_content = f"""
# GPU-Optimized Configuration
DEPLOYMENT_TYPE=gpu_accelerated
CPU_CORES={cpu_cores}
CPU_LIMIT={cpu_cores}.0
MEMORY_LIMIT={min(16, int(memory_gb * 0.8))}G
REDIS_MEMORY=1gb
DB_MEMORY=4G
GPU_COUNT=1

# Model Configuration
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
BATCH_SIZE=32
USE_FP16=true
DEVICE=cuda
"""
    else:
        env_content = f"""
# CPU-Optimized Configuration
DEPLOYMENT_TYPE=cpu_optimized
CPU_CORES={max(2, cpu_cores // 2)}
CPU_LIMIT={cpu_cores}.0
MEMORY_LIMIT={min(8, int(memory_gb * 0.6))}G
REDIS_MEMORY=512mb
DB_MEMORY=2G

# Model Configuration
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
BATCH_SIZE=8
USE_FP16=false
DEVICE=cpu
"""
    
    with open('.env.deploy', 'w') as f:
        f.write(env_content.strip())
    
    print(f"✅ Created optimized .env.deploy file for {deployment_type} deployment")

def deploy_application(deployment_type, system_info):
    """Deploy the application with optimal configuration"""
    compose_cmd = system_info['docker_compose']
    
    if not compose_cmd:
        print("❌ Docker Compose not found. Please install Docker Compose.")
        return False
    
    # Choose appropriate compose file
    if deployment_type == 'gpu':
        compose_file = 'docker-compose.gpu.yml'
    else:
        compose_file = 'docker-compose.cpu.yml'
    
    # Build command
    if compose_cmd == 'compose':
        cmd = ['docker', 'compose', '-f', compose_file, '--env-file', '.env.deploy']
    else:
        cmd = ['docker-compose', '-f', compose_file, '--env-file', '.env.deploy']
    
    # Stop any existing deployment
    print("🛑 Stopping existing deployment...")
    subprocess.run(cmd + ['down'], capture_output=True)
    
    # Build and start
    print("🔨 Building containers...")
    result = subprocess.run(cmd + ['build'], capture_output=False)
    
    if result.returncode != 0:
        print("❌ Build failed")
        return False
    
    print("🚀 Starting services...")
    result = subprocess.run(cmd + ['up', '-d'], capture_output=False)
    
    if result.returncode != 0:
        print("❌ Deployment failed")
        return False
    
    print("✅ Deployment successful!")
    return True

def deploy_local(deployment_type):
    """Deploy locally without Docker"""
    print(f"🏠 Deploying locally with {deployment_type} optimization...")
    
    # Install appropriate requirements
    if deployment_type == 'gpu':
        reqs_file = 'requirements-gpu.txt'
    else:
        reqs_file = 'requirements-cpu.txt'
    
    if os.path.exists(reqs_file):
        print(f"📦 Installing {deployment_type} dependencies...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', reqs_file])
        
        if result.returncode != 0:
            print(f"⚠️  Warning: Failed to install {deployment_type} dependencies")
    
    # Set environment variables
    os.environ['DEPLOYMENT_TYPE'] = f"{deployment_type}_optimized"
    
    print("🚀 Starting application...")
    subprocess.run([sys.executable, 'launcher.py'])

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Smart Trading Bot Deployment')
    parser.add_argument('--mode', choices=['auto', 'cpu', 'gpu', 'local'], default='auto',
                       help='Deployment mode (auto-detect, force CPU, force GPU, or local)')
    parser.add_argument('--no-docker', action='store_true',
                       help='Deploy locally without Docker')
    
    args = parser.parse_args()
    
    print("🤖 Smart Trading Bot Deployment Script")
    print("=" * 50)
    
    # Get system information
    system_info = get_system_info()
    
    print(f"🖥️  System: {system_info['platform']}")
    print(f"⚡ CPU Cores: {system_info['cpu_count']}")
    print(f"💾 Memory: {system_info['memory_gb']:.1f} GB")
    print(f"🎮 GPU Available: {'Yes' if system_info['has_gpu'] else 'No'}")
    print(f"🐳 Docker Available: {'Yes' if system_info['has_docker'] else 'No'}")
    print()
    
    # Determine deployment type
    if args.mode == 'auto':
        if system_info['has_gpu'] and system_info['memory_gb'] >= 8:
            deployment_type = 'gpu'
            print("🎯 Auto-detected: GPU-accelerated deployment")
        else:
            deployment_type = 'cpu'
            print("🎯 Auto-detected: CPU-optimized deployment")
    else:
        deployment_type = args.mode
        print(f"🎯 Manual selection: {deployment_type.upper()} deployment")
    
    # Choose deployment method
    if args.no_docker or not system_info['has_docker']:
        if not system_info['has_docker']:
            print("⚠️  Docker not available, falling back to local deployment")
        deploy_local(deployment_type)
    else:
        # Create optimized environment file
        create_env_file(deployment_type, system_info)
        
        # Deploy with Docker
        success = deploy_application(deployment_type, system_info)
        
        if success:
            print("\n🎉 Deployment Complete!")
            print("📊 Dashboard: http://localhost:8000")
            print("📈 Grafana: http://localhost:3000 (admin/admin)")
            print("🔍 Prometheus: http://localhost:9090")
            
            if deployment_type == 'gpu':
                print("🎮 NVIDIA GPU monitoring: http://localhost:9400/metrics")
            
            print("\n📋 Deployment Summary:")
            print(f"   • Type: {deployment_type.upper()}-optimized")
            print(f"   • CPU Cores: {system_info['cpu_count']}")
            print(f"   • Memory: {system_info['memory_gb']:.1f} GB")
            if deployment_type == 'gpu':
                print("   • GPU: NVIDIA CUDA Enabled")
            
            print("\n🛠️  Management Commands:")
            compose_cmd = 'docker compose' if system_info['docker_compose'] == 'compose' else 'docker-compose'
            compose_file = f"docker-compose.{deployment_type}.yml"
            print(f"   • View logs: {compose_cmd} -f {compose_file} logs -f")
            print(f"   • Stop: {compose_cmd} -f {compose_file} down")
            print(f"   • Restart: {compose_cmd} -f {compose_file} restart")
        else:
            print("❌ Deployment failed. Check the logs above for details.")

if __name__ == "__main__":
    main()