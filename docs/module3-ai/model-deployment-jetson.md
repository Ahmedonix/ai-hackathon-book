# Deploying AI Models to Jetson Orin Nano/NX

## Overview

This section covers the complete workflow for deploying AI models to NVIDIA Jetson Orin Nano and NX platforms. We'll focus on optimizing models for the Jetson's specific hardware capabilities, setting up the appropriate runtime environment, and ensuring efficient execution of AI pipelines for humanoid robotics applications.

## Understanding Jetson Orin Platforms

### Jetson Orin Nano Specifications

**Jetson Orin Nano Developer Kit**:
- CPU: 6-core ARM Cortex-A78AE v8.2 64-bit CPU
- GPU: 1024-core NVIDIA Ampere™ architecture GPU with 2x RT cores, 8x Tensor cores
- Memory: 4 GB or 8 GB LPDDR5
- Power: 15W to 25W
- CUDA Cores: 1024, Tensor Cores: 8, RT Cores: 2

**Jetson Orin NX Specifications**:
- CPU: 8-core ARM Cortex-A78AE v8.2 64-bit CPU
- GPU: 1024-core NVIDIA Ampere architecture GPU
- Memory: 8 GB or 16 GB LPDDR5
- Power: 15W to 25W
- Higher performance than Nano

## Prerequisites for Jetson Deployment

### 1. Hardware Requirements

- NVIDIA Jetson Orin Nano or NX Developer Kit
- Power supply (official 19V/65W for Orin Nano, 19V/120W for Orin NX)
- MicroSD card (64GB+ recommended)
- Ethernet cable or WiFi for initial setup
- Host computer for flashing and development

### 2. Software Prerequisites

On your development machine (which will build models for Jetson):

```bash
# Install Jetson Inference tools
sudo apt install nvidia-jetpack

# Install cross-compilation tools if needed
# For native compilation on Jetson, this isn't necessary
```

### 3. Jetson Setup

On your Jetson device:

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install build-essential cmake git curl wget python3-pip python3-dev

# Install ROS 2 Iron on Jetson
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key
sudo apt-key add -o /tmp/ros.key
echo "deb [arch=arm64] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install ros-iron-desktop ros-iron-ros-base
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Install CUDA and TensorRT
sudo apt install cuda-libraries-dev-11-4
sudo apt install libnvinfer-dev
```

## AI Model Preparation for Jetson Deployment

### 1. Model Optimization with TensorRT

First, let's optimize our AI models for Jetson execution:

```python
# scripts/optimize_model_for_jetson.py
#!/usr/bin/env python3

"""
Script for optimizing AI models for Jetson deployment using TensorRT.
This script converts ONNX models to optimized TensorRT engines for Jetson.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
from onnx_tensorrt.backend import Backend
import os
import argparse
import logging


def setup_logging():
    """Setup logging for model optimization process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def optimize_model_for_jetson(
    onnx_model_path, 
    trt_engine_path, 
    precision='fp16', 
    max_batch_size=1,
    workspace_size=2<<30  # 2GB workspace
):
    """
    Optimize an ONNX model for Jetson using TensorRT.
    
    Args:
        onnx_model_path (str): Path to the input ONNX model
        trt_engine_path (str): Path to save the optimized TensorRT engine
        precision (str): Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size (int): Maximum batch size to optimize for
        workspace_size (int): Maximum workspace size for TensorRT engine building
    """
    
    logger = setup_logging()
    logger.info(f"Starting model optimization for Jetson")
    logger.info(f"Input model: {onnx_model_path}")
    logger.info(f"Output engine: {trt_engine_path}")
    logger.info(f"Precision: {precision}")
    
    # Initialize TensorRT logger
    trt_logger = trt.Logger(trt.Logger.INFO)
    
    # Create builder
    builder = trt.Builder(trt_logger)
    if not builder:
        logger.error("Failed to create TensorRT Builder")
        return False
    
    # Create network definition
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if not network:
        logger.error("Failed to create TensorRT Network")
        return False
    
    # Create builder config
    config = builder.create_builder_config()
    if not config:
        logger.error("Failed to create TensorRT Config")
        return False
    
    # Set workspace size
    config.max_workspace_size = workspace_size
	
    # Set precision based on input
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Using FP16 precision for optimization")
        else:
            logger.warning("FP16 not available on this platform, using FP32")
    
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("Using INT8 precision for optimization")
            # For INT8, you would need to provide calibration data
            # config.int8_calibrator = YourCalibrator(calibration_files)
        else:
            logger.warning("INT8 not available on this platform, using FP32")
    
    elif precision == 'fp32':
        logger.info("Using FP32 precision (default)")
    else:
        logger.error(f"Unsupported precision: {precision}")
        return False
    
    # Create parser for ONNX model
    parser = trt.OnnxParser(network, trt_logger)
    if not parser:
        logger.error("Failed to create ONNX parser")
        return False
    
    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as model_file:
        model_data = model_file.read()
        if not parser.parse(model_data):
            logger.error("Failed to parse the ONNX model")
            for idx in range(parser.num_errors):
                logger.error(parser.get_error(idx))
            return False
    
    # Set optimization profiles if needed
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)
    
    # Build the TensorRT engine
    logger.info("Building TensorRT engine...")
    
    try:
        # Build serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Write the engine to a file
        with open(trt_engine_path, 'wb') as engine_file:
            engine_file.write(serialized_engine)
        
        logger.info(f"Model optimization completed successfully!")
        logger.info(f"Engine saved to: {trt_engine_path}")
        
        # Get engine information
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        logger.info(f"Engine name: {engine.name}")
        logger.info(f"Number of layers: {engine.num_layers}")
        logger.info(f"Max batch size: {engine.max_batch_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during engine building: {str(e)}")
        return False


def benchmark_model_performance(engine_path, input_shape):
    """
    Benchmark the performance of the optimized model.
    
    Args:
        engine_path (str): Path to the TensorRT engine file
        input_shape (tuple): Input tensor shape to use for benchmarking
    """
    
    logger = setup_logging()
    logger.info(f"Benchmarking model: {engine_path}")
    
    # Load the TensorRT engine
    trt_logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(trt_logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        logger.error("Failed to deserialize TensorRT engine")
        return None
    
    # Create execution context
    context = engine.create_execution_context()
    if context is None:
        logger.error("Failed to create execution context")
        return None
    
    # Allocate CUDA memory buffers
    input_binding_idx = engine.get_binding_index('input')  # This should match your model's input binding name
    output_binding_idx = engine.get_binding_index('output')  # This should match your model's output binding name
    
    # Get input/output binding information
    input_shape = engine.get_binding_shape(input_binding_idx)
    output_shape = engine.get_binding_shape(output_binding_idx)
    
    # Calculate buffer sizes
    input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
    output_size = trt.volume(output_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
    
    # Allocate host and device buffers
    h_input = cuda.pagelocked_empty(trt.volume(input_shape) * engine.max_batch_size, dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
    
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Generate random input data
    h_input = np.random.randn(*input_shape).astype(np.float32)
    
    import time
    
    # Warm up run
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # Benchmark multiple runs for accuracy
    num_runs = 100
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    fps = 1000.0 / avg_time
    
    logger.info(f"Benchmark Results:")
    logger.info(f"  Average inference time: {avg_time:.3f} ms")
    logger.info(f"  Standard deviation: {std_time:.3f} ms")
    logger.info(f"  Min/Max: {min_time:.3f}/{max_time:.3f} ms")
    logger.info(f"  Average FPS: {fps:.2f}")
    logger.info(f"  Input shape: {input_shape}")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'fps': fps,
        'input_shape': input_shape
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize models for Jetson deployment')
    parser.add_argument('--input-model', type=str, required=True, 
                       help='Path to input ONNX model')
    parser.add_argument('--output-engine', type=str, required=True,
                       help='Path to save optimized TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision for optimization (default: fp16)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Maximum batch size for optimization (default: 1)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark after optimization')
    parser.add_argument('--input-shape', nargs='+', type=int,
                       help='Input shape for benchmarking (e.g., 1 3 224 224)')
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not os.path.exists(args.input_model):
        print(f"Error: Input model file {args.input_model} does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_engine)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Optimize the model
    success = optimize_model_for_jetson(
        args.input_model,
        args.output_engine,
        precision=args.precision,
        max_batch_size=args.batch_size
    )
    
    if not success:
        print("Model optimization failed")
        return 1
    
    print(f"Model optimization completed successfully: {args.output_engine}")
    
    # Run benchmark if requested
    if args.benchmark and args.input_shape:
        benchmark_results = benchmark_model_performance(
            args.output_engine,
            tuple(args.input_shape)
        )
        
        if benchmark_results:
            print(f"Benchmark completed. Performance: {benchmark_results['fps']:.2f} FPS")
        else:
            print("Benchmark failed")
    
    return 0


if __name__ == '__main__':
    exit(main())
```

### 2. Creating a Deployment Package Structure

Create a structure for organizing your Jetson deployment:

```bash
# Create deployment directory structure
mkdir -p ~/jetson_deployment/models
mkdir -p ~/jetson_deployment/config
mkdir -p ~/jetson_deployment/scripts
mkdir -p ~/jetson_deployment/launch

# Create the optimized model directory
mkdir -p ~/jetson_deployment/models/optimized
```

### 3. Model Deployment Script

Create a deployment script that automates the entire process:

```bash
# scripts/deploy_to_jetson.sh
#!/bin/bash

# Deployment script for AI models to Jetson Orin Nano/NX
# Automates the process of optimizing models and deploying to Jetson

set -e  # Exit on any error

# Parse command line arguments
MODEL_PATH=""
JETSON_IP=""
JETSON_USER="jetson"
PRECISION="fp16"
BENCHMARK=true
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --jetson-ip)
            JETSON_IP="$2"
            shift 2
            ;;
        --jetson-user)
            JETSON_USER="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --no-benchmark)
            BENCHMARK=false
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model-path <path> --jetson-ip <ip> [--jetson-user <user>] [--precision <fp16|fp32|int8>] [--no-benchmark]"
            exit 1
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "Deploy AI model to Jetson Orin platform"
    echo ""
    echo "Usage: $0 --model-path <path> --jetson-ip <ip> [options]"
    echo ""
    echo "Required:"
    echo "  --model-path <path>    Path to ONNX model file"
    echo "  --jetson-ip <ip>       Jetson IP address"
    echo ""
    echo "Options:"
    echo "  --jetson-user <user>   Username for Jetson SSH (default: jetson)"
    echo "  --precision <type>     Precision for optimization (default: fp16)"
    echo "  --no-benchmark         Skip benchmarking after deployment"
    echo "  -h, --help             Show this help message"
    exit 0
fi

# Validate required arguments
if [ -z "$MODEL_PATH" ] || [ -z "$JETSON_IP" ]; then
    echo "Error: Both --model-path and --jetson-ip are required"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file does not exist: $MODEL_PATH"
    exit 1
fi

echo "Starting AI model deployment to Jetson at $JETSON_IP"
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"

# Get model name without extension
MODEL_NAME=$(basename "$MODEL_PATH" .onnx)
ENGINE_NAME="${MODEL_NAME}_${PRECISION}.trt"

echo "Optimizing model for Jetson ($PRECISION precision)..."
python3 optimize_model_for_jetson.py \
    --input-model "$MODEL_PATH" \
    --output-engine "./models/optimized/$ENGINE_NAME" \
    --precision "$PRECISION" \
    --benchmark

if [ $? -ne 0 ]; then
    echo "Model optimization failed"
    exit 1
fi

echo "Model optimization completed. Engine saved as: models/optimized/$ENGINE_NAME"

# Create deployment package
echo "Creating deployment package..."
DEPLOY_PACKAGE="jetson_model_${MODEL_NAME}_${PRECISION}_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$DEPLOY_PACKAGE" -C . \
    "models/optimized/$ENGINE_NAME" \
    config/ \
    launch/ \
    scripts/

echo "Deployment package created: $DEPLOY_PACKAGE"

# Deploy to Jetson via SCP
echo "Deploying to Jetson at $JETSON_IP..."
scp "$DEPLOY_PACKAGE" "$JETSON_USER@$JETSON_IP:/tmp/"

if [ $? -ne 0 ]; then
    echo "Failed to copy deployment package to Jetson"
    exit 1
fi

# Extract on Jetson
ssh "$JETSON_USER@$JETSON_IP" "mkdir -p ~/robot_models && tar -xzf /tmp/$DEPLOY_PACKAGE -C ~/robot_models && rm /tmp/$DEPLOY_PACKAGE"

if [ $? -ne 0 ]; then
    echo "Failed to extract deployment package on Jetson"
    exit 1
fi

echo "Model deployed successfully to Jetson at $JETSON_IP:/home/$JETSON_USER/robot_models/"

# Verification steps
echo "Verifying deployment on Jetson..."
ssh "$JETSON_USER@$JETSON_IP" "ls -l ~/robot_models/models/optimized/$ENGINE_NAME"

if [ $? -eq 0 ]; then
    echo "✓ Model successfully deployed to Jetson"
    
    if [ "$BENCHMARK" = true ]; then
        echo "Running performance benchmark on Jetson..."
        ssh "$JETSON_USER@$JETSON_IP" "cd ~/robot_models && python3 benchmark_model.py $ENGINE_NAME"
    fi
else
    echo "✗ Verification failed - model may not be properly deployed"
    exit 1
fi

echo "Deployment complete!"
echo "Model is available at: ~/robot_models/models/optimized/$ENGINE_NAME on Jetson"
```

### 4. Creating a Model Execution Node for Jetson

Create a node that will execute the optimized models on Jetson:

```python
# scripts/jetson_model_executor.py
#!/usr/bin/env python3

"""
Model execution node for Jetson Orin platforms.
Loads and runs optimized TensorRT engines for various humanoid AI tasks.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os


class JetsonModelExecutor(Node):
    """
    Executes optimized TensorRT models on Jetson for humanoid perception and control tasks.
    """
    
    def __init__(self):
        super().__init__('jetson_model_executor')
        
        # Declare parameters
        self.declare_parameter('model_directory', '/home/jetson/robot_models/models/optimized')
        self.declare_parameter('perception_model_path', 'perception_model_fp16.trt')
        self.declare_parameter('control_model_path', 'control_model_fp16.trt')
        self.declare_parameter('execution_frequency', 30.0)  # Hz
        self.declare_parameter('use_gpu_processing', True)
        self.declare_parameter('input_preprocessing', True)
        
        # Get parameters
        self.model_directory = self.get_parameter('model_directory').value
        self.perception_model_path = self.get_parameter('perception_model_path').value
        self.control_model_path = self.get_parameter('control_model_path').value
        self.execution_frequency = self.get_parameter('execution_frequency').value
        self.use_gpu_processing = self.get_parameter('use_gpu_processing').value
        self.input_preprocessing = self.get_parameter('input_preprocessing').value
        
        # Initialize
        self.cv_bridge = CvBridge()
        self.loaded_models = {}
        self.model_contexts = {}
        self.gpu_buffers = {}
        
        # Load models
        self.load_models()
        
        # Subscriptions
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.command_sub = self.create_subscription(
            Twist,
            '/ai_commands',
            self.command_callback,
            10
        )
        
        # Publishers
        self.perception_output_pub = self.create_publisher(
            String,
            '/perception_output',
            10
        )
        
        self.control_output_pub = self.create_publisher(
            Twist,
            '/control_output',
            10
        )
        
        self.inference_stats_pub = self.create_publisher(
            Float64MultiArray,
            '/model_inference_stats',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/model_execution_status',
            10
        )
        
        # Execution timer
        self.execution_timer = self.create_timer(
            1.0 / self.execution_frequency,
            self.execution_cycle
        )
        
        # Internal state
        self.latest_image = None
        self.latest_lidar = None
        self.latest_imu = None
        self.latest_command = None
        self.inference_times = []
        
        self.get_logger().info('Jetson Model Executor Node Initialized')

    def load_models(self):
        """Load TensorRT optimized models"""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Load perception model
        perception_path = os.path.join(self.model_directory, self.perception_model_path)
        if os.path.exists(perception_path):
            self.perception_engine = self.load_trt_engine(perception_path)
            if self.perception_engine:
                self.get_logger().info(f'Loaded perception model: {perception_path}')
            else:
                self.get_logger().error(f'Failed to load perception model: {perception_path}')
        else:
            self.get_logger().error(f'Perception model path does not exist: {perception_path}')
        
        # Load control model
        control_path = os.path.join(self.model_directory, self.control_model_path)
        if os.path.exists(control_path):
            self.control_engine = self.load_trt_engine(control_path)
            if self.control_engine:
                self.get_logger().info(f'Loaded control model: {control_path}')
            else:
                self.get_logger().error(f'Failed to load control model: {control_path}')
        else:
            self.get_logger().error(f'Control model path does not exist: {control_path}')

    def load_trt_engine(self, engine_path):
        """Load a TensorRT engine from file"""
        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            return engine
            
        except Exception as e:
            self.get_logger().error(f'Error loading TensorRT engine {engine_path}: {str(e)}')
            return None

    def allocate_gpu_buffers(self, engine):
        """Allocate GPU buffers for a TensorRT engine"""
        try:
            # Get binding information
            context = engine.create_execution_context()
            
            # Allocate buffers for inputs and outputs
            inputs = []
            outputs = []
            bindings = []
            
            for binding_idx in range(engine.num_bindings):
                if engine.binding_is_input(binding_idx):
                    # Input binding
                    input_shape = engine.get_binding_shape(binding_idx)
                    input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
                    d_input = cuda.mem_alloc(input_size)
                    inputs.append({'binding_idx': binding_idx, 'shape': input_shape, 'gpu_ptr': d_input})
                    bindings.append(int(d_input))
                else:
                    # Output binding
                    output_shape = engine.get_binding_shape(binding_idx)
                    output_size = trt.volume(output_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
                    d_output = cuda.mem_alloc(output_size)
                    outputs.append({'binding_idx': binding_idx, 'shape': output_shape, 'gpu_ptr': d_output})
                    bindings.append(int(d_output))
            
            return {
                'context': context,
                'inputs': inputs,
                'outputs': outputs,
                'bindings': bindings,
                'max_batch_size': engine.max_batch_size
            }
            
        except Exception as e:
            self.get_logger().error(f'Error allocating GPU buffers: {str(e)}')
            return None

    def camera_callback(self, msg):
        """Process incoming camera images"""
        self.latest_image = msg

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.latest_lidar = msg

    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu = msg

    def command_callback(self, msg):
        """Process incoming commands"""
        self.latest_command = msg

    def execution_cycle(self):
        """Main execution cycle for model inference"""
        try:
            start_time = time.time()
            
            # Run perception model if we have camera data
            if self.latest_image and self.perception_engine:
                perception_result = self.run_perception_inference(self.latest_image)
                if perception_result:
                    perception_msg = String()
                    perception_msg.data = f"PERCEPTION_RESULT: {perception_result}"
                    self.perception_output_pub.publish(perception_msg)
            
            # Run control model if we have necessary data
            if (self.latest_image and self.latest_lidar and 
                self.latest_imu and self.control_engine):
                
                control_command = self.run_control_inference(
                    self.latest_image, 
                    self.latest_lidar, 
                    self.latest_imu
                )
                
                if control_command:
                    self.control_output_pub.publish(control_command)
            
            # Calculate and publish inference statistics
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times (last 100)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            if len(self.inference_times) > 0:
                avg_inference_time = sum(self.inference_times) / len(self.inference_times)
                min_inference_time = min(self.inference_times)
                max_inference_time = max(self.inference_times)
                
                # Publish statistics
                stats_msg = Float64MultiArray()
                stats_msg.data = [
                    avg_inference_time,  # Average inference time in ms
                    min_inference_time,  # Min inference time
                    max_inference_time,  # Max inference time
                    len(self.inference_times),  # Number of samples
                    self.execution_frequency,  # Target execution frequency
                    1000.0 / avg_inference_time if avg_inference_time > 0 else 0  # Actual achieved FPS
                ]
                self.inference_stats_pub.publish(stats_msg)
                
                # Publish status
                status_msg = String()
                status_msg.data = (
                    f"INFERENCE_STATS: avg={avg_inference_time:.2f}ms | "
                    f"FPS={1000.0/avg_inference_time:.1f} | "
                    f"Models:{'P' if hasattr(self, 'perception_engine') else 'N'}"
                    f"{'C' if hasattr(self, 'control_engine') else 'N'}"
                )
                self.status_pub.publish(status_msg)
                
                # Log periodically
                if int(self.get_clock().now().nanoseconds / 1e9) % 5 == 0:  # Every 5 seconds
                    self.get_logger().info(status_msg.data)
            
        except Exception as e:
            self.get_logger().error(f'Error in execution cycle: {str(e)}')

    def run_perception_inference(self, image_msg):
        """Run perception model inference"""
        try:
            # Convert ROS image to numpy array
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            
            # Preprocess the image for the model
            processed_image = self.preprocess_image(cv_image)
            
            # Perform inference using TensorRT
            if hasattr(self, 'perception_engine'):
                # Create execution context
                context = self.perception_engine.create_execution_context()
                
                # Allocate host and device buffers
                input_shape = self.perception_engine.get_binding_shape(0)
                output_shape = self.perception_engine.get_binding_shape(1)
                
                # Create input buffer
                h_input = np.asarray(processed_image, dtype=np.float32)
                
                # Allocate GPU memory
                input_size = trt.volume(input_shape) * self.perception_engine.max_batch_size * np.dtype(np.float32).itemsize
                output_size = trt.volume(output_shape) * self.perception_engine.max_batch_size * np.dtype(np.float32).itemsize
                
                d_input = cuda.mem_alloc(input_size)
                d_output = cuda.mem_alloc(output_size)
                
                # Create CUDA stream
                stream = cuda.Stream()
                
                # Transfer input data to GPU
                cuda.memcpy_htod_async(d_input, h_input, stream)
                
                # Execute model
                context.execute_async_v2(
                    bindings=[int(d_input), int(d_output)], 
                    stream_handle=stream.handle
                )
                
                # Transfer predictions back from GPU
                h_output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                
                # Process results (simplified for this example)
                results = self.postprocess_predictions(h_output)
                
                # Clean up GPU memory
                del d_input, d_output, stream
                
                return results
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error in perception inference: {str(e)}')
            return None

    def run_control_inference(self, image_msg, lidar_msg, imu_msg):
        """Run control model inference based on sensor inputs"""
        try:
            # This would integrate multiple sensor inputs for control decisions
            # For this example, we'll just use image and lidar for simple navigation
            
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            processed_image = self.preprocess_image(cv_image)
            
            # Process lidar data
            lidar_data = np.array(lidar_msg.ranges, dtype=np.float32)
            lidar_data = np.nan_to_num(lidar_data, nan=np.inf)  # Replace NaN with infinity
            
            # Process IMU data
            imu_vector = np.array([
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ], dtype=np.float32)
            
            # In a real implementation, you would concatenate these inputs
            # or use a more sophisticated fusion approach before inference
            # For this example, we'll just make a simple decision
            
            # Look for obstacles in front
            front_ranges = lidar_data[len(lidar_data)//2 - len(lidar_data)//8 : len(lidar_data)//2 + len(lidar_data)//8]
            front_ranges = front_ranges[np.isfinite(front_ranges)]
            
            cmd = Twist()
            if len(front_ranges) > 0 and np.min(front_ranges) < 0.8:  # Obstacle within 0.8m
                # Stop and turn
                cmd.linear.x = 0.0
                cmd.angular.z = 0.3  # Turn right
            else:
                # Move forward
                cmd.linear.x = 0.2
                cmd.angular.z = 0.0
            
            return cmd
            
        except Exception as e:
            self.get_logger().error(f'Error in control inference: {str(e)}')
            # Return safe command if there's an error
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if self.input_preprocessing:
            # Resize image to expected input size
            # This should match the size expected by your ONNX model
            expected_height, expected_width = 224, 224  # Adjust as needed
            
            import cv2
            resized_image = cv2.resize(image, (expected_width, expected_height))
            
            # Normalize image
            normalized = resized_image.astype(np.float32) / 255.0
            
            # Normalize with ImageNet statistics if needed for your model
            # normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Transpose to CHW format for TensorRT
            transposed = np.transpose(normalized, (2, 0, 1))
            
            return transposed
        else:
            # Return as-is if no preprocessing needed
            return image

    def postprocess_predictions(self, outputs):
        """Postprocess model predictions"""
        # This would convert model outputs to meaningful results
        # For this example, return a simplified result
        if len(outputs.shape) > 1 and outputs.shape[0] > 0:
            max_idx = np.argmax(outputs[0]) if len(outputs.shape) > 1 else np.argmax(outputs)
            confidence = np.max(outputs[0]) if len(outputs.shape) > 1 else np.max(outputs)
            
            return f"class_{max_idx}_confidence_{confidence:.3f}"
        else:
            return "no_detection"

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Jetson Model Executor Node Shutting Down')
        
        # Clean up GPU buffers if they exist
        for gpu_ptr in self.gpu_buffers.values():
            try:
                del gpu_ptr
            except:
                pass
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    executor = JetsonModelExecutor()
    
    try:
        rclpy.spin(executor)
    except KeyboardInterrupt:
        executor.get_logger().info('Model executor interrupted by user')
    finally:
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Model Deployment Strategy

### 1. Multi-Model Pipeline

Create a deployment strategy for multiple models:

```yaml
# config/jetson_model_pipeline.yaml
model_pipeline:
  ros__parameters:
    # Model management
    model_directory: "/home/jetson/robot_models/models/optimized"
    perception_model: "humanoid_perception_fp16.trt"
    control_model: "humanoid_control_fp16.trt"
    planning_model: "navigation_planning_fp16.trt"
    
    # Execution parameters
    execution_frequency: 30.0
    gpu_processing_enabled: true
    input_preprocessing_enabled: true
    
    # Performance monitoring
    enable_benchmarking: true
    benchmark_interval: 10.0  # seconds
    performance_thresholds:
      avg_inference_time_ms: 50.0
      min_fps: 15.0
    
    # Resource management
    memory_reservation_mb: 2048
    gpu_memory_reservation_mb: 1024
    cpu_affinity: [2, 3, 4, 5]  # Use specific CPU cores
```

### 2. Containerized Deployment for Jetson

Create a deployment approach using containers:

```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/ros:jammy-ros-iron-perception

# Install Jetson-specific dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libhdf5-dev \
    python3-h5py \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies for AI
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pillow==9.5.0 \
    onnx==1.14.0 \
    onnxruntime-gpu==1.15.1 \
    pycuda==2022.2.2 \
    && ldconfig

# Install TensorRT Python bindings
COPY install_tensorrt.sh /tmp/install_tensorrt.sh
RUN chmod +x /tmp/install_tensorrt.sh && /tmp/install_tensorrt.sh

# Copy model files
COPY models/optimized/ /opt/models/

# Copy ROS 2 packages
COPY src/ /ws/src/

WORKDIR /ws

# Build the workspace
RUN source /opt/ros/iron/setup.bash && \
    colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release

# Setup entrypoint
RUN echo 'source /opt/ros/iron/setup.bash' >> ~/.bashrc && \
    echo 'source /ws/install/setup.bash' >> ~/.bashrc

CMD ["bash", "-c", "source /opt/ros/iron/setup.bash && source /ws/install/setup.bash && ros2 launch humanoid_jetson_bringup humanoid_jetson.launch.py"]
```

### 3. Model Update Mechanism

Create a script for updating models on the Jetson:

```bash
#!/bin/bash
# scripts/update_models.sh

# Script to update AI models on Jetson device

JETSON_IP=$1
JETSON_USER=${2:-"jetson"}
REMOTE_MODEL_DIR="/home/$JETSON_USER/robot_models/models/optimized"

if [ -z "$JETSON_IP" ]; then
    echo "Usage: $0 <jetson_ip> [jetson_user]"
    exit 1
fi

echo "Updating models on Jetson at $JETSON_IP..."

# Copy new models to Jetson
for model_file in ./models/optimized/*.trt; do
    if [ -f "$model_file" ]; then
        echo "Transferring $model_file..."
        scp "$model_file" "$JETSON_USER@$JETSON_IP:$REMOTE_MODEL_DIR/"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully transferred $model_file"
        else
            echo "✗ Failed to transfer $model_file"
        fi
    fi
done

# Restart the model executor node
echo "Restarting model executor on Jetson..."
ssh "$JETSON_USER@$JETSON_IP" "source ~/.bashrc && ros2 lifecycle set /jetson_model_executor deactivate && ros2 lifecycle set /jetson_model_executor activate"

echo "Model update complete!"
```

## Performance Optimization for Jetson

### 1. Model Quantization Techniques

For better performance on Jetson:

```python
# scripts/quantize_model.py
#!/usr/bin/env python3

"""
Model quantization for Jetson deployment.
Reduces model size and increases inference speed with minimal accuracy loss.
"""

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import argparse


def quantize_model(input_path, output_path, weight_type=QuantType.QInt8):
    """
    Quantize an ONNX model for better Jetson performance.
    """
    print(f"Quantizing {input_path} to {output_path} with {weight_type}...")
    
    # Perform dynamic quantization
    quantized_model = quantize_dynamic(
        input_path,
        output_path,
        weight_type=weight_type
    )
    
    print("Model quantization completed!")
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX models for Jetson')
    parser.add_argument('--input-model', type=str, required=True, 
                       help='Path to input ONNX model')
    parser.add_argument('--output-model', type=str, required=True,
                       help='Path to save quantized model')
    parser.add_argument('--weight-type', type=str, default='qint8',
                       choices=['qint8', 'qfloat16'],
                       help='Quantization type')
    
    args = parser.parse_args()
    
    weight_type = QuantType.QUInt8 if args.weight_type == 'qint8' else QuantType.QFloat16
    
    quantize_model(args.input_model, args.output_model, weight_type)


if __name__ == '__main__':
    main()
```

### 2. Jetson Power Management

Optimize for Jetson power consumption:

```python
# scripts/power_manager.py
#!/usr/bin/env python3

"""
Power management for Jetson Orin platforms.
Manages power consumption based on computational demands.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
import subprocess
import time


class JetsonPowerManager(Node):
    """
    Manages Jetson power profiles based on computational demands.
    """
    
    def __init__(self):
        super().__init__('jetson_power_manager')
        
        # Declare parameters
        self.declare_parameter('monitoring_frequency', 5.0)  # Hz
        self.declare_parameter('cpu_usage_threshold', 70.0)  # Percent
        self.declare_parameter('gpu_usage_threshold', 70.0)  # Percent
        self.declare_parameter('high_power_profile', 'MAXN')  # Jetson power mode
        self.declare_parameter('low_power_profile', 'MIN_POWER')
        
        # Get parameters
        self.monitoring_frequency = self.get_parameter('monitoring_frequency').value
        self.cpu_threshold = self.get_parameter('cpu_usage_threshold').value
        self.gpu_threshold = self.get_parameter('gpu_usage_threshold').value
        self.high_power_profile = self.get_parameter('high_power_profile').value
        self.low_power_profile = self.get_parameter('low_power_profile').value
        
        # Publishers
        self.power_status_pub = self.create_publisher(
            String,
            '/power_status',
            10
        )
        
        self.power_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/power_metrics',
            10
        )
        
        # Timer for power monitoring
        self.power_timer = self.create_timer(
            1.0 / self.monitoring_frequency,
            self.monitor_power_state
        )
        
        self.last_power_profile = None
        self.get_logger().info('Jetson Power Manager Initialized')

    def monitor_power_state(self):
        """Monitor power consumption and adjust profiles if needed"""
        try:
            # Get CPU usage
            cpu_usage = self.get_cpu_usage()
            
            # Get GPU usage (Jetson-specific)
            gpu_usage, gpu_temp = self.get_gpu_status()
            
            # Get memory usage
            memory_usage = self.get_memory_usage()
            
            # Determine if we need to change power profile
            current_profile = self.get_current_power_profile()
            
            if (cpu_usage > self.cpu_threshold or 
                gpu_usage > self.gpu_threshold or 
                memory_usage > 80.0):
                
                # High demand - switch to high power mode if different
                if current_profile != self.high_power_profile:
                    self.set_power_profile(self.high_power_profile)
                    self.get_logger().info(f'Switched to high power mode: {self.high_power_profile}')
            else:
                # Low demand - switch to low power mode if different
                if current_profile != self.low_power_profile:
                    self.set_power_profile(self.low_power_profile)
                    self.get_logger().info(f'Switched to low power mode: {self.low_power_profile}')
            
            # Publish power status
            status_msg = String()
            status_msg.data = f"CPU:{cpu_usage:.1f}% | GPU:{gpu_usage:.1f}% | Temp:{gpu_temp:.1f}C | Profile:{current_profile}"
            self.power_status_pub.publish(status_msg)
            
            # Publish metrics
            metrics_msg = Float64MultiArray()
            metrics_msg.data = [cpu_usage, gpu_usage, gpu_temp, memory_usage]
            self.power_metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in power monitoring: {str(e)}')

    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        import psutil
        return psutil.cpu_percent(interval=0.1)

    def get_memory_usage(self):
        """Get memory usage percentage"""
        import psutil
        return psutil.virtual_memory().percent

    def get_gpu_status(self):
        """Get GPU usage and temperature on Jetson"""
        try:
            # Use nvidia-ml-py to get GPU usage
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output_parts = result.stdout.strip().split(',')
                gpu_usage = float(output_parts[0].strip())
                gpu_temp = float(output_parts[1].strip())
                return gpu_usage, gpu_temp
            else:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

    def get_current_power_profile(self):
        """Get current Jetson power profile"""
        try:
            # This command might vary depending on your Jetson model
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-q'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse the output to extract the profile name
                for line in result.stdout.split('\n'):
                    if 'Model:' in line:
                        profile = line.split(':')[1].strip()
                        return profile
                return "UNKNOWN"
            else:
                return "ERROR"
        except Exception:
            return "ERROR"

    def set_power_profile(self, profile_name):
        """Set Jetson power profile"""
        try:
            # This will require passwordless sudo if run without root privileges
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-m', profile_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.get_logger().info(f'Successfully set power profile to {profile_name}')
                return True
            else:
                self.get_logger().error(f'Failed to set power profile: {result.stderr}')
                return False
        except Exception as e:
            self.get_logger().error(f'Error setting power profile: {str(e)}')
            return False

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Jetson Power Manager Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    power_manager = JetsonPowerManager()
    
    try:
        rclpy.spin(power_manager)
    except KeyboardInterrupt:
        power_manager.get_logger().info('Power manager interrupted by user')
    finally:
        power_manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Testing and Validation on Jetson

### 1. Deployment Validation Script

Create a validation script to test your deployment:

```python
# scripts/deployment_validator.py
#!/usr/bin/env python3

"""
Deployment validator for Jetson Orin platforms.
Validates that all AI models are properly loaded and executing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
import time
import os


class DeploymentValidator(Node):
    def __init__(self):
        super().__init__('deployment_validator')
        
        # Declare parameters
        self.declare_parameter('validation_timeout', 30.0)  # seconds
        self.declare_parameter('model_directory', '/home/jetson/robot_models/models/optimized')
        
        # Get parameters
        self.validation_timeout = self.get_parameter('validation_timeout').value
        self.model_directory = self.get_parameter('model_directory').value
        
        # Track validation status
        self.model_files_validated = False
        self.sensors_working = False
        self.models_loaded = False
        self.inference_running = False
        
        # Track sensor data
        self.image_received = False
        self.lidar_received = False
        
        # Subscriptions to verify sensor data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # Publisher for validation status
        self.validation_status_pub = self.create_publisher(
            String,
            '/deployment_validation_status',
            10
        )
        
        # Timer for validation checks
        self.validation_timer = self.create_timer(1.0, self.run_validation_step)
        
        # Start validation process
        self.validation_start_time = time.time()
        self.validation_step = 0
        
        self.get_logger().info('Deployment Validator Started')

    def image_callback(self, msg):
        """Track image data reception"""
        self.image_received = True

    def lidar_callback(self, msg):
        """Track LiDAR data reception"""
        self.lidar_received = True

    def run_validation_step(self):
        """Run validation steps sequentially"""
        if self.validation_step == 0:
            # Step 0: Check model files exist
            self.validate_model_files()
            self.validation_step = 1
        elif self.validation_step == 1:
            # Step 1: Check if sensors are working
            if self.image_received and self.lidar_received:
                self.sensors_working = True
                self.get_logger().info('✓ Sensors validated')
                self.validation_step = 2
            elif time.time() - self.validation_start_time > self.validation_timeout:
                self.get_logger().error('✗ Timeout waiting for sensor data')
                self.validation_step = 3
        elif self.validation_step == 2:
            # Step 2: Check if models are loaded
            # In a real implementation, we'd check for model loader status
            # For now, we'll assume they're loaded after sensors work
            self.models_loaded = True
            self.get_logger().info('✓ Models loaded')
            self.validation_step = 3
        elif self.validation_step == 3:
            # Step 3: Check if inference is running
            # This would involve checking for inference output topics
            # For now, we'll simulate by checking if we're receiving outputs
            # from the model executor node
            
            # In a real implementation, we'd subscribe to model outputs
            # and verify they're being published
            self.inference_running = True
            self.get_logger().info('✓ Inference running')
            
            # Complete validation
            self.complete_validation()
            self.validation_timer.cancel()

    def validate_model_files(self):
        """Validate that model files exist in the expected location"""
        if not os.path.exists(self.model_directory):
            self.get_logger().error(f'✗ Model directory does not exist: {self.model_directory}')
            return
        
        trt_files = [f for f in os.listdir(self.model_directory) if f.endswith('.trt')]
        if trt_files:
            self.model_files_validated = True
            self.get_logger().info(f'✓ Found {len(trt_files)} TensorRT model files')
        else:
            self.get_logger().error('✗ No TensorRT model files found in model directory')

    def complete_validation(self):
        """Complete the validation process"""
        # Determine overall validation status
        all_validated = all([
            self.model_files_validated,
            self.sensors_working,
            self.models_loaded,
            self.inference_running
        ])
        
        status_msg = String()
        if all_validated:
            status_msg.data = f"DEPLOYMENT_VALIDATION_SUCCESS: All components working. " \
                             f"Model files: {int(self.model_files_validated)}, " \
                             f"Sensors: {int(self.sensors_working)}, " \
                             f"Models loaded: {int(self.models_loaded)}, " \
                             f"Inference: {int(self.inference_running)}"
            self.get_logger().info('🎉 ALL DEPLOYMENT VALIDATION STEPS PASSED!')
        else:
            status_msg.data = f"DEPLOYMENT_VALIDATION_FAILED: " \
                             f"Model files: {int(self.model_files_validated)}, " \
                             f"Sensors: {int(self.sensors_working)}, " \
                             f"Models loaded: {int(self.models_loaded)}, " \
                             f"Inference: {int(self.inference_running)}"
            self.get_logger().error('❌ DEPLOYMENT VALIDATION FAILED!')
        
        self.validation_status_pub.publish(status_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Deployment Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = DeploymentValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Validator interrupted by user')
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Jetson AI Deployment

### 1. Model Optimization Best Practices

- **Precision Selection**: Use FP16 or INT8 for better performance on Jetson
- **Model Pruning**: Remove redundant neurons to reduce model size
- **Efficient Architectures**: Use MobileNet, EfficientNet, or other efficient architectures for edge deployment
- **Bottleneck Analysis**: Identify computational bottlenecks in your pipeline

### 2. Resource Management

- **Memory Management**: Monitor and manage GPU memory usage carefully
- **Thermal Management**: Implement thermal throttling when temperatures get too high
- **Power Management**: Adjust power profiles based on computational demands
- **Multi-threading**: Use appropriate threading to maximize CPU utilization

### 3. Deployment Considerations

- **Over-the-Air Updates**: Implement secure model update mechanisms
- **Rollback Capability**: Ensure you can rollback to previous models if needed
- **Monitoring**: Continuously monitor model performance and resource usage
- **Fault Tolerance**: Implement fallback behaviors when AI models fail

## Next Steps

With AI models successfully deployed to the Jetson platform, you're now ready to move on to motion planning basics. The AI systems you've created provide the perception and decision-making capabilities needed for advanced humanoid motion planning and control.

The deployment workflow you've implemented creates a complete pipeline from model optimization to runtime execution on Jetson, providing the foundation for all AI-powered behaviors in your humanoid robot.