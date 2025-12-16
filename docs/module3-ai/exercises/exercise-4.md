---
title: Jetson Deployment Exercise
description: Practical hands-on exercise for deploying AI models to NVIDIA Jetson
sidebar_position: 11
---

# Jetson Deployment Exercise

## Overview

This hands-on exercise focuses on deploying AI models to NVIDIA Jetson platforms for humanoid robotics applications. You'll learn to optimize, deploy, and run AI models on resource-constrained embedded systems while maintaining real-time performance.

## Learning Objectives

- Understand NVIDIA Jetson hardware capabilities and constraints
- Optimize AI models for deployment on Jetson platforms
- Implement TensorRT optimization for inference acceleration
- Create deployment pipelines for Jetson devices
- Evaluate performance and resource utilization

## Prerequisites

- NVIDIA Jetson Orin Nano/NX device with JetPack SDK installed
- Basic knowledge of deep learning frameworks (PyTorch/TensorFlow)
- Understanding of Docker and containerization
- Experience with ROS 2 integration

## Exercise Setup

### Step 1: Hardware and Software Prerequisites

First, let's make sure your Jetson environment is properly set up:

```bash
# Check Jetson hardware information
sudo jetson_clocks
sudo tegrastats

# Verify JetPack version
cat /etc/nv_tegra_release

# Check available memory and storage
free -h
df -h

# Verify CUDA installation
nvidia-smi
nvcc --version
```

### Step 2: Install Required Dependencies

On your Jetson device, install the necessary packages:

```bash
# Update package list
sudo apt update

# Install Python packages for AI deployment
sudo apt install python3-pip python3-dev
pip3 install torch torchvision tensorrt pycuda numpy opencv-python

# Install ROS 2 Iron dependencies for Jetson
sudo apt install python3-ros-iron-perception python3-ros-iron-vision-opencv

# Install additional dependencies
sudo apt install build-essential cmake pkg-config
sudo apt install libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install libatlas-base-dev
sudo apt install libjasper-dev
```

### Step 3: Set Up Development Environment

```bash
# Create workspace for Jetson deployment exercise
mkdir -p ~/jetson_deployment_exercises
cd ~/jetson_deployment_exercises

# Create virtual environment
python3 -m venv jetson_env
source jetson_env/bin/activate

# Install required Python packages
pip3 install --upgrade pip
pip3 install torch torchvision tensorrt pycuda numpy opencv-python
pip3 install onnx onnxruntime-gpu
pip3 install pybind11

# Install ROS 2 Python packages
pip3 install ros-iron-std-msgs ros-iron-sensor-msgs
```

## Model Optimization

### Step 4: Create a Sample Model for Deployment

First, let's create a simple model that we'll optimize for Jetson deployment:

```python
# sample_model.py
import torch
import torch.nn as nn
import torch.onnx

class SimplePerceptionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SimplePerceptionModel, self).__init__()
        
        # Define the network architecture suitable for embedded deployment
        self.features = nn.Sequential(
            # Input: 224x224x3 -> Output: 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Input: 112x112x32 -> Output: 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            
            # Input: 56x56x64 -> Output: 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            
            # Input: 28x28x128 -> Output: 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_and_save_model():
    """Create, train a bit, and save the model"""
    model = SimplePerceptionModel(num_classes=5)
    model.eval()  # Set to evaluation mode
    
    # Create sample input for tracing
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Save the model in PyTorch format
    torch.save(model.state_dict(), "perception_model.pth")
    
    # Export to ONNX format
    torch.onnx.export(
        model,
        sample_input,
        "perception_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Model saved as perception_model.pth and perception_model.onnx")
    return model

if __name__ == "__main__":
    model = create_and_save_model()
    
    # Test the model
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output shape: {output.shape}")
        print(f"Sample output: {output[0, :5]}")  # Print first 5 values
```

### Step 5: Model Optimization with TensorRT

Now, let's create an optimization script that converts the ONNX model to TensorRT format:

```python
# optimize_model.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
import os

def build_engine_from_onnx(onnx_file_path, engine_file_path, input_shape=(1, 3, 224, 224)):
    """
    Build a TensorRT engine from an ONNX file
    
    Args:
        onnx_file_path: Path to the ONNX model
        engine_file_path: Path to save the TensorRT engine
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
    """
    
    # Create a TensorRT builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Create network definition
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX file
    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure the builder
    config = builder.create_builder_config()
    
    # Set memory limit for workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Create optimization profile
    profile = builder.create_optimization_profile()
    
    # Define input dimensions for the profile
    input_name = network.get_input(0).name
    min_shape = input_shape  # Minimum batch size
    opt_shape = input_shape  # Optimal batch size
    max_shape = (4, 3, 224, 224)  # Maximum batch size (for dynamic batching)
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # Build the engine
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build the TensorRT engine.")
        return None
    
    # Save the engine to a file
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_file_path}")
    
    # Create runtime and engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine

def test_engine(engine, input_shape=(1, 3, 224, 224), num_runs=100):
    """
    Test the TensorRT engine performance
    """
    # Create an execution context
    context = engine.create_execution_context()
    
    # Allocate buffers for input and output
    input_binding_idx = engine.get_binding_index("input")
    output_binding_idx = engine.get_binding_index("output")
    
    input_size = trt.volume(engine.get_binding_shape(input_binding_idx)) * engine.max_batch_size * np.dtype(np.float32).itemsize
    output_size = trt.volume(engine.get_binding_shape(output_binding_idx)) * engine.max_batch_size * np.dtype(np.float32).itemsize
    
    # Allocate device memory
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    # Create stream for async execution
    stream = cuda.Stream()
    
    # Prepare input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    import time
    
    # Warm up
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    stream.synchronize()
    
    # Time the inference
    start_time = time.time()
    for i in range(num_runs):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        stream.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
    
    # Copy output back to host
    output = np.empty(engine.get_binding_shape(output_binding_idx), dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    
    return output

def main():
    # Build the TensorRT engine
    engine = build_engine_from_onnx(
        onnx_file_path="perception_model.onnx",
        engine_file_path="perception_model.trt",
        input_shape=(1, 3, 224, 224)
    )
    
    if engine is not None:
        print("Engine built successfully!")
        
        # Test the engine
        print("\nTesting engine performance...")
        output = test_engine(engine)
        print(f"Sample output: {output.flatten()[:5]}")  # Print first 5 values
    else:
        print("Failed to build engine")

if __name__ == "__main__":
    main()
```

## Deployment Implementation

### Step 6: Create TensorRT Inference Engine

Create a class for handling TensorRT inference:

```python
# tensorrt_inference.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
from typing import List, Tuple

class TensorRTInference:
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT inference engine
        
        Args:
            engine_path: Path to the TensorRT engine file (.trt)
        """
        # Load the engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate CUDA stream
        self.stream = cuda.Stream()
        
        # Get input and output information
        self.input_binding_idx = self.engine.get_binding_index("input")
        self.output_binding_idx = self.engine.get_binding_index("output")
        
        # Allocate host and device buffers
        self._allocate_buffers()
        
        print(f"TensorRT engine loaded: {engine_path}")
        print(f"Input shape: {self.engine.get_binding_shape(self.input_binding_idx)}")
        print(f"Output shape: {self.engine.get_binding_shape(self.output_binding_idx)}")
    
    def _allocate_buffers(self):
        """Allocate input and output buffers for inference"""
        # Calculate sizes
        input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        output_shape = self.engine.get_binding_shape(self.output_binding_idx)
        
        # Calculate buffer sizes
        self.input_size = trt.volume(input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
        self.output_size = trt.volume(output_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize
        
        # Allocate host buffers (pinned memory for faster transfers)
        self.host_input = cuda.pagelocked_empty(self.input_size // np.dtype(np.float32).itemsize, dtype=np.float32)
        self.host_output = cuda.pagelocked_empty(self.output_size // np.dtype(np.float32).itemsize, dtype=np.float32)
        
        # Allocate device buffers
        self.device_input = cuda.mem_alloc(self.input_size)
        self.device_output = cuda.mem_alloc(self.output_size)
        
        # Store shapes for later use
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image: Input image as numpy array (H, W, C)
        
        Returns:
            Preprocessed input tensor as numpy array
        """
        # Resize image to model input size
        h, w = self.input_shape[2], self.input_shape[3]
        image_resized = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB if necessary
        if image_resized.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_resized
        
        # Normalize to [0, 1] and transpose from HWC to CHW
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_transposed = image_normalized.transpose(2, 0, 1)  # CHW format
        
        # Add batch dimension if needed
        if len(image_transposed.shape) == 3:  # (C, H, W)
            image_tensor = np.expand_dims(image_transposed, axis=0)  # (1, C, H, W)
        else:
            image_tensor = image_transposed  # (B, C, H, W)
        
        return image_tensor
    
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Perform inference on input tensor
        
        Args:
            input_tensor: Input tensor as numpy array
        
        Returns:
            Output tensor as numpy array
        """
        # Validate input shape
        if input_tensor.shape != self.input_shape:
            print(f"Warning: Input shape {input_tensor.shape} doesn't match expected {self.input_shape}")
            # Reshape if possible, otherwise raise error
            input_tensor = input_tensor.reshape(self.input_shape)
        
        # Copy input to host buffer
        np.copyto(self.host_input, input_tensor.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        
        # Execute inference
        self.context.execute_async_v2(
            bindings=[int(self.device_input), int(self.device_output)], 
            stream_handle=self.stream.handle
        )
        
        # Transfer output data back to host
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape output to expected shape
        output = self.host_output.reshape(self.output_shape)
        
        return output
    
    def infer_with_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference with preprocessing
        
        Args:
            image: Input image as numpy array (H, W, C)
        
        Returns:
            Output tensor as numpy array
        """
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Perform inference
        output = self.infer(input_tensor)
        
        return output

class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.frame_count = 0
        self.start_time = time.time()
    
    def start_inference(self):
        """Start timing an inference operation"""
        return time.time()
    
    def end_inference(self, start_time):
        """End timing an inference operation and record time"""
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return inference_time
    
    def get_fps(self):
        """Calculate current FPS"""
        if self.frame_count == 0:
            return 0
        
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0
    
    def get_avg_inference_time(self):
        """Calculate average inference time"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)
    
    def reset(self):
        """Reset performance counters"""
        self.inference_times = []
        self.frame_count = 0
        self.start_time = time.time()
    
    def print_stats(self):
        """Print performance statistics"""
        fps = self.get_fps()
        avg_inf_time = self.get_avg_inference_time()
        
        print(f"Performance Stats:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Avg Inference Time: {avg_inf_time*1000:.2f} ms")
        print(f"  Total Frames: {self.frame_count}")

def run_inference_example():
    """Example of running inference with performance monitoring"""
    # Initialize TensorRT inference
    trt_infer = TensorRTInference("perception_model.trt")
    perf_monitor = PerformanceMonitor()
    
    # Create a sample image (in real application, this would come from camera)
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    print("Running inference example...")
    
    # Run several inferences to warm up
    for i in range(10):
        output = trt_infer.infer_with_preprocessing(sample_image)
    
    print("Warmup complete. Starting performance measurement...")
    
    # Run performance test
    for i in range(100):
        start_time = perf_monitor.start_inference()
        output = trt_infer.infer_with_preprocessing(sample_image)
        inference_time = perf_monitor.end_inference(start_time)
        
        if i % 20 == 0:
            fps = perf_monitor.get_fps()
            print(f"Frame {i}: Inference time: {inference_time*1000:.2f} ms, FPS: {fps:.2f}")
    
    # Print final stats
    perf_monitor.print_stats()

if __name__ == "__main__":
    run_inference_example()
```

### Step 7: ROS 2 Integration

Now, let's integrate the TensorRT inference with a ROS 2 node:

```python
# jetson_ros_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
from typing import List

from tensorrt_inference import TensorRTInference, PerformanceMonitor

class JetsonInferenceNode(Node):
    def __init__(self):
        super().__init__('jetson_inference_node')
        
        # Initialize TensorRT inference
        self.trt_infer = TensorRTInference('perception_model.trt')
        self.bridge = CvBridge()
        self.perf_monitor = PerformanceMonitor()
        
        # Class labels (for our 5-class model)
        self.class_labels = ['obstacle', 'target', 'person', 'animal', 'unknown']
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.result_pub = self.create_publisher(
            String,
            'inference_results',
            10
        )
        
        # Timer for performance reporting
        self.perf_timer = self.create_timer(5.0, self.report_performance)
        
        self.get_logger().info('Jetson Inference Node initialized')
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Perform inference
            start_time = self.perf_monitor.start_inference()
            output = self.trt_infer.infer_with_preprocessing(cv_image)
            inference_time = self.perf_monitor.end_inference(start_time)
            
            # Process results
            results = self.process_inference_output(output)
            
            # Log results
            self.get_logger().info(
                f'Inference: {results}. Time: {inference_time*1000:.2f}ms'
            )
            
            # Publish results
            result_msg = String()
            result_msg.data = f"{results} (inf_time:{inference_time*1000:.1f}ms)"
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
    
    def process_inference_output(self, output: np.ndarray) -> str:
        """
        Process the inference output and return human-readable results
        """
        # Get the predicted class and confidence
        probabilities = self.softmax(output[0])  # Assuming output is raw logits
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        predicted_class = self.class_labels[predicted_class_idx]
        
        # Format results
        results = f"class: {predicted_class}, confidence: {confidence:.3f}"
        
        return results
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def report_performance(self):
        """Report performance metrics"""
        fps = self.perf_monitor.get_fps()
        avg_inf_time = self.perf_monitor.get_avg_inference_time()
        
        self.get_logger().info(
            f'Performance - FPS: {fps:.2f}, Avg Inference Time: {avg_inf_time*1000:.2f}ms'
        )

def main(args=None):
    rclpy.init(args=args)
    node = JetsonInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Containerization for Deployment

### Step 8: Create Dockerfile for Jetson Deployment

Create a Dockerfile optimized for Jetson deployment:

```dockerfile
# Dockerfile.jetson
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install OpenCV optimized for Jetson
RUN pip3 install opencv-python-headless

# Install TensorRT Python bindings (they should come with the base image)
# But install any additional dependencies if needed
RUN pip3 install numpy pycuda

# Install ROS 2 Iron Python packages
RUN pip3 install rclpy ros-iron-std-msgs ros-iron-sensor-msgs

# Install additional packages for model optimization
RUN pip3 install onnx onnxruntime-gpu

# Set up working directory
WORKDIR /workspace

# Copy application code
COPY . /workspace

# Install any additional requirements
RUN pip3 install -e .

# Set environment variables for Jetson optimization
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDA_CACHE_PATH=/workspace/.cuda_cache

# Create non-root user
RUN useradd -m -s /bin/bash jetson && echo "jetson:jetson" | chpasswd && adduser jetson sudo
USER jetson

# Create directories for models and logs
RUN mkdir -p /workspace/models /workspace/logs

# Set entry point
CMD ["python3", "jetson_ros_node.py"]
```

### Step 9: Create a Deployment Script

```python
# deploy_to_jetson.py
#!/usr/bin/env python3

import os
import subprocess
import sys
import argparse
import shutil
from pathlib import Path

def check_jetson_environment():
    """Check if running on a Jetson device"""
    # Check if we're on Jetson
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            if 'JETSON' in content.upper():
                return True
    except FileNotFoundError:
        pass
    
    # Alternative check
    if os.path.exists('/sys/module/tegra_fuse'):
        return True
    
    return False

def build_tensorrt_engine(model_path, engine_path, input_shape):
    """Build a TensorRT engine from PyTorch model"""
    print(f"Building TensorRT engine from {model_path}")
    
    # Import here to avoid dependency issues
    try:
        import torch
        import torch.onnx
    except ImportError:
        print("PyTorch not available, skipping ONNX export")
        return False
    
    # Load the model
    model = torch.load(model_path)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(input_shape)
    
    # Export to ONNX
    onnx_path = engine_path.replace('.trt', '.onnx')
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print(f"ONNX model saved to {onnx_path}")
    
    # Call optimize_model.py to build TensorRT engine
    subprocess.run(['python3', 'optimize_model.py'], check=True)
    
    return True

def copy_model_to_device(source_model, dest_path, device_address=None, username='jetson'):
    """Copy model to Jetson device if deploying remotely"""
    if device_address:
        print(f"Copying model to remote Jetson at {device_address}")
        subprocess.run([
            'scp', source_model, 
            f'{username}@{device_address}:{dest_path}'
        ], check=True)
    else:
        # Local deployment
        dest_dir = Path(dest_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_model, dest_path)
        print(f"Model copied to {dest_path}")

def build_docker_image(image_name, dockerfile_path):
    """Build Docker image from Dockerfile"""
    print(f"Building Docker image: {image_name}")
    
    result = subprocess.run([
        'docker', 'build', 
        '-t', image_name,
        '-f', dockerfile_path,
        '.'
    ], check=True)
    
    return result.returncode == 0

def run_docker_container(image_name, container_name, port_mapping=None, volume_mapping=None):
    """Run Docker container with the trained model"""
    cmd = ['docker', 'run', '-d', '--name', container_name]
    
    if port_mapping:
        for mapping in port_mapping:
            cmd.extend(['-p', mapping])
    
    if volume_mapping:
        for mapping in volume_mapping:
            cmd.extend(['-v', mapping])
    
    # Add runtime for GPU
    cmd.extend(['--gpus', 'all'])
    
    cmd.append(image_name)
    
    print(f"Running container with command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode == 0

def deploy_model(
    model_path,
    jetson_address=None,
    username='jetson',
    use_docker=False,
    image_name='jetson-ai-model:latest'
):
    """Deploy model to Jetson device"""
    
    print("Starting Jetson deployment process...")
    
    # Check if running on Jetson
    on_jetson = check_jetson_environment()
    
    if not on_jetson and not jetson_address:
        print("Warning: Not running on Jetson and no remote address provided.")
        print("Assuming local deployment for testing purposes.")
    
    # If not on Jetson, optimize model for TensorRT
    if not on_jetson:
        print("Optimizing model for TensorRT...")
        input_shape = (1, 3, 224, 224)  # Adjust based on your model
        trt_engine_path = 'optimized_model.trt'
        
        if not build_tensorrt_engine(model_path, trt_engine_path, input_shape):
            print("Failed to build TensorRT engine")
            return False
        
        model_path = trt_engine_path
    
    # Copy model to Jetson
    if jetson_address:
        dest_path = f'/home/{username}/jetson_models/model.trt'
        copy_model_to_device(model_path, dest_path, jetson_address, username)
    else:
        dest_path = './model.trt'
        copy_model_to_device(model_path, dest_path)
    
    # If using Docker deployment
    if use_docker:
        print("Building Docker image...")
        if not build_docker_image(image_name, 'Dockerfile.jetson'):
            print("Failed to build Docker image")
            return False
        
        print("Running Docker container...")
        if not run_docker_container(
            image_name, 
            'jetson-ai-inference',
            port_mapping=['8080:8080'],  # Example port mapping
            volume_mapping=[f'{os.getcwd()}:/workspace/input_data:ro']  # Example volume mapping
        ):
            print("Failed to run Docker container")
            return False
    
    print("Model successfully deployed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Deploy AI model to Jetson')
    parser.add_argument('--model-path', required=True, help='Path to the model file')
    parser.add_argument('--jetson-address', help='IP address of Jetson device')
    parser.add_argument('--username', default='jetson', help='Username for Jetson device')
    parser.add_argument('--use-docker', action='store_true', help='Use Docker for deployment')
    parser.add_argument('--image-name', default='jetson-ai-model:latest', help='Docker image name')
    
    args = parser.parse_args()
    
    success = deploy_model(
        model_path=args.model_path,
        jetson_address=args.jetson_address,
        username=args.username,
        use_docker=args.use_docker,
        image_name=args.image_name
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

## Exercise Tasks

### Step 10: Complete Implementation Tasks

Complete the following tasks to finish the Jetson deployment exercise:

1. **Model Creation and Optimization**:
   - Create the sample perception model
   - Optimize it for TensorRT deployment
   - Verify performance improvements

2. **TensorRT Integration**:
   - Implement the TensorRT inference class
   - Test with sample images
   - Measure performance gains

3. **ROS 2 Integration**:
   - Create the ROS 2 node for inference
   - Test with simulated image data
   - Verify real-time performance

4. **Deployment Script**:
   - Implement the deployment script
   - Test model transfer to Jetson
   - Validate the deployment process

5. **Performance Evaluation**:
   - Measure inference speed on Jetson
   - Compare performance with CPU
   - Assess power consumption impact

## Performance Evaluation

### Step 11: Deployment Performance Metrics

```python
# jetson_performance_eval.py
import subprocess
import time
import os
import statistics
import matplotlib.pyplot as plt

def measure_power_consumption():
    """Measure power consumption on Jetson (if possible)"""
    try:
        # Check if jetson_clocks is available
        result = subprocess.run(['sudo', 'nvpmodel', '-q'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return "Power info not available"

def benchmark_inference_speed(model_path, num_runs=100):
    """Benchmark inference speed of the model"""
    from tensorrt_inference import TensorRTInference
    
    # Initialize inference
    infer = TensorRTInference(model_path)
    
    # Create dummy input
    import numpy as np
    dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        _ = infer.infer_with_preprocessing(dummy_input)
    
    # Time inference
    inference_times = []
    for _ in range(num_runs):
        start = time.time()
        _ = infer.infer_with_preprocessing(dummy_input)
        end = time.time()
        inference_times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'avg_time_ms': statistics.mean(inference_times),
        'std_time_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
        'min_time_ms': min(inference_times),
        'max_time_ms': max(inference_times),
        'fps': 1000 / statistics.mean(inference_times) if statistics.mean(inference_times) > 0 else 0
    }

def compare_deployments():
    """Compare different deployment methods"""
    results = {}
    
    print("Measuring TensorRT deployment performance...")
    trt_results = benchmark_inference_speed("perception_model.trt")
    results['TensorRT'] = trt_results
    
    # If PyTorch model is available, also benchmark it
    try:
        import torch
        print("Measuring PyTorch deployment performance...")
        # Implementation would require a PyTorch inference class
        # This is a placeholder for comparison
        results['PyTorch (CPU)'] = {'avg_time_ms': 200.0, 'fps': 5.0}  # Placeholder values
    except ImportError:
        print("PyTorch not available for comparison")
    
    return results

def plot_performance_comparison(results):
    """Plot performance comparison"""
    deployment_types = list(results.keys())
    avg_times = [results[d]['avg_time_ms'] for d in deployment_types]
    fps_values = [results[d]['fps'] for d in deployment_types]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot inference time
    ax1.bar(deployment_types, avg_times)
    ax1.set_ylabel('Average Inference Time (ms)')
    ax1.set_title('Inference Time Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot FPS
    ax2.bar(deployment_types, fps_values)
    ax2.set_ylabel('Frames Per Second (FPS)')
    ax2.set_title('FPS Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Starting Jetson deployment performance evaluation...")
    
    # Check hardware information
    print("\nHardware Information:")
    try:
        subprocess.run(['cat', '/proc/device-tree/model'], check=True)
    except:
        print("Could not retrieve Jetson model info")
    
    print("\nPower Consumption (before benchmark):")
    print(measure_power_consumption())
    
    # Run benchmarks
    results = compare_deployments()
    
    print("\nPerformance Results:")
    for deployment, metrics in results.items():
        print(f"\n{deployment}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Plot comparison
    plot_performance_comparison(results)
    
    print("\nPerformance evaluation complete!")

if __name__ == "__main__":
    main()
```

## Challenge: Advanced Deployment

As an additional challenge, try to:

1. Implement model quantization for further optimization
2. Create an auto-scaling deployment that adapts to workload
3. Add model versioning and A/B testing capabilities
4. Implement remote model updates over the network
5. Create a web-based dashboard for monitoring deployed models

## Summary

In this exercise, you've completed a comprehensive Jetson deployment workflow:

1. **Model Optimization**: Learned to optimize models using TensorRT
2. **TensorRT Integration**: Implemented efficient inference engines
3. **ROS 2 Integration**: Connected AI models with robotics framework
4. **Containerization**: Created Docker-based deployment solutions
5. **Performance Evaluation**: Measured and compared deployment performance

This deployment pipeline enables efficient AI inference on resource-constrained Jetson devices, making it possible to run complex perception models on humanoid robots while maintaining real-time performance and managing power consumption effectively.