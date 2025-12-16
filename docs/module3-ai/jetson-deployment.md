---
title: Jetson Deployment
description: Deploying AI models to NVIDIA Jetson Orin Nano/NX
sidebar_position: 6
---

# Deploying AI Models to NVIDIA Jetson

## Overview

In this chapter, we'll learn how to deploy AI models to NVIDIA Jetson Orin Nano/NX devices. The Jetson platform provides edge AI computing capabilities that are essential for running AI algorithms on humanoid robots where power efficiency and real-time processing are critical.

## Learning Objectives

- Understand the Jetson platform's architecture and capabilities
- Prepare AI models for deployment on Jetson devices
- Implement optimized inference pipelines for edge AI
- Test deployed models in simulated and real-world scenarios

## Jetson Platform Introduction

The NVIDIA Jetson family is designed for edge AI applications, providing high-performance computing in a compact, power-efficient form factor. For humanoid robotics, the Jetson Orin Nano/NX are particularly suitable due to their:

- Up to 275 TOPS AI performance (Orin)
- 2-15W power consumption range
- Real-time processing capabilities
- Support for multiple sensors and interfaces
- ROS 2 compatibility

### Hardware Specifications

- **GPU**: 1024-core NVIDIA Ampere architecture GPU
- **CPU**: 8-core ARM A78AE v8.2 64-bit CPU
- **Memory**: 4GB or 8GB LPDDR5
- **Storage**: eMMC 32GB or 64GB
- **Connectivity**: Gigabit Ethernet, CSI/MIPI interfaces, PWM, GPIO

## Preparing AI Models for Jetson

### Model Optimization

Before deploying models to Jetson devices, it's crucial to optimize them for the target hardware:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def optimize_model_for_jetson(model_path):
    """
    Optimize a model for Jetson deployment using TensorRT
    """
    # Create TensorRT builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set memory limit for workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, logger)
    with open(model_path, 'rb') as model_file:
        parser.parse(model_file.read())
    
    # Create optimization profile
    profile = builder.create_optimization_profile()
    # Add configuration for input dimensions based on your model
    # profile.set_shape("input_name", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # Build engine
    engine = builder.build_serialized_network(network, config)
    
    return engine
```

### TensorRT Deployment

TensorRT is NVIDIA's high-performance inference optimizer and runtime that delivers low latency and high throughput for deep learning applications:

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        """
        Initialize TensorRT inference engine
        """
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.bindings = []
        self.input_shape = None
        self.output_shape = None
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_buffer = cuda.pagelocked_empty(size, dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            self.bindings.append({'host': host_buffer, 'device': device_buffer})
            
            if self.engine.binding_is_input(binding):
                self.input_shape = self.engine.get_binding_shape(binding)
            else:
                self.output_shape = self.engine.get_binding_shape(binding)
    
    def load_engine(self, engine_path):
        """
        Load serialized TensorRT engine
        """
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(serialized_engine)
    
    def infer(self, input_data):
        """
        Perform inference on input data
        """
        # Copy input data to device
        np.copyto(self.bindings[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.bindings[0]['device'], self.bindings[0]['host'], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=[int(b['device']) for b in self.bindings], stream_handle=self.stream.handle)
        
        # Copy output data to host
        cuda.memcpy_dtoh_async(self.bindings[1]['host'], self.bindings[1]['device'], self.stream)
        self.stream.synchronize()
        
        # Reshape output
        output = self.bindings[1]['host'].reshape(self.output_shape)
        return output
```

## Jetson Development Environment Setup

### Using JetPack SDK

JetPack SDK provides the complete development environment for Jetson:

1. **Install JetPack**: Download from NVIDIA Developer website
2. **Flash Jetson device**: Use SDK Manager to flash the operating system
3. **Configure development environment**: Install required dependencies

```bash
# Install Python dependencies
pip3 install jetson-inference
pip3 install jetson-utils
pip3 install pycuda
pip3 install tensorrt

# Install ROS 2 dependencies for Jetson
sudo apt update
sudo apt install python3-ros-iron-perception
sudo apt install python3-ros-iron-vision-opencv
```

### Docker Containers for Jetson

Use Docker to create reproducible environments:

```dockerfile
# Dockerfile for Jetson deployment
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Install Python packages
RUN pip3 install numpy scipy
RUN pip3 install opencv-python
RUN pip3 install pycuda
RUN pip3 install tensorrt

# Copy model files
COPY models/ /workspace/models/

# Set working directory
WORKDIR /workspace

COPY deploy_model.py .
CMD ["python3", "deploy_model.py"]
```

## ROS 2 Integration on Jetson

Integrate the deployed model with ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2

class JetsonInferenceNode(Node):
    def __init__(self):
        super().__init__('jetson_inference_node')
        
        # Initialize TensorRT inference
        self.inference = TensorRTInference('models/perception_model.trt')
        self.bridge = CvBridge()
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)
        
        self.result_pub = self.create_publisher(
            String,
            'inference_results',
            10)
        
        self.get_logger().info('Jetson Inference Node initialized')
    
    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Preprocess image for model
        input_tensor = self.preprocess(cv_image)
        
        # Run inference
        result = self.inference.infer(input_tensor)
        
        # Publish results
        result_msg = String()
        result_msg.data = str(result.tolist())
        self.result_pub.publish(result_msg)
        
        self.get_logger().info(f'Published inference result: {result}')
    
    def preprocess(self, image):
        # Resize and normalize image for model
        resized = cv2.resize(image, (224, 224))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0).astype(np.float32)

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

## Practical Exercise: Deploying a Perception Model

### Exercise Objective

Deploy a simple object detection model to the Jetson platform and integrate it with ROS 2.

### Prerequisites

- NVIDIA Jetson Orin Nano/NX with JetPack SDK installed
- ROS 2 Iron installed on Jetson
- Sample perception model in ONNX format

### Exercise Steps

1. **Prepare the Model**
   ```bash
   # Convert your trained model to ONNX format if needed
   python3 convert_to_onnx.py
   ```

2. **Optimize for TensorRT**
   ```python
   # Optimize your model using TensorRT
   python3 optimize_model.py --model_path perception_model.onnx --output_path perception_model.trt
   ```

3. **Deploy to Jetson**
   - Transfer the optimized model to your Jetson device
   - Set up the ROS 2 environment on Jetson
   - Create and run the integration node

4. **Test Integration**
   ```bash
   # Launch the inference node
   ros2 run my_robot_jetson_jetson_inference_node
   ```

5. **Validate Results**
   - Verify that the model runs efficiently on the Jetson
   - Check that inference results are published via ROS 2
   - Ensure real-time performance is achieved

## Performance Optimization

### Memory Management

```python
import gc
import psutil

def optimize_memory_usage():
    """
    Optimize memory usage on Jetson
    """
    # Clear GPU memory
    import torch
    torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Monitor memory usage
    mem = psutil.virtual_memory()
    print(f"Memory usage: {mem.percent}%")
```

### Power Management

```bash
# Check power consumption
sudo tegrastats

# Set power mode
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Lock all clocks to maximum frequency
```

## Summary

In this chapter, we've covered the essential steps for deploying AI models to NVIDIA Jetson devices:

1. **Model optimization** with TensorRT for efficient inference
2. **Development environment setup** on Jetson platforms
3. **ROS 2 integration** to connect AI models with robot systems
4. **Performance optimization** for real-time edge AI applications

This deployment strategy enables humanoid robots to run sophisticated AI algorithms directly on the robot, reducing latency and enabling real-time decision-making capabilities essential for autonomous operation.

The next step is to integrate these deployed models with the robot's control system to enable AI-driven behaviors in real-world scenarios.