# Jetson Deployment Workflow for Humanoid Robotics

## Overview

NVIDIA Jetson platforms provide powerful edge computing capabilities essential for humanoid robotics applications. This section details the complete workflow for deploying AI models and robotics software from simulation to Jetson Orin Nano/NX hardware, ensuring optimal performance and efficient execution on resource-constrained platforms.

## Jetson Platforms for Humanoid Robotics

### 1. Supported Jetson Platforms

For humanoid robotics, we recommend:

- **Jetson Orin Nano**: Best for complex AI tasks with good performance-per-watt
- **Jetson Orin NX**: Balance of performance and form factor
- **Jetson AGX Orin**: For the most complex humanoid behaviors with multiple AI models

### 2. Hardware Specifications for Robotics

**Jetson Orin Nano (Developer Kit)**:
- CPU: 6-core ARM Cortex-A78AE v8.2 64-bit CPU
- GPU: 1024-core NVIDIA Ampere™ architecture GPU with 2x RT cores, 8x Tensor cores
- Memory: 4 GB or 8 GB LPDDR5
- Power: 15W to 25W
- Perfect for: AI perception, navigation, lightweight control

**Jetson Orin NX**:
- CPU: 8-core ARM Cortex-A78AE v8.2 64-bit CPU
- GPU: 1024-core NVIDIA Ampere architecture GPU
- Memory: 8 GB or 16 GB LPDDR5
- Power: 15W to 25W
- Perfect for: Complex perception + navigation + control

## Prerequisites for Jetson Deployment

### 1. Host Development Machine Setup

Before deploying to Jetson, ensure your development environment is properly configured:

```bash
# Verify your simulation environment works
cd ~/humanoid_ws
source /opt/ros/iron/setup.bash
colcon build --packages-select humanoid_simple_robot
source install/setup.bash

# Test that all simulation components work
ros2 launch humanoid_simple_robot sim2real_transfer.launch.py
```

### 2. Jetson Device Preparation

Prepare your Jetson device for AI deployment:

```bash
# 1. Flash JetPack OS (follow NVIDIA official instructions)
# For Jetson Orin Nano, use JetPack 5.1 or later

# 2. Update system packages
sudo apt update && sudo apt upgrade -y

# 3. Install essential development tools
sudo apt install build-essential cmake git curl wget python3-dev python3-pip python3-venv

# 4. Install Docker (for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

## Building Optimized AI Models for Jetson

### 1. Model Optimization with TensorRT

Create optimized models specifically for Jetson deployment:

```python
# scripts/optimize_model_for_jetson.py
#!/usr/bin/env python3

"""
Optimize AI models for Jetson deployment using TensorRT.
This script optimizes various AI models for efficient execution on Jetson platforms.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import onnx
from onnx_tensorrt.backend import Backend
import os
import argparse


def optimize_model_for_jetson(model_path, output_path, precision='fp16'):
    """
    Optimize an ONNX model for deployment on Jetson using TensorRT.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the optimized TensorRT engine
        precision: Precision for optimization ('fp32', 'fp16', 'int8')
    """
    # Initialize TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Build engine based on precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        else:
            print("FP16 not supported on this platform, using FP32")
    
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Set INT8 calibration parameters
            config.int8_calibrator = None  # You would implement a calibration dataset
            print("Using INT8 precision")
        else:
            print("INT8 not supported on this platform, using FP32")
    
    # Optimize for Jetson's target
    config.max_workspace_size = 2 << 30  # 2GB
    
    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine")
        return False
    
    # Save the optimized engine
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Model optimized and saved to: {output_path}")
    return True


def benchmark_model(engine_path, input_shape):
    """
    Benchmark the optimized model performance.
    """
    # Load the TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        print("Failed to load the engine")
        return None
    
    # Create execution context
    context = engine.create_execution_context()
    
    # Allocate memory for input/output
    input_binding_idx = engine.get_binding_index('input')  # Update name based on your model
    output_binding_idx = engine.get_binding_index('output')
    
    # Allocate CUDA memory
    h_input = np.zeros(input_shape, dtype=np.float32)
    h_output = np.zeros((engine.get_binding_shape(output_binding_idx)), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Execute and time the model
    import time
    start_time = time.time()
    
    # Copy input to device
    cuda.memcpy_htod_async(d_input, h_input, stream)
    
    # Run inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # Copy output back to host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"Model inference time: {inference_time:.2f} ms")
    print(f"Throughput: {1000/inference_time:.2f} fps")
    
    return inference_time


def main():
    parser = argparse.ArgumentParser(description='Optimize models for Jetson deployment')
    parser.add_argument('--input-model', type=str, required=True, 
                       help='Path to input ONNX model')
    parser.add_argument('--output-engine', type=str, required=True,
                       help='Path to save optimized TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision for optimization (default: fp16)')
    parser.add_argument('--input-shape', nargs='+', type=int, required=True,
                       help='Input shape for benchmarking (e.g., 1 3 224 224)')
    
    args = parser.parse_args()
    
    print(f"Optimizing {args.input_model} for Jetson deployment...")
    
    # Optimize the model
    success = optimize_model_for_jetson(
        args.input_model, 
        args.output_engine, 
        args.precision
    )
    
    if success:
        print("Model optimization completed successfully")
        
        # Benchmark the optimized model
        print("Benchmarking optimized model...")
        inference_time = benchmark_model(args.output_engine, args.input_shape)
        
        if inference_time:
            print(f"Model ready for Jetson deployment. Performance: {1000/inference_time:.2f} fps")
    else:
        print("Model optimization failed")


if __name__ == '__main__':
    main()
```

### 2. ROS 2 Package Optimization

Optimize your ROS 2 packages for Jetson deployment:

```yaml
# jetson_deployment/docker-compose.jetson.yaml
version: '3.8'

services:
  humanoid-jetson:
    build:
      context: .
      dockerfile: Dockerfile.jetson
    devices:
      - /dev:/dev
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw  # For visualization if needed
      - ./models:/opt/models
    environment:
      - DISPLAY=$DISPLAY
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    privileged: true
    command: >
      bash -c "
        source /opt/ros/iron/setup.bash &&
        source /usr/local/cuda/bin/cuda_profile.sh &&
        source /install/setup.bash &&
        ros2 launch humanoid_jetson_bringup humanoid_jetson.launch.py
      "

volumes:
  jetson_data:
```

Create the Jetson-specific Dockerfile:

```dockerfile
# jetson_deployment/Dockerfile.jetson
FROM nvcr.io/nvidia/ros:jammy-ros-iron-perception

# Install dependencies for humanoid robotics
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies for Jetson
COPY requirements_jetson.txt /tmp/requirements_jetson.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements_jetson.txt

# Copy robot model and AI packages
COPY . /ws/src/humanoid_robot
WORKDIR /ws

# Install dependencies
RUN source /opt/ros/iron/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN source /opt/ros/iron/setup.bash && \
    colcon build --packages-select humanoid_simple_robot --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
RUN echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc && \
    echo "source /ws/install/setup.bash" >> ~/.bashrc

CMD ["bash", "-c", "source /opt/ros/iron/setup.bash && source /ws/install/setup.bash && ros2 launch humanoid_jetson_bringup humanoid_jetson.launch.py"]
```

Create a requirements file for Jetson:

```txt
# jetson_deployment/requirements_jetson.txt
numpy==1.24.3
scipy==1.10.1
opencv-python-headless==4.8.0.74
torch==2.0.0+cu118
torchvision==0.15.1+cu118
torchaudio==2.0.1+cu118
onnx==1.14.0
onnxruntime==1.15.1
tensorrt==8.6.1
pycuda==2022.2.2
transforms3d==0.4.1
matplotlib==3.7.1
```

## Containerized Deployment Approach

### 1. Creating Jetson-Optimized Containers

For efficient deployment, containerize your solution:

```bash
# scripts/build_jetson_container.sh
#!/bin/bash

# Build script for Jetson-optimized ROS 2 container

echo "Building Jetson-optimized container..."

# Set platform to ARM64 for Jetson
PLATFORM=linux/arm64

# Build the container for Jetson
docker build \
  --platform $PLATFORM \
  -f Dockerfile.jetson \
  -t humanoid-ros-jetson:latest \
  .

# Tag for your registry
REGISTRY_URL="your-registry.com/humanoid-ros-jetson:latest"
docker tag humanoid-ros-jetson:latest $REGISTRY_URL

echo "Container built successfully!"
echo "To deploy: docker push $REGISTRY_URL"
echo "On Jetson: docker pull $REGISTRY_URL && docker run humanoid-ros-jetson"
```

### 2. Jetson-Specific Launch File

Create a jetson-specific launch file: `launch/humanoid_jetson.launch.py`

```python
# launch/humanoid_jetson.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_visualization = LaunchConfiguration('enable_visualization', default='false')
    model_precision = LaunchConfiguration('model_precision', default='fp16')
    jetson_performance_mode = LaunchConfiguration('jetson_performance_mode', default='max')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_jetson_bringup')
    
    # Set Jetson to max performance mode (optional, requires root)
    set_performance_mode = TimerAction(
        period=1.0,
        actions=[
            # This would set Jetson to max performance mode
            # In practice, this might be done via a systemd service instead
            ExecuteProcess(
                cmd=['nvpmodel', '-m', '0'],  # Max performance mode
                condition=IfCondition(jetson_performance_mode)
            )
        ]
    )
    
    # Robot State Publisher (for visualization and transforms)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_description'),
                'urdf',
                'optimzed_humanoid.urdf'
            ])}
        ],
        output='screen'
    )
    
    # Perception pipeline optimized for Jetson
    perception_pipeline = Node(
        package='humanoid_perception',
        executable='jetson_perception_pipeline',
        name='jetson_perception_pipeline',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'model_precision': model_precision},  # Use optimized precision
            {'enable_gpu_processing': True},       # Enable GPU processing
            {'max_processing_rate': 30.0},         # Reduced rate for Jetson
            {'gpu_processing': True}
        ],
        output='screen'
    )
    
    # AI planner with optimized parameters
    ai_planner = Node(
        package='humanoid_navigation',
        executable='jetson_ai_planner',
        name='jetson_ai_planner',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'planning_frequency': 2.0},  # Lower frequency for Jetson
            {'max_plan_length': 50},       # Shorter plans for faster computation
            {'gpu_acceleration': True},    # Enable GPU acceleration
            {'model_path': PathJoinSubstitution([
                FindPackageShare('humanoid_models'),
                'optimized',
                'nav_plan_fp16.trt'
            ])}
        ],
        output='screen'
    )
    
    # Control pipeline optimized for Jetson
    control_pipeline = Node(
        package='humanoid_control',
        executable='jetson_control_pipeline',
        name='jetson_control_pipeline',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'control_frequency': 100.0},  # Higher frequency for stability
            {'gpu_processing': True},      # Enable GPU for control algorithms
            {'model_precision': 'fp16'}
        ],
        output='screen'
    )
    
    # Hardware interface for Jetson-specific peripherals
    hardware_interface = Node(
        package='humanoid_hardware_interface',
        executable='jetson_hardware_interface',
        name='jetson_hardware_interface',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'camera_interface': 'usb'},
            {'imu_interface': 'i2c'},
            {'motor_interface': 'pwm'}
        ],
        output='screen'
    )
    
    # Performance monitor for Jetson
    jetson_monitor = Node(
        package='humanoid_system_monitor',
        executable='jetson_performance_monitor',
        name='jetson_performance_monitor',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'monitor_frequency': 1.0}  # Lower frequency to reduce overhead
        ],
        output='screen'
    )
    
    # RViz for visualization (only if enabled)
    rviz = Node(
        condition=IfCondition(enable_visualization),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', PathJoinSubstitution([
                FindPackageShare('humanoid_jetson_bringup'),
                'rviz',
                'jetson_humanoid.rviz'
            ])
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    return LaunchDescription([
        # Set performance mode
        set_performance_mode,
        
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch robot state publisher
        robot_state_publisher,
        
        # Launch hardware interface first
        TimerAction(
            period=2.0,
            actions=[hardware_interface]
        ),
        
        # Launch perception pipeline after hardware is ready
        TimerAction(
            period=4.0,
            actions=[perception_pipeline]
        ),
        
        # Launch AI planner after perception is ready
        TimerAction(
            period=6.0,
            actions=[ai_planner]
        ),
        
        # Launch control pipeline after planner is ready
        TimerAction(
            period=8.0,
            actions=[control_pipeline]
        ),
        
        # Launch performance monitor
        TimerAction(
            period=10.0,
            actions=[jetson_monitor]
        ),
        
        # Launch RViz if visualization enabled
        TimerAction(
            period=12.0,
            actions=[rviz]
        ),
    ])
```

## Jetson-Specific Hardware Interface

### 1. Hardware Abstraction Layer

Create a hardware abstraction layer optimized for Jetson:

```python
# scripts/jetson_hardware_interface.py
#!/usr/bin/env python3

"""
Hardware interface abstraction for Jetson Orin platforms.
Handles sensor data acquisition and actuator control specifically optimized for Jetson hardware.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState, Temperature
from std_msgs.msg import String, Float64MultiArray
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CameraInfo
import cv2
import numpy as np
from cv_bridge import CvBridge
import time
import threading
from threading import Lock


class JetsonHardwareInterface(Node):
    """
    Hardware interface for Jetson Orin platforms, managing sensors and actuators.
    Includes power and thermal management specifically for Jetson.
    """
    
    def __init__(self):
        super().__init__('jetson_hardware_interface')
        
        # Declare parameters
        self.declare_parameter('camera_interface', 'usb', ParameterDescriptor())
        self.declare_parameter('imu_interface', 'i2c', ParameterDescriptor())
        self.declare_parameter('motor_interface', 'pwm', ParameterDescriptor())
        self.declare_parameter('camera_fps', 30, ParameterDescriptor())
        self.declare_parameter('thermal_monitoring_enabled', True, ParameterDescriptor())
        self.declare_parameter('power_management_enabled', True, ParameterDescriptor())
        
        # Get parameters
        self.camera_interface = self.get_parameter('camera_interface').value
        self.imu_interface = self.get_parameter('imu_interface').value
        self.motor_interface = self.get_parameter('motor_interface').value
        self.camera_fps = self.get_parameter('camera_fps').value
        self.thermal_monitoring_enabled = self.get_parameter('thermal_monitoring_enabled').value
        self.power_management_enabled = self.get_parameter('power_management_enabled').value
        
        # Initialize internal state
        self.cv_bridge = CvBridge()
        self.hardware_lock = Lock()
        self.camera_thread = None
        self.imu_thread = None
        self.running = True
        
        # Camera interface initialization
        self.camera = None
        self.camera_publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )
        
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/camera/camera_info',
            10
        )
        
        # IMU interface initialization
        self.imu_publisher = self.create_publisher(
            Imu,
            '/imu',
            10
        )
        
        # Joint state publisher (for feedback from actual hardware)
        self.joint_state_publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        # Hardware status monitoring
        self.status_publisher = self.create_publisher(
            String,
            '/hardware_status',
            10
        )
        
        # Thermal monitoring
        if self.thermal_monitoring_enabled:
            self.thermal_publisher = self.create_publisher(
                Temperature,
                '/jetson_temperature',
                10
            )
            self.thermal_timer = self.create_timer(1.0, self.monitor_thermal)
        
        # Initialize hardware connections
        self.initialize_hardware_interfaces()
        
        self.get_logger().info('Jetson Hardware Interface Initialized')

    def initialize_hardware_interfaces(self):
        """Initialize Jetson-specific hardware interfaces"""
        # Initialize camera
        self.initialize_camera()
        
        # Initialize IMU
        self.initialize_imu()
        
        # Start sensor acquisition threads
        self.start_sensor_threads()

    def initialize_camera(self):
        """Initialize camera interface"""
        if self.camera_interface == 'usb':
            # Initialize USB camera
            try:
                self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
                
                if self.camera.isOpened():
                    self.get_logger().info(f'USB Camera initialized at {self.camera_fps} FPS')
                else:
                    self.get_logger().error('Failed to open USB camera')
                    self.camera = None
            except Exception as e:
                self.get_logger().error(f'Error initializing USB camera: {str(e)}')
                self.camera = None

    def initialize_imu(self):
        """Initialize IMU interface"""
        # In a real implementation, this would initialize the I2C connection to the IMU
        # For simulation, we'll just log the initialization
        self.get_logger().info(f'Initializing IMU via {self.imu_interface}')
        
        # Simulate IMU initialization
        self.imu_initialized = True

    def start_sensor_threads(self):
        """Start sensor acquisition threads"""
        # Start camera acquisition thread
        if self.camera:
            self.camera_thread = threading.Thread(target=self.acquire_camera_data, daemon=True)
            self.camera_thread.start()
        
        # Start IMU acquisition thread
        self.imu_thread = threading.Thread(target=self.acquire_imu_data, daemon=True)
        self.imu_thread.start()

    def acquire_camera_data(self):
        """Acquire camera data in a separate thread"""
        if not self.camera:
            return
            
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    ros_image = self.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = 'camera_link'
                    
                    self.camera_publisher.publish(ros_image)
                    
                    # Publish camera info
                    camera_info = self.create_camera_info()
                    camera_info.header.stamp = ros_image.header.stamp
                    camera_info.header.frame_id = 'camera_link'
                    self.camera_info_publisher.publish(camera_info)
                    
                except Exception as e:
                    self.get_logger().error(f'Error publishing camera data: {str(e)}')
            
            time.sleep(1.0 / self.camera_fps)

    def acquire_imu_data(self):
        """Acquire IMU data in a separate thread"""
        while self.running:
            try:
                # In a real implementation, this would read from the physical IMU
                # For simulation, we'll create synthetic data
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = 'imu_link'
                
                # Create synthetic IMU data (in real hardware, this would come from actual sensor)
                # For this example, we'll simulate stable upright position
                imu_msg.orientation.w = 1.0  # No rotation
                imu_msg.orientation.x = 0.0
                imu_msg.orientation.y = 0.0
                imu_msg.orientation.z = 0.0
                
                # Add some small noise to make it realistic
                from random import random
                imu_msg.angular_velocity.x = (random() - 0.5) * 0.01
                imu_msg.angular_velocity.y = (random() - 0.5) * 0.01
                imu_msg.angular_velocity.z = (random() - 0.5) * 0.01
                
                imu_msg.linear_acceleration.x = (random() - 0.5) * 0.1
                imu_msg.linear_acceleration.y = (random() - 0.5) * 0.1
                # Include gravity component
                imu_msg.linear_acceleration.z = 9.8 + (random() - 0.5) * 0.1
                
                self.imu_publisher.publish(imu_msg)
                
            except Exception as e:
                self.get_logger().error(f'Error publishing IMU data: {str(e)}')
            
            time.sleep(0.01)  # 100Hz for IMU

    def create_camera_info(self):
        """Create camera info message"""
        camera_info = CameraInfo()
        
        # Set camera parameters (these should come from calibration in a real system)
        camera_info.width = 640
        camera_info.height = 480
        
        # Basic calibration parameters
        camera_info.k = [640.0, 0.0, 320.0,  # fx, 0, cx
                        0.0, 640.0, 240.0,  # 0, fy, cy
                        0.0, 0.0, 1.0]      # 0, 0, 1
        camera_info.r = [1.0, 0.0, 0.0,  # Identity
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        camera_info.p = [640.0, 0.0, 320.0, 0.0,  # fx, 0, cx, 0
                        0.0, 640.0, 240.0, 0.0,  # 0, fy, cy, 0
                        0.0, 0.0, 1.0, 0.0]      # 0, 0, 1, 0
        
        return camera_info

    def monitor_thermal(self):
        """Monitor Jetson thermal status"""
        try:
            # Read thermal zones (on Jetson devices)
            thermal_data = []
            for thermal_zone in ['/sys/class/thermal/thermal_zone0', 
                                '/sys/class/thermal/thermal_zone1',
                                '/sys/class/thermal/thermal_zone2']:
                try:
                    with open(f'{thermal_zone}/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000.0  # Convert from millidegrees to degrees
                        type_file = f'{thermal_zone}/type'
                        if os.path.exists(type_file):
                            with open(type_file, 'r') as type_f:
                                zone_type = type_f.read().strip()
                        
                        thermal_reading = Temperature()
                        thermal_reading.header.stamp = self.get_clock().now().to_msg()
                        thermal_reading.header.frame_id = f'thermal_zone_{zone_type}'
                        thermal_reading.temperature = temp
                        thermal_reading.variance = 0.0
                        
                        self.thermal_publisher.publish(thermal_reading)
                        thermal_data.append(f"{zone_type}:{temp:.1f}C")
                        
                except Exception:
                    continue  # Skip unavailable thermal zones
            
            # Determine overall thermal status
            thermal_status = "NORMAL"
            for temp_str in thermal_data:
                temp_val = float(temp_str.split(':')[1].replace('C', ''))
                if temp_val > 80:
                    thermal_status = "CRITICAL"
                elif temp_val > 70 and thermal_status == "NORMAL":
                    thermal_status = "WARNING"
            
            # Publish status
            status_msg = String()
            status_msg.data = f"THERMAL:{thermal_status} | {' | '.join(thermal_data)}"
            self.status_publisher.publish(status_msg)
            
            if thermal_status == "CRITICAL":
                self.get_logger().error(f'CRITICAL THERMAL CONDITIONS: {status_msg.data}')
            elif thermal_status == "WARNING":
                self.get_logger().warn(f'THERMAL WARNING: {status_msg.data}')
            else:
                self.get_logger().debug(f'Thermal status: {status_msg.data}')
                
        except Exception as e:
            self.get_logger().error(f'Error monitoring thermal status: {str(e)}')

    def destroy_node(self):
        """Cleanup hardware before node destruction"""
        self.running = False
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Wait for threads to finish
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        if self.imu_thread:
            self.imu_thread.join(timeout=1.0)
        
        self.get_logger().info('Jetson Hardware Interface Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    hw_interface = JetsonHardwareInterface()
    
    try:
        rclpy.spin(hw_interface)
    except KeyboardInterrupt:
        hw_interface.get_logger().info('Hardware interface interrupted by user')
    finally:
        hw_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Performance and Resource Management

### 1. Resource Optimization for Jetson

Implement resource management for Jetson platforms:

```python
# scripts/jetson_resource_manager.py
#!/usr/bin/env python3

"""
Resource manager for Jetson platforms.
Monitors and manages CPU, GPU, and memory usage for humanoid AI applications.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import BatteryState
import psutil
import subprocess
import os
import time


class JetsonResourceManager(Node):
    """
    Resource manager for Jetson Orin platforms.
    Monitors system resources and manages AI workload based on available resources.
    """
    
    def __init__(self):
        super().__init__('jetson_resource_manager')
        
        # Declare parameters
        self.declare_parameter('monitoring_frequency', 2.0)  # Hz
        self.declare_parameter('cpu_threshold', 85.0)     # Percent
        self.declare_parameter('gpu_threshold', 90.0)     # Percent
        self.declare_parameter('memory_threshold', 90.0)  # Percent
        self.declare_parameter('temperature_threshold', 80.0)  # Celsius
        self.declare_parameter('throttling_enabled', True)  # Enable performance throttling
        
        # Get parameters
        self.monitoring_frequency = self.get_parameter('monitoring_frequency').value
        self.cpu_threshold = self.get_parameter('cpu_threshold').value
        self.gpu_threshold = self.get_parameter('gpu_threshold').value
        self.memory_threshold = self.get_parameter('memory_threshold').value
        self.temperature_threshold = self.get_parameter('temperature_threshold').value
        self.throttling_enabled = self.get_parameter('throttling_enabled').value
        
        # Publishers
        self.resource_status_pub = self.create_publisher(
            String,
            '/jetson_resource_status',
            10
        )
        
        self.resource_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/jetson_resource_metrics',
            10
        )
        
        self.battery_pub = self.create_publisher(
            BatteryState,
            '/battery_state',
            10
        )
        
        # Timer for monitoring
        self.monitor_timer = self.create_timer(
            1.0 / self.monitoring_frequency,
            self.monitor_resources
        )
        
        self.get_logger().info('Jetson Resource Manager Initialized')

    def monitor_resources(self):
        """Monitor Jetson system resources"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Get GPU usage (Jetson-specific)
            gpu_percent, gpu_temp = self.get_gpu_status()
            
            # Get thermal status
            thermal_status = self.get_thermal_status()
            
            # Get battery status (if available)
            battery_status = self.get_battery_status()
            
            # Create resource metrics message
            metrics_msg = Float64MultiArray()
            metrics_msg.data = [
                cpu_percent,
                gpu_percent,
                memory_percent,
                disk_percent,
                thermal_status['max_temp'],
                gpu_temp if gpu_temp else 0.0,
                battery_status['voltage'] if battery_status else 0.0,
                battery_status['percentage'] if battery_status else 0.0
            ]
            self.resource_metrics_pub.publish(metrics_msg)
            
            # Determine overall status
            status_parts = []
            if cpu_percent > self.cpu_threshold:
                status_parts.append(f"CPU_HIGH:{cpu_percent:.1f}%")
            if gpu_percent > self.gpu_threshold:
                status_parts.append(f"GPU_HIGH:{gpu_percent:.1f}%")
            if memory_percent > self.memory_threshold:
                status_parts.append(f"MEMORY_HIGH:{memory_percent:.1f}%")
            if thermal_status['max_temp'] > self.temperature_threshold:
                status_parts.append(f"TEMP_HIGH:{thermal_status['max_temp']:.1f}C")
            
            # Build status message
            status_msg = String()
            if status_parts:
                status_msg.data = f"RESOURCE_WARNING: {' | '.join(status_parts)}"
                
                # Take action based on excessive resource usage
                if self.throttling_enabled:
                    self.take_resource_management_action(status_parts)
            else:
                status_msg.data = f"RESOURCES_OK: CPU:{cpu_percent:.1f}%, GPU:{gpu_percent:.1f}%, MEM:{memory_percent:.1f}%, TEMP:{thermal_status['max_temp']:.1f}C"
            
            self.resource_status_pub.publish(status_msg)
            
            # Log if any resource is concerning
            if status_parts:
                self.get_logger().warn(f'Jetson Resource Issue: {" | ".join(status_parts)}')
            else:
                self.get_logger().debug(f'Jetson Resources OK: {status_msg.data}')
            
        except Exception as e:
            self.get_logger().error(f'Error monitoring resources: {str(e)}')

    def get_gpu_status(self):
        """Get GPU status on Jetson platform"""
        try:
            # For Jetson, we can use nvidia-smi or jetson-stats
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output_parts = result.stdout.strip().split(',')
                gpu_util = float(output_parts[0].strip())
                gpu_temp = float(output_parts[1].strip())
                return gpu_util, gpu_temp
            else:
                return 0.0, 0.0  # If command fails, return defaults
        except Exception:
            return 0.0, 0.0  # If any error, return defaults

    def get_thermal_status(self):
        """Get thermal status from Jetson thermal zones"""
        max_temp = 0.0
        thermal_zones = []
        
        for i in range(10):  # Check first 10 thermal zones
            thermal_path = f'/sys/class/thermal/thermal_zone{i}'
            if os.path.exists(thermal_path):
                try:
                    with open(f'{thermal_path}/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000.0  # Convert from millidegrees
                        max_temp = max(max_temp, temp)
                        
                        with open(f'{thermal_path}/type', 'r') as type_f:
                            zone_type = type_f.read().strip()
                        
                        thermal_zones.append(f"{zone_type}:{temp:.1f}C")
                except Exception:
                    continue
        
        return {
            'zones': thermal_zones,
            'max_temp': max_temp,
            'timestamp': time.time()
        }

    def get_battery_status(self):
        """Get battery status (simplified for Jetson)"""
        # In a real implementation, this would read from actual battery interface
        # For now, we'll simulate battery status
        try:
            # Try to get battery status if available
            result = subprocess.run(['upower', '-i', '/org/freedesktop/UPower/devices/battery_BAT0'], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse battery information (simplified)
                lines = result.stdout.split('\n')
                voltage = 0.0
                percentage = 100.0
                
                for line in lines:
                    if 'voltage:' in line:
                        # Extract voltage value
                        voltage_str = ''.join(filter(str.isdigit, line.replace('.', '')))
                        voltage = float(voltage_str) / 1000.0  # Convert to volts
                    if 'percentage:' in line:
                        # Extract percentage
                        pct_str = ''.join(filter(lambda x: x.isdigit() or x == '.', line))
                        percentage = float(pct_str) if pct_str else 100.0
                
                battery_msg = BatteryState()
                battery_msg.header.stamp = self.get_clock().now().to_msg()
                battery_msg.voltage = voltage
                battery_msg.percentage = percentage
                battery_msg.charge = 1.0  # Simplified
                battery_msg.capacity = 1.0  # Simplified
                battery_msg.design_capacity = 1.0  # Simplified
                battery_msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING  # Simplified
                battery_msg.present = True  # Simplified
                
                self.battery_pub.publish(battery_msg)
                
                return {
                    'voltage': voltage,
                    'percentage': percentage,
                    'timestamp': time.time()
                }
        except Exception:
            # Return default values if battery monitoring fails
            pass
        
        return None

    def take_resource_management_action(self, issues):
        """Take actions when resources are being overutilized"""
        for issue in issues:
            if "CPU_HIGH" in issue:
                self.reduce_cpu_intensive_tasks()
            elif "GPU_HIGH" in issue:
                self.reduce_gpu_intensive_tasks()
            elif "MEMORY_HIGH" in issue:
                self.trigger_memory_cleanup()
            elif "TEMP_HIGH" in issue:
                self.reduce_computation_intensity()

    def reduce_cpu_intensive_tasks(self):
        """Reduce CPU-intensive processes"""
        self.get_logger().info('Reducing CPU-intensive tasks to manage resources')
        
        # Example: Reduce perception pipeline update rate
        # In a real system, this would send commands to reduce processing rate
        pass

    def reduce_gpu_intensive_tasks(self):
        """Reduce GPU-intensive processes"""
        self.get_logger().info('Reducing GPU-intensive tasks to manage resources')
        
        # Example: Reduce model precision or processing rate
        # In a real system, this might switch to lower-resolution models
        pass

    def trigger_memory_cleanup(self):
        """Trigger memory cleanup tasks"""
        self.get_logger().info('Triggering memory cleanup')
        
        # Example: Clear cached data
        # In a real system, this would clear temporary data
        pass

    def reduce_computation_intensity(self):
        """Reduce overall computational intensity"""
        self.get_logger().info('Reducing computational intensity to manage temperature')
        
        # Example: Reduce AI model update rate, simplify perception algorithms
        pass

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Jetson Resource Manager Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    resource_manager = JetsonResourceManager()
    
    try:
        rclpy.spin(resource_manager)
    except KeyboardInterrupt:
        resource_manager.get_logger().info('Resource manager interrupted by user')
    finally:
        resource_manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Deployment Validation

### 1. Complete Deployment Verification

Create a script to verify the complete deployment works properly:

```python
# scripts/deploy_validator.py
#!/usr/bin/env python3

"""
Deployment validator for Jetson platform.
Verifies that all components work correctly in the Jetson environment.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import time


class DeploymentValidatorNode(Node):
    """
    Validates that the entire humanoid AI pipeline works correctly on Jetson.
    Checks that each component is operating within acceptable parameters.
    """
    
    def __init__(self):
        super().__init__('deployment_validator_node')
        
        # Declare parameters
        self.declare_parameter('validation_frequency', 5.0)  # Hz
        self.declare_parameter('test_duration', 60.0)      # seconds
        self.declare_parameter('critical_components', [
            'hardware_interface',
            'perception_pipeline',
            'ai_planner',
            'control_pipeline'
        ])
        
        # Get parameters
        self.validation_frequency = self.get_parameter('validation_frequency').value
        self.test_duration = self.get_parameter('test_duration').value
        self.critical_components = self.get_parameter('critical_components').value
        
        # Component status tracking
        self.component_status = {comp: False for comp in self.critical_components}
        self.component_last_seen = {comp: 0 for comp in self.critical_components}
        
        # Validation data
        self.validation_start_time = self.get_clock().now()
        self.validation_results = {}
        
        # Subscriptions to key topics
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            '/hardware_status',
            self.status_callback,
            10
        )
        
        self.resource_sub = self.create_subscription(
            String,
            '/jetson_resource_status',
            self.resource_callback,
            10
        )
        
        # Validation result publisher
        self.validation_result_pub = self.create_publisher(
            String,
            '/deployment_validation_result',
            10
        )
        
        # Timer for validation
        self.validation_timer = self.create_timer(
            1.0 / self.validation_frequency,
            self.run_validation_cycle
        )
        
        self.get_logger().info('Deployment Validator Node Initialized')

    def image_callback(self, msg):
        """Track image reception from camera"""
        self.component_status['camera'] = True
        self.component_last_seen['camera'] = self.get_clock().now().nanoseconds / 1e9

    def imu_callback(self, msg):
        """Track IMU reception"""
        self.component_status['imu'] = True
        self.component_last_seen['imu'] = self.get_clock().now().nanoseconds / 1e9

    def status_callback(self, msg):
        """Track status messages"""
        # Update status of the component mentioned in the message
        if "HARDWARE" in msg.data:
            self.component_status['hardware_interface'] = True
            self.component_last_seen['hardware_interface'] = self.get_clock().now().nanoseconds / 1e9

    def resource_callback(self, msg):
        """Track resource status"""
        if "RESOURCES" in msg.data:
            self.component_status['resource_manager'] = True
            self.component_last_seen['resource_manager'] = self.get_clock().now().nanoseconds / 1e9

    def run_validation_cycle(self):
        """Run validation cycle"""
        # Check component status
        self.update_component_status()
        
        # Generate validation report
        validation_report = self.generate_validation_report()
        
        # Publish report
        report_msg = String()
        report_msg.data = validation_report
        self.validation_result_pub.publish(report_msg)
        
        # Log validation status
        self.get_logger().info(f'Deployment Validation: {validation_report}')
        
        # Check if test duration has been reached
        elapsed_time = (self.get_clock().now() - self.validation_start_time).nanoseconds / 1e9
        if elapsed_time > self.test_duration:
            self.completion_validation()
            self.validation_timer.cancel()

    def update_component_status(self):
        """Update status of all components"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for comp in self.critical_components:
            # If we haven't seen the component in 5 seconds, mark as inactive
            if current_time - self.component_last_seen[comp] > 5.0:
                self.component_status[comp] = False

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        active_components = [comp for comp, status in self.component_status.items() if status]
        inactive_components = [comp for comp, status in self.component_status.items() if not status]
        
        report_parts = []
        report_parts.append(f"COMPONENTS_ACTIVE:{len(active_components)}/{len(self.component_status)}")
        
        if active_components:
            report_parts.append(f"ACTIVE:[{','.join(active_components)}]")
        
        if inactive_components:
            report_parts.append(f"INACTIVE:[{','.join(inactive_components)}]")
        
        # Calculate validation score
        total_components = len(self.component_status)
        active_count = len(active_components)
        validation_score = active_count / total_components if total_components > 0 else 0.0
        
        report_parts.append(f"SCORE:{validation_score:.2f}")
        
        # Determine overall status
        if validation_score >= 1.0:
            overall_status = "DEPLOYMENT_COMPLETE"
        elif validation_score >= 0.8:
            overall_status = "DEPLOYMENT_PARTIAL"
        elif validation_score >= 0.5:
            overall_status = "DEPLOYMENT_INCOMPLETE"
        else:
            overall_status = "DEPLOYMENT_FAILED"
        
        report_parts.insert(0, overall_status)
        
        return " | ".join(report_parts)

    def completion_validation(self):
        """Final validation when test is complete"""
        final_report = self.generate_validation_report()
        
        # Determine final assessment
        active_count = len([s for s in self.component_status.values() if s])
        total_count = len(self.component_status)
        
        if active_count == total_count:
            status = "SUCCESS"
            message = f"Full deployment validated: {active_count}/{total_count} components active"
        elif active_count >= total_count * 0.8:
            status = "PARTIAL_SUCCESS"
            message = f"Deployment mostly successful: {active_count}/{total_count} components active"
        else:
            status = "FAILED"
            message = f"Deployment failed: {active_count}/{total_count} components active"
        
        final_message = f"VALIDATION_COMPLETE_{status}: {message}"
        
        self.get_logger().info(f'Final Validation Result: {final_message}')
        
        # Publish final result
        result_msg = String()
        result_msg.data = final_message
        self.validation_result_pub.publish(result_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Deployment Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = DeploymentValidatorNode()
    
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

## Deployment Configuration

### 1. Jetson Deployment Manifest

Create a deployment manifest for Jetson systems:

```yaml
# jetson_deployment/deployment_manifest.yaml
deployment_version: 1.0
target_platform: "nvidia_jetson_orin"
robot_model: "simple_humanoid"
ros_distro: "iron"

components:
  perception_system:
    package: "humanoid_perception"
    nodes:
      - name: "jetson_perception_pipeline"
        cpu_affinity: [2, 3, 4, 5]  # Use specific CPU cores
        memory_reservation: "2GB"
        gpu_reservation: "high"
        parameters:
          model_precision: "fp16"
          update_rate: 30.0
          enable_gpu_processing: true
  
  ai_planning:
    package: "humanoid_navigation"
    nodes:
      - name: "jetson_ai_planner"
        cpu_affinity: [0, 1]
        memory_reservation: "1GB"
        gpu_reservation: "medium"
        parameters:
          planning_frequency: 2.0
          model_path: "/opt/models/navigation_plan_fp16.trt"
          gpu_acceleration: true

  control_system:
    package: "humanoid_control"
    nodes:
      - name: "jetson_control_pipeline"
        cpu_affinity: [6, 7]
        memory_reservation: "1GB"
        real_time_priority: 50
        parameters:
          control_frequency: 100.0
          enable_gpu_processing: true
  
  hardware_interface:
    package: "humanoid_hardware_interface"
    nodes:
      - name: "jetson_hardware_interface"
        cpu_affinity: [0]
        memory_reservation: "512MB"
        real_time_priority: 60
        parameters:
          camera_fps: 30
          thermal_monitoring_enabled: true

resources:
  cpu_cores: [0, 1, 2, 3, 4, 5, 6, 7]
  memory_limit: "6GB"
  gpu_memory_limit: "4GB"
  
performance_settings:
  # Optimize for inference on Jetson
  tensorrt_precision: "fp16"
  cuda_context: "shared"
  multi_threading: true
  pipeline_parallelization: true

deployment_instructions:
  - "Flash JetPack 5.1+ on target Jetson device"
  - "Install ROS 2 Iron following official instructions"
  - "Run 'rosdep install --from-paths src --ignore-src -r -y'"
  - "Build with 'colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release'"
  - "Launch with 'ros2 launch humanoid_jetson_bringup humanoid_jetson.launch.py'"

startup_sequence:
  - "jetson_hardware_interface"
  - "robot_state_publisher"
  - "jetson_perception_pipeline"
  - "jetson_ai_planner"
  - "jetson_control_pipeline"
  - "jetson_resource_manager"
```

## Monitoring and Maintenance

### 1. Creating a Deployment Status Monitor

```bash
# jetson_deployment/deploy_monitor.sh
#!/bin/bash

# Deployment monitoring script for Jetson humanoid robot

LOG_FILE="/var/log/humanoid_deploy.log"

echo "$(date): Starting Jetson deployment monitoring..." | tee -a $LOG_FILE

# Function to check ROS 2 nodes
check_ros_nodes() {
    echo "Checking ROS 2 nodes:"
    ros2 node list | grep humanoid
}

# Function to check system resources
check_resources() {
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)"
    
    echo "Memory Usage:"
    free -h
    
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    
    echo "Temperature:"
    cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | while read temp; do 
        echo "  $(($temp/1000))°C"
    done
}

# Function to check critical services
check_services() {
    echo "Checking critical services:"
    
    # Check if main ROS processes are running
    if pgrep -f "ros2.*launch.*humanoid"; then
        echo "  ✓ Humanoid processes running"
    else
        echo "  ✗ Humanoid processes not running"
    fi
    
    # Check if camera is accessible
    if ls /dev/video0 >/dev/null 2>&1; then
        echo "  ✓ Camera accessible"
    else
        echo "  ✗ Camera not accessible"
    fi
}

# Continuous monitoring
while true; do
    echo "--- $(date) ---" | tee -a $LOG_FILE
    
    check_resources | tee -a $LOG_FILE
    check_services | tee -a $LOG_FILE
    
    # Log to journalctl as well
    check_ros_nodes > /tmp/ros_nodes_status
    if [ -s /tmp/ros_nodes_status ]; then
        echo "Active ROS nodes:" | systemd-cat -t humanoid-deploy -p info
        cat /tmp/ros_nodes_status | systemd-cat -t humanoid-deploy -p info
    else
        echo "No ROS nodes active!" | systemd-cat -t humanoid-deploy -p warning
    fi
    
    sleep 10
done
```

## Best Practices for Jetson Deployment

### 1. Resource Management

- **GPU Memory**: Monitor GPU memory usage closely and optimize models accordingly
- **Thermal Management**: Implement automatic throttling when temperatures exceed safe limits
- **Power Management**: Consider power constraints when planning computations
- **Real-time Priority**: Set appropriate real-time priorities for control systems

### 2. Model Optimization

- Always optimize models with TensorRT for Jetson deployment
- Consider precision trade-offs: FP16 for speed vs. FP32 for accuracy
- Profile your specific use case to find optimal model-accuracy tradeoffs

### 3. Deployment Testing

- Test in controlled environments before complex deployments
- Monitor resource usage during operation
- Implement graceful degradation when resources are constrained

## Next Steps

With the complete Jetson deployment workflow documented, you now have the foundational knowledge to successfully deploy your simulation-trained humanoid robotics system to real Jetson hardware. The workflow includes hardware abstraction, resource optimization, and validation techniques essential for reliable deployment.

The next step is to explore bipedal humanoid motion planning, which builds on the perception, navigation, and control foundations you've established.