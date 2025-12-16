# Isaac ROS Perception Stack Documentation

## Overview

The Isaac ROS perception stack provides a set of hardware-accelerated perception packages that leverage NVIDIA's GPUs for efficient processing of sensor data in robotic applications. This stack is particularly valuable for humanoid robotics, where real-time perception of the environment is critical for navigation, manipulation, and interaction.

## Isaac ROS Perception Components

### 1. Core Perception Packages

The Isaac ROS perception stack includes several specialized packages:

#### Isaac ROS Apriltag
- **Purpose**: Real-time fiducial marker detection
- **GPU Acceleration**: CUDA-based AprilTag detection
- **Use Cases**: Robot localization, object pose estimation, calibration

#### Isaac ROS DNN Inference
- **Purpose**: Deep neural network inference for perception tasks
- **GPU Acceleration**: TensorRT optimization for faster inference
- **Use Cases**: Object detection, semantic segmentation, classification

#### Isaac ROS Visual SLAM
- **Purpose**: Visual simultaneous localization and mapping
- **GPU Acceleration**: GPU-accelerated tracking and mapping
- **Use Cases**: Indoor navigation, map building

#### Isaac ROS Stereo DNN
- **Purpose**: Stereo vision-based deep learning inference
- **GPU Acceleration**: GPU-accelerated stereo matching and DNN inference
- **Use Cases**: 3D object detection, depth estimation

#### Isaac ROS Stereo Image Proc
- **Purpose**: Stereo image processing and rectification
- **GPU Acceleration**: GPU-accelerated stereo rectification
- **Use Cases**: Stereo camera calibration, disparity map generation

#### Isaac ROS OAK
- **Purpose**: Integration with StereoLabs OAK devices
- **GPU Acceleration**: GPU-accelerated processing of OAK device data
- **Use Cases**: Depth sensing, spatial AI

### 2. Installation

#### Installing Isaac ROS Packages

```bash
# Update package list
sudo apt update

# Install Isaac ROS packages
sudo apt install ros-iron-isaac-ros-* ros-iron-isaac-ros-perception

# For specific packages:
sudo apt install \
  ros-iron-isaac-ros-apriltag \
  ros-iron-isaac-ros-dnn-inference \
  ros-iron-isaac-ros-visual-slame \
  ros-iron-isaac-ros-stereo-image-proc \
  ros-iron-isaac-ros-bitbots
```

#### Verifying Installation

```bash
# Check that Isaac ROS packages are installed
dpkg -l | grep isaac-ros

# Verify Isaac ROS nodes are available
find /opt/ros/iron/lib/ -name "*isaac*"
```

## 3. Isaac ROS Apriltag Integration

### 3.1. Understanding Apriltags

Apriltags are visual fiducial markers that provide precise 6DOF pose estimation. They're particularly useful for humanoid robots for:

- Accurate localization in known environments
- Calibration of sensors and actuators
- Reference points for navigation and manipulation tasks

### 3.2. Configuration Example

Create a launch file for Apriltag detection: `launch/apriltag_detection.launch.py`

```python
# launch/apriltag_detection.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    apriltag_family = LaunchConfiguration('apriltag_family', default='tag36h11')
    
    # Isaac ROS Apriltag node
    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='isaac_ros_apriltag_exe',
        name='apriltag_node',
        parameters=[
            {
                'family': apriltag_family,
                'max_tags': 64,
                'tag_layout_file': PathJoinSubstitution([
                    FindPackageShare('humanoid_simple_robot'),
                    'config',
                    'apriltag_layout.json'
                ]),
                'publish_tag_detections_image': True,
                'publish_approximate_sync': True,
                'approximate_sync': True,
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('image', [camera_namespace, '/image_rect']),
            ('camera_info', [camera_namespace, '/camera_info']),
            ('tag_detections', 'apriltag_detections'),
            ('tag_detections_image', 'tag_detections_image')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='/camera',
            description='Namespace for camera topics'
        ),
        
        DeclareLaunchArgument(
            'apriltag_family',
            default_value='tag36h11',
            description='AprilTag family to use'
        ),
        
        # Launch Apriltag node
        apriltag_node,
    ])
```

### 3.3. Apriltag Configuration File

Create `config/apriltag_layout.json`:

```json
{
  "layout": [
    {
      "id": 0,
      "size": 0.16,  /* Tag size in meters */
      "x": 0.0,      /* X position in meters */
      "y": 0.0,      /* Y position in meters */
      "z": 1.0,      /* Z position in meters */
      "qw": 1.0,     /* Quaternion w */
      "qx": 0.0,     /* Quaternion x */
      "qy": 0.0,     /* Quaternion y */
      "qz": 0.0      /* Quaternion z */
    },
    {
      "id": 1,
      "size": 0.16,
      "x": 1.0,
      "y": 0.0,
      "z": 1.0,
      "qw": 1.0,
      "qx": 0.0,
      "qy": 0.0,
      "qz": 0.0
    },
    {
      "id": 2,
      "size": 0.16,
      "x": 0.0,
      "y": 1.0,
      "z": 1.0,
      "qw": 0.707,
      "qx": 0.0,
      "qy": 0.0,
      "qz": 0.707
    }
  ]
}
```

## 4. Isaac ROS DNN Inference Pipeline

### 4.1. Understanding DNN Inference in Isaac ROS

The Isaac ROS DNN inference pipeline provides GPU-accelerated neural network inference for perception tasks. It includes:

- TensorRT optimization for faster inference
- Efficient preprocessing and postprocessing
- Integration with ROS 2 message types
- Support for various model formats (ONNX, TensorRT engine)

### 4.2. DNN Inference Configuration

Create a DNN inference launch file: `launch/dnn_inference.launch.py`

```python
# launch/dnn_inference.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    engine_file_path = LaunchConfiguration('engine_file_path', 
                                          default='/tmp/resnet50_plan.plan')
    input_tensor_name = LaunchConfiguration('input_tensor_name', default='input')
    output_tensor_name = LaunchConfiguration('output_tensor_name', default='output')
    input_image_width = LaunchConfiguration('input_image_width', default='224')
    input_image_height = LaunchConfiguration('input_image_height', default='224')
    
    # Isaac ROS DNN Inference node
    dnn_inference_node = Node(
        package='isaac_ros_dnn_inference',
        executable='isaac_ros_dnn_inference_encoder_tensor_rt',
        name='dnn_inference_node',
        parameters=[
            {
                'model_file_path': engine_file_path,
                'input_tensor_names': [input_tensor_name],
                'output_tensor_names': [output_tensor_name],
                'input_binding_names': [input_tensor_name],
                'output_binding_names': [output_tensor_name],
                'max_batch_size': 1,
                'input_tensor_formats': ['nitros_tensor_list_nchw'],
                'output_tensor_formats': ['nitros_tensor_list_nhwc'],
                'use_sim_time': use_sim_time,
                'image_mean': [0.0, 0.0, 0.0],
                'image_stddev': [1.0, 1.0, 1.0]
            }
        ],
        remappings=[
            ('image', 'input_image'),
            ('tensor', 'tensor_output')
        ],
        output='screen'
    )
    
    # Image to Nitros bridge node to convert ROS Image to Isaac ROS format
    image_converter_node = Node(
        package='isaac_ros_image_proc',
        executable='image_format_converter_node',
        name='image_converter',
        parameters=[
            {
                'encoding_desired': 'rgb8',
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('image', 'input_image'),
            ('image_out', 'converted_image')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='/tmp/resnet50_plan.plan',
            description='Path to the TensorRT engine file'
        ),
        
        DeclareLaunchArgument(
            'input_tensor_name',
            default_value='input',
            description='Name of the input tensor'
        ),
        
        DeclareLaunchArgument(
            'output_tensor_name',
            default_value='output',
            description='Name of the output tensor'
        ),
        
        DeclareLaunchArgument(
            'input_image_width',
            default_value='224',
            description='Width of input image'
        ),
        
        DeclareLaunchArgument(
            'input_image_height',
            default_value='224',
            description='Height of input image'
        ),
        
        # Launch nodes
        image_converter_node,
        dnn_inference_node
    ])
```

### 4.3. Object Detection Pipeline

For object detection tasks, create a more specific pipeline: `launch/object_detection.launch.py`

```python
# launch/object_detection.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    model_file_path = LaunchConfiguration('model_file_path', 
                                         default='/tmp/yolov5_plan.plan')
    
    # Isaac ROS DNN Inference for object detection
    object_detection_node = Node(
        package='isaac_ros_dnn_inference',
        executable='isaac_ros_dnn_inference_tensor_rt',
        name='object_detection_node',
        parameters=[
            {
                'model_file_path': model_file_path,
                'input_tensor_name': 'input',
                'output_tensor_names': ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'],
                'image_mean': [0.0, 0.0, 0.0],
                'image_stddev': [1.0, 1.0, 1.0],
                'threshold': 0.5,
                'top_k': 100,
                'max_batch_size': 1,
                'input_binding_names': ['input'],
                'output_binding_names': ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'],
                'input_tensor_formats': ['nitros_tensor_list_nchw'],
                'output_tensor_formats': ['nitros_tensor_list_nhwc'],
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('image', 'input_image'),
            ('detections', 'object_detections')
        ],
        output='screen'
    )
    
    # Post-processing node to convert detection output to ROS format
    detection_postprocessor_node = Node(
        package='isaac_ros_detectnet',
        executable='isaac_ros_detectnet',
        name='detection_postprocessor',
        parameters=[
            {
                'model_name': 'yolov5',
                'max_batch_size': 1,
                'input_format': 'nitros_tensor_list_nchw',
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('detections', 'object_detections'),
            ('image_size', 'input_image_size'),
            ('detection_image', 'detection_viz')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'model_file_path',
            default_value='/tmp/yolov5_plan.plan',
            description='Path to the TensorRT engine file for detection model'
        ),
        
        object_detection_node,
        detection_postprocessor_node
    ])
```

## 5. Isaac ROS Visual SLAM

### 5.1. Understanding Visual SLAM

Visual SLAM (Simultaneous Localization and Mapping) allows humanoid robots to build maps of their environment while simultaneously determining their position within that map. The Isaac ROS Visual SLAM package provides GPU-accelerated visual SLAM capabilities.

### 5.2. Visual SLAM Configuration

Create `launch/visual_slam.launch.py`:

```python
# launch/visual_slam.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    rectify_stereo = LaunchConfiguration('rectify_stereo', default='true')
    enable_fisheye = LaunchConfiguration('enable_fisheye', default='false')
    enable_rectification = LaunchConfiguration('enable_rectification', default='true')
    enable_debug = LaunchConfiguration('enable_debug', default='false')
    
    # Isaac ROS Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_rectification': enable_rectification,
            'rectified_images_input': rectify_stereo,
            'enable_debug_mode': enable_debug,
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'input_voxel': '0.01',
            'cutoff_distance': 1.0,
            'max_duration_seconds': 10.0,
            'max_distance': 1.0,
            'minimum_keyframe_travel_distance': 0.5,
            'minimum_keyframe_rotation': 0.261799,  # 15 degrees in radians
        }],
        remappings=[
            ('/stereo_camera/left/image', '/camera/left/image_rect_color'),
            ('/stereo_camera/right/image', '/camera/right/image_rect_color'),
            ('/stereo_camera/left/camera_info', '/camera/left/camera_info'),
            ('/stereo_camera/right/camera_info', '/camera/right/camera_info'),
            ('visual_slam/pose', 'visual_slam/pose_graph/pose'),
            ('visual_slam/imu', 'visual_slam/imu/data'),
        ],
        output='screen'
    )
    
    # If using fisheye cameras
    fisheye_stereo_rectifier = Node(
        condition=IfCondition(enable_fisheye),
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_fisheye_stereo_rectify_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'alpha': 0.0,  # For full cropping of invalid pixels
            'fisheye_model_type': 'equidistant',
        }],
        remappings=[
            ('left/image_raw', '/camera/left/image_raw'),
            ('left/camera_info', '/camera/left/camera_info'),
            ('right/image_raw', '/camera/right/image_raw'),
            ('right/camera_info', '/camera/right/camera_info'),
            ('left/image_rect', '/camera/left/image_rect'),
            ('right/image_rect', '/camera/right/image_rect'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'rectify_stereo',
            default_value='true',
            description='Rectify stereo images'
        ),
        
        DeclareLaunchArgument(
            'enable_fisheye',
            default_value='false',
            description='Enable fisheye stereo rectification'
        ),
        
        DeclareLaunchArgument(
            'enable_rectification',
            default_value='true',
            description='Enable image rectification'
        ),
        
        DeclareLaunchArgument(
            'enable_debug',
            default_value='false',
            description='Enable debug mode'
        ),
        
        visual_slam_node,
        fisheye_stereo_rectifier
    ])
```

## 6. Isaac ROS Stereo Pipeline

### 6.1. Understanding Stereo Vision

Stereo vision allows humanoid robots to perceive depth by using two cameras separated by a baseline distance. The Isaac ROS stereo pipeline provides GPU-accelerated stereo rectification and disparity computation.

### 6.2. Stereo Pipeline Configuration

Create `launch/stereo_vision.launch.py`:

```python
# launch/stereo_vision.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    max_disparity = LaunchConfiguration('max_disparity', default='64')
    disparity_processor = LaunchConfiguration('disparity_processor', default='center')
    
    # Isaac ROS Stereo Image Proc node for rectification
    stereo_rectification_node = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_stereo_rectify_node',
        name='stereo_rectification',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'alpha': 0.0,  # 0 for full cropping, 1 for no cropping
                'max_disparity': max_disparity,
            }
        ],
        remappings=[
            ('left/image_raw', 'left/image_raw'),
            ('left/camera_info', 'left/camera_info'),
            ('right/image_raw', 'right/image_raw'),
            ('right/camera_info', 'right/camera_info'),
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
        ],
        output='screen'
    )
    
    # Isaac ROS Disparity node for computing depth
    disparity_node = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_disparity_node',
        name='disparity_node',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'min_disparity': 0,
                'max_disparity': max_disparity,
                'disparity_processor': disparity_processor,
                'prefilter_size': 9,
                'prefilter_cap': 63,
                'correlation_window_size': 15,
                'disp12_max_diff': 1,
                'uniqueness_ratio': 15,
                'speckle_range': 3,
                'speckle_window': 0,
            }
        ],
        remappings=[
            ('left/image_rect', 'left/image_rect'),
            ('right/image_rect', 'right/image_rect'),
            ('left/camera_info', 'left/camera_info'),
            ('right/camera_info', 'right/camera_info'),
            ('disparity', 'disparity_output'),
        ],
        output='screen'
    )
    
    # Point cloud generation from disparity
    point_cloud_node = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_point_cloud_node',
        name='point_cloud_generator',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'queue_size': 1,
            }
        ],
        remappings=[
            ('left/image_rect', 'left/image_rect'),
            ('left/camera_info', 'left/camera_info'),
            ('disparity', 'disparity_output'),
            ('points', 'point_cloud'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'max_disparity',
            default_value='64',
            description='Maximum disparity for stereo computation'
        ),
        
        DeclareLaunchArgument(
            'disparity_processor',
            default_value='center',
            description='Disparity computation method'
        ),
        
        stereo_rectification_node,
        disparity_node,
        point_cloud_node
    ])
```

## 7. Integration with Humanoid Robot Perception

### 7.1. Multi-Sensor Fusion Configuration

For humanoid robots, perception often requires fusing data from multiple sensors. Create a fusion configuration: `launch/perception_fusion.launch.py`

```python
# launch/perception_fusion.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Launch multiple perception nodes together
    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='isaac_ros_apriltag_exe',
        name='apriltag_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('image', '/camera/image_rect'),
            ('camera_info', '/camera/camera_info'),
        ],
        output='screen'
    )
    
    dnn_inference_node = Node(
        package='isaac_ros_dnn_inference',
        executable='isaac_ros_dnn_inference_encoder_tensor_rt',
        name='dnn_inference_node',
        parameters=[
            {
                'model_file_path': '/tmp/yolov5_plan.plan',
                'input_tensor_names': ['input'],
                'output_tensor_names': ['output'],
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            ('image', '/camera/image_rect'),
            ('tensor', 'dnn_tensor_output')
        ],
        output='screen'
    )
    
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='visual_slam_node',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'enable_slam_visualization': True,
                'map_frame': 'map',
                'odom_frame': 'odom',
                'base_frame': 'base_link',
            }
        ],
        remappings=[
            ('/stereo_camera/left/image', '/stereo_camera/left/image_rect'),
            ('/stereo_camera/right/image', '/stereo_camera/right/image_rect'),
        ],
        output='screen'
    )
    
    # Perception aggregator node to combine outputs
    perception_aggregator_node = Node(
        package='humanoid_simple_robot',
        executable='perception_aggregator',
        name='perception_aggregator',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        # Launch all perception nodes
        apriltag_node,
        dnn_inference_node,
        visual_slam_node,
        perception_aggregator_node
    ])
```

### 7.2. Perception Aggregator Node

Create `scripts/perception_aggregator.py`:

```python
#!/usr/bin/env python3

"""
Perception aggregator for combining multiple sensor inputs into unified perception outputs.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np


class PerceptionAggregator(Node):
    def __init__(self):
        super().__init__('perception_aggregator')
        
        # Use_sim_time parameter
        self.declare_parameter('use_sim_time', False)
        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        
        # Subscriptions for different perception outputs
        self.apriltag_sub = self.create_subscription(
            String,  # Simplified - actual message type would be different
            '/apriltag_detections',
            self.apriltag_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            String,  # Simplified - actual would be a detection message type
            '/object_detections',
            self.detection_callback,
            10
        )
        
        self.slam_pose_sub = self.create_subscription(
            PoseStamped,
            '/visual_slam/pose',
            self.slam_pose_callback,
            10
        )
        
        # Publishers for aggregated perception
        self.perception_map_pub = self.create_publisher(
            MarkerArray,
            '/perception_map',
            10
        )
        
        self.environment_status_pub = self.create_publisher(
            String,
            '/environment_status',
            10
        )
        
        # Internal state
        self.apriltag_detections = []
        self.object_detections = []
        self.robot_pose = None
        
        # Timer for publishing aggregated perception
        self.aggregation_timer = self.create_timer(0.1, self.aggregate_perception)
        
        self.get_logger().info('Perception Aggregator Node Initialized')

    def apriltag_callback(self, msg):
        """Process AprilTag detections"""
        # This would parse the actual AprilTag detection message
        # For now, store as placeholder
        self.apriltag_detections.append({
            'timestamp': self.get_clock().now(),
            'data': msg.data
        })
        
        # Keep only recent detections
        self.cleanup_old_detections(self.apriltag_detections)

    def detection_callback(self, msg):
        """Process object detections"""
        # This would parse the actual object detection message
        self.object_detections.append({
            'timestamp': self.get_clock().now(),
            'data': msg.data
        })
        
        # Keep only recent detections
        self.cleanup_old_detections(self.object_detections)

    def slam_pose_callback(self, msg):
        """Process SLAM pose estimates"""
        self.robot_pose = msg

    def cleanup_old_detections(self, detection_list):
        """Remove old detections to prevent memory buildup"""
        current_time = self.get_clock().now()
        
        # Remove detections older than 1 second
        detection_list[:] = [
            det for det in detection_list 
            if (current_time - det['timestamp']).nanoseconds < 1e9
        ]

    def aggregate_perception(self):
        """Aggregate all perception data into unified output"""
        if not self.robot_pose:
            return  # Need robot pose for aggregation
        
        # Create aggregated perception output
        perception_output = {
            'robot_pose': self.robot_pose,
            'apriltags': self.get_recent_apriltags(),
            'objects': self.get_recent_objects(),
            'timestamp': self.get_clock().now()
        }
        
        # Create visualization markers
        marker_array = self.create_perception_markers(perception_output)
        self.perception_map_pub.publish(marker_array)
        
        # Create environment status message
        status_msg = String()
        status_msg.data = self.create_environment_status(perception_output)
        self.environment_status_pub.publish(status_msg)
        
        # Log perception summary
        self.get_logger().info(f'Aggregated perception: {status_msg.data}')

    def get_recent_apriltags(self):
        """Get recently detected AprilTags"""
        return [det for det in self.apriltag_detections 
                if (self.get_clock().now() - det['timestamp']).nanoseconds < 5e8]  # 0.5 seconds

    def get_recent_objects(self):
        """Get recently detected objects"""
        return [det for det in self.object_detections 
                if (self.get_clock().now() - det['timestamp']).nanoseconds < 5e8]  # 0.5 seconds

    def create_perception_markers(self, perception_data):
        """Create visualization markers for perception data"""
        marker_array = MarkerArray()
        
        # Create markers for AprilTags
        for i, tag_detection in enumerate(perception_data['apriltags']):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = perception_data['timestamp'].to_msg()
            marker.ns = "apriltags"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Extract position from perception data (would need proper parsing in full implementation)
            marker.pose.position.x = 1.0  # Placeholder
            marker.pose.position.y = 1.0  # Placeholder
            marker.pose.position.z = 1.0  # Placeholder
            marker.pose.orientation.w = 1.0
            
            marker.scale.z = 0.2  # Text size
            marker.color.a = 1.0  # Alpha
            marker.color.r = 1.0  # Red
            marker.color.g = 1.0  # Green
            marker.color.b = 0.0  # Blue
            
            marker.text = f"AprilTag: {i}"
            
            marker_array.markers.append(marker)
        
        # Create markers for detected objects
        for i, obj_detection in enumerate(perception_data['objects']):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = perception_data['timestamp'].to_msg()
            marker.ns = "objects"
            marker.id = i + 1000  # Different ID range
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Extract position and size from perception data (would need proper parsing)
            marker.pose.position.x = 2.0  # Placeholder
            marker.pose.position.y = 2.0  # Placeholder
            marker.pose.position.z = 1.0  # Placeholder
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3  # Object size
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 0.7  # Alpha
            marker.color.r = 0.0  # Blue
            marker.color.g = 0.0
            marker.color.b = 1.0
            
            marker_array.markers.append(marker)
        
        return marker_array

    def create_environment_status(self, perception_data):
        """Create environment status string"""
        status_parts = []
        
        # Add AprilTag information
        if perception_data['apriltags']:
            status_parts.append(f"AprilTags: {len(perception_data['apriltags'])}")
        
        # Add object detection information
        if perception_data['objects']:
            status_parts.append(f"Objects: {len(perception_data['objects'])}")
        
        # Add pose information
        if perception_data['robot_pose']:
            status_parts.append(f"At: ({perception_data['robot_pose'].pose.position.x:.2f}, {perception_data['robot_pose'].pose.position.y:.2f})")
        
        return ", ".join(status_parts) if status_parts else "No perception data"

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Perception Aggregator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    aggregator = PerceptionAggregator()
    
    try:
        rclpy.spin(aggregator)
    except KeyboardInterrupt:
        aggregator.get_logger().info('Node interrupted by user')
    finally:
        aggregator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 8. Performance Considerations

### 8.1. Optimizing Isaac ROS Perception

The Isaac ROS perception stack is computationally intensive. Here are optimization strategies:

1. **Use appropriate hardware**: Ensure NVIDIA GPU with TensorRT support
2. **Optimize model sizes**: Use quantized or pruned models for faster inference
3. **Pipeline optimization**: Use Isaac ROS Nitros for optimized data transport
4. **Parameter tuning**: Adjust parameters based on your specific requirements

### 8.2. Resource Management

```python
# Example resource management for perception nodes
def configure_perception_resources(node_name, gpu_memory_fraction=0.5, cpu_threads=4):
    """
    Configure resource allocation for perception nodes
    """
    # This would use Isaac ROS Nitros for optimized resource allocation
    # In practice, GPU memory management is handled differently in Isaac ROS
    pass
```

## 9. Troubleshooting

### 9.1. Common Issues

1. **CUDA/TensorRT Not Found**
   - Verify NVIDIA GPU drivers are correctly installed
   - Check that CUDA and TensorRT are properly installed and linked

2. **Performance Issues**
   - Reduce image resolution
   - Lower update rates
   - Use simpler neural network models

3. **Synchronization Issues**
   - Verify timestamp synchronization between cameras
   - Check that all required topics are properly connected

### 9.2. Debugging Commands

```bash
# Check Isaac ROS perception nodes
ros2 component types | grep -i isaac

# Monitor perception topics
ros2 topic hz /apriltag_detections
ros2 topic hz /object_detections
ros2 topic hz /visual_slam/pose

# Check for errors in perception nodes
ros2 run rqt_console rqt_console
```

## 10. Best Practices for Humanoid Robotics

### 10.1. Sensor Fusion Strategy

For humanoid robots, implement a multi-layered perception approach:

1. **Low-level processing**: Direct sensor processing (IMU, joint encoders)
2. **Mid-level processing**: Feature extraction (AprilTags, object detection)
3. **High-level processing**: Semantic understanding and decision making

### 10.2. Fail-Safe Mechanisms

Always implement backup perception methods:

```python
def emergency_stop_if_no_perception(self):
    """
    Emergency stop procedure if perception fails
    """
    if not self.has_recent_perception_data():
        # Send emergency stop command to controllers
        self.emergency_stop_publisher.publish(Empty())
        self.get_logger().error('PERCEPTION FAILURE - EMERGENCY STOP!')
        return True
    return False
```

## Next Steps

With the Isaac ROS perception stack properly configured, you'll next implement a complete perception pipeline that combines multiple sensors to create a robust perception system for your humanoid robot. This will include integrating visual SLAM, object detection, and fiducial tracking to create a comprehensive understanding of the robot's environment.