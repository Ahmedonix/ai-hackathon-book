# Building VSLAM Pipeline using Isaac ROS

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is essential for autonomous humanoid robots to navigate unknown environments. In this section, we'll build a comprehensive VSLAM pipeline using Isaac ROS, which provides GPU-accelerated visual processing for real-time performance. This pipeline will form the foundation for autonomous navigation in our humanoid robot simulation.

## Understanding Isaac ROS Visual SLAM Components

### 1. Core VSLAM Components

The Isaac ROS Visual SLAM system includes:
- **Visual Feature Detection**: Identifies distinctive features in images
- **Stereo Matching**: Calculates depth from stereo camera pairs
- **Pose Estimation**: Determines robot position and orientation
- **Map Building**: Constructs a persistent map of the environment
- **Loop Closure**: Recognizes previously visited locations to correct drift

### 2. GPU-Accelerated Processing

Isaac ROS VSLAM leverages NVIDIA GPUs for:
- Real-time feature extraction and matching
- Parallel processing of image data
- Fast pose estimation and optimization
- Efficient map representation and updates

## Implementing the VSLAM Pipeline

### Step 1: Creating the VSLAM Configuration

First, let's create a comprehensive configuration file for the VSLAM system: `config/vslam_config.yaml`

```yaml
# VSLAM Configuration for Humanoid Robot
humanoid_vslam:
  ros__parameters:
    # Input topics
    input_left_camera_topic: "/camera/left/image_rect_color"
    input_right_camera_topic: "/camera/right/image_rect_color"
    input_left_camera_info_topic: "/camera/left/camera_info"
    input_right_camera_info_topic: "/camera/right/camera_info"
    
    # Output topics
    output_tracking_topic: "/vslam/tracking"
    output_map_topic: "/vslam/map"
    output_trajectory_topic: "/vslam/trajectory"
    output_pose_graph_pose_topic: "/vslam/pose_graph/pose"
    output_pose_graph_odom_topic: "/vslam/pose_graph/odometry"
    
    # Frame parameters
    map_frame: "map"
    odom_frame: "odom"
    base_frame: "base_link"
    publish_odom_tf: true
    publish_base_tf: true
    
    # Performance parameters
    enable_visualization: false  # Set to true to visualize features
    tracking_rate: 30.0  # Hz
    optimization_rate: 10.0  # Hz
    max_map_size: 5000  # Maximum number of features in map
    min_track_length: 5  # Minimum number of frames for feature tracking
    
    # Feature detection parameters
    detector_type: "ORB"  # Options: ORB, SIFT, SURF
    max_features: 1000
    scale_factor: 1.2  # Pyramid scale factor
    levels: 8  # Number of pyramid levels
    edge_threshold: 19  # FAST threshold
    patch_size: 31  # Patch size for feature descriptor
    
    # Stereo parameters
    stereo_baseline: 0.2  # Distance between stereo cameras (meters)
    min_disparity: 0.5
    max_disparity: 128.0
    speckle_range: 2
    speckle_window: 200
    
    # Tracking parameters
    tracker_type: "LK"  # Lucas-Kanade tracker
    max_level: 3  # Pyramid levels for LK tracking
    max_iteration: 30  # Max iterations for LK tracking
    epsilon: 0.01  # Convergence threshold for LK tracking
    
    # Mapping parameters
    min_depth: 0.1  # Minimum depth for valid observations (meters)
    max_depth: 20.0  # Maximum depth for valid observations (meters)
    depth_noise: 0.05  # Depth uncertainty as fraction of depth
    
    # Loop closure parameters
    enable_loop_closure: true
    loop_closure_frequency: 1.0  # Hz
    loop_closure_threshold: 0.5  # Distance threshold for candidate retrieval
    min_loop_closure_matches: 20  # Minimum matches for valid loop closure
    
    # Optimization parameters
    enable_optimization: true
    optimization_frequency: 5.0  # Hz
    max_optimization_iterations: 100
    robust_kernel: "cauchy"  # Options: cauchy, huber, none
    
    # Use simulation time
    use_sim_time: true
```

### Step 2: Creating the VSLAM Launch File

Create a launch file that orchestrates the complete VSLAM pipeline: `launch/vslam_pipeline.launch.py`

```python
# launch/vslam_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    enable_mapping = LaunchConfiguration('enable_mapping', default='true')
    enable_localization = LaunchConfiguration('enable_localization', default='true')
    vslam_config_file = LaunchConfiguration('vslam_config_file', 
                                           default=PathJoinSubstitution([
                                               FindPackageShare('humanoid_vslam'),
                                               'config',
                                               'vslam_config.yaml'
                                           ]))
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_vslam')
    
    # Robot State Publisher (if needed)
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('robot_state_publisher'),
            '/launch/robot_state_publisher.launch.py'
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # Stereo Rectification Node (preprocessing step)
    stereo_rectify = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_stereo_rectify_node',
        name='stereo_rectify',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'alpha': 0.0}  # Full rectification with zero-disparity
        ],
        remappings=[
            ('left/image_raw', [camera_namespace, '/left/image_raw']),
            ('left/camera_info', [camera_namespace, '/left/camera_info']),
            ('right/image_raw', [camera_namespace, '/right/image_raw']),
            ('right/camera_info', [camera_namespace, '/right/camera_info']),
            ('left/image_rect', [camera_namespace, '/left/image_rect_color']),
            ('right/image_rect', [camera_namespace, '/right/image_rect_color']),
        ],
        output='screen'
    )
    
    # Isaac ROS Visual SLAM Node
    visual_slam = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='visual_slam',
        parameters=[
            vslam_config_file,
            {'use_sim_time': use_sim_time},
            {'enable_mapping': enable_mapping},
            {'enable_localization': enable_localization}
        ],
        remappings=[
            ('/visual_slam/image_left', [camera_namespace, '/left/image_rect_color']),
            ('/visual_slam/image_right', [camera_namespace, '/right/image_rect_color']),
            ('/visual_slam/camera_info_left', [camera_namespace, '/left/camera_info']),
            ('/visual_slam/camera_info_right', [camera_namespace, '/right/camera_info']),
            ('visual_slam/tracking/pose', '/vslam/tracking/pose'),
            ('visual_slam/map', '/vslam/map'),
            ('visual_slam/trajectory', '/vslam/trajectory'),
            ('visual_slam/pose_graph/pose', '/vslam/pose_graph/pose'),
            ('visual_slam/pose_graph/optimization_trigger', '/vslam/optimization_trigger'),
        ],
        output='screen'
    )
    
    # Localization Mode Selector Node
    localization_selector = Node(
        package='humanoid_vslam',
        executable='localization_selector',
        name='localization_selector',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'mode': 'slam'}  # Options: 'slam', 'localization', 'mapping'
        ],
        output='screen'
    )
    
    # VSLAM Performance Monitor Node
    performance_monitor = Node(
        package='humanoid_vslam',
        executable='vslam_performance_monitor',
        name='vslam_performance_monitor',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'monitor_rate': 5.0},  # Hz
            {'publish_diagnostics': True}
        ],
        output='screen'
    )
    
    # Create worlds directory if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    os.makedirs(worlds_dir, exist_ok=True)
    
    # Create a simple world file if it doesn't exist
    world_file = os.path.join(worlds_dir, 'vslam_test.sdf')
    if not os.path.exists(world_file):
        with open(world_file, 'w') as f:
            f.write("""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="vslam_test_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add some visual landmarks for feature detection -->
    <model name="landmark_1">
      <pose>5 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 1.0</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>
    
    <model name="landmark_2">
      <pose>-3 2 0.5 0 0 0.785</pose> <!-- 45-degree rotation -->
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
      </link>
    </model>
    
    <model name="landmark_3">
      <pose>0 -4 0.5 0 0 1.57</pose> <!-- 90-degree rotation -->
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.3</radius>
            </sphere>
          </geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>""")
    
    return LaunchDescription([
        # Set parameter for all nodes
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch robot state publisher
        robot_state_publisher,
        
        # Launch stereo rectification first (as preprocessing)
        stereo_rectify,
        
        # Launch VSLAM after stereo rectification is ready
        TimerAction(
            period=2.0,  # Wait 2 seconds before launching VSLAM
            actions=[visual_slam]
        ),
        
        # Launch localization selector
        TimerAction(
            period=3.0,
            actions=[localization_selector]
        ),
        
        # Launch performance monitor
        TimerAction(
            period=4.0,
            actions=[performance_monitor]
        ),
    ])
```

### Step 3: Creating the Localization Mode Selector Node

Create `scripts/localization_selector.py` to manage different operating modes:

```python
#!/usr/bin/env python3

"""
VSLAM Localization Mode Selector Node.
Controls the operational mode of the VSLAM system (SLAM, Localization, or Mapping).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool
import numpy as np


class LocalizationSelectorNode(Node):
    """
    Controls the operational mode of the VSLAM system.
    Modes: SLAM (mapping and localization), Localization (using existing map),
    and Mapping (building map without localization).
    """
    
    def __init__(self):
        super().__init__('localization_selector_node')
        
        # Declare parameters
        self.declare_parameter('initial_mode', 'slam')  # Default to SLAM mode
        self.declare_parameter('map_save_path', '~/vslam_maps/')
        self.declare_parameter('map_load_path', '')
        self.declare_parameter('use_sim_time', True)
        
        # Get parameters
        self.initial_mode = self.get_parameter('initial_mode').value
        self.map_save_path = self.get_parameter('map_save_path').value
        self.map_load_path = self.get_parameter('map_load_path').value
        
        # Current VSLAM mode
        self.current_mode = self.initial_mode
        self.mode_options = ['slam', 'localization', 'mapping']
        
        # Service clients for VSLAM control
        self.reset_map_client = self.create_client(SetBool, '/visual_slam/reset_map')
        self.save_map_client = self.create_client(SetBool, '/visual_slam/save_map')
        self.load_map_client = self.create_client(SetBool, '/visual_slam/load_map')
        
        # Publishers for mode control commands
        self.mode_control_pub = self.create_publisher(
            String,
            '/vslam_mode_control',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vslam_mode_status',
            10
        )
        
        # Subscription for mode change requests
        self.mode_request_sub = self.create_subscription(
            String,
            '/request_vslam_mode',
            self.mode_request_callback,
            10
        )
        
        # Timer to periodically report mode status
        self.status_timer = self.create_timer(2.0, self.publish_status)
        
        # Initialize VSLAM to initial mode
        self.change_mode(self.initial_mode)
        
        self.get_logger().info(f'Localization Selector Node initialized in {self.initial_mode} mode')

    def mode_request_callback(self, msg):
        """Handle requests to change VSLAM mode"""
        requested_mode = msg.data.lower()
        
        if requested_mode in self.mode_options:
            if requested_mode != self.current_mode:
                success = self.change_mode(requested_mode)
                if success:
                    self.get_logger().info(f'VSLAM mode changed from {self.current_mode} to {requested_mode}')
                    self.current_mode = requested_mode
                else:
                    self.get_logger().error(f'Failed to change VSLAM mode to {requested_mode}')
            else:
                self.get_logger().info(f'VSLAM already in {requested_mode} mode')
        else:
            self.get_logger().warn(f'Invalid VSLAM mode requested: {requested_mode}. Valid options: {self.mode_options}')

    def change_mode(self, new_mode):
        """Change the VSLAM operational mode"""
        if new_mode == 'slam':
            # Reset map for simultaneous localization and mapping
            return self.enable_slam_mode()
        elif new_mode == 'localization':
            # Enable localization-only mode
            return self.enable_localization_mode()
        elif new_mode == 'mapping':
            # Enable mapping-only mode
            return self.enable_mapping_mode()
        else:
            self.get_logger().error(f'Unknown mode: {new_mode}')
            return False

    def enable_slam_mode(self):
        """Enable SLAM mode (both localization and mapping)"""
        try:
            # In a real implementation, this would send commands to the VSLAM node
            # to enable both localization and mapping simultaneously
            mode_msg = String()
            mode_msg.data = "SLAM_ENABLED"
            self.mode_control_pub.publish(mode_msg)
            
            # Reset the map to start fresh
            if self.reset_map_client.wait_for_service(timeout_sec=1.0):
                future = self.reset_map_client.call_async(SetBool.Request(data=True))
                # We'll handle the response asynchronously
            else:
                self.get_logger().warn('Reset map service not available')
            
            return True
        except Exception as e:
            self.get_logger().error(f'Error enabling SLAM mode: {str(e)}')
            return False

    def enable_localization_mode(self):
        """Enable localization-only mode"""
        try:
            mode_msg = String()
            mode_msg.data = "LOCALIZATION_ENABLED"
            self.mode_control_pub.publish(mode_msg)
            
            # In localization mode, we'd typically load a pre-existing map
            if self.load_map_client.wait_for_service(timeout_sec=1.0):
                req = SetBool.Request()
                req.data = True
                future = self.load_map_client.call_async(req)
            else:
                self.get_logger().warn('Load map service not available')
            
            return True
        except Exception as e:
            self.get_logger().error(f'Error enabling localization mode: {str(e)}')
            return False

    def enable_mapping_mode(self):
        """Enable mapping-only mode"""
        try:
            mode_msg = String()
            mode_msg.data = "MAPPING_ENABLED"
            self.mode_control_pub.publish(mode_msg)
            
            # Reset map to start building new one
            if self.reset_map_client.wait_for_service(timeout_sec=1.0):
                req = SetBool.Request()
                req.data = True
                future = self.reset_map_client.call_async(req)
            else:
                self.get_logger().warn('Reset map service not available')
            
            return True
        except Exception as e:
            self.get_logger().error(f'Error enabling mapping mode: {str(e)}')
            return False

    def publish_status(self):
        """Publish current VSLAM mode status"""
        status_msg = String()
        status_msg.data = f"MODE:{self.current_mode.upper()}"
        self.status_publisher.publish(status_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Localization Selector Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    selector_node = LocalizationSelectorNode()
    
    try:
        rclpy.spin(selector_node)
    except KeyboardInterrupt:
        selector_node.get_logger().info('Node interrupted by user')
    finally:
        selector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 4: Creating the VSLAM Performance Monitor

Create `scripts/vslam_performance_monitor.py`:

```python
#!/usr/bin/env python3

"""
VSLAM Performance Monitor Node.
Monitors and analyzes the performance of the VSLAM system.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import time
import numpy as np
from collections import deque


class VSLAMPerformanceMonitorNode(Node):
    """
    Monitors the performance of the VSLAM system by analyzing:
    - Feature tracking quality
    - Map building progress
    - Computational performance
    - Localization accuracy indicators
    """
    
    def __init__(self):
        super().__init__('vslam_performance_monitor')
        
        # Declare parameters
        self.declare_parameter('monitor_rate', 5.0)  # Hz
        self.declare_parameter('publish_diagnostics', True)
        self.declare_parameter('use_sim_time', True)
        
        # Get parameters
        self.monitor_rate = self.get_parameter('monitor_rate').value
        self.publish_diagnostics = self.get_parameter('publish_diagnostics').value
        
        # Track performance metrics
        self.feature_counts = deque(maxlen=100)  # Last 100 measurements
        self.tracking_poses = deque(maxlen=100)
        self.image_timestamps = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Time tracking
        self.last_pose_time = None
        self.last_image_time = None
        
        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/vslam/pose_graph/pose',
            self.pose_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect_color',  # Assuming left camera
            self.image_callback,
            10
        )
        
        # Publishers
        self.performance_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/vslam/performance_metrics',
            10
        )
        
        self.diagnostic_pub = self.create_publisher(
            DiagnosticArray,
            '/vslam/diagnostics',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vslam/performance_status',
            10
        )
        
        # Timer for performance analysis
        self.performance_timer = self.create_timer(1.0/self.monitor_rate, self.analyze_performance)
        
        self.get_logger().info(f'VSLAM Performance Monitor running at {self.monitor_rate} Hz')

    def pose_callback(self, msg):
        """Track VSLAM pose updates"""
        self.tracking_poses.append({
            'timestamp': msg.header.stamp,
            'position': (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            'orientation': (msg.pose.orientation.x, msg.pose.orientation.y, 
                           msg.pose.orientation.z, msg.pose.orientation.w)
        })
        
        # Calculate time since last pose (for performance analysis)
        if self.last_pose_time:
            time_diff = (msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9) - \
                       (self.last_pose_time.sec + self.last_pose_time.nanosec / 1e9)
            self.processing_times.append(time_diff)
        
        self.last_pose_time = msg.header.stamp

    def image_callback(self, msg):
        """Track image input for performance calculation"""
        self.image_timestamps.append(msg.header.stamp)
        
        # Calculate image processing delay
        if self.last_pose_time:
            image_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            pose_time = self.last_pose_time.sec + self.last_pose_time.nanosec / 1e9
            processing_delay = abs(pose_time - image_time)
            
            self.processing_times.append(processing_delay)

    def analyze_performance(self):
        """Analyze VSLAM performance metrics"""
        metrics = self.calculate_performance_metrics()
        
        # Publish performance metrics
        metrics_msg = Float64MultiArray()
        metrics_msg.data = list(metrics.values())
        self.performance_metrics_publisher.publish(metrics_msg)
        
        # Publish diagnostic information
        if self.publish_diagnostics:
            diagnostic_msg = self.create_diagnostics(metrics)
            self.diagnostic_pub.publish(diagnostic_msg)
        
        # Publish status string
        status_msg = String()
        status_msg.data = self.format_status_string(metrics)
        self.status_publisher.publish(status_msg)
        
        # Log performance if degraded
        if metrics['localization_accuracy'] < 0.7:  # Arbitrary threshold
            self.get_logger().warn(f'VSLAM performance degraded: {status_msg.data}')
        elif int(self.get_clock().now().nanoseconds / 1e9) % 10 == 0:  # Every 10 seconds
            self.get_logger().info(f'VSLAM Performance: {status_msg.data}')

    def calculate_performance_metrics(self):
        """Calculate various VSLAM performance metrics"""
        metrics = {}
        
        # Feature tracking quality (if available in a real implementation)
        avg_features = np.mean(self.feature_counts) if self.feature_counts else 0
        metrics['avg_feature_count'] = float(avg_features)
        
        # Tracking stability based on pose differences
        if len(self.tracking_poses) > 1:
            positions = [p['position'] for p in self.tracking_poses]
            distances = [np.sqrt(sum((p2[i]-p1[i])**2 for i in range(3))) 
                        for p1, p2 in zip(positions[:-1], positions[1:])]
            avg_movement = np.mean(distances) if distances else 0
            metrics['avg_movement'] = float(avg_movement)
        else:
            metrics['avg_movement'] = 0.0
        
        # Processing time metrics
        if self.processing_times:
            avg_proc_time = np.mean(self.processing_times)
            max_proc_time = np.max(self.processing_times)
            min_proc_time = np.min(self.processing_times)
            std_proc_time = np.std(self.processing_times)
            
            metrics['avg_processing_time'] = float(avg_proc_time)
            metrics['max_processing_time'] = float(max_proc_time)
            metrics['min_processing_time'] = float(min_proc_time)
            metrics['std_processing_time'] = float(std_proc_time)
        else:
            metrics['avg_processing_time'] = 0.0
            metrics['max_processing_time'] = 0.0
            metrics['min_processing_time'] = 0.0
            metrics['std_processing_time'] = 0.0
        
        # Calculate localization accuracy indicator
        # This is a simplified measure - real implementation would use more sophisticated metrics
        if avg_movement < 0.01 and len(self.processing_times) > 0:  # Robot stationary
            # Check if poses are stable
            stability_score = 1.0 - min(std_proc_time, 0.1) if std_proc_time else 1.0
        else:
            # Calculate based on movement consistency
            movement_consistency = 1.0 / (1.0 + avg_movement) if avg_movement > 0 else 1.0
            time_consistency = 1.0 / (1.0 + avg_proc_time) if avg_proc_time > 0 else 1.0
            stability_score = (movement_consistency + time_consistency) / 2.0
        
        metrics['localization_accuracy'] = min(1.0, stability_score)
        
        # Map building progress indicator
        map_size = len(self.tracking_poses)  # Simplified map size
        metrics['map_size'] = float(map_size)
        
        # Calculate map expansion rate
        if len(self.tracking_poses) > 10:
            # Calculate the bounding box of the poses to estimate map expansion
            positions = np.array([p['position'] for p in self.tracking_poses])
            x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
            y_range = np.max(positions[:, 1]) - np.min(positions[:, 1])
            map_area = x_range * y_range
            metrics['map_coverage_area'] = float(map_area)
        else:
            metrics['map_coverage_area'] = 0.0
        
        return metrics

    def create_diagnostics(self, metrics):
        """Create diagnostic messages from performance metrics"""
        diagnostic_array = DiagnosticArray()
        diagnostic_array.header.stamp = self.get_clock().now().to_msg()
        
        # Create diagnostic status for VSLAM
        status = DiagnosticStatus()
        status.name = "VSLAM Performance"
        
        # Determine overall status
        if metrics['localization_accuracy'] > 0.8:
            status.level = DiagnosticStatus.OK
            status.message = "VSLAM performing well"
        elif metrics['localization_accuracy'] > 0.5:
            status.level = DiagnosticStatus.WARN
            status.message = "VSLAM performance degraded"
        else:
            status.level = DiagnosticStatus.ERROR
            status.message = "VSLAM performance critical"
        
        # Add key metrics as key-value pairs
        status.values.extend([
            {'key': 'avg_feature_count', 'value': f"{metrics['avg_feature_count']:.2f}"},
            {'key': 'avg_processing_time', 'value': f"{metrics['avg_processing_time']:.4f}"},
            {'key': 'localization_accuracy', 'value': f"{metrics['localization_accuracy']:.2f}"},
            {'key': 'map_size', 'value': f"{int(metrics['map_size'])}"},
            {'key': 'map_coverage_area', 'value': f"{metrics['map_coverage_area']:.2f}m²"}
        ])
        
        diagnostic_array.status.append(status)
        return diagnostic_array

    def format_status_string(self, metrics):
        """Format performance status as a readable string"""
        return (
            f"Acc:{metrics['localization_accuracy']:.2f} | "
            f"Proc:{metrics['avg_processing_time']:.3f}s | "
            f"Feat:{metrics['avg_feature_count']:.1f} | "
            f"Map:{int(metrics['map_size'])} poses | "
            f"Area:{metrics['map_coverage_area']:.1f}m²"
        )

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('VSLAM Performance Monitor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    monitor_node = VSLAMPerformanceMonitorNode()
    
    try:
        rclpy.spin(monitor_node)
    except KeyboardInterrupt:
        monitor_node.get_logger().info('Performance monitor interrupted by user')
    finally:
        monitor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Creating a VSLAM Data Processor

Create `scripts/vslam_data_processor.py` to process and utilize VSLAM data:

```python
#!/usr/bin/env python3

"""
VSLAM Data Processor Node.
Processes VSLAM outputs and makes them available for navigation and other downstream tasks.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
import numpy as np
from scipy.spatial import cKDTree


class VSLAMDataProcessorNode(Node):
    """
    Processes VSLAM outputs and converts them to formats useful for navigation
    and other downstream nodes in the humanoid robotics system.
    """
    
    def __init__(self):
        super().__init__('vslam_data_processor')
        
        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('map_resolution', 0.1)  # meters per cell
        self.declare_parameter('map_width', 20.0)     # meters
        self.declare_parameter('map_height', 20.0)    # meters
        self.declare_parameter('robot_radius', 0.3)   # meters for occupancy grid
        
        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.robot_radius = self.get_parameter('robot_radius').value
        
        # Initialize TF broadcaster and buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Store VSLAM data
        self.pose_history = []  # Store pose history
        self.map_points = []    # Store landmark points for map generation
        self.current_pose = None
        
        # Subscriptions
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            '/vslam/pose_graph/pose',
            self.vslam_pose_callback,
            10
        )
        
        self.vslam_map_sub = self.create_subscription(
            PointCloud2,
            '/vslam/map',
            self.vslam_map_callback,
            10
        )
        
        # Publishers
        self.global_map_pub = self.create_publisher(
            OccupancyGrid,
            '/vslam_global_map',
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/vslam_trajectory',
            10
        )
        
        self.robot_pose_pub = self.create_publisher(
            PoseStamped,
            '/vslam_robot_pose',
            10
        )
        
        self.landmarks_pub = self.create_publisher(
            PointCloud2,
            '/vslam_landmarks',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vslam_data_processor_status',
            10
        )
        
        # Timer for processing and publishing data
        self.processing_timer = self.create_timer(0.1, self.process_and_publish)
        
        self.get_logger().info('VSLAM Data Processor Node Initialized')

    def vslam_pose_callback(self, msg):
        """Process VSLAM pose updates"""
        self.current_pose = msg
        
        # Store pose in history for trajectory generation
        self.pose_history.append({
            'timestamp': msg.header.stamp,
            'position': (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            'orientation': (msg.pose.orientation.x, msg.pose.orientation.y,
                           msg.pose.orientation.z, msg.pose.orientation.w)
        })
        
        # Keep only recent poses to prevent memory issues
        if len(self.pose_history) > 1000:  # Keep last 100 poses
            self.pose_history = self.pose_history[-100:]

    def vslam_map_callback(self, msg):
        """Process VSLAM map points"""
        # In a real implementation, this would convert PointCloud2 to usable format
        # For now, we'll keep the message for later publishing
        self.map_points.append(msg)
        
        # Keep only recent map points
        if len(self.map_points) > 10:  # Keep last 10 map updates
            self.map_points = self.map_points[-10:]

    def process_and_publish(self):
        """Process VSLAM data and publish processed outputs"""
        try:
            # Publish current robot pose
            if self.current_pose:
                self.robot_pose_pub.publish(self.current_pose)
                
                # Publish robot's position as a transform
                self.publish_robot_transform(self.current_pose)
            
            # Publish trajectory path
            self.publish_trajectory()
            
            # Publish landmarks
            self.publish_landmarks()
            
            # Publish occupancy grid if we have enough data
            if len(self.pose_history) > 10:
                occupancy_grid = self.create_occupancy_grid()
                self.global_map_pub.publish(occupancy_grid)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Poses:{len(self.pose_history)}, Maps:{len(self.map_points)}"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in VSLAM data processing: {str(e)}')

    def publish_robot_transform(self, pose_msg):
        """Publish TF transform for robot pose"""
        from geometry_msgs.msg import TransformStamped
        
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'vslam_odom'
        
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        t.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)

    def publish_trajectory(self):
        """Publish the robot's trajectory as a Path message"""
        if len(self.pose_history) < 2:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for pose_data in self.pose_history:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = pose_data['timestamp']
            
            pos = pose_data['position']
            quat = pose_data['orientation']
            
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = pos[2]
            
            from geometry_msgs.msg import Quaternion
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)

    def publish_landmarks(self):
        """Publish landmark points from the VSLAM map"""
        # This would convert the VSLAM point cloud to a filtered landmark representation
        # For now, we'll just publish the raw map points if available
        if self.map_points:
            # Publish the most recent map points
            self.landmarks_pub.publish(self.map_points[-1])

    def create_occupancy_grid(self):
        """Create an occupancy grid from VSLAM pose history"""
        from nav_msgs.msg import OccupancyGrid
        from geometry_msgs.msg import Point
        
        # Create occupancy grid message
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        
        # Calculate grid dimensions
        width_cells = int(self.map_width / self.map_resolution)
        height_cells = int(self.map_height / self.map_resolution)
        
        grid.info.resolution = self.map_resolution
        grid.info.width = width_cells
        grid.info.height = height_cells
        
        # Calculate map origin (centered on robot's starting position)
        if self.pose_history:
            start_pos = self.pose_history[0]['position']
            origin_x = start_pos[0] - self.map_width/2.0
            origin_y = start_pos[1] - self.map_height/2.0
        else:
            origin_x = -self.map_width/2.0
            origin_y = -self.map_height/2.0
        
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0  # No rotation
        
        # Initialize with unknown (value -1)
        grid.data = [-1] * (width_cells * height_cells)
        
        # Mark the robot's path as free space (value 0)
        for pose_data in self.pose_history:
            pos = pose_data['position']
            
            # Convert world coordinates to grid coordinates
            grid_x = int((pos[0] - origin_x) / self.map_resolution)
            grid_y = int((pos[1] - origin_y) / self.map_resolution)
            
            # Check if coordinates are within grid bounds
            if 0 <= grid_x < width_cells and 0 <= grid_y < height_cells:
                # Mark as free (0)
                grid.data[grid_y * width_cells + grid_x] = 0
        
        # Mark positions with landmarks as occupied (value 100) - simplified
        # In a real implementation, we'd use the actual VSLAM landmarks
        
        return grid

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('VSLAM Data Processor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    processor_node = VSLAMDataProcessorNode()
    
    try:
        rclpy.spin(processor_node)
    except KeyboardInterrupt:
        processor_node.get_logger().info('Node interrupted by user')
    finally:
        processor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 7: Testing and Validation

### Step 7.1: Create a Test Launch File

Create `launch/test_vslam_pipeline.launch.py`:

```python
# launch/test_vslam_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_vslam')
    
    # Launch the main VSLAM pipeline
    vslam_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('humanoid_vslam'),
            '/launch/vslam_pipeline.launch.py'
        ])
    )
    
    # Navigation interface using VSLAM data
    nav_interface = Node(
        package='humanoid_vslam',
        executable='vslam_navigation_interface',
        name='vslam_navigation_interface',
        parameters=[
            {'use_sim_time': True},
            {'vslam_pose_topic': '/vslam/pose_graph/pose'},
            {'vslam_trajectory_topic': '/vslam/trajectory'},
            {'robot_radius': 0.3}
        ],
        output='screen'
    )
    
    # Performance monitor
    perf_monitor = Node(
        package='humanoid_vslam',
        executable='vslam_performance_monitor',
        name='vslam_performance_monitor',
        parameters=[
            {'use_sim_time': True},
            {'monitor_rate': 10.0},
            {'publish_diagnostics': True}
        ],
        output='screen'
    )
    
    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', 
            PathJoinSubstitution([
                FindPackageShare('humanoid_vslam'),
                'rviz',
                'vslam_demo.rviz'
            ])
        ],
        output='screen'
    )
    
    # Test trajectory follower
    trajectory_follower = Node(
        package='humanoid_vslam',
        executable='test_trajectory_follower',
        name='test_trajectory_follower',
        parameters=[
            {'use_sim_time': True},
            {'trajectory_topic': '/vslam/trajectory'},
            {'control_frequency': 20.0}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Launch VSLAM pipeline first
        vslam_pipeline,
        
        # Launch performance monitor after a delay
        TimerAction(
            period=3.0,
            actions=[perf_monitor]
        ),
        
        # Launch navigation interface after VSLAM is running
        TimerAction(
            period=5.0,
            actions=[nav_interface]
        ),
        
        # Launch trajectory follower
        TimerAction(
            period=6.0,
            actions=[trajectory_follower]
        ),
        
        # Launch RViz for visualization
        TimerAction(
            period=8.0,
            actions=[rviz]
        ),
    ])
```

### Step 7.2: Build and Run the VSLAM Pipeline

```bash
# Navigate to the workspace
cd ~/humanoid_ws

# Source the ROS 2 environment
source /opt/ros/iron/setup.bash

# Build the humanoid_vslam package
colcon build --packages-select humanoid_vslam

# Source the workspace
source install/setup.bash

# Launch the VSLAM pipeline
ros2 launch humanoid_vslam test_vslam_pipeline.launch.py
```

In a separate terminal, you can monitor the system's performance:

```bash
# Monitor VSLAM status
ros2 topic echo /vslam_mode_status

# Monitor VSLAM performance metrics
ros2 topic echo /vslam/performance_metrics

# Monitor pose estimates
ros2 topic echo /vslam/pose_graph/pose

# Monitor the generated map
ros2 topic echo /vslam_global_map

# Check for available topics
ros2 topic list | grep -i vslam
```

## Part 8: Troubleshooting Common VSLAM Issues

### 1. Feature Starvation (Poor Textures)

**Issue**: VSLAM fails in textureless environments like plain walls or floors.

**Solution**: Add more visual landmarks or adjust feature detection parameters:

```yaml
# Increase number of features to detect
max_features: 1500

# Use less restrictive feature detector
detector_type: "FAST"

# Lower the threshold for feature detection
edge_threshold: 10
```

### 2. Drift Issues

**Issue**: Pose estimates drift over time.

**Solution**: 
- Verify loop closure is enabled
- Check that the robot moves through recognizable areas repeatedly
- Verify stereo camera calibration

### 3. Performance Issues

**Issue**: VSLAM running slowly or dropping frames.

**Solution**:
- Reduce resolution of input images
- Lower feature detection count
- Reduce optimization frequency
- Ensure GPU acceleration is properly configured

## Part 9: Integration with Navigation

The VSLAM pipeline you've created provides the essential mapping and localization capabilities for your humanoid robot's navigation system. The pose estimates and maps generated by VSLAM feed directly into the Nav2 stack to enable autonomous navigation in unknown environments.

The processed VSLAM data (including the occupancy grid, trajectory, and landmarks) provides Nav2 with the environmental information it needs to plan and execute navigation commands for your humanoid robot.

## Next Steps

With the VSLAM pipeline successfully implemented, you're now ready to configure Nav2 for obstacle avoidance, which will use the data from your VSLAM system to plan safe paths for your humanoid robot. The VSLAM provides the environmental understanding that Nav2 uses for safe navigation.