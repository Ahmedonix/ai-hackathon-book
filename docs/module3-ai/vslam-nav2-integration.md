# VSLAM and Navigation with Nav2 for Humanoid Robotics

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) combined with ROS 2's Navigation Stack (Nav2) forms the basis of autonomous navigation for humanoid robots. This section covers how to integrate Isaac ROS Visual SLAM with Nav2 to create a complete navigation system for humanoid robots in simulation and eventually in the real world.

## Understanding VSLAM for Humanoid Robotics

### 1. VSLAM vs Traditional SLAM

Traditional SLAM typically uses LiDAR sensors, while VSLAM uses visual input (cameras). For humanoid robots:

- **Advantages of VSLAM**:
  - Rich semantic information from images
  - Works in GPS-denied environments
  - Can operate without artificial fiducials
  - Lower power consumption than LiDAR systems

- **Challenges of VSLAM**:
  - Susceptible to lighting changes
  - Performance degradation in textureless environments
  - More computationally intensive
  - Requires more sophisticated algorithms for drift correction

### 2. Isaac ROS Visual SLAM Components

The Isaac ROS Visual SLAM system provides:
- GPU-accelerated feature extraction
- Real-time pose estimation
- Loop closure detection
- Map building and optimization
- Integration with ROS 2 navigation stack

## Setting Up Isaac ROS Visual SLAM

### 1. Installing Isaac ROS Visual SLAM

```bash
# Install Isaac ROS Visual SLAM packages
sudo apt install ros-iron-isaac-ros-visual-slame ros-iron-isaac-ros-stereo-image-proc ros-iron-isaac-ros-rectify

# Verify installation
dpkg -l | grep isaac-ros-visual
```

### 2. Visual SLAM Configuration

Create a configuration file for your Visual SLAM system: `config/visual_slam_config.yaml`

```yaml
# Visual SLAM Configuration for Humanoid Robot
visual_slam:
  ros__parameters:
    # Input parameters
    input_left_camera_topic: "/camera/left/image_rect_color"
    input_right_camera_topic: "/camera/right/image_rect_color"
    input_left_camera_info_topic: "/camera/left/camera_info"
    input_right_camera_info_topic: "/camera/right/camera_info"
    
    # Output parameters
    output_tracking_topic: "/visual_slam/tracking"
    output_map_topic: "/visual_slam/map"
    output_trajectory_topic: "/visual_slam/trajectory"
    output_pose_topic: "/visual_slam/pose_graph/pose"
    
    # Frame parameters
    map_frame: "map"
    odom_frame: "odom"
    base_frame: "base_link"
    publish_odom_tf: true
    publish_base_tf: true
    
    # Algorithm parameters
    enable_localization: true
    enable_mapping: true
    enable_diagnostics: true
    
    # Optimization parameters
    optimization_frequency: 10.0  # Hz
    tracking_frequency: 30.0      # Hz
    max_map_size: 10000           # Maximum number of map points
    min_num_features: 50          # Minimum number of features to track
    
    # Loop closure parameters
    enable_loop_closure: true
    loop_closure_frequency: 1.0   # Loop closure detection frequency in Hz
    loop_closure_minimum_travel_distance: 1.0  # Meters
    loop_closure_minimum_travel_heading: 0.5   # Radians
    
    # Feature detection parameters
    feature_detector_type: "orb"
    max_num_keypoints: 1000
    scale_factor: 1.2
    level_pyramid: 8
    fast_threshold: 20
    brief_descriptor: true
    
    # Stereo matching parameters
    stereo_baseline: 0.2  # Baseline distance in meters
    min_disparity: 0.1
    max_disparity: 128.0
    speckle_range: 3
    speckle_window: 0
    
    # Optimization parameters
    num_frames_fix_graph: 5
    fix_when_inf: true
    
    # Use simulation time if applicable
    use_sim_time: true
```

### 3. Launch File for Visual SLAM

Create a launch file for the Visual SLAM system: `launch/visual_slam.launch.py`

```python
# launch/visual_slam.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    enable_mapping = LaunchConfiguration('enable_mapping', default='true')
    enable_localization = LaunchConfiguration('enable_localization', default='true')
    
    # Gazebo launch (if needed for simulation)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Isaac ROS Visual SLAM Node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='visual_slam',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_navigation'),
                'config',
                'visual_slam_config.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('/camera/left/image_rect_color', [camera_namespace, '/left/image_rect_color']),
            ('/camera/right/image_rect_color', [camera_namespace, '/right/image_rect_color']),
            ('/camera/left/camera_info', [camera_namespace, '/left/camera_info']),
            ('/camera/right/camera_info', [camera_namespace, '/right/camera_info']),
            ('visual_slam/tracking/pose', '/visual_slam/pose'),
            ('visual_slam/map', '/visual_slam/map'),
            ('visual_slam/trajectory', '/visual_slam/trajectory')
        ],
        output='screen'
    )
    
    # Stereo Rectification Node (for preprocessing stereo images)
    stereo_rectify_node = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_stereo_rectify_node',
        name='stereo_rectify',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'alpha': 0.0}  # Full cropping to eliminate invalid pixels
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
    
    # Transform publisher for SLAM frames
    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='slam_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo (uncomment if needed)
        # gazebo,
        
        # Launch robot state publisher
        robot_state_publisher,
        
        # Launch stereo rectification first
        stereo_rectify_node,
        
        # Then launch visual SLAM after a short delay
        # (to ensure stereo images are ready)
        TimerAction(
            period=2.0,
            actions=[visual_slam_node]
        ),
        
        # Launch TF publisher
        tf_publisher
    ])
```

## Nav2 Navigation Stack Integration

### 1. Nav2 Installation and Configuration

```bash
# Install Nav2 packages
sudo apt install ros-iron-navigation2 ros-iron-nav2-bringup

# Install Isaac ROS Nav2 components
sudo apt install ros-iron-isaac-ros-navigation
```

### 2. Nav2 Configuration for Humanoid Robots

Create `config/nav2_config.yaml` for humanoid-specific navigation:

```yaml
# Nav2 Configuration for Humanoid Robot
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    goal_addition_distance: 0.5  # Minimum distance between waypoints (important for humanoid locomotion)
    goal_addition_angle: 1.047  # Minimum angle between waypoints

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: true

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 20.0  # Lower frequency for humanoid stability
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIC"
      debug_visualizations: false
      control_horizon: 10
      dt: 0.05
      max_heading_change: 1.57  # Max 90 degrees per control cycle (for humanoid stability)
      speed_units: "m/s"
      speed_lim_v: 0.5  # Conservative speed for humanoid stability
      speed_lim_w: 1.0
      acc_lim_v: 0.5   # Conservative acceleration for humanoid balance
      acc_lim_w: 1.5
      decel_factor: 1.0
      oscillation_score_penalty: 0.05
      oscillation_magic_number: 0.05
      oscillation_reset_angle: 0.17
      xy_goal_tolerance: 0.2  # Increased for humanoid stepping accuracy
      yaw_goal_tolerance: 0.2
      stateful: true

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: true

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0  # Lower for humanoid - less reactive but more stable
      publish_frequency: 8.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 6  # Wider window for humanoid navigation planning
      height: 6
      resolution: 0.05  # Higher resolution for precise humanoid stepping
      robot_radius: 0.3  # Radius for humanoid collision checking
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher cost scaling for humanoid safety
        inflation_radius: 0.5     # Larger inflation for humanoid safety margin
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: false
        origin_z: 0.0
        z_resolution: 0.2   # Higher resolution for humanoid foot placement
        z_voxels: 10
        max_obstacle_height: 2.0  # Humanoid can step over low obstacles
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.05

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 5.0  # Lower update for humanoid path planning
      publish_frequency: 2.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      robot_radius: 0.3  # Same as local costmap
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.05
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher for humanoid safety
        inflation_radius: 0.7     # Larger for humanoid safety margin

planner_server:
  ros__parameters:
    expected_planner_frequency: 5.0  # Lower for humanoid path planning
    use_sim_time: true
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5  # Increased tolerance for humanoid navigation
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: true

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
      ideal_spin_angle: 1.57  # 90 degrees for tighter turns
      max_rotational_vel: 0.4  # Conservative spinning for humanoid stability
      min_rotational_vel: 0.1
    backup:
      plugin: "nav2_recoveries::BackUp"
      backup_dist: 0.3  # Conservative backing up for humanoid
      backup_speed: 0.05
    wait:
      plugin: "nav2_recoveries::Wait"
      wait_duration: 5s

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1s
```

### 3. Nav2 Launch File

Create `launch/nav2_humano.launch.py`:

```python
# launch/nav2_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    autostart = LaunchConfiguration('autostart', default='true')
    params_file = LaunchConfiguration('params_file', 
                                     default=PathJoinSubstitution([
                                         FindPackageShare('humanoid_navigation'),
                                         'config',
                                         'nav2_config.yaml'
                                     ]))
    default_bt_xml_filename = LaunchConfiguration(
        'default_bt_xml_filename',
        default=PathJoinSubstitution([
            FindPackageShare('nav2_bt_navigator'),
            'behavior_trees',
            'navigate_w_replanning_and_recovery.xml'
        ])
    )
    
    # Setup environment variables
    stdout_linebuf_envvar = SetEnvironmentVariable(
        name='RCUTILS_LOGGING_BUFFERED_STREAM',
        value='1'
    )
    
    # Navigation launch
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/navigation_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'params_file': params_file,
            'default_bt_xml_filename': default_bt_xml_filename
        }.items()
    )
    
    # Lifecycle manager for navigation
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': autostart},
            {'node_names': [
                'controller_server',
                'planner_server', 
                'recoveries_server',
                'bt_navigator',
                'waypoint_follower'
            ]}
        ]
    )
    
    # Local costmap for humanoid-specific navigation
    local_costmap_node = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d_node',
        name='local_costmap',
        namespace='',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ],
        output='screen'
    )
    
    # Global costmap for humanoid-specific navigation
    global_costmap_node = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d_node',
        name='global_costmap',
        namespace='',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Set environment variables
        stdout_linebuf_envvar,
        
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch navigation
        navigation,
        
        # Launch lifecycle manager
        lifecycle_manager,
    ])
```

## Integrating Visual SLAM and Nav2

### 1. Coordinate Frame Integration

Create a TF bridge node to integrate Visual SLAM and Nav2 coordinate frames:

```python
# scripts/vslam_nav2_integration.py
#!/usr/bin/env python3

"""
Coordinate frame integration between Isaac ROS Visual SLAM and Nav2.
This node manages the relationship between visual SLAM map frame and navigation frames.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R


class VSLAMNav2Integrator(Node):
    """
    Integrates Isaac ROS Visual SLAM pose estimates with Nav2 navigation system.
    Manages coordinate frame relationships between visual SLAM and navigation.
    """
    
    def __init__(self):
        super().__init__('vslam_nav2_integrator')
        
        # Declare parameters
        self.declare_parameter('vslam_pose_topic', '/visual_slam/pose_graph/pose')
        self.declare_parameter('nav_odom_topic', '/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_sim_time', True)
        
        # Get parameters
        self.vslam_pose_topic = self.get_parameter('vslam_pose_topic').value
        self.nav_odom_topic = self.get_parameter('nav_odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.use_sim_time = self.get_parameter('use_sim_time').value
        
        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            self.vslam_pose_topic,
            self.vslam_pose_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            self.nav_odom_topic,
            self.odom_callback,
            10
        )
        
        # Publishers
        self.integrated_pose_pub = self.create_publisher(
            PoseStamped,
            '/integrated_pose',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/vslam_nav2_integration_status',
            10
        )
        
        # Internal state
        self.last_vslam_pose = None
        self.last_odom = None
        self.vslam_to_odom_transform = None  # Transform from VSLAM to ODOM frame
        self.initial_alignment_performed = False
        
        # Timer for broadcasting transforms
        self.transform_timer = self.create_timer(0.05, self.broadcast_transforms)  # 20Hz
        
        self.get_logger().info(
            f'VSLAM-Nav2 Integration node initialized\n'
            f'  VSLAM Pose Topic: {self.vslam_pose_topic}\n'
            f'  Odom Topic: {self.nav_odom_topic}\n'
            f'  Map Frame: {self.map_frame}\n'
            f'  Odom Frame: {self.odom_frame}\n'
            f'  Base Frame: {self.base_frame}'
        )

    def vslam_pose_callback(self, msg):
        """Handle Visual SLAM pose updates"""
        self.last_vslam_pose = msg
        
        if not self.initial_alignment_performed:
            self.perform_initial_alignment()
        
        # Log pose information
        self.get_logger().debug(
            f'VSLAM Pose: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )

    def odom_callback(self, msg):
        """Handle odometry updates from wheel encoders or other sources"""
        self.last_odom = msg
        
        if not self.initial_alignment_performed:
            self.perform_initial_alignment()
        
        # Log odometry information
        self.get_logger().debug(
            f'Odom Pose: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})'
        )

    def perform_initial_alignment(self):
        """
        Align the VSLAM map frame with the navigation odom frame.
        This should be called when both VSLAM and odometry data is available.
        """
        if self.last_vslam_pose is None or self.last_odom is None:
            return
        
        # Calculate transform between VSLAM map and odom frame
        # This aligns the visual SLAM coordinate system with the robot's odometry system
        vslam_pos = self.last_vslam_pose.pose.position
        odom_pos = self.last_odom.pose.pose.position
        
        # Create transform from VSLAM map to ODOM frame
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame  # Map frame from VSLAM
        transform.child_frame_id = self.odom_frame  # Odom frame for navigation
        
        # Calculate offset
        transform.transform.translation.x = odom_pos.x - vslam_pos.x
        transform.transform.translation.y = odom_pos.y - vslam_pos.y
        transform.transform.translation.z = odom_pos.z - vslam_pos.z
        
        # For rotation, assume initial alignment (identity quaternion)
        transform.transform.rotation.w = 1.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        
        self.vslam_to_odom_transform = transform
        self.initial_alignment_performed = True
        
        self.get_logger().info(
            f'Initial alignment performed. VSLAM to ODOM offset: '
            f'({transform.transform.translation.x:.2f}, {transform.transform.translation.y:.2f}, {transform.transform.translation.z:.2f})'
        )

    def broadcast_transforms(self):
        """Broadcast necessary transforms for navigation system"""
        if not self.initial_alignment_performed or self.vslam_to_odom_transform is None:
            return
        
        # Update timestamp
        current_time = self.get_clock().now()
        self.vslam_to_odom_transform.header.stamp = current_time.to_msg()
        
        # Broadcast VSLAM to ODOM transform
        self.tf_broadcaster.sendTransform(self.vslam_to_odom_transform)
        
        # If we also have current VSLAM pose, broadcast ODOM to BASE_LINK transform
        # using the combined VSLAM pose + calculated offset
        if self.last_vslam_pose is not None:
            try:
                # Calculate robot pose in odom frame by combining VSLAM pose with fixed offset
                odom_to_base_transform = TransformStamped()
                odom_to_base_transform.header.stamp = current_time.to_msg()
                odom_to_base_transform.header.frame_id = self.odom_frame
                odom_to_base_transform.child_frame_id = self.base_frame
                
                # Apply VSLAM pose to get base position in VSLAM frame
                # then apply inverse of VSLAM->ODOM transform to get in ODOM frame
                vslam_pos = self.last_vslam_pose.pose.position
                vslam_rot = self.last_vslam_pose.pose.orientation
                
                # Transform position from VSLAM frame to ODOM frame
                transformed_pos = self.transform_position(
                    vslam_pos,
                    self.vslam_to_odom_transform.transform
                )
                
                odom_to_base_transform.transform.translation.x = transformed_pos.x
                odom_to_base_transform.transform.translation.y = transformed_pos.y
                odom_to_base_transform.transform.translation.z = transformed_pos.z
                odom_to_base_transform.transform.rotation = vslam_rot  # Keep rotation from VSLAM
                
                self.tf_broadcaster.sendTransform(odom_to_base_transform)
                
                # Publish integrated pose
                integrated_pose = PoseStamped()
                integrated_pose.header = self.last_vslam_pose.header
                integrated_pose.header.frame_id = self.odom_frame
                integrated_pose.pose.position = transformed_pos
                integrated_pose.pose.orientation = vslam_rot
                self.integrated_pose_pub.publish(integrated_pose)
                
            except Exception as e:
                self.get_logger().error(f'Error in transform calculation: {str(e)}')
        
        # Publish status
        status_msg = String()
        if self.initial_alignment_performed:
            status_msg.data = "INTEGRATION_ACTIVE"
        else:
            status_msg.data = "WAITING_ALIGNMENT"
        self.status_pub.publish(status_msg)

    def transform_position(self, position, transform):
        """
        Transform a position by a transform.
        This applies translation offset from the transform to the position.
        """
        # Apply translation offset
        transformed = Point()
        transformed.x = position.x + transform.translation.x
        transformed.y = position.y + transform.translation.y
        transformed.z = position.z + transform.translation.z
        return transformed

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('VSLAM-Nav2 Integration Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    integrator = VSLAMNav2Integrator()
    
    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        integrator.get_logger().info('Node interrupted by user')
    finally:
        integrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Creating a Navigation Command Interface

Create a node to accept navigation commands and integrate them with VSLAM:

```python
# scripts/navigation_interface.py
#!/usr/bin/env python3

"""
Navigation command interface that integrates VSLAM information with Nav2 commands.
Provides high-level navigation capabilities for humanoid robots.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.spatial.distance import euclidean


class NavigationInterface(Node):
    """
    Provides high-level navigation interface that combines VSLAM and Nav2.
    Offers path planning and obstacle avoidance capabilities for humanoid robots.
    """
    
    def __init__(self):
        super().__init__('navigation_interface')
        
        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('laser_topic', '/scan')
        self.declare_parameter('vslam_pose_topic', '/visual_slam/pose_graph/pose')
        self.declare_parameter('goal_tolerance', 0.3)  # meters
        self.declare_parameter('obstacle_threshold', 0.6)  # meters
        self.declare_parameter('waypoint_distance', 0.5)  # meters between waypoints
        
        # Get parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_topic = self.get_parameter('laser_topic').value
        self.vslam_pose_topic = self.get_parameter('vslam_pose_topic').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.waypoint_distance = self.get_parameter('waypoint_distance').value
        
        # Navigation state
        self.current_pose = None
        self.target_pose = None
        self.navigation_active = False
        self.path_to_follow = None
        self.laser_data = None
        self.vslam_pose = None
        
        # Subscriptions
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            self.vslam_pose_topic,
            self.vslam_pose_callback,
            10
        )
        
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.laser_topic,
            self.laser_callback,
            10
        )
        
        # Navigation goal publisher
        self.nav_goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        
        # Navigation control commands
        self.nav_cancel_pub = self.create_publisher(
            Bool,
            '/navigate_to_pose/_action/cancel_goal',
            10
        )
        
        # Path visualization
        self.path_pub = self.create_publisher(
            Path,
            '/navigation_path',
            10
        )
        
        # Obstacle visualization
        self.obstacle_pub = self.create_publisher(
            MarkerArray,
            '/obstacles',
            10
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/navigation_status',
            10
        )
        
        # Command subscription for high-level navigation
        self.command_sub = self.create_subscription(
            PointStamped,
            '/navigation_command',
            self.navigation_command_callback,
            10
        )
        
        # Timer for navigation control loop
        self.nav_control_timer = self.create_timer(0.1, self.navigation_control_loop)
        
        self.get_logger().info(
            f'Navigation Interface initialized\n'
            f'  Laser topic: {self.laser_topic}\n'
            f'  VSLAM topic: {self.vslam_pose_topic}\n'
            f'  Goal tolerance: {self.goal_tolerance}m\n'
            f'  Obstacle threshold: {self.obstacle_threshold}m'
        )

    def vslam_pose_callback(self, msg):
        """Update current pose from Visual SLAM"""
        self.vslam_pose = msg
        self.current_pose = msg.pose  # Store just the pose part

    def laser_callback(self, msg):
        """Update laser scan data for obstacle detection"""
        self.laser_data = msg

    def navigation_command_callback(self, msg):
        """Handle high-level navigation commands"""
        if not self.current_pose:
            self.get_logger().warn('No current pose available, cannot set navigation goal')
            return
        
        # Create navigation goal from command
        goal = PoseStamped()
        goal.header = msg.header
        goal.header.frame_id = self.map_frame  # Use map frame from VSLAM
        goal.pose.position = msg.point
        # Set orientation to face the goal direction
        goal_direction = np.array([
            msg.point.x - self.current_pose.position.x,
            msg.point.y - self.current_pose.position.y
        ])
        
        if np.linalg.norm(goal_direction) > 0.1:
            # Calculate orientation to face the goal
            yaw = np.arctan2(goal_direction[1], goal_direction[0])
            from geometry_msgs.msg import Quaternion
            self.set_yaw_orientation(goal.pose, yaw)
        
        # Publish goal to Nav2
        self.nav_goal_pub.publish(goal)
        self.target_pose = goal.pose
        
        self.get_logger().info(
            f'Navigation goal set: ({msg.point.x:.2f}, {msg.point.y:.2f})'
        )
        self.navigation_active = True

    def set_yaw_orientation(self, pose, yaw):
        """Set orientation for a pose based on yaw angle"""
        import math
        from geometry_msgs.msg import Quaternion
        
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = 1.0  # No pitch
        sp = 0.0
        cr = 1.0  # No roll
        sr = 0.0
        
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        
        pose.orientation = q

    def navigation_control_loop(self):
        """Main navigation control loop"""
        if not self.current_pose or not self.target_pose:
            return
        
        # Calculate distance to goal
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        
        target_pos = np.array([
            self.target_pose.position.x,
            self.target_pose.position.y
        ])
        
        distance_to_goal = np.linalg.norm(target_pos - current_pos)
        
        # Check if we've reached the goal
        if distance_to_goal < self.goal_tolerance:
            self.navigation_active = False
            status_msg = String()
            status_msg.data = f"GOAL_REACHED: Distance {distance_to_goal:.2f}m < {self.goal_tolerance:.2f}m tolerance"
            self.status_pub.publish(status_msg)
            self.get_logger().info(status_msg.data)
            return
        
        # Check for obstacles in path
        if self.laser_data:
            obstacles_ahead = self.check_obstacles_in_path(self.laser_data)
            
            if obstacles_ahead:
                status_msg = String()
                status_msg.data = f"OBSTACLE_DETECTED: Navigation paused, {obstacles_ahead} obstacles ahead"
                self.status_pub.publish(status_msg)
                self.get_logger().warn(status_msg.data)
                
                # Optionally cancel navigation and wait
                # cancel_msg = Bool()
                # cancel_msg.data = True
                # self.nav_cancel_pub.publish(cancel_msg)
        
        # Publish navigation status
        status_msg = String()
        status_msg.data = f"NAVIGATING: {distance_to_goal:.2f}m to goal"
        self.status_pub.publish(status_msg)
        
        self.get_logger().debug(status_msg.data)

    def check_obstacles_in_path(self, laser_msg):
        """Check if obstacles are in the direct path to target"""
        if not self.current_pose or not self.target_pose:
            return 0
        
        # Calculate direction to target
        direction_vector = np.array([
            self.target_pose.position.x - self.current_pose.position.x,
            self.target_pose.position.y - self.current_pose.position.y
        ])
        
        direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
        
        # Define cone to check (30 degrees half-angle)
        cone_half_angle = np.deg2rad(15)  # 30 degree cone
        
        obstacles_found = 0
        center_idx = len(laser_msg.ranges) // 2
        
        # Check laser readings in the cone toward the goal
        # This is a simplified approach; real implementation would be more sophisticated
        for i, range_val in enumerate(laser_msg.ranges):
            if not np.isfinite(range_val) or range_val > laser_msg.range_max:
                continue
            
            # Calculate angle of this laser beam relative to forward direction
            angle = laser_msg.angle_min + i * laser_msg.angle_increment
            
            # Check if this angle is within our cone toward the target
            if abs(angle) < cone_half_angle and range_val < self.obstacle_threshold:
                obstacles_found += 1
        
        return obstacles_found

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Navigation Interface Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    nav_interface = NavigationInterface()
    
    try:
        rclpy.spin(nav_interface)
    except KeyboardInterrupt:
        nav_interface.get_logger().info('Navigation interface interrupted by user')
    finally:
        nav_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Complete Integration Launch File

Create a launch file that brings everything together: `launch/humano_vslam_nav2.launch.py`

```python
# launch/humano_vslam_nav2.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    enable_mapping = LaunchConfiguration('enable_mapping', default='true')
    enable_localization = LaunchConfiguration('enable_localization', default='true')
    nav2_params_file = LaunchConfiguration('nav2_params_file', 
                                          default=PathJoinSubstitution([
                                              FindPackageShare('humanoid_navigation'),
                                              'config',
                                              'nav2_config.yaml'
                                          ]))
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_navigation')
    
    # Launch Gazebo (if needed)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', 'navigation_test.sdf']),
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Isaac ROS Stereo Rectification
    stereo_rectify = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_stereo_rectify_node',
        name='stereo_rectify',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'alpha': 0.0}
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
    
    # Isaac ROS Visual SLAM
    visual_slam = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='visual_slam',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_navigation'),
                'config',
                'visual_slam_config.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('/camera/left/image_rect_color', [camera_namespace, '/left/image_rect_color']),
            ('/camera/right/image_rect_color', [camera_namespace, '/right/image_rect_color']),
            ('/camera/left/camera_info', [camera_namespace, '/left/camera_info']),
            ('/camera/right/camera_info', [camera_namespace, '/right/camera_info']),
            ('visual_slam/tracking/pose', '/visual_slam/pose'),
            ('visual_slam/map', '/visual_slam/map'),
            ('visual_slam/trajectory', '/visual_slam/trajectory')
        ],
        output='screen'
    )
    
    # VSLAM-Nav2 Integration
    vslam_nav2_integration = Node(
        package='humanoid_navigation',
        executable='vslam_nav2_integration',
        name='vslam_nav2_integration',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'map_frame': 'map'},
            {'odom_frame': 'odom'},
            {'base_frame': 'base_link'}
        ],
        output='screen'
    )
    
    # Nav2 Navigation Stack
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('nav2_bringup'),
            '/launch/navigation_launch.py'
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file
        }.items()
    )
    
    # Navigation Interface
    nav_interface = Node(
        package='humanoid_navigation',
        executable='navigation_interface',
        name='navigation_interface',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'map_frame': 'map'},
            {'odom_frame': 'odom'},
            {'base_frame': 'base_link'},
            {'laser_topic': '/scan'},
            {'vslam_pose_topic': '/visual_slam/pose'},
            {'goal_tolerance': 0.3},
            {'obstacle_threshold': 0.6}
        ],
        output='screen'
    )
    
    # Create worlds directory if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    os.makedirs(worlds_dir, exist_ok=True)
    
    # Create a simple navigation test world
    world_file = os.path.join(worlds_dir, 'navigation_test.sdf')
    if not os.path.exists(world_file):
        with open(world_file, 'w') as f:
            f.write("""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="navigation_test">
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
    
    <!-- Add some simple obstacles for navigation testing -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.5 0.5 1.0</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.5 0.5 1.0</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="obstacle_2">
      <pose>-2 2 0.5 0 0 0.785</pose>  <!-- 45 degrees rotation -->
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.3 1.5 1.0</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.3 1.5 1.0</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>""")
    
    return LaunchDescription([
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo (if needed)
        # gazebo,
        
        # Launch robot state publisher
        robot_state_publisher,
        
        # Launch stereo rectification
        stereo_rectify,
        
        # Launch VSLAM after a delay
        TimerAction(
            period=2.0,
            actions=[visual_slam]
        ),
        
        # Launch VSLAM-Nav2 integration after VSLAM
        TimerAction(
            period=4.0,
            actions=[vslam_nav2_integration]
        ),
        
        # Launch Nav2 after integration
        TimerAction(
            period=6.0,
            actions=[nav2]
        ),
        
        # Launch navigation interface after Nav2
        TimerAction(
            period=8.0,
            actions=[nav_interface]
        ),
    ])
```

## Testing the Complete VSLAM-Nav2 Integration

### 1. Build the Package

```bash
cd ~/humanoid_ws
source /opt/ros/iron/setup.bash
colcon build --packages-select humanoid_navigation
source install/setup.bash
```

### 2. Run the Complete System

```bash
# Terminal 1: Launch the complete system
ros2 launch humanoid_navigation humano_vslam_nav2.launch.py

# In a separate terminal, send navigation commands
# Terminal 2: Send a navigation goal
ros2 topic pub /navigation_command geometry_msgs/msg/PointStamped "header:
  frame_id: 'map'
  stamp:
    sec: 0
    nanosec: 0
point:
  x: 3.0
  y: 0.0
  z: 0.0"
```

### 3. Monitor the System

```bash
# Monitor navigation status
ros2 topic echo /navigation_status

# Monitor VSLAM pose
ros2 topic echo /visual_slam/pose_graph/pose

# Monitor Nav2 paths
ros2 topic echo /navigation_path

# Visualize in RViz
ros2 run rviz2 rviz2
```

## Troubleshooting VSLAM and Navigation Issues

### Common Issues:

1. **VSLAM Not Converging**:
   - Check camera calibration
   - Ensure sufficient visual features in environment
   - Verify stereo baseline distance

2. **Navigation Not Working**:
   - Check coordinate frame relationships
   - Verify obstacle detection parameters
   - Confirm Nav2 costmaps are updating

3. **Integration Problems**:
   - Ensure VSLAM map and Nav2 odom frames are properly connected
   - Check timing synchronization between systems
   - Verify TF tree is complete

## Next Steps

With VSLAM and Nav2 properly integrated, you'll next implement the complete perception-to-navigation pipeline that uses the sensor data from your humanoid robot to enable autonomous navigation in complex environments. The integration you've completed provides the foundation for all navigation and path planning capabilities for your humanoid robot.