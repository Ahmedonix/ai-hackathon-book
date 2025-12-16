---
title: Navigation Demo in Simulation
description: Creating a complete navigation demonstration in simulation for humanoid robots
sidebar_position: 5
---

# Navigation Demo in Simulation

## Overview

In this chapter, we'll create a complete navigation demonstration for humanoid robots in simulation. This includes setting up the navigation stack, configuring path planning algorithms, and implementing obstacle avoidance behaviors. We'll use the Navigation2 stack integrated with Gazebo simulation to demonstrate how humanoid robots can navigate complex environments.

## Learning Objectives

- Set up Navigation2 stack for humanoid robots
- Configure navigation components (local and global planners)
- Implement obstacle avoidance behaviors
- Create simulation scenarios for navigation testing
- Evaluate navigation performance metrics

## Navigation Stack Setup

### Understanding Navigation2 Architecture

Navigation2 is the navigation stack for ROS 2, providing a complete solution for robot navigation. For humanoid robots, we need to consider specific constraints like stability, dynamic motion, and sensor positioning.

```python
# navigation_launch.py - Launch file for Navigation2 stack
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'namespace', default_value='',
        description='Top-level namespace'))

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value='nav2_params_humanoid.yaml',
        description='Full path to the ROS2 parameters file to use for all launched nodes'))

    ld.add_action(DeclareLaunchArgument(
        'default_bt_xml_filename',
        default_value='navigate_w_replanning_and_recovery.xml',
        description='Full path to the behavior tree xml file to use'))

    ld.add_action(DeclareLaunchArgument(
        'map_subscribe_transient_local', default_value='false',
        description='Whether to set the map subscriber QoS to transient local'))

    # Create parameter substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_bt_xml_filename': default_bt_xml_filename,
        'map_subscribe_transient_local': map_subscribe_transient_local}

    # Create renamed parameters
    configured_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites=param_substitutions,
        convert_types=True)

    # Add navigation nodes
    # 1. Lifecycle Manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time},
                   {'autostart': autostart},
                   {'node_names': ['map_server',
                                  'planner_server',
                                  'controller_server',
                                  'recoveries_server',
                                  'bt_navigator',
                                  'waypoint_follower']}],
        output='screen')

    # 2. Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # 3. Planner Server (Global Planner)
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # 4. Controller Server (Local Planner)
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # 5. Behavior Tree Navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # 6. Recovery Server
    recoveries_server = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # 7. Waypoint Follower
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        namespace=namespace,
        parameters=[configured_params],
        output='screen')

    # Add nodes to launch description
    ld.add_action(lifecycle_manager)
    ld.add_action(map_server)
    ld.add_action(planner_server)
    ld.add_action(controller_server)
    ld.add_action(bt_navigator)
    ld.add_action(recoveries_server)
    ld.add_action(waypoint_follower)

    return ld
```

### Navigation Parameters Configuration

```yaml
# nav2_params_humanoid.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    scan_topic: "scan"
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.2
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_have_remaining_waypoints_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller configuration
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_horizon: 1.5
      dt: 0.05
      vx_std: 0.2
      vy_std: 0.2
      wz_std: 0.3
      model_dt: 0.05
      batch_size: 2000
      ctrl_freq: 20.0
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.1
      motion_model: "DiffDrive"
      visualize: false
      transform_tolerance: 0.1
      heading_lookahead_dist: 0.4
      rotation_lookahead_time: 1.0
      max_linear_speed: 0.8
      min_linear_speed: 0.0
      max_angular_speed: 1.0
      min_angular_speed: 0.0
      adaptive_lookahead_dist: false
      lookahead_dist: 0.6
      speed_scaling_dist: 0.8
      speed_scaling_min_speed: 0.15
      track_obstacles: true
      use_global_plan_overrides: true
      max_global_plan_lookahead_dist: 3.0
      global_plan_prune_distance: 1.0
      occ_grid_type: "nav2_costmap_2d::ObstacleLayer"
      inflation_cost_scaling: 3.0
      inflation_radius: 0.5

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Humanoid robot radius
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    recovery_plugin_types: ["nav2_recoveries::Spin", "nav2_recoveries::BackUp", "nav2_recoveries::Wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
      enabled: True
      simulate_ahead_time: 2.0
      max_rotational_vel: 1.0
      min_rotational_vel: 0.4
      rotational_acc_lim: 3.2
    backup:
      plugin: "nav2_recoveries::BackUp"
      enabled: True
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_recoveries::Wait"
      enabled: True
      wait_duration: 1.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True
```

## Navigation Behavior Trees

Navigation2 uses behavior trees to control navigation behaviors. For humanoid robots, these need to be adapted to account for their unique movement patterns.

```xml
<!-- navigate_w_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <RecoveryNode number_of_retries="1" name="ComputePathToPose">
            <ReactiveSequence>
              <RemovePassedGoals input_goals="{goals}" output_goals="{goals_to_process}"/>
              <ComputePathToPose goal="{goals_to_process[0]}" path="{path}" planner_id="GridBased"/>
            </ReactiveSequence>
          </RecoveryNode>
        </RateController>
        <ReactiveSequence>
          <UpdatePathWithCostmap input_path="{path}" output_path="{updated_path}"/>
          <FollowPath path="{updated_path}" controller_id="FollowPath"/>
        </ReactiveSequence>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RecoveryNode number_of_retries="1" name="ClearingRotation">
          <ClearCostmapService service_name="clearing_rotation_global_costmap"/>
        </RecoveryNode>
        <RecoveryNode number_of_retries="1" name="GlobalLocalizationService">
          <ReinitializeGlobalLocalizationService/>
        </RecoveryNode>
        <RecoveryNode number_of_retries="1" name="Wait">
          <Wait wait_duration="5"/>
        </RecoveryNode>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

## Humanoid-Specific Navigation Components

### Gait-Based Path Following

For humanoid robots, we need to adapt the navigation to work with their walking patterns:

```python
import rclpy
from rclpy.node import Node
from nav2_core.controller import Controller
from nav2_core.types import (
    Costmap, 
    GlobalPlanner, 
    LocalPlanner, 
    Controller as ControllerInterface
)
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np


class HumanoidPathFollower(Controller):
    def __init__(self, name):
        super().__init__(name)
        self.logger = self.get_logger()
        self.active_goal = None
        self.path = None
        self.current_pose = None
        self.velocity_publisher = None
        self.laser_subscriber = None
        
    def configure(self, tf_buffer, costmap_ros, lifecycle_name):
        """
        Configure the controller
        """
        self.logger.info(f"Configuring {self.name}")
        
        # Initialize TF buffer and listener
        self.tf_buffer = tf_buffer
        self.costmap_ros = costmap_ros
        self.lifecycle_name = lifecycle_name
        
        # Create velocity command publisher
        self.velocity_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # Create laser scan subscriber for local obstacle detection
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )
        
        # Initialize local planner for humanoid-specific path following
        self.local_planner = HumanoidLocalPlanner()
        
    def activate(self):
        """Activate the controller"""
        self.logger.info(f"Activating {self.name}")
        
    def deactivate(self):
        """Deactivate the controller"""
        self.logger.info(f"Deactivating {self.name}")
        
    def cleanup(self):
        """Clean up the controller"""
        self.logger.info(f"Cleaning up {self.name}")
        
    def setPlan(self, path):
        """
        Set the global plan for the controller
        """
        self.logger.info("Setting path for humanoid path follower")
        self.path = path
    
    def computeVelocityCommands(self, pose, velocity):
        """
        Compute velocity commands based on current pose and path
        """
        # Calculate appropriate velocity command for humanoid robot
        cmd_vel = self.local_planner.compute_velocity_command(
            pose=pose,
            path=self.path,
            current_velocity=velocity
        )
        
        return cmd_vel, True, "Computed velocity command"
    
    def isGoalReached(self):
        """
        Check if the goal has been reached
        """
        if self.path and self.active_goal:
            # Calculate distance to goal
            goal_dist = self.calculate_distance_to_goal()
            return goal_dist < 0.3  # Goal tolerance for humanoid
        return False
    
    def calculate_distance_to_goal(self):
        """
        Calculate distance to the goal position
        """
        if self.path.poses:
            goal_pose = self.path.poses[-1]
            current_pose = self.current_pose
            if current_pose:
                dist = np.sqrt(
                    (goal_pose.pose.position.x - current_pose.pose.position.x)**2 +
                    (goal_pose.pose.position.y - current_pose.pose.position.y)**2
                )
                return dist
        return float('inf')
    
    def laser_callback(self, msg):
        """
        Handle laser scan data for obstacle detection
        """
        # Process laser scan to check for obstacles ahead
        min_distance = min(msg.ranges)
        if min_distance < 0.5:  # Obstacle within 50cm
            self.logger.warning(f"Obstacle detected at {min_distance:.2f}m, reducing speed")


class HumanoidLocalPlanner:
    def __init__(self):
        # Humanoid-specific parameters
        self.lookahead_distance = 0.8  # meters
        self.max_linear_speed = 0.6    # m/s
        self.max_angular_speed = 0.5   # rad/s
        self.min_linear_speed = 0.1    # m/s
        self.safety_distance = 0.4     # minimum distance to obstacles
        self.gait_adaptation_enabled = True
        
    def compute_velocity_command(self, pose, path, current_velocity):
        """
        Compute velocity command for humanoid robot considering gait constraints
        """
        # Get current position and orientation
        current_x = pose.pose.position.x
        current_y = pose.pose.position.y
        current_yaw = self.yaw_from_quaternion(pose.pose.orientation)
        
        # Find next waypoint based on lookahead distance
        target_idx = self.find_next_waypoint(path, current_x, current_y)
        
        if target_idx is None:
            # Stop if no path or no waypoints
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel
        
        # Calculate distance to target
        target_pose = path.poses[target_idx]
        dx = target_pose.pose.position.x - current_x
        dy = target_pose.pose.position.y - current_y
        distance_to_target = np.sqrt(dx**2 + dy**2)
        
        # Calculate desired orientation to target
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - current_yaw)
        
        # Adjust speed based on distance to target and obstacles
        linear_speed = min(self.max_linear_speed, distance_to_target * 0.5)
        linear_speed = max(self.min_linear_speed, linear_speed)
        
        # Apply gait adaptation if enabled
        if self.gait_adaptation_enabled:
            linear_speed = self.adapt_speed_for_gait(linear_speed, current_velocity)
        
        # Calculate angular velocity based on yaw error
        angular_speed = min(self.max_angular_speed, abs(yaw_error) * 1.5)
        angular_speed = np.sign(yaw_error) * angular_speed
        
        # Create and return velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_speed
        cmd_vel.angular.z = angular_speed
        
        return cmd_vel
    
    def find_next_waypoint(self, path, current_x, current_y):
        """
        Find the next waypoint along the path based on lookahead distance
        """
        if not path.poses:
            return None
        
        for i in range(len(path.poses)-1, -1, -1):
            waypoint = path.poses[i]
            dist = np.sqrt(
                (waypoint.pose.position.x - current_x)**2 + 
                (waypoint.pose.position.y - current_y)**2
            )
            if dist >= self.lookahead_distance:
                return i
        
        # If no waypoint is far enough, return the last one
        return len(path.poses) - 1
    
    def yaw_from_quaternion(self, orientation):
        """
        Extract yaw from quaternion
        """
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def adapt_speed_for_gait(self, desired_speed, current_velocity):
        """
        Adapt speed for humanoid gait constraints
        """
        # Humanoid robots need smooth acceleration/deceleration
        max_acc = 0.2  # m/s^2
        max_dec = 0.4  # m/s^2 (can brake faster than accelerate)
        
        dt = 0.05  # Assuming 20Hz control frequency
        
        # Calculate max allowed speed change
        if desired_speed > current_velocity.x:
            max_speed_change = max_acc * dt
        else:
            max_speed_change = max_dec * dt
        
        # Limit speed change
        actual_speed = current_velocity.x + np.sign(
            desired_speed - current_velocity.x
        ) * min(max_speed_change, abs(desired_speed - current_velocity.x))
        
        # Keep within bounds
        actual_speed = np.clip(actual_speed, 0.0, self.max_linear_speed)
        
        return actual_speed
```

### Obstacle Avoidance for Humanoid Robots

```python
class HumanoidObstacleAvoidance:
    def __init__(self):
        self.safety_distance = 0.5  # meters
        self.escape_threshold = 0.3  # meters (when to initiate escape)
        self.max_escape_attempts = 5  # maximum attempts to escape before re-planning
        
    def check_for_obstacles(self, laser_scan):
        """
        Check laser scan for obstacles in the path
        """
        # Check if obstacles exist within safety distance
        min_range = min([r for r in laser_scan.ranges if r > laser_scan.range_min])
        
        if min_range < self.safety_distance:
            # Find closest angle to obstacle
            obstacle_angle_idx = laser_scan.ranges.index(min_range)
            angle_increment = laser_scan.angle_increment
            obstacle_angle = laser_scan.angle_min + obstacle_angle_idx * angle_increment
            
            return True, min_range, obstacle_angle
        
        return False, float('inf'), 0.0
    
    def compute_escape_velocity(self, obstacle_distance, obstacle_angle):
        """
        Compute escape velocity when obstacle is too close
        """
        cmd_vel = Twist()
        
        # If obstacle is very close, stop and potentially back up
        if obstacle_distance < self.escape_threshold:
            # Decide escape direction based on sensor data
            # Look at sectors to find the clearest direction
            sector_size = 0.785  # 45 degrees in radians
            sectors = {
                'front': 0,
                'left': 0,
                'right': 0
            }
            
            # This is a simplified approach - in practice you'd use the full scan
            # For now, we'll just move laterally away from the obstacle
            cmd_vel.linear.x = 0.0
            cmd_vel.linear.y = 0.2 if obstacle_angle > 0 else -0.2  # Move right or left
        else:
            # Reduce speed but continue forward
            cmd_vel.linear.x = 0.2  # Slow speed
            cmd_vel.angular.z = 0.1 * np.sign(obstacle_angle)  # Slight turn away
        
        return cmd_vel
```

## Simulation Environment Setup

### Creating a Navigation Test Environment

```xml
<!-- world_navigation_test.world -->
<sdf version="1.7">
  <world name="navigation_test">
    <!-- Include empty world -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Add walls to create a maze-like environment -->
    <model name="wall_1">
      <pose>2 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="wall_2">
      <pose>-2 0 0 0 0 1.5707</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Add obstacles -->
    <model name="obstacle_1">
      <pose>1 1 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>1.0</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.5 0.5 1 1</ambient>
            <diffuse>0.5 0.5 1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="obstacle_2">
      <pose>-1 -1 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.8 1.0</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.8 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0.5 0.5 1</ambient>
            <diffuse>1 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Set up for humanoid robot -->
    <light name="directional_light" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>20</range>
      </attenuation>
      <direction>-0.5 -0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

## Navigation Performance Evaluation

### Creating Metrics and Evaluation Tools

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class NavigationEvaluator:
    def __init__(self):
        self.trajectory = []
        self.execution_times = []
        self.success_count = 0
        self.failure_count = 0
        self.path_efficiencies = []
        
    def record_pose(self, pose):
        """Record robot pose during navigation"""
        self.trajectory.append({
            'x': pose.pose.position.x,
            'y': pose.pose.position.y,
            'time': self.get_clock().now().nanoseconds / 1e9
        })
    
    def calculate_metrics(self, start_pose, goal_pose, planned_path):
        """Calculate navigation performance metrics"""
        if not self.trajectory:
            return {}
        
        # Calculate actual path length
        actual_length = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i]['x'] - self.trajectory[i-1]['x']
            dy = self.trajectory[i]['y'] - self.trajectory[i-1]['y']
            actual_length += np.sqrt(dx**2 + dy**2)
        
        # Calculate planned path length
        planned_length = 0.0
        for i in range(1, len(planned_path.poses)):
            dx = (planned_path.poses[i].pose.position.x - 
                  planned_path.poses[i-1].pose.position.x)
            dy = (planned_path.poses[i].pose.position.y - 
                  planned_path.poses[i-1].pose.position.y)
            planned_length += np.sqrt(dx**2 + dy**2)
        
        # Calculate efficiency
        path_efficiency = planned_length / actual_length if actual_length > 0 else 0
        self.path_efficiencies.append(path_efficiency)
        
        # Calculate success metrics
        final_pose = self.trajectory[-1] if self.trajectory else None
        if final_pose:
            goal_distance = np.sqrt(
                (final_pose['x'] - goal_pose.pose.position.x)**2 +
                (final_pose['y'] - goal_pose.pose.position.y)**2
            )
            success = goal_distance < 0.5  # Within 50cm of goal
        else:
            success = False
        
        # Calculate execution time
        if len(self.trajectory) > 0:
            execution_time = self.trajectory[-1]['time'] - self.trajectory[0]['time']
        else:
            execution_time = 0.0
        self.execution_times.append(execution_time)
        
        # Update success/failure counts
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        return {
            'success': success,
            'path_efficiency': path_efficiency,
            'execution_time': execution_time,
            'actual_path_length': actual_length,
            'planned_path_length': planned_length,
            'goal_distance': goal_distance if 'goal_distance' in locals() else float('inf')
        }
    
    def plot_trajectory(self):
        """Plot the navigation trajectory"""
        if not self.trajectory:
            print("No trajectory data to plot")
            return
        
        x_coords = [p['x'] for p in self.trajectory]
        y_coords = [p['y'] for p in self.trajectory]
        
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Actual Trajectory')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Navigation Trajectory')
        plt.legend()
        plt.show()
    
    def print_summary(self):
        """Print navigation performance summary"""
        total_attempts = self.success_count + self.failure_count
        
        if total_attempts > 0:
            success_rate = (self.success_count / total_attempts) * 100
            avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
            avg_efficiency = np.mean(self.path_efficiencies) if self.path_efficiencies else 0
            
            print(f"\n=== Navigation Performance Summary ===")
            print(f"Total Attempts: {total_attempts}")
            print(f"Successes: {self.success_count}")
            print(f"Failures: {self.failure_count}")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Average Execution Time: {avg_execution_time:.2f}s")
            print(f"Average Path Efficiency: {avg_efficiency:.2f}")
        else:
            print("No navigation attempts recorded")
```

## Practical Exercise: Creating a Navigation Demo

### Exercise Objective

Create a complete navigation demo for a humanoid robot in Gazebo simulation, including path planning, obstacle avoidance, and performance evaluation.

### Prerequisites

- Gazebo simulation environment
- Navigation2 stack installed
- Humanoid robot model in simulation
- Understanding of ROS 2 navigation concepts

### Exercise Steps

1. **Set up the simulation environment**:
   ```bash
   # Launch Gazebo with the navigation test world
   ros2 launch gazebo_ros gazebo.launch.py world:=$(pwd)/world_navigation_test.world
   
   # In another terminal, spawn the humanoid robot
   ros2 run gazebo_ros spawn_entity.py -entity humanoid_robot -file $(pwd)/humanoid_model.sdf -x 0 -y 0 -z 1
   ```

2. **Launch the navigation stack**:
   ```bash
   # Launch the navigation stack with humanoid parameters
   ros2 launch [your_robot_name]_nav2_bringup navigation_launch.py
   ```

3. **Implement the navigation node**:
   ```python
   # navigation_demo.py
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Twist
   from nav_msgs.msg import Path, Odometry
   from sensor_msgs.msg import LaserScan
   from builtin_interfaces.msg import Duration
   from rclpy.qos import QoSProfile
   import numpy as np
   
   class NavigationDemo(Node):
       def __init__(self):
           super().__init__('navigation_demo')
           
           # Publishers and subscribers
           self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
           self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
           self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
           self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
           
           # Robot state
           self.current_pose = None
           self.scan_data = None
           self.navigation_active = False
           
           # Navigation goals
           self.navigation_goals = [
               {'x': 2.0, 'y': 2.0},
               {'x': -2.0, 'y': 2.0},
               {'x': -2.0, 'y': -2.0},
               {'x': 2.0, 'y': -2.0},
           ]
           self.current_goal_index = 0
           
           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)
           
           # Timer for sending goals
           self.goal_timer = self.create_timer(5.0, self.send_next_goal)
           
           self.get_logger().info('Navigation demo node initialized')
       
       def odom_callback(self, msg):
           self.current_pose = msg.pose.pose
       
       def scan_callback(self, msg):
           self.scan_data = msg
       
       def send_next_goal(self):
           if self.current_goal_index < len(self.navigation_goals):
               goal = self.navigation_goals[self.current_goal_index]
               
               goal_msg = PoseStamped()
               goal_msg.header.stamp = self.get_clock().now().to_msg()
               goal_msg.header.frame_id = 'map'
               goal_msg.pose.position.x = goal['x']
               goal_msg.pose.position.y = goal['y']
               goal_msg.pose.position.z = 0.0
               goal_msg.pose.orientation.w = 1.0
               
               self.goal_pub.publish(goal_msg)
               self.get_logger().info(f'Sent goal: ({goal["x"]}, {goal["y"]})')
               
               self.current_goal_index += 1
       
       def control_loop(self):
           if not self.scan_data or not self.current_pose:
               return
           
           # Check if close to current goal (simple implementation)
           if self.current_goal_index > 0:
               current_goal = self.navigation_goals[self.current_goal_index - 1]
               distance_to_goal = np.sqrt(
                   (self.current_pose.position.x - current_goal['x'])**2 +
                   (self.current_pose.position.y - current_goal['y'])**2
               )
               
               if distance_to_goal < 0.5:  # Within 50cm
                   self.get_logger().info(f'Reached goal: ({current_goal["x"]}, {current_goal["y"]})')
                   return
           
           # Implement simple obstacle avoidance
           if self.scan_data:
               min_range = min([r for r in self.scan_data.ranges if r > self.scan_data.range_min])
               if min_range < 0.8:  # Obstacle within 80cm
                   self.get_logger().info(f'Obstacle detected at {min_range:.2f}m, avoiding')
                   # Stop robot
                   cmd_vel = Twist()
                   cmd_vel.linear.x = 0.0
                   cmd_vel.angular.z = 0.5  # Turn
                   self.cmd_vel_pub.publish(cmd_vel)
                   return
           
           # If no obstacles, continue navigation
           # In a real implementation, this would use the Navigation2 stack
           # For this demo, we'll just publish a simple command
           cmd_vel = Twist()
           cmd_vel.linear.x = 0.3  # Move forward slowly
           self.cmd_vel_pub.publish(cmd_vel)
   
   def main(args=None):
       rclpy.init(args=args)
       node = NavigationDemo()
       
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

4. **Run the complete demo**:
   ```bash
   # Terminal 1: Start Gazebo simulation
   ros2 launch gazebo_ros gazebo.launch.py world:=$(pwd)/world_navigation_test.world
   
   # Terminal 2: Spawn the robot
   ros2 run gazebo_ros spawn_entity.py -entity humanoid_robot -file $(pwd)/humanoid_model.sdf -x 0 -y 0 -z 1
   
   # Terminal 3: Start the navigation stack
   ros2 launch [your_robot_name]_nav2_bringup navigation_launch.py
   
   # Terminal 4: Start the navigation demo
   ros2 run [your_robot_name]_nav_demo navigation_demo.py
   ```

5. **Monitor the navigation performance**:
   ```bash
   # In another terminal, monitor the navigation status
   ros2 topic echo /navigation/status
   
   # Monitor robot pose
   ros2 topic echo /odom
   
   # Check the costmap
   ros2 run rviz2 rviz2 -d [your_robot_name]_nav_config.rviz
   ```

6. **Analyze results**:
   - Observe how the robot navigates through the environment
   - Check how obstacle avoidance works
   - Monitor the robot's path efficiency
   - Record performance metrics using the NavigationEvaluator

## Summary

In this chapter, we've created a complete navigation demonstration for humanoid robots in simulation:

1. **Navigation Stack Setup**: Configured Navigation2 for humanoid-specific requirements
2. **Humanoid Path Following**: Implemented gait-aware path following
3. **Obstacle Avoidance**: Created obstacle avoidance strategies for humanoid robots
4. **Simulation Environment**: Built test environments for navigation validation
5. **Performance Evaluation**: Implemented metrics and evaluation tools
6. **Practical Implementation**: Created a complete navigation demo

The navigation system we've built considers the unique challenges of humanoid robots, including their dynamic movement patterns, stability requirements, and sensor positioning. This provides a foundation for autonomous navigation in real-world environments.