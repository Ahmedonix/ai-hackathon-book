# Configuring Nav2 for Obstacle Avoidance in Humanoid Robotics

## Overview

Navigation 2 (Nav2) is ROS 2's navigation stack for mobile robots. When applied to humanoid robots, Nav2 provides essential path planning and obstacle avoidance capabilities. This section covers configuring Nav2 specifically for humanoid robotics applications, considering the unique challenges and requirements of bipedal navigation.

## Understanding Nav2 Architecture for Humanood Robots

### 1. Key Nav2 Components

Nav2 consists of several key components:

- **Navigation Server**: Coordinates navigation tasks
- **Global Planner**: Plans long-term routes
- **Local Planner**: Handles immediate obstacle avoidance
- **Controller**: Executes low-level motion commands
- **Costmap**: Represents obstacles and free space
- **Lifecycle Manager**: Manages node lifecycle states

### 2. Humanoid-Specific Considerations

For humanoid robots, Nav2 must account for:

- **Bipedal kinematics**: Different from wheeled robots
- **Balance constraints**: Limited foot placement options
- **Dynamic stability**: Need for continuous balance
- **Step height limitations**: Cannot climb steep obstacles
- **Footprint awareness**: Different collision geometry

## Configuring Global Planner for Humanoid Robots

### 1. NavFn Global Planner Configuration

For humanoid robots, the global planner needs special considerations regarding the robot's size and movement patterns:

```yaml
# config/global_planner.yaml
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: false
      width: 200
      height: 200
      resolution: 0.05  # Higher resolution for precise humanoid navigation
      origin_x: -50.0
      origin_y: -50.0
      
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0  # Humanoid can step over low obstacles
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
        cost_scaling_factor: 5.0  # Higher for humanoid safety
        inflation_radius: 1.0     # Larger than typical wheeled robots (was 0.55)
        inflate_unknown: false
        inflate_around_unknown: true
      
      always_send_full_costmap: true

planner_server:
  ros__parameters:
    expected_planner_frequency: 2.0  # Slower for more stable humanoid planning
    use_sim_time: true
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 1.0  # Allow larger tolerance for humanoid stepping
      use_astar: true  # Use A* for better path quality
      allow_unknown: false  # More conservative approach
```

### 2. Creating a Custom Global Planner Configuration

For advanced humanoid navigation, you might need a custom planner configuration:

```yaml
# config/humanoid_global_planner.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 1.0  # More conservative for humanoid
    use_sim_time: true
    planner_plugins: ["HumanoidPathPlanner"]
    HumanoidPathPlanner:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.8  # Allow some tolerance for stepped navigation
      use_astar: true
      allow_unknown: false
      planner_window_x: 0.0  # Disable windowing for consistent planning
      planner_window_y: 0.0
      default_tolerance: 0.5
```

## Configuring Local Planner for Humanoid Obstacle Avoidance

### 1. DWA Local Planner Configuration

For humanoid robots, we'll configure the DWA (Dynamic Window Approach) planner with humanoid-specific parameters:

```yaml
# config/local_planner.yaml
controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 20.0  # 20Hz - appropriate for humanoid balance
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "nav2_mppi_controller::MPPIC"  # Humanoid-friendly controller
      debug_visualizations: false
      control_horizon: 15  # Longer horizon for humanoid stability
      dt: 0.05  # 20Hz control loop
      max_heading_change: 0.78  # About 45 degrees per control cycle (conservative for humanoid balance)
      speed_units: "m/s"
      speed_lim_v: 0.4  # Conservative speed for humanoid stability
      speed_lim_w: 0.8
      acc_lim_v: 0.4   # Lower acceleration for humanoid balance
      acc_lim_w: 1.0
      decel_factor: 1.1  # Slightly more deceleration for safety
      oscillation_score_penalty: 0.1  # Reduced to allow some oscillation for balance recovery
      oscillation_magic_number: 0.85
      oscillation_reset_angle: 0.34  # Reduce this for humanoid (20 degrees vs default 45)
      xy_goal_tolerance: 0.3  # Increased for humanoid stepping accuracy
      yaw_goal_tolerance: 0.3  # Increased for humanoid rotation accuracy
      stateful: true
      adaptive_lookahead: true  # Enable adaptive lookahead for smoother motion

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0  # Match controller frequency
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 6.0  # Larger window for humanoid navigation planning
      height: 6.0
      resolution: 0.05  # Higher resolution for precise humanoid navigation
      robot_radius: 0.4  # Larger radius for humanoid (typical humanoid torso width)
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 4.0  # Higher cost scaling for humanoid safety
        inflation_radius: 0.8     # Larger inflation for humanoid safety margin
        inflate_unknown: false
        inflate_around_unknown: true
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: false
        origin_z: 0.0
        z_resolution: 0.2   # Higher resolution for humanoid foot placement
        z_voxels: 16        # More voxels to cover humanoid height
        max_obstacle_height: 1.8  # Obstacles higher than humanoid need special handling
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0  # Humanoid can step over low obstacles
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.05
      always_send_full_costmap: true

progress_checker:
  ros__parameters:
    plugin: "nav2_controller::SimpleProgressChecker"
    required_movement_radius: 0.5  # Reduced for humanoid stepping
    movement_time_allowance: 10.0

goal_checker:
  ros__parameters:
    plugin: "nav2_controller::SimpleGoalChecker"
    xy_goal_tolerance: 0.3  # Increased for humanoid stepping accuracy
    yaw_goal_tolerance: 0.3  # Increased for humanoid rotation accuracy
    stateful: true
```

### 2. Alternative: TEB Local Planner for Humanoid Robots

TEB (Timed Elastic Band) planner can also be configured for humanoid robots:

```yaml
# config/teb_local_planner.yaml
controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 10.0  # Slower for TEB with humanoid constraints
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "nav2_te_b_local_planner::TEBLocalPlanner"
      # Robot configuration
      max_vel_x: 0.4  # Conservative for humanoid stability
      max_vel_x_backwards: 0.2
      max_vel_theta: 0.6
      acc_lim_x: 0.4   # Conservative acceleration for humanoid balance
      acc_lim_theta: 1.0
      
      # TEB specific configuration
      min_samples: 3
      max_samples: 50
      global_plan_overwrite_orientation: false  # Maintain orientation for humanoid
      allow_init_with_backwards_motion: true
      max_global_plan_lookahead_dist: 3.0
      global_plan_viapoint_sep: 0.5  # Larger separation for humanoid steps
      feasibility_check_no_poses: 5
      publish_feedback: true
      
      # Optimization parameters
      no_inner_iterations: 5
      no_outer_iterations: 4
      optimization_activate: true
      optimization_verbose: false
      penalty_epsilon: 0.1
      weight_max_vel_x: 2.0
      weight_max_vel_theta: 1.0
      weight_acc_lim_x: 1.0
      weight_acc_lim_theta: 1.0
      weight_kinematics_nh: 1000.0
      weight_kinematics_forward_drive: 1.0
      weight_kinematics_turning_radius: 1.0
      weight_optimaltime: 1.0
      weight_shortest_path: 0.5
      weight_obstacle: 100.0
      weight_inflation: 0.2
      weight_dynamic_obstacle: 10.0
      selection_alternative_time_cost: false

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 6.0
      height: 6.0
      resolution: 0.05
      robot_radius: 0.4
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 4.0
        inflation_radius: 0.8
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: false
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 16
        max_obstacle_height: 1.8
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
```

## Behavior Trees for Humanoid Navigation

### 1. Custom Behavior Tree for Humanoid Robots

Create a customized behavior tree that accounts for humanoid-specific navigation challenges:

```xml
<!-- bt_navigator.xml for humanoid robots -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="NavigateWithReplanning">
      <RecoveryNode number_of_retries="6" name="NavigateRecovery">
        <PipelineSequence name="NavigateWithSmoothing">
          <RateController hz="1.0">
            <ComputePathToPose goal_updater_node="goal_updater" planner_node="GridBased" />
          </RateController>
          <RecoveryNode number_of_retries="1" name="SmoothPathRecovery">
            <SmoothPath path_smoothing_node="PathSmoother" />
            <ReactiveFallback name="SmoothPathFallback">
              <GoalReached goal_checker_node="goal_checker" />
              <PathUnreachable />
            </ReactiveFallback>
          </RecoveryNode>
          <TruncatePath distance="1.0" truncation_node="truncate_path" />
        </PipelineSequence>
        <ReactiveFallback name="FollowPathRecoveryFallback">
          <GoalReached goal_checker_node="goal_checker" />
          <FollowPath path_executor_node="FollowPath" />
          <StuckOnUnreachableGoalUpdater goal_updater_node="goal_updater" />  <!-- Humanoid-specific recovery -->
        </ReactiveFallback>
      </RecoveryNode>
      <ReactiveFallback name="RecoveryFallback">
        <GoalReached goal_checker_node="goal_checker" />
        <RecoveryNode number_of_retries="2" name="Clearing">
          <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap" />
          <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap" />
        </RecoveryNode>
        <RecoveryNode number_of_retries="2" name="Wait10s">
          <Wait wait_duration="10" />
        </RecoveryNode>
      </ReactiveFallback>
    </Sequence>
  </BehaviorTree>
  
  <BehaviorTree ID="StuckOnUnreachableGoalUpdater">
    <UpdateGoalFromRobot position_node="GoalPositionShifter" tolerance_node="GoalToleranceShifter" />
  </BehaviorTree>
</root>
```

### 2. Behavior Tree Configuration

Create a behavior tree configuration file:

```yaml
# config/behavior_trees/humanoid_navigator.xml
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    max_loop_rate: 100
    # Use the humanoid-specific behavior tree
    default_bt_xml_filename: "package://humanoid_nav2_config/behavior_trees/humanoid_navigator.xml"
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    interrupt_on_task_preemption: true
    # Humanoid-specific parameters
    goal_addition_distance: 0.8  # Minimum distance between waypoints for humanoid
    goal_addition_angle: 0.524  # Approximately 30 degrees
```

## Recovery Behaviors for Humanood Robots

### 1. Humanoid-Specific Recovery Behaviors

Humanoid robots need specialized recovery behaviors due to their kinematic constraints:

```yaml
# config/recovery_behaviors.yaml
recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait", "humanoid_escape"]
    spin:
      plugin: "nav2_recoveries::Spin"
      ideal_spin_angle: 1.57  # 90 degrees for tight turns
      max_rotation_attempts: 10  # More attempts for humanoid stability
      min_rotational_vel: 0.2  # Conservative for humanoid balance
      max_rotational_vel: 0.4  # Conservative spinning for humanoid stability
      rotational_acc_lim: 0.5
    backup:
      plugin: "nav2_recoveries::BackUp"
      backup_dist: 0.5  # Conservative backup for humanoid
      backup_speed: 0.05  # Slow reverse for humanoid stability
      sim_time: 2.0
      vx_max: 0.05
      vx_min: -0.05
    wait:
      plugin: "nav2_recoveries::Wait"
      wait_duration: 5s
    humanoid_escape:
      plugin: "humanoid_nav2_plugins::HumanoidEscape"
      escape_distance: 0.6  # Distance to move to escape obstacle
      escape_attempts: 3    # Number of escape direction attempts
      step_size: 0.2        # Size of each step during escape maneuver
      max_escape_time: 30.0 # Max time for escape maneuver in seconds
```

### 2. Creating a Custom Escape Recovery Plugin

Create a Python node that implements a humanoid-specific escape recovery:

```python
# scripts/humanoid_escape_recovery.py
#!/usr/bin/env python3

"""
Humanoid-specific escape recovery behavior.
Used when the robot gets stuck in a tight space.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from nav2_msgs.action import Recover
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import time


class HumanoidEscapeRecovery(Node):
    def __init__(self):
        super().__init__('humanoid_escape_recovery')
        
        # Declare parameters
        self.declare_parameter('escape_distance', 0.6)  # meters
        self.declare_parameter('escape_attempts', 3)
        self.declare_parameter('step_size', 0.2)  # meters
        self.declare_parameter('max_escape_time', 30.0)  # seconds
        self.declare_parameter('max_forward_attempts', 5)
        
        # Get parameters
        self.escape_distance = self.get_parameter('escape_distance').value
        self.escape_attempts = self.get_parameter('escape_attempts').value
        self.step_size = self.get_parameter('step_size').value
        self.max_escape_time = self.get_parameter('max_escape_time').value
        self.max_forward_attempts = self.get_parameter('max_forward_attempts').value
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # Store current state
        self.current_odom = None
        self.current_scan = None
        self.escape_active = False
        
        # Action server
        self.recover_action_server = ActionServer(
            self,
            Recover,
            'humanoid_escape_recovery',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.get_logger().info('Humanoid Escape Recovery Initialized')

    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom = msg

    def scan_callback(self, msg):
        """Store current laser scan"""
        self.current_scan = msg

    def execute_callback(self, goal_handle):
        """Execute the escape recovery behavior"""
        self.get_logger().info('Executing humanoid escape recovery...')
        
        feedback = Recover.Feedback()
        result = Recover.Result()
        
        start_time = self.get_clock().now()
        
        # Try different escape directions
        escape_success = False
        for attempt in range(self.escape_attempts):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.status = Recover.Result.Failed
                return result
            
            self.get_logger().info(f'Attempt {attempt + 1}/{self.escape_attempts} for escape')
            
            # Try forward movement first
            escape_success = self.attempt_forward_escape()
            
            if not escape_success:
                # Try sideways movements
                escape_success = self.attempt_sideways_escape()
            
            if escape_success:
                self.get_logger().info('Escape succeeded!')
                break
            
            # Check if we've exceeded maximum time
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > self.max_escape_time:
                self.get_logger().warn('Max escape time exceeded')
                break
            
            # Brief pause between attempts
            time.sleep(0.5)
        
        if escape_success:
            goal_handle.succeed()
            result.status = Recover.Result.SUCCEEDED
        else:
            goal_handle.succeed()  # Consider as completed even if unsuccessful
            result.status = Recover.Result.Failed
        
        self.escape_active = False
        return result

    def attempt_forward_escape(self):
        """Attempt forward movement for escape"""
        if not self.current_scan:
            return False
        
        # Check if forward path is clear
        if not self.is_forward_path_clear(self.current_scan):
            self.get_logger().info('Forward path not clear, aborting forward escape')
            return False
        
        # Move forward in steps
        steps_needed = int(self.escape_distance / self.step_size)
        current_position = self.get_current_position()
        
        for step in range(steps_needed):
            if not self.escape_active:
                return False  # Recovery was terminated
            
            # Calculate target position
            target_x = current_position[0] + (step + 1) * self.step_size
            target_y = current_position[1]  # Stay in same y position
            
            # Move to target position
            success = self.move_to_position(target_x, target_y)
            
            if not success:
                self.get_logger().warn(f'Failed to move to position step {step + 1}')
                return False
            
            # Check if path is still clear
            if not self.current_scan or not self.is_forward_path_clear(self.current_scan):
                self.get_logger().info('Path became obstructed during forward escape')
                return False
        
        return True

    def attempt_sideways_escape(self):
        """Attempt sideways movement for escape"""
        if not self.current_scan:
            return False
        
        # Try moving left or right
        directions = ['left', 'right']
        
        for direction in directions:
            if not self.escape_active:
                return False
            
            if self.is_sideways_path_clear(self.current_scan, direction):
                self.get_logger().info(f'Trying {direction} escape path')
                
                # Calculate target position
                current_pos = self.get_current_position()
                if direction == 'left':
                    target_x = current_pos[0]
                    target_y = current_pos[1] + self.escape_distance
                else:  # right
                    target_x = current_pos[0]
                    target_y = current_pos[1] - self.escape_distance
                
                # Move to target position
                steps_needed = int(self.escape_distance / self.step_size)
                
                for step in range(1, steps_needed + 1):
                    if not self.escape_active:
                        return False
                    
                    step_multiplier = 1 if direction == 'left' else -1
                    step_target_y = current_pos[1] + step_multiplier * (step * self.step_size)
                    
                    success = self.move_to_position(current_pos[0], step_target_y)
                    
                    if not success:
                        self.get_logger().warn(f'Failed to move {direction} on step {step}')
                        break
                    
                    # Check if path is still clear
                    if not self.current_scan or not self.is_sideways_path_clear(self.current_scan, direction):
                        self.get_logger().info(f'Sideways {direction} path became obstructed')
                        break
                
                if success:
                    return True
        
        return False

    def is_forward_path_clear(self, scan_msg, distance=1.0):
        """Check if forward path is clear of obstacles"""
        if not scan_msg:
            return False
        
        # Check middle section of scan (forward direction)
        center_idx = len(scan_msg.ranges) // 2
        sector_size = max(1, len(scan_msg.ranges) // 12)  # 30-degree sector
        
        forward_sector = scan_msg.ranges[
            center_idx - sector_size:center_idx + sector_size
        ]
        
        for range_val in forward_sector:
            if not math.isinf(range_val) and not math.isnan(range_val) and range_val < distance:
                return False  # Obstacle detected in forward path
        
        return True

    def is_sideways_path_clear(self, scan_msg, direction, distance=1.0):
        """Check if left or right path is clear of obstacles"""
        if not scan_msg:
            return False
        
        # Check left or right sectors (90 degrees from forward)
        if direction == 'left':
            start_idx = len(scan_msg.ranges) // 4  # 45 degrees left
            end_idx = len(scan_msg.ranges) // 4 + len(scan_msg.ranges) // 12  # 30-degree sector
        else:  # right
            start_idx = 3 * len(scan_msg.ranges) // 4 - len(scan_msg.ranges) // 12  # 30-degree sector on right
            end_idx = 3 * len(scan_msg.ranges) // 4  # 45 degrees right
        
        sector = scan_msg.ranges[max(0, start_idx):min(len(scan_msg.ranges), end_idx)]
        
        for range_val in sector:
            if not math.isinf(range_val) and not math.isnan(range_val) and range_val < distance:
                return False  # Obstacle detected in sideways path
        
        return True

    def move_to_position(self, target_x, target_y, tolerance=0.05):
        """Move robot to a specific position using simple proportional control"""
        if not self.current_odom:
            return False
        
        start_time = self.get_clock().now()
        
        while True:
            if not self.current_odom:
                continue
            
            # Calculate current position and distance to target
            current_x = self.current_odom.pose.pose.position.x
            current_y = self.current_odom.pose.pose.position.y
            distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # Check if we've reached the target
            if distance < tolerance:
                # Stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                return True
            
            # Check if we've exceeded time limit
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 10.0:  # 10 second timeout for reaching target
                self.get_logger().warn('Timeout while moving to target position')
                return False
            
            # Calculate velocity command
            cmd = Twist()
            cmd.linear.x = min(0.2, max(0.05, distance * 0.5))  # Proportional control
            cmd.linear.y = 0.0
            
            # Calculate required angular adjustment to face target
            target_angle = math.atan2(target_y - current_y, target_x - current_x)
            current_angle = self.get_robot_yaw()
            angle_diff = self.normalize_angle(target_angle - current_angle)
            
            cmd.angular.z = min(0.4, max(-0.4, angle_diff * 2.0))  # Proportional control for rotation
            
            self.cmd_vel_pub.publish(cmd)
            
            # Sleep briefly (simulating control loop)
            time.sleep(0.1)

    def get_current_position(self):
        """Get current position from odometry"""
        if self.current_odom:
            return (
                self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y
            )
        return (0, 0)

    def get_robot_yaw(self):
        """Get robot's yaw angle from odometry orientation"""
        if not self.current_odom:
            return 0.0
        
        orientation = self.current_odom.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Humanoid Escape Recovery Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    escape_node = HumanoidEscapeRecovery()
    
    try:
        rclpy.spin(escape_node)
    except KeyboardInterrupt:
        escape_node.get_logger().info('Node interrupted by user')
    finally:
        escape_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Configuration for Humanoid Nav2

### 1. Complete Nav2 Launch File

Create a launch file that incorporates all humanoid-specific configurations:

```python
# launch/humanoid_nav2.launch.py
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
    autostart = LaunchConfiguration('autostart', default='true')
    params_file = LaunchConfiguration('params_file',
                                    default=PathJoinSubstitution([
                                        FindPackageShare('humanoid_nav2_config'),
                                        'config',
                                        'humanoid_nav2_params.yaml'
                                    ]))
    default_bt_xml_filename = LaunchConfiguration(
        'default_bt_xml_filename',
        default=PathJoinSubstitution([
            FindPackageShare('humanoid_nav2_config'),
            'behavior_trees',
            'humanoid_navigator.xml'
        ])
    )
    
    # Navigation launch (using default Nav2 bringup with custom params)
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
                'map_server',
                'planner_server',
                'controller_server', 
                'recoveries_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother'
            ]}
        ]
    )
    
    # Humanoid-specific recovery behavior
    humanoid_escape_recovery = Node(
        package='humanoid_nav2_config',
        executable='humanoid_escape_recovery',
        name='humanoid_escape_recovery',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Navigation goal publisher (for testing)
    navigation_goal_publisher = Node(
        package='nav2_msgs',
        executable='navigation_goal_publisher',
        name='navigation_goal_publisher',
        output='screen'
    )
    
    # Transform publisher for Nav2 coordinate frames
    robot_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_nav2_config'),
                'config',
                'ekf.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch lifecycle manager
        lifecycle_manager,
        
        # Launch navigation stack after a delay
        TimerAction(
            period=2.0,
            actions=[navigation]
        ),
        
        # Launch humanoid-specific recovery after navigation is up
        TimerAction(
            period=4.0,
            actions=[humanoid_escape_recovery]
        ),
        
        # Launch robot localization after recovery
        TimerAction(
            period=5.0,
            actions=[robot_localization]
        ),
    ])
```

### 2. Parameter Configuration File

Create the complete parameter file: `config/humanoid_nav2_params.yaml`

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
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
    resample_thresh: 0.5
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

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
    default_bt_xml_filename: "package://humanoid_nav2_config/behavior_trees/humanoid_navigator.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_updated_goal_condition_bt_node
    - nav2_is_path_valid_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
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
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node
    - nav2_is_stuck_on_unreachable_goal_bt_node
    - nav2_update_origin_action_bt_node

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

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIC"
      debug_visualizations: false
      control_horizon: 15
      dt: 0.05
      max_heading_change: 0.78
      speed_units: "m/s"
      speed_lim_v: 0.4
      speed_lim_w: 0.8
      acc_lim_v: 0.4
      acc_lim_w: 1.0
      decel_factor: 1.1
      oscillation_score_penalty: 0.1
      oscillation_magic_number: 0.85
      oscillation_reset_angle: 0.34
      xy_goal_tolerance: 0.3
      yaw_goal_tolerance: 0.3
      stateful: true

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: True
      width: 6.0
      height: 6.0
      resolution: 0.05
      robot_radius: 0.4
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 4.0
        inflation_radius: 0.8
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 16
        max_obstacle_height: 1.8
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
          obstacle_min_range: 0.05
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: False
      width: 200
      height: 200
      resolution: 0.05
      origin_x: -50.0
      origin_y: -50.0
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
          obstacle_min_range: 0.05
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0
        inflation_radius: 1.0
        inflate_unknown: False
        inflate_around_unknown: True
      always_send_full_costmap: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65
    map_subscribe_transient_local: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 2.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 1.0
      use_astar: True
      allow_unknown: False

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait", "humanoid_escape"]
    spin:
      plugin: "nav2_recoveries::Spin"
      ideal_spin_angle: 1.57
      max_rotation_attempts: 10
      min_rotational_vel: 0.2
      max_rotational_vel: 0.4
      rotational_acc_lim: 0.5
    backup:
      plugin: "nav2_recoveries::BackUp"
      backup_dist: 0.5
      backup_speed: 0.05
      sim_time: 2.0
      vx_max: 0.05
      vx_min: -0.05
    wait:
      plugin: "nav2_recoveries::Wait"
      wait_duration: 5s
    humanoid_escape:
      plugin: "humanoid_nav2_plugins::HumanoidEscape"
      escape_distance: 0.6
      escape_attempts: 3
      step_size: 0.2
      max_escape_time: 30.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      wait_time: 1s
```

## Testing the Humanoid Navigation System

### 1. Build and Launch

```bash
# Build the package
cd ~/humanoid_ws
source /opt/ros/iron/setup.bash
colcon build --packages-select humanoid_nav2_config
source install/setup.bash

# Launch the complete navigation system
ros2 launch humanoid_nav2_config humanoid_nav2.launch.py
```

### 2. Send Navigation Commands

In a separate terminal:

```bash
# Send a navigation goal
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

### 3. Monitor Navigation Performance

```bash
# Monitor navigation status
ros2 topic echo /behavior_tree_log

# Monitor costmaps
ros2 run rviz2 rviz2

# In RViz, add the global and local costmaps to visualize them
```

## Performance Optimization for Humanoid Robots

### 1. Tuning for Stability

For better stability with humanoid robots:

```yaml
# controller-specific adjustments for humanoid stability
FollowPath:
  plugin: "nav2_mppi_controller::MPPIC"
  control_horizon: 20  # Longer horizon for smoother movements
  dt: 0.1  # Slower control loop for stability
  max_heading_change: 0.5  # More conservative heading changes
  speed_lim_v: 0.3  # Slower movement for stability
  acc_lim_v: 0.2   # Gentler acceleration for balance
  oscillation_reset_angle: 0.17  # Reduce to minimize oscillation
```

### 2. Costmap Tuning for Humanoid Steps

```yaml
# Costmap settings optimized for humanoid stepping
inflation_layer:
  plugin: "nav2_costmap_2d::InflationLayer"
  cost_scaling_factor: 6.0  # Higher for extra safety margin
  inflation_radius: 1.2     # Larger for humanoid safety
```

## Troubleshooting Common Issues

### 1. Robot Oscillation in Tight Spaces

**Problem**: Robot oscillates when trying to navigate through tight spaces.

**Solution**: 
- Increase `oscillation_reset_angle` in the goal checker
- Reduce the `xy_goal_tolerance` to allow closer approach
- Increase the costmap inflation radius

### 2. Getting Stuck Near Obstacles

**Problem**: Robot gets stuck near obstacles and doesn't trigger recovery behaviors.

**Solution**:
- Decrease `max_rotation_attempts` to trigger backup sooner
- Lower the costmap resolution for better detail
- Ensure the robot's radius is properly configured

### 3. Poor Path Quality

**Problem**: Generated paths are not suitable for humanoid robots.

**Solution**:
- Use the TEB planner instead of DWA for more flexible paths
- Increase the global plan lookahead distance
- Tune the global planner's tolerance settings

## Next Steps

With Nav2 properly configured for humanoid navigation and obstacle avoidance, you're now ready to build complete AI pipelines that incorporate perception, navigation, and control components. The navigation stack you've configured provides the foundation for autonomous movement that accounts for the challenges of humanoid locomotion.