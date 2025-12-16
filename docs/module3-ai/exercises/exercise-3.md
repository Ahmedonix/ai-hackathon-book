---
title: Navigation Exercise
description: Practical hands-on exercise for humanoid robot navigation in simulation
sidebar_position: 10
---

# Navigation Exercise

## Overview

This hands-on exercise focuses on implementing and testing navigation systems for humanoid robots in simulation environments. You'll create a complete navigation pipeline that includes path planning, obstacle avoidance, locomotion control, and integration with perception systems.

## Learning Objectives

- Implement path planning algorithms for humanoid robots
- Integrate navigation with perception systems
- Create locomotion controllers that work with navigation
- Test navigation in complex environments
- Evaluate navigation performance metrics

## Prerequisites

- Completed perception pipeline exercise
- Understanding of ROS 2 navigation concepts
- Isaac Sim or Gazebo simulation environment
- Basic knowledge of control theory

## Exercise Setup

### Step 1: Environment Initialization

Let's start by setting up our navigation workspace:

```bash
# Create navigation exercise directory
mkdir -p ~/isaac_sim_projects/navigation_exercises
cd ~/isaac_sim_projects/navigation_exercises
```

### Step 2: Create Navigation Configuration Files

First, we'll create the configuration files for our navigation system:

```yaml
# navigation_params.yaml
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

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_remove_passed_goals_action_bt_node

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
      robot_radius: 0.3
      plugins: ["obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
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
```

## Path Planning Implementation

### Step 3: Create Path Planning Module

Let's implement our path planning system:

```python
# path_planner.py
import numpy as np
from scipy.spatial.distance import cdist
import heapq
from typing import List, Tuple, Optional

class PathPlanner:
    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.map_data = None
        self.origin = None
        
    def set_map(self, map_data: np.ndarray, origin: Tuple[float, float, float]):
        """
        Set the occupancy map for path planning
        
        Args:
            map_data: 2D numpy array with occupancy values (0: free, 100: occupied)
            origin: (x, y, theta) origin of the map in world coordinates
        """
        self.map_data = map_data
        self.origin = origin
        
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal using A* algorithm
        """
        if self.map_data is None:
            raise ValueError("Map data not set")
        
        # Convert world coordinates to map indices
        start_idx = self.world_to_map(start)
        goal_idx = self.world_to_map(goal)
        
        # Check if start and goal are valid
        if not self.is_valid_index(start_idx) or not self.is_valid_index(goal_idx):
            print("Start or goal position is out of map bounds")
            return None
            
        if self.is_occupied(start_idx) or self.is_occupied(goal_idx):
            print("Start or goal position is occupied")
            return None
        
        # Run A* path planning
        path_indices = self.a_star(start_idx, goal_idx)
        
        if path_indices is None:
            print("No path found")
            return None
        
        # Convert path indices back to world coordinates
        path = [self.map_to_world(idx) for idx in path_indices]
        return path
    
    def world_to_map(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to map indices"""
        x, y = world_pos
        origin_x, origin_y, _ = self.origin
        map_x = int((x - origin_x) / self.resolution)
        map_y = int((y - origin_y) / self.resolution)
        return (map_x, map_y)
    
    def map_to_world(self, map_idx: Tuple[int, int]) -> Tuple[float, float]:
        """Convert map indices to world coordinates"""
        map_x, map_y = map_idx
        origin_x, origin_y, _ = self.origin
        x = map_x * self.resolution + origin_x
        y = map_y * self.resolution + origin_y
        return (x, y)
    
    def is_valid_index(self, idx: Tuple[int, int]) -> bool:
        """Check if map index is within valid bounds"""
        x, y = idx
        h, w = self.map_data.shape
        return 0 <= x < w and 0 <= y < h
    
    def is_occupied(self, idx: Tuple[int, int]) -> bool:
        """Check if a map cell is occupied"""
        x, y = idx
        if not self.is_valid_index(idx):
            return True  # Treat out-of-bounds as occupied
        return self.map_data[y, x] >= 50  # Consider cells with 50+ as occupied
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* path planning algorithm"""
        # Define possible movements (8-connected)
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Movement costs (diagonal movement costs more)
        move_costs = [
            1.414, 1.0, 1.414,
            1.0,         1.0,
            1.414, 1.0, 1.414
        ]
        
        # Initialize data structures
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct path
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            # Check neighbors
            for i, (dx, dy) in enumerate(movements):
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_index(neighbor):
                    continue
                
                if self.is_occupied(neighbor):
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + move_costs[i]
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic distance between two points"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # Using Chebyshev distance (allows diagonal movement)
        return max(dx, dy)

# Example usage and testing
def test_path_planner():
    # Create a simple test map (0 = free, 100 = occupied)
    test_map = np.zeros((20, 20))
    # Add some obstacles
    test_map[5:15, 10] = 100  # Vertical wall
    test_map[10, 5:15] = 100  # Horizontal wall
    
    planner = PathPlanner(resolution=0.1)
    planner.set_map(test_map, (0, 0, 0))
    
    start = (0, 0)
    goal = (1.5, 1.5)
    
    path = planner.plan_path(start, goal)
    
    if path:
        print(f"Found path with {len(path)} waypoints")
        for i, point in enumerate(path[:10]):  # Show first 10 points
            print(f"  Waypoint {i}: {point}")
        if len(path) > 10:
            print(f"  ... and {len(path) - 10} more points")
    else:
        print("No path found")

if __name__ == "__main__":
    test_path_planner()
```

### Step 4: Create Navigation Controller

Now, let's implement the locomotion controller for humanoid robots:

```python
# locomotion_controller.py
import numpy as np
from typing import Tuple, List
import math

class HumanoidLocomotionController:
    def __init__(self):
        # Gait parameters for humanoid walking
        self.step_length = 0.3  # meters
        self.step_height = 0.05  # meters
        self.step_duration = 0.5  # seconds
        self.max_linear_speed = 0.6  # m/s
        self.max_angular_speed = 0.8  # rad/s
        
        # Walking state
        self.left_support = True
        self.step_in_progress = False
        self.step_phase = 0.0  # 0.0 to 1.0
        
        # PID controllers for position tracking
        self.linear_pid = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}
        self.angular_pid = {'kp': 2.0, 'ki': 0.1, 'kd': 0.05}
        
        # Internal state
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0
        self.integral_linear_error = 0.0
        self.integral_angular_error = 0.0
        
    def compute_command(self, 
                       current_pose: Tuple[float, float, float],  # (x, y, theta)
                       target_pose: Tuple[float, float, float],   # (x, y, theta)
                       dt: float = 0.05) -> Tuple[float, float]:  # (linear_vel, angular_vel)
        """
        Compute velocity command to move toward target pose
        """
        current_x, current_y, current_theta = current_pose
        target_x, target_y, target_theta = target_pose
        
        # Calculate distance and angle to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance_to_target = math.sqrt(dx*dx + dy*dy)
        
        # Calculate desired heading
        desired_theta = math.atan2(dy, dx)
        
        # Calculate errors
        linear_error = distance_to_target
        angular_error = self.normalize_angle(desired_theta - current_theta)
        
        # Use PID controller for linear velocity
        self.integral_linear_error += linear_error * dt
        derivative_linear = (linear_error - self.prev_linear_error) / dt if dt > 0 else 0
        linear_vel = (
            self.linear_pid['kp'] * linear_error + 
            self.linear_pid['ki'] * self.integral_linear_error + 
            self.linear_pid['kd'] * derivative_linear
        )
        
        # Use PID controller for angular velocity
        self.integral_angular_error += angular_error * dt
        derivative_angular = (angular_error - self.prev_angular_error) / dt if dt > 0 else 0
        angular_vel = (
            self.angular_pid['kp'] * angular_error + 
            self.angular_pid['ki'] * self.integral_angular_error + 
            self.angular_pid['kd'] * derivative_angular
        )
        
        # Update previous errors
        self.prev_linear_error = linear_error
        self.prev_angular_error = angular_error
        
        # Limit velocities based on humanoid constraints
        linear_vel = np.clip(linear_vel, -self.max_linear_speed, self.max_linear_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        # If very close to target, slow down
        if distance_to_target < 0.2:
            linear_vel *= 0.5  # Slow down when close
        
        return linear_vel, angular_vel
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def is_at_waypoint(self, 
                      current_pose: Tuple[float, float, float],
                      waypoint: Tuple[float, float, float],
                      pos_tolerance: float = 0.3,
                      angle_tolerance: float = 0.2) -> bool:
        """Check if robot is close enough to a waypoint"""
        current_x, current_y, current_theta = current_pose
        wp_x, wp_y, wp_theta = waypoint
        
        # Calculate distance to waypoint
        distance = math.sqrt((current_x - wp_x)**2 + (current_y - wp_y)**2)
        
        # Check if position is within tolerance
        pos_reached = distance < pos_tolerance
        
        # Check if orientation is within tolerance
        angle_diff = abs(self.normalize_angle(current_theta - wp_theta))
        angle_reached = angle_diff < angle_tolerance
        
        return pos_reached and angle_reached
    
    def generate_footstep_plan(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Generate footstep plan from navigation path for humanoid robot
        """
        footsteps = []
        
        if len(path) < 2:
            return footsteps
        
        # Start with initial position
        footsteps.append((path[0][0], path[0][1], path[0][2]))
        
        # Add footsteps along the path
        for i in range(1, len(path)):
            # For humanoid, we don't generate individual footstep positions
            # but we could if we needed to plan specific foot placements
            # For now, just return the path as waypoints
            footsteps.append(path[i])
        
        # In a real humanoid implementation, you would:
        # 1. Plan specific left/right foot positions
        # 2. Ensure stability at each step
        # 3. Consider COM control during stepping
        
        return footsteps

# Example usage
def test_locomotion_controller():
    controller = HumanoidLocomotionController()
    
    # Test following a simple path
    current_pose = (0.0, 0.0, 0.0)  # Starting position
    target_pose = (2.0, 1.0, 0.0)   # Target position
    
    print("Testing locomotion controller:")
    for i in range(100):  # Simulate 5 seconds of control (20 Hz)
        linear_vel, angular_vel = controller.compute_command(current_pose, target_pose)
        
        # Simple pose integration (in real system, use proper odometry)
        dt = 0.05
        current_pose = (
            current_pose[0] + linear_vel * math.cos(current_pose[2]) * dt,
            current_pose[1] + linear_vel * math.sin(current_pose[2]) * dt,
            current_pose[2] + angular_vel * dt
        )
        
        if i % 20 == 0:  # Print every second
            distance = math.sqrt(
                (current_pose[0] - target_pose[0])**2 + 
                (current_pose[1] - target_pose[1])**2
            )
            print(f"Step {i}: Position {current_pose[:2]}, Distance to target: {distance:.2f}m")
        
        # Check if reached
        if controller.is_at_waypoint(current_pose, target_pose):
            print(f"Reached target at step {i}")
            break

if __name__ == "__main__":
    test_locomotion_controller()
```

## Navigation System Integration

### Step 5: Create the Complete Navigation System

Now, let's integrate all components into a complete navigation system:

```python
# navigation_system.py
import numpy as np
import math
from typing import List, Tuple, Optional
from collections import deque
import time

from path_planner import PathPlanner
from locomotion_controller import HumanoidLocomotionController

class NavigationSystem:
    def __init__(self):
        self.path_planner = PathPlanner()
        self.locomotion_controller = HumanoidLocomotionController()
        
        # Navigation state
        self.current_goal = None
        self.current_path = []
        self.path_index = 0
        self.navigation_active = False
        self.goal_reached = False
        
        # Obstacle avoidance
        self.local_map = None
        self.safe_distance = 0.5  # meters
        self.escape_threshold = 0.3
        
        # Waypoint tracking
        self.waypoint_queue = deque()
        self.current_waypoint = None
        
        # Performance metrics
        self.path_length = 0.0
        self.execution_time = 0.0
        self.collision_avoidance_engaged = 0
        
    def set_map(self, map_data: np.ndarray, origin: Tuple[float, float, float]):
        """Set the occupancy map for navigation"""
        self.path_planner.set_map(map_data, origin)
        
    def set_goal(self, goal: Tuple[float, float, float]):
        """
        Set a navigation goal
        
        Args:
            goal: (x, y, theta) goal position and orientation in world coordinates
        """
        self.current_goal = goal
        self.navigation_active = True
        self.goal_reached = False
        self.path_index = 0
        
        # Plan path to goal
        start_pos = (0.0, 0.0)  # This should be current robot position in real implementation
        goal_pos = (goal[0], goal[1])
        
        self.current_path = self.path_planner.plan_path(start_pos, goal_pos)
        
        if self.current_path is not None:
            # Add orientation to each waypoint
            oriented_path = []
            for i, (x, y) in enumerate(self.current_path):
                # Calculate desired orientation toward next waypoint
                if i < len(self.current_path) - 1:
                    next_x, next_y = self.current_path[i+1]
                    desired_theta = math.atan2(next_y - y, next_x - x)
                else:
                    desired_theta = goal[2]  # Use final goal orientation
                
                oriented_path.append((x, y, desired_theta))
            
            self.current_path = oriented_path
            self.path_length = self.calculate_path_length(self.current_path)
            
            # Initialize waypoint queue
            self.waypoint_queue = deque(self.current_path)
            self.current_waypoint = self.waypoint_queue.popleft() if self.waypoint_queue else None
        else:
            print("Failed to plan path to goal")
            self.navigation_active = False
    
    def calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate total length of the path"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length
    
    def update(self, 
               current_pose: Tuple[float, float, float],
               sensor_data: dict = None,
               dt: float = 0.05) -> Tuple[float, float]:
        """
        Update navigation system and return velocity command
        
        Args:
            current_pose: Current robot pose (x, y, theta)
            sensor_data: Dictionary with sensor readings
            dt: Time step
        
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        if not self.navigation_active:
            return 0.0, 0.0
        
        # Check for obstacles in sensor data
        if sensor_data and self.avoid_obstacles(sensor_data):
            self.collision_avoidance_engaged += 1
            # Return escape velocity
            return self.escape_velocity(sensor_data)
        
        # Check if we need a new waypoint
        if (self.current_waypoint is not None and 
            self.locomotion_controller.is_at_waypoint(current_pose, self.current_waypoint)):
            if self.waypoint_queue:
                self.current_waypoint = self.waypoint_queue.popleft()
            else:
                # Reached the end of the path
                # Check if close enough to final goal
                if self.locomotion_controller.is_at_waypoint(current_pose, self.current_goal):
                    self.navigation_active = False
                    self.goal_reached = True
                    return 0.0, 0.0
                else:
                    # Set the final goal as the current waypoint
                    self.current_waypoint = self.current_goal
        
        # Calculate command using locomotion controller
        if self.current_waypoint is not None:
            linear_vel, angular_vel = self.locomotion_controller.compute_command(
                current_pose, self.current_waypoint, dt
            )
        else:
            # No more waypoints, stop the robot
            linear_vel, angular_vel = 0.0, 0.0
        
        return linear_vel, angular_vel
    
    def avoid_obstacles(self, sensor_data: dict) -> bool:
        """Check if obstacles require navigation adjustment"""
        if 'lidar' in sensor_data:
            lidar_ranges = sensor_data['lidar']
            min_distance = min(lidar_ranges) if lidar_ranges else float('inf')
            
            return min_distance < self.safe_distance
        elif 'range' in sensor_data:
            # General range sensor data
            ranges = sensor_data['range']
            min_distance = min(ranges) if ranges else float('inf')
            
            return min_distance < self.safe_distance
        
        return False
    
    def escape_velocity(self, sensor_data: dict) -> Tuple[float, float]:
        """Generate velocity command to escape from obstacles"""
        # Find the direction with the most free space
        if 'lidar' in sensor_data:
            lidar_ranges = sensor_data['lidar']
            
            # Divide LIDAR readings into sectors
            sector_size = len(lidar_ranges) // 8  # 8 sectors
            sector_distances = []
            
            for i in range(8):
                start_idx = i * sector_size
                end_idx = min(start_idx + sector_size, len(lidar_ranges))
                sector_avg = sum(lidar_ranges[start_idx:end_idx]) / len(lidar_ranges[start_idx:end_idx])
                sector_distances.append(sector_avg)
            
            # Find safest direction (max distance)
            safest_sector = sector_distances.index(max(sector_distances))
            sector_angle = safest_sector * (2 * math.pi / 8)  # Convert to radians
            
            # Generate command to move in safest direction
            linear_vel = 0.2  # Slow, careful movement
            angular_vel = 0.5 * (sector_angle - math.pi)  # Turn toward safest direction
            
            return linear_vel, angular_vel
        else:
            # If no specific sensor data, just slow down and turn
            return 0.1, 0.3
    
    def is_goal_reached(self) -> bool:
        """Check if the goal has been reached"""
        return self.goal_reached
    
    def get_navigation_status(self) -> dict:
        """Get current navigation status"""
        return {
            'active': self.navigation_active,
            'goal_reached': self.goal_reached,
            'path_length': self.path_length,
            'waypoints_remaining': len(self.waypoint_queue),
            'collision_avoidance_engagements': self.collision_avoidance_engaged
        }

# Example usage
def test_navigation_system():
    # Create a simple test map
    test_map = np.zeros((40, 40))
    # Add some obstacles
    test_map[15:25, 15] = 100  # Vertical wall
    test_map[20, 10:20] = 100  # Horizontal wall
    
    nav_system = NavigationSystem()
    nav_system.set_map(test_map, (-2.0, -2.0, 0.0))
    
    # Set a goal
    nav_system.set_goal((1.0, 1.0, 0.0))
    
    # Simulate navigation
    current_pose = (0.0, 0.0, 0.0)
    print("Starting navigation...")
    
    for step in range(500):  # Max 25 seconds of simulation
        # Simple sensor simulation (in real implementation, use actual sensor data)
        # For this example, we'll add some dummy range data
        sensor_data = {
            'lidar': [2.0 for _ in range(360)]  # Simulated LIDAR data
        }
        
        # Add some obstacles to the simulation
        if 1.0 < current_pose[0] < 1.5 and 0.0 < current_pose[1] < 1.5:
            # Add obstacle in a specific region
            sensor_data['lidar'][180] = 0.3  # Obstacle straight ahead
            sensor_data['lidar'][135] = 0.8  # More space to the right
            sensor_data['lidar'][225] = 0.8  # More space to the left
        
        linear_vel, angular_vel = nav_system.update(current_pose, sensor_data)
        
        # Simple pose integration (in real system, use proper odometry)
        dt = 0.05
        current_pose = (
            current_pose[0] + linear_vel * math.cos(current_pose[2]) * dt,
            current_pose[1] + linear_vel * math.sin(current_pose[2]) * dt,
            current_pose[2] + angular_vel * dt
        )
        
        if step % 50 == 0:  # Print every 2.5 seconds
            target_distance = math.sqrt(
                (current_pose[0] - 1.0)**2 + (current_pose[1] - 1.0)**2
            )
            print(f"Step {step}: Position {current_pose[:2]}, Velocity: ({linear_vel:.2f}, {angular_vel:.2f})")
            print(f"  Distance to goal: {target_distance:.2f}m")
        
        if nav_system.is_goal_reached():
            print(f"Goal reached at step {step}!")
            break
    else:
        print("Max steps reached without reaching goal")
    
    # Print final status
    status = nav_system.get_navigation_status()
    print(f"Final status: {status}")

if __name__ == "__main__":
    test_navigation_system()
```

## Simulation Environment

### Step 6: Create Simulation Environment

Let's create a simulation environment to test our navigation system:

```python
# navigation_simulation.py
#!/usr/bin/env python3

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Tuple

from navigation_system import NavigationSystem

class NavigationSimulation:
    def __init__(self):
        # Initialize navigation system
        self.nav_system = NavigationSystem()
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # Heading angle in radians
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Simulation state
        self.simulation_time = 0.0
        self.dt = 0.05  # 20 Hz
        self.max_simulation_time = 60.0  # 60 seconds max
        
        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.robot_patch = None
        self.goal_circle = None
        self.path_line = None
        self.robot_trajectory = []
        
    def setup_environment(self):
        """Set up the simulation environment"""
        # Create a map with obstacles
        map_size = 40
        resolution = 0.1  # meters per cell
        self.map_data = np.zeros((map_size, map_size))
        
        # Add obstacles (100 = occupied)
        # Vertical wall
        self.map_data[150:250, 150] = 100
        # Horizontal wall
        self.map_data[200, 100:200] = 100
        # Diagonal obstacle
        for i in range(50):
            self.map_data[50+i, 300+i] = 100
        
        # Set the map in the navigation system
        self.nav_system.set_map(self.map_data, (-2.0, -2.0, 0.0))
        
        # Set a goal for the robot
        goal_pos = (1.5, 1.0, 0.0)
        self.nav_system.set_goal(goal_pos)
        
        # Set up visualization
        self.ax.imshow(
            self.map_data.T, 
            extent=[-2, 2, -2, 2], 
            origin='lower', 
            cmap='gray', 
            alpha=0.6
        )
        
        # Draw goal
        self.goal_circle = plt.Circle(goal_pos[:2], 0.1, color='green', label='Goal')
        self.ax.add_patch(self.goal_circle)
        
        # Draw robot
        robot_size = 0.15
        self.robot_patch = plt.Rectangle(
            (self.robot_x - robot_size/2, self.robot_y - robot_size/2),
            robot_size, robot_size, 
            color='blue', 
            label='Robot'
        )
        self.ax.add_patch(self.robot_patch)
        
        # Direction indicator for robot
        self.robot_arrow = plt.arrow(
            self.robot_x, self.robot_y, 
            0.1 * math.cos(self.robot_theta), 0.1 * math.sin(self.robot_theta),
            head_width=0.05, head_length=0.05, 
            fc='blue', ec='blue'
        )
        self.ax.add_patch(self.robot_arrow)
        
        # Set up trajectory line
        self.trajectory_line, = self.ax.plot([], [], 'r-', linewidth=2, label='Trajectory')
        
        # Set axis properties
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Humanoid Robot Navigation Simulation')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
    def update_robot_state(self):
        """Update robot state based on current velocities"""
        # Update position using differential drive model
        self.robot_x += self.linear_vel * math.cos(self.robot_theta) * self.dt
        self.robot_y += self.linear_vel * math.sin(self.robot_theta) * self.dt
        self.robot_theta += self.angular_vel * self.dt
        
        # Keep theta in [-pi, pi]
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta), 
            math.cos(self.robot_theta)
        )
        
        # Add current position to trajectory
        self.robot_trajectory.append((self.robot_x, self.robot_y))
    
    def get_sensor_data(self):
        """Simulate sensor data based on robot position"""
        # Simulate LIDAR data
        num_rays = 360
        lidar_ranges = [2.0] * num_rays  # Default range 2m
        
        # Add some simple obstacle detection
        for i in range(num_rays):
            angle = self.robot_theta + (i * 2 * math.pi / num_rays)
            
            # Cast a ray in this direction to detect obstacles
            step_size = 0.05
            max_range = 2.0
            current_range = 0.0
            
            while current_range < max_range:
                test_x = self.robot_x + current_range * math.cos(angle)
                test_y = self.robot_y + current_range * math.sin(angle)
                
                # Convert to map coordinates
                map_x = int((test_x + 2.0) / 0.1)  # Assuming 0.1m resolution
                map_y = int((test_y + 2.0) / 0.1)
                
                if (0 <= map_x < 40 and 0 <= map_y < 40 and 
                    self.map_data[map_x, map_y] >= 50):
                    # Obstacle detected
                    lidar_ranges[i] = current_range
                    break
                
                current_range += step_size
        
        return {
            'lidar': lidar_ranges
        }
    
    def update_simulation(self, frame):
        """Update simulation for animation"""
        if self.simulation_time < self.max_simulation_time:
            # Get sensor data
            sensor_data = self.get_sensor_data()
            
            # Update navigation system
            current_pose = (self.robot_x, self.robot_y, self.robot_theta)
            self.linear_vel, self.angular_vel = self.nav_system.update(
                current_pose, sensor_data, self.dt
            )
            
            # Update robot state
            self.update_robot_state()
            
            # Update visualization
            # Update robot position
            self.robot_patch.set_xy((self.robot_x - 0.075, self.robot_y - 0.075))
            
            # Update direction indicator
            self.ax.patches.remove(self.robot_arrow)
            self.robot_arrow = plt.arrow(
                self.robot_x, self.robot_y,
                0.1 * math.cos(self.robot_theta), 0.1 * math.sin(self.robot_theta),
                head_width=0.05, head_length=0.05,
                fc='blue', ec='blue'
            )
            self.ax.add_patch(self.robot_arrow)
            
            # Update trajectory
            if len(self.robot_trajectory) > 1:
                traj_x, traj_y = zip(*self.robot_trajectory)
                self.trajectory_line.set_data(traj_x, traj_y)
            
            # Check if goal is reached
            if self.nav_system.is_goal_reached():
                print(f"Goal reached at time {self.simulation_time:.2f}s!")
                print(f"Final position: ({self.robot_x:.2f}, {self.robot_y:.2f})")
                
                # Print navigation statistics
                stats = self.nav_system.get_navigation_status()
                print(f"Navigation statistics: {stats}")
                
                # Stop simulation
                plt.close()
            
            # Update time
            self.simulation_time += self.dt
            
            # Print status every few seconds
            if int(self.simulation_time) % 5 == 0 and hasattr(self, '_last_print_time'):
                if self.simulation_time - self._last_print_time >= 5.0:
                    print(f"Time: {self.simulation_time:.1f}s, "
                          f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f}), "
                          f"Velocity: ({self.linear_vel:.2f}, {self.angular_vel:.2f})")
                    self._last_print_time = self.simulation_time
            elif not hasattr(self, '_last_print_time'):
                self._last_print_time = self.simulation_time
        
        return [self.robot_patch, self.robot_arrow, self.trajectory_line]
    
    def run_simulation(self):
        """Run the navigation simulation"""
        print("Setting up navigation simulation...")
        self.setup_environment()
        
        print("Starting simulation...")
        print("Goal position: (1.5, 1.0)")
        print("Robot starting position: (0.0, 0.0)")
        
        # Create animation
        anim = FuncAnimation(
            self.fig, 
            self.update_simulation, 
            frames=1200,  # 60 seconds at 20 Hz
            interval=50,  # 50ms per frame = 20 FPS
            blit=False,
            repeat=False
        )
        
        plt.show()

def main():
    """Main function to run the navigation simulation"""
    try:
        sim = NavigationSimulation()
        sim.run_simulation()
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
        return 0
    except Exception as e:
        print(f"Error during simulation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Exercise Tasks

### Step 7: Complete Implementation Tasks

Complete the following tasks to finish the navigation exercise:

1. **Implement Path Planning**:
   - Create the path planner module
   - Test with different map configurations
   - Verify path optimality and obstacle avoidance

2. **Fine-tune Locomotion Controller**:
   - Adjust PID parameters for stable movement
   - Test with different terrain types
   - Ensure smooth transitions between waypoints

3. **Integrate Navigation System**:
   - Connect all components together
   - Test obstacle avoidance behavior
   - Validate goal reaching accuracy

4. **Run Simulation**:
   - Execute the navigation simulation
   - Observe robot behavior in complex environments
   - Analyze performance metrics

5. **Evaluate Performance**:
   - Measure path efficiency
   - Check obstacle avoidance effectiveness
   - Validate timing constraints

## Performance Evaluation

### Step 8: Navigation Metrics

Implement metrics to evaluate navigation performance:

```python
# navigation_metrics.py
import numpy as np
from typing import List, Tuple
import math

class NavigationMetrics:
    def __init__(self):
        self.trajectory = []
        self.execution_times = []
        self.success_count = 0
        self.failure_count = 0
        self.path_efficiencies = []
        self.obstacle_encounters = []
        
    def add_position(self, position: Tuple[float, float]):
        """Add robot position to trajectory"""
        self.trajectory.append(position)
        
    def calculate_metrics(self, start_pos: Tuple[float, float], 
                         goal_pos: Tuple[float, float], 
                         planned_path: List[Tuple[float, float]]) -> dict:
        """Calculate navigation performance metrics"""
        
        if not self.trajectory:
            return {}
        
        # Calculate actual path length
        actual_length = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i-1][0]
            dy = self.trajectory[i][1] - self.trajectory[i-1][1]
            actual_length += math.sqrt(dx*dx + dy*dy)
        
        # Calculate planned path length
        planned_length = 0.0
        for i in range(1, len(planned_path)):
            dx = planned_path[i][0] - planned_path[i-1][0]
            dy = planned_path[i][1] - planned_path[i-1][1]
            planned_length += math.sqrt(dx*dx + dy*dy)
        
        # Calculate efficiency
        path_efficiency = planned_length / actual_length if actual_length > 0 else 0
        
        # Calculate final distance to goal
        final_pos = self.trajectory[-1] if self.trajectory else start_pos
        goal_distance = math.sqrt(
            (final_pos[0] - goal_pos[0])**2 + 
            (final_pos[1] - goal_pos[1])**2
        )
        
        # Determine success
        success = goal_distance < 0.3  # Within 30cm of goal
        
        # Calculate metrics
        metrics = {
            'success': success,
            'path_efficiency': path_efficiency,
            'actual_path_length': actual_length,
            'planned_path_length': planned_length,
            'final_goal_distance': goal_distance,
            'trajectory_points_count': len(self.trajectory),
            'trajectory': self.trajectory.copy()
        }
        
        return metrics
    
    def plot_trajectory(self, ax, start_pos, goal_pos):
        """Plot the navigation trajectory"""
        if not self.trajectory:
            return
        
        # Extract x and y coordinates
        x_coords, y_coords = zip(*self.trajectory)
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Actual Path')
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        ax.plot(goal_pos[0], goal_pos[1], 'gs', markersize=12, label='Goal')
        
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.set_title('Navigation Trajectory')
    
    def print_summary(self):
        """Print navigation performance summary"""
        if not self.trajectory:
            print("No trajectory data available")
            return
        
        print("\n=== Navigation Performance Summary ===")
        print(f"Trajectory points: {len(self.trajectory)}")
        print(f"Total distance traveled: {self.calculate_total_distance():.2f} m")
        
        if len(self.trajectory) > 1:
            start_pos = self.trajectory[0]
            end_pos = self.trajectory[-1]
            displacement = math.sqrt(
                (end_pos[0] - start_pos[0])**2 + 
                (end_pos[1] - start_pos[1])**2
            )
            print(f"Net displacement: {displacement:.2f} m")
            print(f"Path efficiency: {displacement/self.calculate_total_distance():.2f}")
    
    def calculate_total_distance(self) -> float:
        """Calculate total distance of the trajectory"""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_dist = 0.0
        for i in range(1, len(self.trajectory)):
            dx = self.trajectory[i][0] - self.trajectory[i-1][0]
            dy = self.trajectory[i][1] - self.trajectory[i-1][1]
            total_dist += math.sqrt(dx*dx + dy*dy)
        
        return total_dist
```

## Challenge: Advanced Navigation

As an additional challenge, try to:

1. Implement more sophisticated navigation algorithms (D* Lite, RRT*)
2. Add dynamic obstacle avoidance
3. Create a multi-goal navigation task
4. Integrate with perception system from previous exercise
5. Add navigation recovery behaviors

## Summary

In this exercise, you've implemented a complete navigation system for humanoid robots:

1. **Path Planning**: Created an A* path planner for finding optimal routes
2. **Locomotion Control**: Developed a controller for humanoid-specific movement
3. **Navigation System**: Integrated planning, control, and obstacle avoidance
4. **Simulation Environment**: Tested the system in a simulated environment
5. **Performance Evaluation**: Implemented metrics to assess navigation quality

This navigation system provides the foundation for autonomous humanoid robot operation in complex environments, enabling them to reach goals while avoiding obstacles and adapting to changing conditions.