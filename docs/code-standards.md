# Code Standards and Formatting Guidelines: Physical AI & Humanoid Robotics Book

## Overview

This document establishes code standards and formatting guidelines for the Physical AI & Humanoid Robotics curriculum. These standards ensure consistency across all code examples, promote best practices, and facilitate learning and maintenance of robotics software.

## Python Coding Standards

### 1. PEP 8 Compliance
All Python code should comply with PEP 8, the official Python style guide, with the following specifications:

#### Naming Conventions
```python
# Variables and functions: lowercase with underscores
robot_position = 0.0
calculate_distance = lambda x, y: (x**2 + y**2)**0.5

# Constants: uppercase with underscores
MAX_LINEAR_VELOCITY = 0.5  # m/s
WHEEL_RADIUS = 0.05  # meters

# Classes: PascalCase
class DifferentialDriveController:
    pass

class URDFConverter:
    pass

# Private members: prefix with underscore
class RobotNode:
    def __init__(self):
        self._internal_state = 0  # Internal state
        self._private_helper()    # Private method
    
    def _private_helper(self):
        pass
```

#### Import Organization
```python
# Standard library imports first
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import cv2
import rclpy

# ROS imports
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import LaserScan, Image

# Local imports
from .utils import calculate_transform
from .constants import ROBOT_CONFIG
```

### 2. Line Length and Whitespace
- Maximum line length: 88 characters (preferred by Black formatter)
- Use hanging indents for multi-line statements
- Add whitespace around operators and after commas

```python
# Correct
def calculate_trajectory(
    initial_position, 
    target_position, 
    max_velocity=0.5, 
    tolerance=0.01
):
    """
    Calculate trajectory to move from initial to target position.
    
    Args:
        initial_position (tuple): Starting (x, y, theta)
        target_position (tuple): Target (x, y, theta)  
        max_velocity (float): Maximum allowed velocity (m/s)
        tolerance (float): Position tolerance for completion (m)
        
    Returns:
        list: List of intermediate waypoints
    """
    trajectory_points = []
    current_pos = initial_position
    
    while distance(current_pos, target_position) > tolerance:
        next_waypoint = plan_next_waypoint(
            current_pos, target_position, max_velocity
        )
        trajectory_points.append(next_waypoint)
        current_pos = next_waypoint
    
    return trajectory_points

# Wrong - too long
def calculate_trajectory(initial_position, target_position, max_velocity=0.5, tolerance=0.01):
    trajectory_points = []
    # This line is getting way too long and exceeds the recommended character limit

# Correct - properly broken up
if (robot_state.position_x > boundary.x_max or 
    robot_state.position_y > boundary.y_max or
    robot_state.position_x < boundary.x_min or
    robot_state.position_y < boundary.y_min):
    self.get_logger().warn("Robot is approaching boundary limits")
```

### 3. Comments and Docstrings
Use the Google Python Style Guide for docstrings:

```python
class PerceptionPipeline:
    """A class to handle perception pipeline operations for humanoid robots.
    
    This class manages the processing of sensor data to detect objects,
    recognize features, and provide environmental understanding for 
    autonomous humanoid robots.
    
    Attributes:
        model_path (str): Path to the perception model
        confidence_threshold (float): Minimum confidence for detections
        sensor_topics (dict): Mapping of sensor types to ROS topics
    """
    
    def __init__(self, model_path, confidence_threshold=0.7):
        """Initialize the perception pipeline.
        
        Args:
            model_path (str): Path to the trained perception model
            confidence_threshold (float, optional): Confidence threshold 
                for accepting detections. Defaults to 0.7.
                
        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If confidence_threshold is not between 0 and 1
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.sensor_topics = {
            'camera': '/camera/color/image_raw',
            'lidar': '/scan',
            'imu': '/imu/data'
        }
    
    def process_lidar_data(self, scan_msg):
        """Process LiDAR scan data to detect obstacles.
        
        Args:
            scan_msg (sensor_msgs.msg.LaserScan): Raw LiDAR scan data
            
        Returns:
            list: List of obstacle positions in [x, y] format relative 
                to robot coordinate frame
            
        Example:
            >>> obstacles = pipeline.process_lidar_data(scan_msg)
            >>> print(obstacles)
            [[1.2, 0.5], [0.8, -1.3]]
        """
        # Implementation here
        pass
```

### 4. ROS 2 Node Structure
Follow a consistent structure for ROS 2 nodes:

```python
#!/usr/bin/env python3

"""ROS 2 node for humanoid robot navigation controller.

This node implements a controller for humanoid robot navigation
using a model predictive control approach.
"""

import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class NavigationController(LifecycleNode):
    """Navigation controller for humanoid robot.
    
    Implements model predictive control for humanoid robot navigation.
    """
    
    def __init__(self):
        """Initialize the navigation controller node."""
        super().__init__('navigation_controller')
        
        # Declare parameters
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('control_frequency', 10.0)
        
        # Get parameters
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.control_freq = self.get_parameter('control_frequency').value
        
        # Initialize publishers and subscribers
        self.velocity_pub = None
        self.scan_sub = None
        self.obstacle_pub = None
        
        # Node state
        self.current_pose = None
        self.target_pose = None
        self.obstacle_detected = False
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure node.
        
        Args:
            state: Current lifecycle state
            
        Returns:
            TransitionCallbackReturn.SUCCESS if configuration successful
        """
        self.get_logger().info("Configuring navigation controller")
        
        # Create publishers and subscribers
        self.velocity_pub = self.create_publisher(
            Twist, 
            'cmd_vel', 
            10
        )
        self.obstacle_pub = self.create_publisher(
            Bool, 
            'obstacle_detected', 
            10
        )
        
        # Create timers
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self._control_loop
        )
        
        return TransitionCallbackReturn.SUCCESS
    
    def _control_loop(self):
        """Main control loop for navigation."""
        if not self.target_pose:
            return
        
        # Calculate control commands
        twist_cmd = self._calculate_velocity_command()
        
        # Publish command
        if twist_cmd:
            self.velocity_pub.publish(twist_cmd)
    
    def _calculate_velocity_command(self) -> Optional[Twist]:
        """Calculate velocity command based on current state.
        
        Returns:
            Twist command or None if no command should be sent
        """
        # Implementation here
        pass


def main(args=None):
    """Main function to run the navigation controller node."""
    rclpy.init(args=args)
    
    try:
        node = NavigationController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## C++ Coding Standards (When Applicable)

### 1. Naming Conventions
```cpp
// Variables and functions: camelCase
double robotPosition = 0.0;
std::vector<double> calculateJointAngles();

// Constants: SCREAMING_SNAKE_CASE
constexpr double MAX_LINEAR_VELOCITY = 0.5;
const std::string DEFAULT_FRAME_ID = "base_link";

// Classes: PascalCase
class ForwardKinematicsSolver {
public:
    ForwardKinematicsSolver() = default;
    ~ForwardKinematicsSolver() = default;
    
    // Public methods
    bool calculatePose(const std::vector<double>& joint_angles,
                      geometry_msgs::msg::Pose& pose);
    
private:
    // Private members: trailing underscore
    std::vector<std::vector<double>> transformation_matrices_;
    rclcpp::Logger logger_{rclcpp::get_logger("forward_kinematics")};
};
```

### 2. Header Guards and Includes
```cpp
#ifndef NAVIGATION_CONTROLLER_HPP
#define NAVIGATION_CONTROLLER_HPP

#include <memory>
#include <vector>

// ROS 2 includes
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

// Standard library includes
#include <cmath>
#include <limits>

namespace navigation {

class NavigationController : public rclcpp::Node {
public:
    explicit NavigationController(const rclcpp::NodeOptions& options);
    
    // Public interface methods
    void setTargetPosition(double x, double y);
    bool isNavigationComplete() const;
    
private:
    // Private implementation details
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void controlTimerCallback();
    
    // Member variables
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    
    // Navigation state
    double target_x_{0.0};
    double target_y_{0.0};
    double current_x_{0.0};
    double current_y_{0.0};
    
    // Parameters
    double linear_velocity_limit_{0.5}; // m/s
    double angular_velocity_limit_{1.0}; // rad/s
};

} // namespace navigation

#endif // NAVIGATION_CONTROLLER_HPP
```

## URDF/XML Standards

### 1. Structure and Organization
```xml
<?xml version="1.0"?>
<!-- Base URDF for humanoid robot -->
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Include common definitions -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/properties.xacro"/>
  
  <!-- Base link definition -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia 
        ixx="0.1" ixy="0.0" ixz="0.0"
        iyy="0.1" iyz="0.0"
        izz="0.2"/>
    </inertial>
  </link>
  
  <!-- Use Xacro macros for repeated elements -->
  <xacro:macro name="leg_link" params="prefix side">
    <link name="${prefix}_${side}_leg">
      <visual>
        <geometry>
          <cylinder length="0.4" radius="0.03"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.4" radius="0.03"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
        <inertia ixx="0.01"  ixy="0"  ixz="0"
                 iyy="0.01"  iyz="0"
                 izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Use the macro to create both legs -->
  <xacro:leg_link prefix="right" side="front"/>
  <xacro:leg_link prefix="left" side="front"/>
  
</robot>
```

## Launch File Standards

### 1. YAML Configuration and Launch Files
```yaml
# config/navigation_params.yaml
# Navigation controller parameters
nav2_controller:
  ros__parameters:
    # Controller parameters
    controller_frequency: 20.0
    velocity_scaling_factor: 1.0
    
    # Trajectory generator
    trajectory_generator:
      max_velocity: 0.5
      max_acceleration: 1.0
      look_ahead_distance: 0.5
    
    # Safety parameters
    safety:
      obstacle_proximity_threshold: 0.5
      emergency_stop_distance: 0.2
```

```xml
<!-- launch/navigation_stack.launch.py -->
<launch>
  <!-- Launch arguments -->
  <arg name="use_sim_time" default="false"/>
  <arg name="robot_namespace" default=""/>
  
  <!-- Parameter files -->
  <arg name="params_file" 
       default="$(find-pkg-share my_robot_navigation)/config/navigation_params.yaml"/>
  
  <!-- Navigation controller node -->
  <node pkg="my_robot_navigation" 
        exec="navigation_controller" 
        name="navigation_controller"
        namespace="$(var robot_namespace)"
        respawn="true"
        respawn_delay="2">
    
    <!-- Pass parameters -->
    <param from="$(var params_file)"/>
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    
    <!-- Remappings -->
    <remap from="cmd_vel" to="cmd_vel_nav"/>
    <remap from="scan" to="front_lidar/scan"/>
  </node>
  
  <!-- Obstacle detector node -->
  <node pkg="my_robot_perception" 
        exec="obstacle_detector" 
        name="obstacle_detector"
        namespace="$(var robot_namespace)"
        respawn="true">
    
    <param from="$(var params_file)"/>
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    
    <!-- Node-specific parameters -->
    <param name="min_obstacle_dist" value="0.3"/>
    <param name="detection_range" value="3.0"/>
  </node>
  
</launch>
```

## Documentation Standards

### 1. Code Documentation
Every function and class should have a docstring explaining:
- Purpose of the code
- Parameters and return values
- Possible exceptions
- Usage examples when appropriate

```python
def plan_path(start_pose, goal_pose, obstacle_map=None, 
              path_resolution=0.05, step_size=0.1):
    """Plan path from start pose to goal pose using RRT algorithm.
    
    Args:
        start_pose (Pose): Starting pose with x, y, theta
        goal_pose (Pose): Goal pose with x, y, theta
        obstacle_map (OccupancyGrid, optional): Static obstacle map
        path_resolution (float): Minimum distance between path points
        step_size (float): Maximum distance for single RRT step
        
    Returns:
        List[Pose]: Waypoints forming the planned path, or empty list
                   if no path is found
            
    Example:
        >>> start = Pose(x=0.0, y=0.0, theta=0.0)
        >>> goal = Pose(x=2.0, y=2.0, theta=0.0)
        >>> path = plan_path(start, goal)
        
    Raises:
        ValueError: If start or goal positions are invalid
    """
    # Implementation
    pass
```

### 2. Inline Code Comments
Use comments to explain complex logic, but not obvious code:

```python
# Good: Explains WHY this value is chosen
# Using 0.7 confidence threshold based on empirical tests
# showing optimal balance between precision and recall
confidence_threshold = 0.7

# Good: Explains complex algorithm logic
# Calculate shortest path using Dijkstra's algorithm
# with distance heuristic for efficiency
distance_map = dijkstra_with_heuristic(graph, start_point, end_point)

# Bad: Comments stating the obvious  
velocity_msg.linear.x = 0.5  # Set linear velocity to 0.5 (unnecessary)

# Acceptable: Clarifying complex expressions
# Convert RPM to rad/s: multiply by 2π and divide by 60
angular_velocity = rpm_value * 2 * math.pi / 60
```

## Performance and Efficiency

### 1. Memory Management
```python
# Efficient: Pre-allocate when size is known
num_sensors = len(self.sensor_list)
sensor_readings = [0.0] * num_sensors

# Inefficient: Growing lists dynamically
sensor_readings = []
for sensor in self.sensor_list:
    sensor_readings.append(sensor.read())  # Expensive growth

# Use generators for large datasets
def process_sensor_stream(self, sensor_data_stream):
    """Process sensor data stream without loading everything into memory."""
    for data_point in sensor_data_stream:
        processed_data = self.preprocess(data_point)
        if self.meets_criteria(processed_data):
            yield processed_data
```

### 2. Error Handling
```python
# Proper error handling with specific exceptions
class RobotController:
    def move_to_waypoint(self, waypoint):
        """Move robot to specified waypoint.
        
        Args:
            waypoint (Pose): Target pose for the robot
            
        Returns:
            bool: True if movement completed successfully
            
        Raises:
            RobotException: If robot is in error state
            NavigationException: If path is blocked or unreachable
            TimeoutException: If movement takes longer than expected
        """
        if not self.is_ready():
            raise RobotException("Robot is not ready for navigation")
        
        if not self.validate_waypoint(waypoint):
            raise NavigationException(f"Invalid waypoint: {waypoint}")
        
        try:
            path = self.path_planner.plan_path(self.current_pose, waypoint)
            if not path:
                raise NavigationException(f"No valid path to {waypoint}")
                
            return self.follow_path(path)
            
        except PathPlanningException as e:
            self.get_logger().error(f"Path planning failed: {e}")
            raise NavigationException(f"Cannot navigate to {waypoint}") from e
        except Exception as e:
            self.get_logger().error(f"Unexpected navigation error: {e}")
            raise
```

## Testing Standards

### 1. Unit Test Structure
```python
import unittest
from unittest.mock import Mock, patch
import numpy as np
from my_robot_navigation.controller import NavigationController


class TestNavigationController(unittest.TestCase):
    """Test suite for NavigationController class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.controller = NavigationController()
        self.test_pose = Mock()
        self.test_pose.x = 1.0
        self.test_pose.y = 1.0
        self.test_pose.theta = 0.0
    
    def test_move_to_waypoint_success(self):
        """Test successful movement to waypoint."""
        # Arrange
        self.controller.current_pose = self.test_pose
        goal_pose = Mock()
        goal_pose.x = 2.0
        goal_pose.y = 2.0
        goal_pose.theta = 0.0
        
        # Mock path planning
        with patch.object(self.controller.path_planner, 'plan_path') as mock_plan:
            mock_plan.return_value = [Mock(), Mock()]  # Valid path
            
            # Act
            result = self.controller.move_to_waypoint(goal_pose)
            
            # Assert
            self.assertTrue(result)
            mock_plan.assert_called_once()
    
    def test_move_to_waypoint_invalid_input(self):
        """Test movement with invalid waypoint raises exception."""
        # Arrange
        invalid_pose = Mock()
        invalid_pose.x = float('inf')  # Invalid coordinate
        
        # Act & Assert
        with self.assertRaises(ValueError):
            self.controller.move_to_waypoint(invalid_pose)


if __name__ == '__main__':
    unittest.main()
```

## Security and Safety Considerations

### 1. Input Validation
```python
def set_robot_velocity(self, linear_vel, angular_vel):
    """Set robot velocity with safety validation."""
    # Validate input bounds
    if not -self.max_linear_vel <= linear_vel <= self.max_linear_vel:
        raise ValueError(f"Linear velocity {linear_vel} exceeds limits")
    
    if not -self.max_angular_vel <= angular_vel <= self.max_angular_vel:
        raise ValueError(f"Angular velocity {angular_vel} exceeds limits")
    
    # Check for NaN or infinity values
    if math.isnan(linear_vel) or math.isinf(linear_vel):
        raise ValueError("Linear velocity contains invalid value")
        
    if math.isnan(angular_vel) or math.isinf(angular_vel):
        raise ValueError("Angular velocity contains invalid value")
    
    # Additional safety checks
    if self.emergency_stop_activated:
        raise RobotException("Cannot set velocity: emergency stop active")
    
    # Set velocity
    self.current_linear_vel = linear_vel
    self.current_angular_vel = angular_vel
```

## Code Review Checklist

### Before Submitting Code
- [ ] Code follows established naming conventions
- [ ] Functions have clear, comprehensive docstrings
- [ ] Code handles potential errors gracefully
- [ ] Comments explain complex logic (not obvious parts)
- [ ] Variable names are descriptive and meaningful
- [ ] Code avoids unnecessary complexity
- [ ] Imports are organized correctly
- [ ] Tests exist for new functionality
- [ ] Performance considerations are addressed
- [ ] Security and safety requirements are met

### During Code Review
- [ ] Logic is correct and handles edge cases
- [ ] Follows established standards and conventions
- [ ] Is maintainable and readable
- [ ] Performance is reasonable for real-time applications
- [ ] Security vulnerabilities are addressed
- [ ] Documentation is comprehensive
- [ ] Code is appropriately tested
- [ ] Follows ROS 2 best practices
- [ ] Error handling is robust
- [ ] Resource management is proper (especially in loops)

By adhering to these coding standards, we ensure that the Physical AI & Humanoid Robotics curriculum maintains high code quality, promotes best practices in robotics development, and provides students with examples of professional-grade code they can reference in their own projects.