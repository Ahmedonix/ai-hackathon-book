---
sidebar_position: 13
---

# Exercise 4: Building Multi-Node ROS 2 Systems

## Objective

In this exercise, you will learn how to create and orchestrate multi-node ROS 2 systems. You'll build a complete humanoid robot control system that includes multiple coordinated nodes exchanging messages, services, and actions. This exercise combines all the concepts learned in previous exercises.

## Prerequisites

Before starting this exercise, ensure you have:
- Completed all previous exercises (topics, services, actions, robot description)
- ROS 2 Iron installed
- Basic understanding of launch files
- Basic Python and C++ programming skills

## Step 1: Designing the Multi-Node System Architecture

Our multi-node humanoid system will include:
1. **Robot State Publisher**: Publishes the robot's joint states and transforms
2. **Motion Controller**: Controls joint positions based on high-level commands
3. **Sensor Simulator**: Simulates sensor data (IMU, joint encoders)
4. **Navigation Planner**: Plans navigation paths
5. **Behavior Manager**: Coordinates high-level behaviors

## Step 2: Creating the Motion Controller Node

Create `motion_controller.py`:

```python
#!/usr/bin/env python3

"""
Motion Controller Node
This node receives high-level motion commands and controls robot joints accordingly.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import math


class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        self.behavior_sub = self.create_subscription(
            String, 'behavior_command', self.behavior_callback, 10)
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.status_pub = self.create_publisher(String, 'motion_status', 10)
        self.motor_enable_pub = self.create_publisher(Bool, 'motor_enable', 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz
        
        # Robot state variables
        self.current_cmd_vel = Twist()
        self.current_behavior = "idle"  # idle, walking, balancing, etc.
        self.current_joint_positions = {}
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow'
        ]
        
        # Initialize joint positions
        for joint_name in self.joint_names:
            self.current_joint_positions[joint_name] = 0.0
        
        # Behavior-specific state
        self.walk_phase = 0.0
        self.is_active = True
        
        # Enable motors initially
        self.enable_motors(True)
        
        self.get_logger().info('Motion Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.current_cmd_vel = msg
        self.get_logger().debug(f'Received cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

    def behavior_callback(self, msg):
        """Handle behavior commands"""
        self.current_behavior = msg.data.lower()
        self.get_logger().info(f'Switched to behavior: {self.current_behavior}')

    def enable_motors(self, enable):
        """Enable or disable robot motors"""
        msg = Bool()
        msg.data = enable
        self.motor_enable_pub.publish(msg)
        self.is_active = enable

    def control_loop(self):
        """Main control loop that executes at 50Hz"""
        if not self.is_active:
            return  # Don't publish commands if motors are disabled
        
        # Create joint state message
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'
        
        # Set joint names
        joint_cmd.name = self.joint_names[:]
        
        # Calculate joint positions based on current behavior
        joint_positions = []
        
        if self.current_behavior == "walking":
            joint_positions = self.calculate_walking_gait()
        elif self.current_behavior == "balancing":
            joint_positions = self.calculate_balancing_posture()
        elif self.current_behavior == "idle":
            joint_positions = self.calculate_idle_posture()
        else:
            # Default to idle posture for unknown behaviors
            joint_positions = self.calculate_idle_posture()
        
        joint_cmd.position = joint_positions
        joint_cmd.velocity = [0.0] * len(joint_positions)
        joint_cmd.effort = [0.0] * len(joint_positions)
        
        # Publish joint commands
        self.joint_cmd_pub.publish(joint_cmd)
        
        # Publish status
        status_msg = String()
        status_msg.data = f'Behavior: {self.current_behavior}, Position: ({self.current_cmd_vel.linear.x:.2f}, {self.current_cmd_vel.linear.y:.2f})'
        self.status_pub.publish(status_msg)

    def calculate_idle_posture(self):
        """Calculate joint positions for idle posture"""
        positions = []
        for joint_name in self.joint_names:
            # Return to neutral position
            positions.append(0.0)
        return positions

    def calculate_balancing_posture(self):
        """Calculate joint positions for balancing posture"""
        positions = []
        for joint_name in self.joint_names:
            if 'hip' in joint_name:
                # Slightly bent knees for stability
                positions.append(0.1 if 'left' in joint_name else -0.1)
            elif 'knee' in joint_name:
                # Bent knees for stability
                positions.append(-0.2 if 'left' in joint_name else 0.2)
            elif 'ankle' in joint_name:
                # Ankle adjustments for balance
                positions.append(0.05 if 'left' in joint_name else -0.05)
            elif 'shoulder' in joint_name:
                # Relaxed arm position
                positions.append(0.1 if 'pitch' in joint_name else 0.05)
            else:
                positions.append(0.0)  # Neutral position
        return positions

    def calculate_walking_gait(self):
        """Calculate joint positions for walking gait"""
        # Update walk phase
        self.walk_phase += 0.1  # Controls speed of gait
        
        # Calculate gait parameters based on commanded velocity
        forward_speed = self.current_cmd_vel.linear.x
        turn_speed = self.current_cmd_vel.angular.z
        
        positions = []
        for joint_name in self.joint_names:
            if 'hip' in joint_name:
                # Hip movement - coordinated for walking
                base_pos = 0.1 * forward_speed  # Forward/back movement
                turn_mod = 0.2 * turn_speed if 'left' in joint_name else -0.2 * turn_speed
                # Add oscillatory motion for walking
                oscillation = 0.2 * math.sin(self.walk_phase) if 'left' in joint_name else 0.2 * math.sin(self.walk_phase + math.pi)
                positions.append(base_pos + turn_mod + oscillation)
            elif 'knee' in joint_name:
                # Knee movement synchronized with hip
                knee_phase = self.walk_phase if 'left' in joint_name else self.walk_phase + math.pi
                knee_bend = -0.3 * forward_speed + 0.4 * math.sin(knee_phase)
                positions.append(knee_bend)
            elif 'ankle' in joint_name:
                # Ankle adjustments for foot clearance and support
                ankle_phase = self.walk_phase + math.pi/2 if 'left' in joint_name else self.walk_phase + 3*math.pi/2
                ankle_pos = 0.1 * math.sin(ankle_phase)
                positions.append(ankle_pos)
            elif 'shoulder' in joint_name:
                # Arm swing coordinated with leg movement for balance
                arm_phase = self.walk_phase + math.pi if 'left' in joint_name else self.walk_phase
                arm_swing = 0.1 * math.sin(arm_phase) if 'pitch' in joint_name else 0.05 * math.sin(arm_phase)
                positions.append(arm_swing)
            elif 'elbow' in joint_name:
                # Elbow position for arm swing
                elbow_pos = 0.1 * math.sin(self.walk_phase + (math.pi if 'left' in joint_name else 0))
                positions.append(elbow_pos)
            else:
                positions.append(0.0)  # Default position
        
        return positions


def main(args=None):
    rclpy.init(args=args)
    controller = MotionController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Controller interrupted by user')
    finally:
        controller.enable_motors(False)  # Safely disable motors
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 3: Creating the Sensor Simulator Node

Create `sensor_simulator.py`:

```python
#!/usr/bin/env python3

"""
Sensor Simulator Node
This node simulates sensor data for the humanoid robot system.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Header
import math
import numpy as np


class SensorSimulator(Node):
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        
        # Timer for sensor simulation
        self.sensor_timer = self.create_timer(0.01, self.publish_sensor_data)  # 100Hz
        
        # Robot state tracking
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow'
        ]
        
        # Initialize joint positions to slightly bent position
        self.joint_positions = [0.0] * len(self.joint_names)
        for i, name in enumerate(self.joint_names):
            if 'knee' in name:
                self.joint_positions[i] = -0.1  # Slightly bent knees
            elif 'ankle' in name:
                self.joint_positions[i] = 0.05  # Slightly flexed ankles
        
        # Simulation parameters
        self.simulation_time = 0.0
        self.robot_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        
        self.get_logger().info('Sensor Simulator initialized')

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        self.simulation_time += 0.01  # timer period
        
        # Publish joint states
        self.publish_joint_states()
        
        # Publish IMU data
        self.publish_imu_data()

    def publish_joint_states(self):
        """Publish simulated joint states"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.name = self.joint_names[:]
        
        # Add some realistic noise to positions
        noisy_positions = []
        for pos in self.joint_positions:
            # Add small amount of noise (< 0.01 rad)
            noisy_pos = pos + np.random.normal(0.0, 0.005)
            noisy_positions.append(noisy_pos)
        
        msg.position = noisy_positions
        
        # Calculate velocities (derivative of positions)
        # In a real system, this would come from encoders
        msg.velocity = [0.0] * len(msg.position)  # Simplified
        
        # Calculate efforts (simulated torques)
        msg.effort = [0.0] * len(msg.position)  # Simplified
        
        self.joint_state_pub.publish(msg)

    def publish_imu_data(self):
        """Publish simulated IMU data"""
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'  # Assuming robot has an IMU mounted
        
        # Simulate robot orientation (add some movement for realism)
        roll = 0.01 * math.sin(self.simulation_time * 0.5)  # Small roll movement
        pitch = 0.02 * math.cos(self.simulation_time * 0.3)  # Small pitch movement
        yaw = self.simulation_time * 0.1  # Slow yaw rotation
        
        # Convert Euler angles to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        msg.orientation = Quaternion()
        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy
        
        # Add covariance (representing uncertainty)
        msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        # Simulate angular velocity (derivative of orientation)
        msg.angular_velocity = Vector3()
        msg.angular_velocity.x = 0.01 * 0.5 * math.cos(self.simulation_time * 0.5)  # Roll rate
        msg.angular_velocity.y = -0.02 * 0.3 * math.sin(self.simulation_time * 0.3)  # Pitch rate
        msg.angular_velocity.z = 0.1  # Constant yaw rate
        
        # Add covariance for angular velocity
        msg.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        # Simulate linear acceleration (in robot frame)
        msg.linear_acceleration = Vector3()
        # Simulate gravity and small accelerations
        msg.linear_acceleration.x = 0.1 * math.sin(self.simulation_time * 2.0)  # Small x-accel
        msg.linear_acceleration.y = 0.1 * math.cos(self.simulation_time * 1.5)  # Small y-accel
        msg.linear_acceleration.z = 9.81 + 0.2 * math.sin(self.simulation_time * 3.0)  # Gravity + small variation
        
        # Add covariance for linear acceleration
        msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        
        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    simulator = SensorSimulator()

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        simulator.get_logger().info('Sensor simulator interrupted by user')
    finally:
        simulator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 4: Creating the Behavior Manager Node

Create `behavior_manager.py`:

```python
#!/usr/bin/env python3

"""
Behavior Manager Node
This node manages high-level behaviors and coordinates between subsystems.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time


class BehaviorManager(Node):
    def __init__(self):
        super().__init__('behavior_manager')
        
        # Publishers
        self.behavior_cmd_pub = self.create_publisher(String, 'behavior_command', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.motion_status_sub = self.create_subscription(
            String, 'motion_status', self.motion_status_callback, 10)
        
        # Timer for behavior decision making
        self.behavior_timer = self.create_timer(1.0, self.evaluate_behavior)  # 1Hz decision making
        
        # Emergency timer
        self.emergency_timer = self.create_timer(0.1, self.emergency_check)  # 10Hz safety check
        
        # State variables
        self.current_behavior = "idle"  # idle, walking, balancing, etc.
        self.last_motion_status = "Unknown"
        self.joint_states_received = False
        self.emergency_stop_triggered = False
        self.behavior_start_time = self.get_clock().now()
        
        # Behavior priorities and transitions
        self.behavior_hierarchy = {
            'emergency': 100,
            'balancing': 50,
            'walking': 10,
            'idle': 1
        }
        
        # Initialize with idle behavior
        self.switch_behavior("idle")
        
        self.get_logger().info('Behavior Manager initialized')

    def joint_state_callback(self, msg):
        """Handle incoming joint states"""
        self.joint_states_received = True
        # Could add joint position analysis here to detect anomalies

    def motion_status_callback(self, msg):
        """Handle motion controller status updates"""
        self.last_motion_status = msg.data

    def switch_behavior(self, new_behavior):
        """Switch to a new behavior and publish the command"""
        if new_behavior != self.current_behavior:
            self.get_logger().info(f'Switching from {self.current_behavior} to {new_behavior}')
            self.current_behavior = new_behavior
            self.behavior_start_time = self.get_clock().now()
            
            # Publish behavior command
            cmd_msg = String()
            cmd_msg.data = new_behavior
            self.behavior_cmd_pub.publish(cmd_msg)

    def evaluate_behavior(self):
        """Evaluate the current situation and decide on behavior"""
        if self.emergency_stop_triggered:
            return  # Emergency stop takes precedence
            
        # Example behavior logic based on time and system status
        elapsed = (self.get_clock().now() - self.behavior_start_time).nanoseconds / 1e9
        
        # Simple behavior sequence for demonstration
        if self.current_behavior == "idle":
            if elapsed > 5.0:  # Stay in idle for 5 seconds
                self.switch_behavior("balancing")  # Switch to balancing
        elif self.current_behavior == "balancing":
            if elapsed > 10.0:  # Stay in balancing for 10 seconds
                self.switch_behavior("walking")  # Switch to walking
        elif self.current_behavior == "walking":
            if elapsed > 15.0:  # Stay in walking for 15 seconds
                self.switch_behavior("idle")  # Back to idle
        else:
            # For any unknown behavior, return to idle
            self.switch_behavior("idle")

    def emergency_check(self):
        """Check for emergency conditions"""
        # Example implementation - in a real robot this would check:
        # - joint position limits
        # - collision sensors
        # - fall detection from IMU
        # - communication timeouts
        # - other safety sensors
        
        # For simulation, just check if we've been in walking too long without joint states
        if (self.current_behavior == "walking" and 
            not self.joint_states_received and
            (self.get_clock().now() - self.behavior_start_time).nanoseconds / 1e9 > 2.0):
            self.trigger_emergency_stop("No joint feedback during walking")
        else:
            # Reset flag if we're receiving joint states
            if self.joint_states_received:
                self.joint_states_received = False

    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop procedure"""
        if not self.emergency_stop_triggered:
            self.get_logger().error(f'EMERGENCY STOP TRIGGERED: {reason}')
            self.emergency_stop_triggered = True
            
            # Publish emergency stop message to the system
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)
            
            # Switch to safe behavior
            self.switch_behavior("idle")


def main(args=None):
    rclpy.init(args=args)
    manager = BehaviorManager()

    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        manager.get_logger().info('Behavior manager interrupted by user')
    finally:
        if not manager.emergency_stop_triggered:
            # Publish an idle command on shutdown
            idle_cmd = String()
            idle_cmd.data = "idle"
            manager.behavior_cmd_pub.publish(idle_cmd)
        manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 5: Creating the Complete Launch File

Create the launch file `humanoid_system.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('multi_node_exercise')
    
    # Launch configuration variables
    use_rviz = LaunchConfiguration('use_rviz')
    use_gui = LaunchConfiguration('use_gui')
    
    # Declare launch arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    declare_use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Whether to launch joint state publisher GUI'
    )
    
    # Path to URDF/XACRO file
    robot_description_path = os.path.join(
        pkg_share,
        'urdf',
        'humanoid_robot.xacro'
    )
    
    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'robot_description': Command(['xacro ', robot_description_path])
            }
        ],
        remappings=[
            ('/joint_states', 'joint_states')
        ]
    )
    
    # Joint State Publisher node
    joint_state_publisher_node = Node(
        condition=lambda context: LaunchConfiguration('use_gui').perform(context) == 'true',
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[
            {'rate': 50}
        ]
    )
    
    # Joint State Publisher node (non-GUI version for headless)
    joint_state_publisher_node_headless = Node(
        condition=lambda context: LaunchConfiguration('use_gui').perform(context) != 'true',
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'rate': 50}
        ]
    )
    
    # Motion Controller node
    motion_controller_node = Node(
        package='multi_node_exercise',
        executable='motion_controller',
        name='motion_controller',
        parameters=[
            {'use_sim_time': False}
        ],
        remappings=[
            ('joint_commands', '/joint_commands')
        ]
    )
    
    # Sensor Simulator node
    sensor_simulator_node = Node(
        package='multi_node_exercise',
        executable='sensor_simulator',
        name='sensor_simulator',
        parameters=[
            {'use_sim_time': False}
        ]
    )
    
    # Behavior Manager node
    behavior_manager_node = Node(
        package='multi_node_exercise',
        executable='behavior_manager',
        name='behavior_manager',
        parameters=[
            {'use_sim_time': False}
        ]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'humanoid_system.rviz')],
        parameters=[
            {'use_sim_time': False}
        ]
    )
    
    # Create launch description and add actions
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_use_gui)
    
    # Add nodes
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(joint_state_publisher_node_headless)
    ld.add_action(motion_controller_node)
    ld.add_action(sensor_simulator_node)
    ld.add_action(behavior_manager_node)
    
    # Conditionally add RViz
    # We'll add a timer to start RViz after other nodes have initialized
    rviz_timer = TimerAction(
        period=1.0,
        actions=[rviz_node],
        condition=lambda context: LaunchConfiguration('use_rviz').perform(context) == 'true'
    )
    ld.add_action(rviz_timer)
    
    return ld
```

## Step 6: Creating the Package Configuration

Create the `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.8)
project(multi_node_exercise)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/motion_controller.py
  scripts/sensor_simulator.py
  scripts/behavior_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install URDF files
install(DIRECTORY
  urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# Install RViz configs
install(DIRECTORY
  rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

Create the `package.xml` file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>multi_node_exercise</name>
  <version>0.1.0</version>
  <description>Multi-node system exercise for Module 1</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>urdf</depend>
  <depend>xacro</depend>
  <depend>joint_state_publisher</depend>
  <depend>joint_state_publisher_gui</depend>
  <depend>robot_state_publisher</depend>
  <depend>rviz2</depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>