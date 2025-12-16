# Testing Humanoid Motion in Simulation

## Overview

Testing humanoid motion in simulation is a critical step in developing reliable and safe humanoid robots. Unlike wheeled robots, humanoid robots have complex dynamics with multiple degrees of freedom, balance challenges, and intricate control requirements. This section covers comprehensive testing methodologies for validating humanoid motion in simulation environments.

## Understanding Humanoid Motion Challenges

### 1. Balance and Stability

Humanoid robots face unique balance challenges:
- **Center of Mass**: High and narrow support polygon makes balance difficult
- **Dynamic Stability**: Requires constant adjustment of stance and movements
- **Contact Transitions**: Complex dynamics during foot contact changes
- **Environmental Interactions**: Response to external forces and terrain variations

### 2. Locomotion Patterns

Humanoid locomotion requires testing of multiple movement patterns:
- **Static Walking**: Slow, stable footsteps with continuous support
- **Dynamic Walking**: Natural gait with periods of single support
- **Running**: Extended periods of flight phases
- **Specialized Gaits**: Stair climbing, uneven terrain navigation

### 3. Control Complexity

Humanoid motion involves multiple control systems:
- **Low-level Joint Control**: Motor control and feedback
- **Balance Control**: Center of mass and Zero Moment Point (ZMP) management
- **Trajectory Planning**: Path planning and obstacle avoidance
- **High-level Motion Planning**: Task-based movement sequences

## Preparing Test Environments

### 1. Simulation Test Scenarios

Set up various test scenarios in your simulation environment:

```xml
<!-- Basic flat terrain for fundamental motion testing -->
<model name="flat_terrain">
  <static>true</static>
  <pose>0 0 0 0 0 0</pose>
  <link name="ground">
    <collision name="collision">
      <geometry>
        <plane><normal>0 0 1</normal></plane>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <plane><normal>0 0 1</normal><size>20 20</size></plane>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Step test for stair climbing validation -->
<model name="step_test">
  <static>true</static>
  <link name="step1">
    <pose>0 0 0.1 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box><size>2 2 0.2</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>2 2 0.2</size></box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="step2">
    <pose>0 0 0.2 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box><size>2 2 0.2</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>2 2 0.2</size></box>
      </geometry>
      <material>
        <ambient>0.6 0.6 0.6 1</ambient>
        <diffuse>0.6 0.6 0.6 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Slope test for incline walking -->
<model name="slope_test">
  <static>true</static>
  <pose>0 0 0 0 0.2 0</pose>  <!-- 0.2 radians ~ 11.4 degrees -->
  <link name="slope">
    <collision name="collision">
      <geometry>
        <box><size>5 3 0.1</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>5 3 0.1</size></box>
      </geometry>
      <material>
        <ambient>0.4 0.6 0.4 1</ambient>
        <diffuse>0.4 0.6 0.4 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Obstacle course for navigation testing -->
<model name="obstacle_course">
  <static>true</static>
  <!-- Series of obstacles at various heights and positions -->
  <link name="obstacle_1">
    <pose>-2 0 0.2 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box><size>0.3 0.3 0.4</size></box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box><size>0.3 0.3 0.4</size></box>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
        <diffuse>0.8 0.2 0.2 1</diffuse>
      </material>
    </visual>
  </link>
</model>
```

### 2. Test Environment Documentation

Create a documentation template for test environments:

```yaml
# Test Environment: Basic Walking Validation
environment_name: "flat_terrain_test"
description: "Flat terrain for basic walking pattern validation"
dimensions: "20m x 20m"
features:
  - flat ground
  - markers every meter for distance measurement
  - starting position marked
robot_starting_position: [0, 0, 1.0]
robot_starting_orientation: [0, 0, 0, 1]  # quaternion

# Test Parameters
test_category: "locomotion"
test_type: "walking"
required_capabilities:
  - basic walking gait
  - balance maintenance
  - forward locomotion

# Success Criteria
success_criteria:
  - robot maintains balance throughout test
  - robot completes 10m walk without falling
  - robot maintains upright posture (less than 15 degrees from vertical)
  - step timing within acceptable range (0.5-2.0 seconds per step)

# Metrics to Track
metrics:
  - COM_position_trajectory
  - ZMP_deviation
  - step_timing
  - energy_consumption
  - joint_torques
```

## Motion Testing Methodologies

### 1. Kinematic Testing

Test the kinematic properties of humanoid motion:

```python
# scripts/kinematic_tester.py
#!/usr/bin/env python3

"""
Kinematic tester for validating humanoid motion in simulation.
Tests forward and inverse kinematics, joint limits, and workspace.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import numpy as np
import math


class KinematicTester(Node):
    def __init__(self):
        super().__init__('kinematic_tester')
        
        # Publishers and subscribers
        self.joint_command_pub = self.create_publisher(
            JointState,
            '/position_commands',
            10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.test_results_pub = self.create_publisher(
            Float64MultiArray,
            '/kinematic_test_results',
            10
        )
        
        # Test parameters
        self.test_positions = [
            # Each position is [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],           # Standing position
            [0.1, 0.2, -0.1, 0.1, 0.2, -0.1],        # Slight forward lean
            [-0.1, 0.2, -0.1, -0.1, 0.2, -0.1],      # Slight backward lean
            [0.2, 0.5, -0.3, 0.0, 0.0, 0.0],         # Left leg forward
            [0.0, 0.0, 0.0, 0.2, 0.5, -0.3],         # Right leg forward
        ]
        
        self.current_test_index = 0
        self.test_timer = self.create_timer(5.0, self.run_next_test)
        self.test_results = []
        
        self.get_logger().info('Kinematic Tester Initialized')

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def run_next_test(self):
        """Execute the next kinematic test"""
        if self.current_test_index >= len(self.test_positions):
            self.publish_results()
            return
        
        test_position = self.test_positions[self.current_test_index]
        self.execute_position_test(test_position)
        self.current_test_index += 1

    def execute_position_test(self, target_position):
        """Execute a single position validation test"""
        # Create joint command message
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        
        # Define joint names for humanoid model
        joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        joint_cmd.name = joint_names
        joint_cmd.position = target_position
        joint_cmd.velocity = [0.0] * len(target_position)
        joint_cmd.effort = [0.0] * len(target_position)
        
        # Publish command
        self.joint_command_pub.publish(joint_cmd)
        
        # Log test
        self.get_logger().info(f'Executing kinematic test {self.current_test_index + 1}: {target_position}')
        
        # Calculate expected results
        test_result = self.calculate_kinematic_metrics(target_position)
        self.test_results.append(test_result)

    def calculate_kinematic_metrics(self, target_position):
        """Calculate kinematic metrics for the test"""
        # This would typically involve forward kinematics calculations
        # For simplicity, we'll return some basic metrics
        
        # Calculate joint range utilization (0 to 1)
        max_joint_range = 3.0  # Assuming maximum joint range of 3 radians
        utilization = [abs(pos) / max_joint_range for pos in target_position]
        
        # Average utilization
        avg_utilization = sum(utilization) / len(utilization)
        
        # Maximum joint position
        max_position = max(abs(pos) for pos in target_position)
        
        return [avg_utilization, max_position, len([p for p in target_position if abs(p) > max_joint_range * 0.8])]

    def publish_results(self):
        """Publish final test results"""
        result_msg = Float64MultiArray()
        
        # Flatten all test results
        flattened_results = []
        for result in self.test_results:
            flattened_results.extend(result)
        
        result_msg.data = flattened_results
        self.test_results_pub.publish(result_msg)
        
        self.get_logger().info(f'Kinematic testing complete. Results: {flattened_results}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Kinematic Tester Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    tester = KinematicTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Node interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Dynamic Stability Testing

Test dynamic stability and balance during motion:

```python
# scripts/dynamic_stability_tester.py
#!/usr/bin/env python3

"""
Dynamic stability tester for humanoid robot simulation.
Tests balance during various movements and external disturbances.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Wrench, Point
from std_msgs.msg import Float64MultiArray, String
from builtin_interfaces.msg import Duration
import numpy as np
import math


class DynamicStabilityTester(Node):
    def __init__(self):
        super().__init__('dynamic_stability_tester')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Publishers
        self.apply_force_pub = self.create_publisher(
            Wrench,
            '/apply_force',
            10
        )
        
        self.test_results_pub = self.create_publisher(
            Float64MultiArray,
            '/stability_test_results',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/stability_test_status',
            10
        )
        
        # Internal state
        self.current_joint_state = JointState()
        self.current_imu_data = None
        self.test_sequence = [
            self.balance_test,
            self.push_recovery_test,
            self.step_disturbance_test
        ]
        self.current_test_index = 0
        self.test_timer = self.create_timer(10.0, self.run_next_test)
        self.test_results = []
        
        # Stability metrics
        self.roll_threshold = 0.3  # radians
        self.pitch_threshold = 0.3  # radians
        self.zmp_threshold = 0.1  # meters

        self.get_logger().info('Dynamic Stability Tester Initialized')

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Process IMU data for balance metrics"""
        self.current_imu_data = msg

    def run_next_test(self):
        """Execute the next stability test in sequence"""
        if self.current_test_index >= len(self.test_sequence):
            self.publish_results()
            return
        
        # Execute current test
        test_func = self.test_sequence[self.current_test_index]
        test_results = test_func()
        
        self.test_results.append(test_results)
        self.current_test_index += 1
        
        # Log test completion
        test_names = ["Balance Test", "Push Recovery", "Step Disturbance"]
        self.get_logger().info(f'Completed: {test_names[self.current_test_index - 1]}')

    def balance_test(self):
        """Test basic balance maintenance"""
        # Monitor stability over time
        stability_metrics = []
        initial_time = self.get_clock().now()
        
        while (self.get_clock().now() - initial_time).nanoseconds < 5e9:  # 5 seconds
            if self.current_imu_data:
                # Calculate roll and pitch from orientation
                orientation = self.current_imu_data.orientation
                roll, pitch, yaw = self.quaternion_to_euler(
                    orientation.x, orientation.y, orientation.z, orientation.w
                )
                
                # Check if robot is within stability bounds
                is_stable = (abs(roll) < self.roll_threshold and
                            abs(pitch) < self.pitch_threshold)
                
                stability_metrics.append({
                    'time': (self.get_clock().now() - initial_time).nanoseconds / 1e9,
                    'roll': abs(roll),
                    'pitch': abs(pitch),
                    'stable': is_stable
                })
            # Small delay to allow simulation to update
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Calculate metrics
        stable_time = sum(1 for m in stability_metrics if m['stable'])
        total_time = len(stability_metrics)
        stability_percentage = (stable_time / total_time * 100) if total_time > 0 else 0
        
        # Count falls (large angle deviations)
        falls = sum(1 for m in stability_metrics if max(m['roll'], m['pitch']) > 0.785)  # ~45 degrees
        
        return [stability_percentage, falls, len(stability_metrics)]

    def push_recovery_test(self):
        """Test recovery from external pushes"""
        # Apply a push force
        push_force = Wrench()
        push_force.force.x = 50.0  # Newtons in forward direction
        push_force.force.y = 0.0
        push_force.force.z = 0.0
        push_force.torque.x = 0.0
        push_force.torque.y = 0.0
        push_force.torque.z = 0.0
        
        # Publish the push for a short duration
        for _ in range(10):  # Apply for 0.1 second at 100Hz
            self.apply_force_pub.publish(push_force)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Stop the push
        zero_force = Wrench()
        for _ in range(20):  # Clear for 0.2 second
            self.apply_force_pub.publish(zero_force)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Monitor recovery
        recovery_metrics = []
        recovery_start = self.get_clock().now()
        
        while (self.get_clock().now() - recovery_start).nanoseconds < 10e9:  # 10 seconds
            if self.current_imu_data:
                orientation = self.current_imu_data.orientation
                roll, pitch, yaw = self.quaternion_to_euler(
                    orientation.x, orientation.y, orientation.z, orientation.w
                )
                
                recovery_metrics.append({
                    'time': (self.get_clock().now() - recovery_start).nanoseconds / 1e9,
                    'roll': abs(roll),
                    'pitch': abs(pitch),
                    'recovered': abs(roll) < 0.1 and abs(pitch) < 0.1  # Within 5.7 degrees
                })
            
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Find time to recovery (when robot is within tolerance)
        recovery_time = 10.0  # Default to test duration if not recovered
        for metric in recovery_metrics:
            if metric['recovered']:
                recovery_time = metric['time']
                break
        
        # Calculate stability during recovery
        stable_during_recovery = sum(1 for m in recovery_metrics if m['recovered'])
        total_recovery_time = len(recovery_metrics)
        recovery_success = 1.0 if stable_during_recovery > 0 else 0.0
        
        return [recovery_time, recovery_success, stable_during_recovery / total_recovery_time if total_recovery_time > 0 else 0.0]

    def step_disturbance_test(self):
        """Test stability during stepping"""
        # This would involve commanding the robot to take steps
        # For simulation, we'll monitor natural sway during static stance
        stability_metrics = []
        initial_time = self.get_clock().now()
        
        while (self.get_clock().now() - initial_time).nanoseconds < 15e9:  # 15 seconds
            if self.current_imu_data:
                orientation = self.current_imu_data.orientation
                roll, pitch, yaw = self.quaternion_to_euler(
                    orientation.x, orientation.y, orientation.z, orientation.w
                )
                
                stability_metrics.append({
                    'time': (self.get_clock().now() - initial_time).nanoseconds / 1e9,
                    'roll': abs(roll),
                    'pitch': abs(pitch),
                    'within_bounds': abs(roll) < 0.2 and abs(pitch) < 0.2
                })
            
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # Calculate metrics
        within_bounds_time = sum(1 for m in stability_metrics if m['within_bounds'])
        total_time = len(stability_metrics)
        stability_percentage = (within_bounds_time / total_time * 100) if total_time > 0 else 0
        
        max_roll = max(m['roll'] for m in stability_metrics) if stability_metrics else 0
        max_pitch = max(m['pitch'] for m in stability_metrics) if stability_metrics else 0
        
        return [stability_percentage, max_roll, max_pitch]

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def publish_results(self):
        """Publish final test results"""
        result_msg = Float64MultiArray()
        
        # Flatten all test results
        flattened_results = []
        for result in self.test_results:
            flattened_results.extend(result)
        
        result_msg.data = flattened_results
        self.test_results_pub.publish(result_msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Testing complete. Balance: {self.test_results[0][0]:.1f}%, Push Recovery: {self.test_results[1][0]:.2f}s, Step Stability: {self.test_results[2][0]:.1f}%"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f'Dynamic stability testing complete. Results: {flattened_results}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Dynamic Stability Tester Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    tester = DynamicStabilityTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Node interrupted by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Motion Validation Framework

### 1. Comprehensive Motion Validator

Create a framework that validates multiple aspects of humanoid motion:

```python
# scripts/motion_validator.py
#!/usr/bin/env python3

"""
Comprehensive motion validator for humanoid robots in simulation.
Validates kinematics, dynamics, and stability during various motions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Float64MultiArray, String
from builtin_interfaces.msg import Duration
import numpy as np
import math
import time


class MotionValidator(Node):
    def __init__(self):
        super().__init__('motion_validator')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.validation_results_pub = self.create_publisher(
            Float64MultiArray,
            '/motion_validation_results',
            10
        )
        
        self.validation_report_pub = self.create_publisher(
            String,
            '/motion_validation_report',
            10
        )
        
        # Internal state
        self.joint_states = []
        self.imu_data = []
        self.odom_data = []
        self.test_start_time = None
        self.is_testing = False
        
        # Validation parameters
        self.validation_window = 10.0  # seconds to accumulate data
        self.validation_timer = self.create_timer(1.0, self.check_validation)
        self.data_collection_timer = self.create_timer(0.01, self.collect_data)  # 100Hz data collection
        
        # Metrics thresholds
        self.metrics_thresholds = {
            'max_roll_deg': 15.0,
            'max_pitch_deg': 15.0, 
            'max_lateral_deviation': 0.5,  # meters
            'min_forward_progress': 0.1,   # meters per second
            'max_energy_consumption': 100.0,  # normalized units
            'max_joint_velocity': 5.0,  # rad/s
            'max_joint_torque': 100.0  # N*m
        }
        
        self.get_logger().info('Motion Validator Initialized')

    def joint_state_callback(self, msg):
        """Store joint state data"""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Store IMU data"""
        self.current_imu = msg

    def odom_callback(self, msg):
        """Store odometry data"""
        self.current_odom = msg

    def start_validation(self):
        """Start a new validation cycle"""
        self.is_testing = True
        self.test_start_time = self.get_clock().now()
        self.joint_states = []
        self.imu_data = []
        self.odom_data = []
        
        self.get_logger().info('Motion validation started')

    def stop_validation(self):
        """Stop validation and report results"""
        self.is_testing = False
        
        # Process collected data and generate report
        validation_results = self.process_validation_data()
        self.publish_validation_report(validation_results)
        
        self.get_logger().info('Motion validation finished')

    def collect_data(self):
        """Collect data at high frequency"""
        if not self.is_testing:
            return
            
        if hasattr(self, 'current_joint_state') and self.current_joint_state:
            # Record joint state with timestamp
            self.joint_states.append({
                'timestamp': self.get_clock().now(),
                'data': self.current_joint_state
            })
            
        if hasattr(self, 'current_imu') and self.current_imu:
            self.imu_data.append({
                'timestamp': self.get_clock().now(),
                'data': self.current_imu
            })
            
        if hasattr(self, 'current_odom') and self.current_odom:
            self.odom_data.append({
                'timestamp': self.get_clock().now(),
                'data': self.current_odom
            })

    def check_validation(self):
        """Check if validation period has ended"""
        if not self.is_testing or self.test_start_time is None:
            return
            
        current_time = self.get_clock().now()
        elapsed = (current_time - self.test_start_time).nanoseconds / 1e9
        
        if elapsed >= self.validation_window:
            self.stop_validation()

    def process_validation_data(self):
        """Process collected data and calculate validation metrics"""
        results = {}
        
        # Calculate balance metrics from IMU data
        if self.imu_data:
            results.update(self.calculate_balance_metrics())
        
        # Calculate mobility metrics from odometry data
        if self.odom_data:
            results.update(self.calculate_mobility_metrics())
        
        # Calculate efficiency metrics from joint data
        if self.joint_states:
            results.update(self.calculate_efficiency_metrics())
        
        # Calculate safety metrics
        results.update(self.calculate_safety_metrics())
        
        return results

    def calculate_balance_metrics(self):
        """Calculate balance-related metrics from IMU data"""
        if not self.imu_data:
            return {}
        
        rolls = []
        pitches = []
        
        for imu_point in self.imu_data:
            # Extract orientation and convert to Euler angles
            orientation = imu_point['data'].orientation
            roll, pitch, yaw = self.quaternion_to_euler(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            
            rolls.append(abs(roll))
            pitches.append(abs(pitch))
        
        if not rolls:
            return {}
        
        avg_roll = np.mean(rolls)
        avg_pitch = np.mean(pitches)
        max_roll = np.max(rolls)
        max_pitch = np.max(pitches)
        
        # Calculate stability percentage (within safe bounds)
        roll_threshold = math.radians(self.metrics_thresholds['max_roll_deg'])
        pitch_threshold = math.radians(self.metrics_thresholds['max_pitch_deg'])
        
        stable_points = sum(1 for r, p in zip(rolls, pitches) 
                          if r <= roll_threshold and p <= pitch_threshold)
        stability_percentage = (stable_points / len(rolls)) * 100
        
        return {
            'avg_roll_deg': math.degrees(avg_roll),
            'avg_pitch_deg': math.degrees(avg_pitch),
            'max_roll_deg': math.degrees(max_roll),
            'max_pitch_deg': math.degrees(max_pitch),
            'stability_percentage': stability_percentage
        }

    def calculate_mobility_metrics(self):
        """Calculate mobility-related metrics from odometry data"""
        if not self.odom_data or len(self.odom_data) < 2:
            return {}
        
        # Calculate path metrics
        start_pos = self.odom_data[0]['data'].pose.pose.position
        end_pos = self.odom_data[-1]['data'].pose.pose.position
        
        displacement = math.sqrt(
            (end_pos.x - start_pos.x)**2 + 
            (end_pos.y - start_pos.y)**2 + 
            (end_pos.z - start_pos.z)**2
        )
        
        # Calculate average velocity
        time_diff = (self.odom_data[-1]['timestamp'] - self.odom_data[0]['timestamp']).nanoseconds / 1e9
        avg_velocity = displacement / time_diff if time_diff > 0 else 0.0
        
        # Calculate path straightness (if applicable)
        if len(self.odom_data) > 1:
            # Simple straightness calculation
            total_distance = 0.0
            for i in range(1, len(self.odom_data)):
                prev_pos = self.odom_data[i-1]['data'].pose.pose.position
                curr_pos = self.odom_data[i]['data'].pose.pose.position
                step_distance = math.sqrt(
                    (curr_pos.x - prev_pos.x)**2 +
                    (curr_pos.y - prev_pos.y)**2 +
                    (curr_pos.z - prev_pos.z)**2
                )
                total_distance += step_distance
            
            straightness_ratio = displacement / total_distance if total_distance > 0 else 0.0
        else:
            straightness_ratio = 1.0
        
        return {
            'displacement_m': displacement,
            'avg_velocity_ms': avg_velocity,
            'path_efficiency': straightness_ratio,
            'duration_s': time_diff
        }

    def calculate_efficiency_metrics(self):
        """Calculate efficiency-related metrics from joint state data"""
        if not self.joint_states:
            return {}
        
        # Calculate energy consumption estimates
        total_torque = 0.0
        total_velocity = 0.0
        total_power = 0.0
        
        for i in range(1, len(self.joint_states)):
            state = self.joint_states[i]['data']
            prev_state = self.joint_states[i-1]['data']
            
            # Calculate approximate power for each joint
            for j in range(len(state.position)):
                if j < len(state.effort) and j < len(state.velocity):
                    torque = abs(state.effort[j])
                    velocity = abs(state.velocity[j])
                    power = torque * velocity
                    
                    total_torque += torque
                    total_velocity += velocity
                    total_power += power
        
        avg_torque = total_torque / len(self.joint_states) if self.joint_states else 0
        avg_velocity = total_velocity / len(self.joint_states) if self.joint_states else 0
        avg_power = total_power / len(self.joint_states) if self.joint_states else 0
        
        return {
            'avg_torque': avg_torque,
            'avg_velocity': avg_velocity,
            'avg_power': avg_power,
            'total_joints': len(self.joint_states[0]['data'].position) if self.joint_states else 0
        }

    def calculate_safety_metrics(self):
        """Calculate safety-related metrics"""
        safety_metrics = {}
        
        # Check for joint limit violations
        if self.joint_states:
            joint_limit_violations = 0
            
            # This would typically involve comparing to actual joint limits
            # For now, we'll use a simple check against common limits
            for state in self.joint_states:
                for velocity in state['data'].velocity:
                    if abs(velocity) > self.metrics_thresholds['max_joint_velocity']:
                        joint_limit_violations += 1
                        
                for torque in state['data'].effort:
                    if abs(torque) > self.metrics_thresholds['max_joint_torque']:
                        joint_limit_violations += 1
            
            safety_metrics['joint_limit_violations'] = joint_limit_violations
        
        # Check for falls
        if self.imu_data:
            falls = 0
            severe_angle_threshold = math.radians(45)  # Robot considered fallen
            
            for imu_point in self.imu_data:
                orientation = imu_point['data'].orientation
                roll, pitch, _ = self.quaternion_to_euler(
                    orientation.x, orientation.y, orientation.z, orientation.w
                )
                
                if abs(roll) > severe_angle_threshold or abs(pitch) > severe_angle_threshold:
                    falls += 1
            
            safety_metrics['falls'] = falls
        
        return safety_metrics

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def publish_validation_report(self, metrics):
        """Publish the validation report"""
        # Create detailed report
        report_parts = []
        report_parts.append("=== HUMANOID MOTION VALIDATION REPORT ===")
        report_parts.append(f"Validation Window: {self.validation_window}s")
        
        # Add balance metrics
        if 'stability_percentage' in metrics:
            report_parts.append("\nBALANCE METRICS:")
            report_parts.append(f"  Stability: {metrics['stability_percentage']:.1f}%")
            report_parts.append(f"  Avg Roll: {metrics['avg_roll_deg']:.2f}°")
            report_parts.append(f"  Avg Pitch: {metrics['avg_pitch_deg']:.2f}°")
            report_parts.append(f"  Max Roll: {metrics['max_roll_deg']:.2f}°")
            report_parts.append(f"  Max Pitch: {metrics['max_pitch_deg']:.2f}°")
        
        # Add mobility metrics
        if 'displacement_m' in metrics:
            report_parts.append("\nMOBILITY METRICS:")
            report_parts.append(f"  Displacement: {metrics['displacement_m']:.2f}m")
            report_parts.append(f"  Avg Velocity: {metrics['avg_velocity_ms']:.2f}m/s")
            report_parts.append(f"  Path Efficiency: {metrics['path_efficiency']:.2f}")
        
        # Add efficiency metrics
        if 'avg_power' in metrics:
            report_parts.append("\nEFFICIENCY METRICS:")
            report_parts.append(f"  Avg Torque: {metrics['avg_torque']:.2f}N*m")
            report_parts.append(f"  Avg Power: {metrics['avg_power']:.2f}W")
        
        # Add safety metrics
        if 'falls' in metrics:
            report_parts.append("\nSAFETY METRICS:")
            report_parts.append(f"  Falls: {metrics['falls']}")
            report_parts.append(f"  Joint Limit Violations: {metrics['joint_limit_violations']}")
        
        report = "\n".join(report_parts)
        
        # Publish detailed report
        report_msg = String()
        report_msg.data = report
        self.validation_report_pub.publish(report_msg)
        
        # Publish quantitative results
        results_msg = Float64MultiArray()
        results_msg.data = [
            metrics.get('stability_percentage', 0.0),
            metrics.get('avg_roll_deg', 0.0),
            metrics.get('avg_pitch_deg', 0.0),
            metrics.get('displacement_m', 0.0),
            metrics.get('avg_velocity_ms', 0.0),
            metrics.get('avg_power', 0.0),
            metrics.get('falls', 0),
            metrics.get('joint_limit_violations', 0)
        ]
        self.validation_results_pub.publish(results_msg)
        
        self.get_logger().info(report)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Motion Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = MotionValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Node interrupted by user')
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Gait Analysis and Validation

### 1. Gait Pattern Testing

Create specialized tools for gait analysis:

```python
# scripts/gait_analyzer.py
#!/usr/bin/env python3

"""
Gait analyzer for humanoid robot locomotion validation.
Analyzes walking patterns, step timing, and gait efficiency.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import math
from collections import deque


class GaitAnalyzer(Node):
    def __init__(self):
        super().__init__('gait_analyzer')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publishers
        self.gait_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/gait_metrics',
            10
        )
        
        self.gait_report_pub = self.create_publisher(
            String,
            '/gait_report',
            10
        )
        
        # Internal state
        self.joint_history = deque(maxlen=1000)  # Store last 1000 joint states
        self.step_events = []  # Store detected step events
        self.stride_analysis = []  # Store stride information
        self.foot_positions = {'left': [], 'right': []}  # Track foot positions
        
        # Gait parameters
        self.left_foot_joints = ['left_ankle_joint']  # Adjust based on your robot
        self.right_foot_joints = ['right_ankle_joint']
        self.gait_analysis_timer = self.create_timer(0.1, self.analyze_gait)
        self.step_detection_threshold = 0.05  # meters for step detection
        self.stride_time_window = 5.0  # seconds for stride analysis
        
        # Step timing parameters
        self.min_step_duration = 0.3  # seconds
        self.max_step_duration = 2.0  # seconds
        self.nominal_step_length = 0.3  # meters (adjust for your robot)
        
        self.get_logger().info('Gait Analyzer Initialized')

    def joint_state_callback(self, msg):
        """Store joint state for gait analysis"""
        self.joint_history.append({
            'timestamp': self.get_clock().now(),
            'data': msg
        })

    def analyze_gait(self):
        """Analyze gait patterns from joint data"""
        if len(self.joint_history) < 10:  # Need minimum data points
            return
        
        # Analyze the recent joint history
        recent_data = list(self.joint_history)[-50:]  # Analyze last 50 points
        
        # Calculate gait metrics
        metrics = {}
        
        # 1. Calculate step detection from joint positions
        step_events = self.detect_steps(recent_data)
        metrics['steps_detected'] = len(step_events)
        
        # 2. Calculate step timing characteristics
        if len(step_events) > 1:
            step_durations = [
                (step_events[i+1]['time'] - step_events[i]['time']) 
                for i in range(len(step_events)-1)
            ]
            
            if step_durations:
                avg_step_duration = np.mean(step_durations)
                std_step_duration = np.std(step_durations)
                metrics['avg_step_duration'] = avg_step_duration
                metrics['std_step_duration'] = std_step_duration
        
        # 3. Calculate stride length (if locomotion is detected)
        displacement, velocity = self.calculate_locomotion_metrics(recent_data)
        metrics['displacement'] = displacement
        metrics['avg_velocity'] = velocity
        
        # 4. Calculate gait symmetry
        symmetry_ratio = self.calculate_symmetry(step_events)
        metrics['symmetry_ratio'] = symmetry_ratio
        
        # 5. Calculate dynamic stability
        stability_metrics = self.calculate_dynamic_stability(recent_data)
        metrics.update(stability_metrics)
        
        # Publish results
        self.publish_gait_metrics(metrics)
        self.publish_gait_report(metrics)

    def detect_steps(self, joint_data):
        """Detect steps from joint position data"""
        if not joint_data:
            return []
        
        step_events = []
        
        # This is a simplified step detection based on ankle joint movement
        # In practice, you would use foot contact sensors or more sophisticated methods
        left_ankle_positions = []
        right_ankle_positions = []
        
        # Extract ankle joint positions over time
        for data_point in joint_data:
            joint_msg = data_point['data']
            
            # Find left ankle position (simplified - in reality you'd use FK)
            # This is a placeholder - real implementation would use forward kinematics
            try:
                left_ankle_idx = joint_msg.name.index('left_ankle_joint')
                right_ankle_idx = joint_msg.name.index('right_ankle_joint')
                
                left_ankle_positions.append(joint_msg.position[left_ankle_idx])
                right_ankle_positions.append(joint_msg.position[right_ankle_idx])
            except ValueError:
                # Joint not found in this message
                continue
        
        # Simple step detection based on position changes
        # In real implementation, you'd use inverse kinematics or contact sensors
        for i in range(1, len(left_ankle_positions)):
            left_change = abs(left_ankle_positions[i] - left_ankle_positions[i-1])
            right_change = abs(right_ankle_positions[i] - right_ankle_positions[i-1])
            
            if left_change > 0.1:  # Threshold for step detection
                step_events.append({
                    'time': joint_data[i]['timestamp'],
                    'leg': 'left',
                    'type': 'step'
                })
            
            if right_change > 0.1:  # Threshold for step detection
                step_events.append({
                    'time': joint_data[i]['timestamp'],
                    'leg': 'right',
                    'type': 'step'
                })
        
        return step_events

    def calculate_locomotion_metrics(self, joint_data):
        """Calculate locomotion metrics like displacement and velocity"""
        if not joint_data or len(joint_data) < 2:
            return 0.0, 0.0
        
        # Calculate approximate forward displacement
        # This is simplified and would require full kinematic analysis in practice
        start_data = joint_data[0]['data']
        end_data = joint_data[-1]['data']
        
        # In a real implementation, you would use odometry or forward kinematics
        # For now, we'll use a simplified approach
        displacement = 0.0
        for i in range(len(start_data.position)):
            # Look for joints that typically indicate forward movement
            if 'hip' in start_data.name[i] or 'thigh' in start_data.name[i]:
                displacement += abs(end_data.position[i] - start_data.position[i])
        
        # Calculate time difference
        time_diff = (joint_data[-1]['timestamp'] - joint_data[0]['timestamp']).nanoseconds / 1e9
        
        velocity = displacement / time_diff if time_diff > 0 else 0.0
        
        return displacement, velocity

    def calculate_symmetry(self, step_events):
        """Calculate gait symmetry between left and right legs"""
        if not step_events:
            return 0.0
        
        left_steps = [e for e in step_events if e['leg'] == 'left']
        right_steps = [e for e in step_events if e['leg'] == 'right']
        
        if not left_steps or not right_steps:
            return 0.0
        
        # Calculate the ratio of left to right steps
        symmetry_ratio = min(len(left_steps), len(right_steps)) / max(len(left_steps), len(right_steps))
        
        # Calculate step timing symmetry
        if len(left_steps) > 1 and len(right_steps) > 1:
            left_durations = [
                (left_steps[i+1]['time'] - left_steps[i]['time']).nanoseconds / 1e9
                for i in range(len(left_steps)-1)
            ]
            
            right_durations = [
                (right_steps[i+1]['time'] - right_steps[i]['time']).nanoseconds / 1e9
                for i in range(len(right_steps)-1)
            ]
            
            if left_durations and right_durations:
                avg_left_duration = np.mean(left_durations)
                avg_right_duration = np.mean(right_durations)
                
                duration_symmetry = min(avg_left_duration, avg_right_duration) / max(avg_left_duration, avg_right_duration)
                symmetry_ratio = (symmetry_ratio + duration_symmetry) / 2
        
        return symmetry_ratio

    def calculate_dynamic_stability(self, joint_data):
        """Calculate dynamic stability metrics"""
        if not joint_data:
            return {}
        
        # Calculate joint velocity variations as a proxy for stability
        velocity_changes = []
        
        for i in range(1, len(joint_data)):
            prev_msg = joint_data[i-1]['data']
            curr_msg = joint_data[i]['data']
            
            # Calculate average joint velocity change
            velocity_change = 0.0
            valid_joints = 0
            
            for j in range(min(len(prev_msg.velocity), len(curr_msg.velocity))):
                vel_change = abs(curr_msg.velocity[j] - prev_msg.velocity[j])
                velocity_change += vel_change
                valid_joints += 1
            
            if valid_joints > 0:
                avg_vel_change = velocity_change / valid_joints
                velocity_changes.append(avg_vel_change)
        
        stability_metrics = {}
        if velocity_changes:
            avg_velocity_change = np.mean(velocity_changes)
            stability_metrics['avg_velocity_change'] = avg_velocity_change
            
            # Lower velocity changes indicate more stable gait
            stability_score = 1.0 / (1.0 + avg_velocity_change)  # Normalize to 0-1 scale
            stability_metrics['stability_score'] = stability_score
        
        return stability_metrics

    def publish_gait_metrics(self, metrics):
        """Publish quantitative gait metrics"""
        results_msg = Float64MultiArray()
        
        # Organize metrics into a consistent order
        ordered_metrics = [
            metrics.get('steps_detected', 0),
            metrics.get('avg_step_duration', 0),
            metrics.get('std_step_duration', 0),
            metrics.get('displacement', 0),
            metrics.get('avg_velocity', 0),
            metrics.get('symmetry_ratio', 0),
            metrics.get('stability_score', 0),
            metrics.get('avg_velocity_change', 0)
        ]
        
        results_msg.data = ordered_metrics
        self.gait_metrics_pub.publish(results_msg)

    def publish_gait_report(self, metrics):
        """Publish detailed gait analysis report"""
        report_parts = []
        report_parts.append("=== GAIT ANALYSIS REPORT ===")
        
        report_parts.append(f"Steps Detected: {metrics.get('steps_detected', 0)}")
        
        if 'avg_step_duration' in metrics:
            report_parts.append(f"Average Step Duration: {metrics['avg_step_duration']:.3f}s")
            report_parts.append(f"Step Duration Variance: {metrics['std_step_duration']:.3f}s")
        
        if 'displacement' in metrics:
            report_parts.append(f"Displacement: {metrics['displacement']:.3f}m")
            report_parts.append(f"Average Velocity: {metrics['avg_velocity']:.3f}m/s")
        
        if 'symmetry_ratio' in metrics:
            symmetry_percent = metrics['symmetry_ratio'] * 100
            report_parts.append(f"Gait Symmetry: {symmetry_percent:.1f}%")
        
        if 'stability_score' in metrics:
            stability_percent = metrics['stability_score'] * 100
            report_parts.append(f"Dynamic Stability: {stability_percent:.1f}%")
        
        # Determine quality assessment
        quality = self.assess_gait_quality(metrics)
        report_parts.append(f"Gait Quality Assessment: {quality}")
        
        report = "\n".join(report_parts)
        
        report_msg = String()
        report_msg.data = report
        self.gait_report_pub.publish(report_msg)

    def assess_gait_quality(self, metrics):
        """Assess overall gait quality based on metrics"""
        scores = []
        
        # Step duration assessment (should be within reasonable range)
        avg_duration = metrics.get('avg_step_duration', 0)
        if 0.5 <= avg_duration <= 1.5:  # Good walking pace
            scores.append(1.0)
        elif 0.3 <= avg_duration <= 2.0:  # Acceptable range
            scores.append(0.7)
        else:
            scores.append(0.3)  # Outside normal range
        
        # Symmetry assessment
        symmetry = metrics.get('symmetry_ratio', 0)
        if symmetry >= 0.9:
            scores.append(1.0)
        elif symmetry >= 0.7:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # Stability assessment
        stability = metrics.get('stability_score', 0)
        if stability >= 0.8:
            scores.append(1.0)
        elif stability >= 0.6:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # Calculate average quality score
        avg_score = np.mean(scores) if scores else 0.0
        
        # Map to qualitative assessment
        if avg_score >= 0.8:
            return "EXCELLENT"
        elif avg_score >= 0.6:
            return "GOOD"
        elif avg_score >= 0.4:
            return "FAIR"
        else:
            return "POOR"

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Gait Analyzer Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    analyzer = GaitAnalyzer()
    
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        analyzer.get_logger().info('Node interrupted by user')
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Testing Procedures and Protocols

### 1. Standardized Test Protocols

Create standardized testing protocols for consistent validation:

```yaml
# Standardized Test Protocol: Humanoid Walking Validation
test_protocol_version: 1.0
test_category: "locomotion"
test_type: "walking_validation"

test_setup:
  environment: "flat_terrain"
  robot_configuration: "standard_humanoid"
  starting_position: [0, 0, 1.0]
  starting_orientation: [0, 0, 0, 1]

test_execution:
  - command: "walk_forward"
    parameters:
      distance: 10.0  # meters
      speed: 0.5      # m/s
      duration: 20.0  # seconds maximum
    verification:
      - "robot_maintains_balance"
      - "completes_distance_within_time"
      - "no_falls"
      - "gait_symmetry_above_threshold"
  
  - command: "turn_in_place"
    parameters:
      angle: 90.0  # degrees
      direction: "right"
    verification:
      - "robot_maintains_balance"
      - "achieves_target_orientation"
      - "no_excessive_sway"
  
  - command: "walk_around_obstacle"
    parameters:
      obstacle_position: [5, 0, 0]
      obstacle_size: [1, 1, 1]  # width, depth, height
    verification:
      - "robot_avoids_obstacle"
      - "robot_reaches_destination"
      - "navigation_success"

success_criteria:
  balance_maintenance: 
    description: "Robot maintains upright posture"
    threshold: 
      max_roll: 15.0  # degrees
      max_pitch: 15.0  # degrees
    measurement: "imu_orientation"
  
  stability_metrics:
    description: "Robot exhibits stable motion"
    threshold:
      stability_score: 0.7  # normalized 0-1 scale
    measurement: "dynamic_stability_analyzer"
  
  gait_quality:
    description: "Robot demonstrates natural gait"
    threshold:
      symmetry_ratio: 0.8
      step_timing_consistency: 0.9
    measurement: "gait_analyzer"
  
  task_completion:
    description: "Robot completes assigned tasks"
    threshold:
      success_rate: 0.9  # 90% success rate
    measurement: "task_performance_monitor"

metrics_collection:
  - name: "balance_metrics"
    source: "/imu"
    frequency: 100  # Hz
    duration: "entire_test"
  
  - name: "joint_states"
    source: "/joint_states" 
    frequency: 100  # Hz
    duration: "entire_test"
  
  - name: "gait_metrics"
    source: "/gait_metrics"
    frequency: 10  # Hz
    duration: "entire_test"
  
  - name: "validation_results"
    source: "/motion_validation_results"
    frequency: 1  # Hz
    duration: "entire_test"

reporting:
  required_fields:
    - "test_id"
    - "robot_id" 
    - "test_protocol_version"
    - "start_time"
    - "end_time"
    - "success_criteria_met"
    - "metrics_summary"
    - "anomalies_detected"
    - "recommendations"
  
  output_formats:
    - "text_summary"
    - "detailed_csv"
    - "visualization_export"
```

## Next Steps

With comprehensive testing methodologies documented, you'll next validate humanoid locomotion using Gazebo physics. This will involve applying the testing frameworks in the actual physics simulation environment to validate the real performance of your humanoid robot.

The testing framework you've implemented provides a solid foundation for systematically validating all aspects of humanoid motion in simulation.