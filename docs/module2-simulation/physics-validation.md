# Validating Humanoid Locomotion using Gazebo Physics

## Overview

Validating humanoid locomotion in simulation requires leveraging Gazebo's physics engine to ensure that the robot's movement patterns are physically realistic and stable. This section details the process of using Gazebo's physics capabilities to verify that your humanoid robot can successfully execute various locomotion patterns while maintaining balance and stability.

## Understanding Gazebo Physics for Humanoid Locomotion

### 1. Physics Engine Selection

Gazebo supports multiple physics engines, each with different strengths for humanoid simulation:

**ODE (Open Dynamics Engine):**
- Best for articulated figure simulation
- Good performance for multi-body systems
- Well-suited for humanoid robots with many joints
- Configurable solver parameters for accuracy vs. performance

**DART (Dynamic Animation and Robotics Toolkit):**
- Advanced contact modeling
- Better for complex articulated figures
- Robust constraint handling
- Recommended for high-fidelity humanoid simulation

**Configure ODE for humanoid use:**
```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- 1ms time step for accuracy -->
  <real_time_update_rate>1000.0</real_time_update_rate>  <!-- 1000 Hz -->
  <real_time_factor>1.0</real_time_factor>
  <gravity>0 0 -9.8</gravity>
  
  <ode>
    <solver>
      <type>quick</type>  <!-- Fast solver for real-time sim -->
      <iters>200</iters>   <!-- Higher iterations for stability -->
      <sor>1.3</sor>       <!-- Successive Over-Relaxation parameter -->
    </solver>
    <constraints>
      <cfm>0.000001</cfm>  <!-- Constraint Force Mixing -->
      <erp>0.2</erp>       <!-- Error Reduction Parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### 2. Contact Modeling for Humanoid Feet

Proper contact modeling is critical for stable locomotion:

```xml
<!-- In your URDF/robot model definition -->
<gazebo reference="left_ankle">
  <collision name="left_foot_collision">
    <surface>
      <friction>
        <ode>
          <!-- High friction for stable walking -->
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <!-- Soft CFM for stable contacts -->
          <soft_cfm>0.0001</soft_cfm>
          <!-- ERP for error correction -->
          <erp>0.2</erp>
          <!-- Maximum contact velocity -->
          <max_vel>100.0</max_vel>
          <!-- Contact layer depth -->
          <min_depth>0.002</min_depth>
        </ode>
      </contact>
      <bounce>
        <restitution_coefficient>0.01</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
    </surface>
  </collision>
</gazebo>
```

## Physics-Based Locomotion Validation

### 1. Center of Mass (CoM) Analysis

Validate that the robot's CoM remains stable during locomotion:

```python
# scripts/com_analyzer.py
#!/usr/bin/env python3

"""
Center of Mass analyzer for validating humanoid balance during locomotion.
Uses physics simulation to track CoM trajectory and stability.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
import numpy as np
import math
import tf2_ros
from rclpy.qos import QoSProfile


class CenterOfMassAnalyzer(Node):
    def __init__(self):
        super().__init__('com_analyzer')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10)
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            QoSProfile(depth=10)
        )
        
        # Publishers
        self.com_trajectory_pub = self.create_publisher(
            Marker,
            '/center_of_mass_trajectory',
            QoSProfile(depth=10)
        )
        
        self.com_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/com_metrics',
            QoSProfile(depth=10)
        )
        
        self.zmp_marker_pub = self.create_publisher(
            Marker,
            '/zero_moment_point',
            QoSProfile(depth=10)
        )
        
        # Internal state
        self.current_joint_state = JointState()
        self.com_trajectory = []
        self.foot_positions = {'left': Point(), 'right': Point()}
        
        # Analysis parameters
        self.trajectory_window = 50  # Number of points to track
        self.analysis_timer = self.create_timer(0.1, self.analyze_com)  # 10Hz analysis
        
        # Robot parameters (these should be loaded from URDF in production)
        self.robot_segments = {
            'torso': {'mass': 5.0, 'position_offset': [0, 0, 0.5]},
            'head': {'mass': 1.0, 'position_offset': [0, 0, 0.9]},
            'left_thigh': {'mass': 2.0, 'position_offset': [-0.1, 0, 0.3]},
            'right_thigh': {'mass': 2.0, 'position_offset': [0.1, 0, 0.3]},
            'left_shin': {'mass': 1.5, 'position_offset': [-0.1, 0, 0.1]},
            'right_shin': {'mass': 1.5, 'position_offset': [0.1, 0, 0.1]},
            'left_foot': {'mass': 0.5, 'position_offset': [-0.1, 0, 0.05]},
            'right_foot': {'mass': 0.5, 'position_offset': [0.1, 0, 0.05]},
        }
        
        # Initialize trajectory marker
        self.trajectory_marker = self.create_trajectory_marker()
        
        self.get_logger().info('Center of Mass Analyzer Initialized')

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def odom_callback(self, msg):
        """Process odometry data"""
        # Extract base position for reference
        self.base_position = msg.pose.pose.position

    def calculate_com(self):
        """Calculate current center of mass position"""
        if not self.current_joint_state.name:
            return Point()
        
        total_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0
        
        # For now, simplified calculation
        # In a real implementation, you'd use forward kinematics
        # to calculate the actual position of each body segment
        
        # Calculate contribution from each joint's effect on body segments
        for i, joint_name in enumerate(self.current_joint_state.name):
            if i < len(self.current_joint_state.position):
                joint_pos = self.current_joint_state.position[i]
                # Simplified model: each joint affects the CoM based on a simple model
                # Real implementation would use full kinematics
                
                # Weight factor for this joint in CoM calculation
                weight = 0.1  # Simplified - in reality this would be calculated from link masses
        
        # Calculate overall CoM using kinematic chain
        com = self.inverse_kinematics_based_com_calculation()
        return com

    def inverse_kinematics_based_com_calculation(self):
        """Calculate CoM based on inverse kinematics and link masses"""
        # This is a simplified approach
        # Real implementation would use full URDF kinematics
        
        # For demonstration, calculate from joint positions
        if not self.current_joint_state.position:
            return Point(x=0.0, y=0.0, z=0.7)  # Default standing position
        
        # Simplified CoM calculation (not physically accurate, just for demonstration)
        avg_x = np.mean([pos for pos in self.current_joint_state.position[::2]]) if len(self.current_joint_state.position) > 0 else 0
        avg_y = np.mean([pos for pos in self.current_joint_state.position[1::2]]) if len(self.current_joint_state.position) > 1 else 0
        
        # Return a representative CoM position
        return Point(
            x=self.current_joint_state.position[0] * 0.1 if len(self.current_joint_state.position) > 0 else 0,
            y=self.current_joint_state.position[1] * 0.1 if len(self.current_joint_state.position) > 1 else 0,
            z=0.7  # Approximate torso height
        )

    def calculate_zmp(self, com_position, com_velocity, com_acceleration):
        """Calculate Zero Moment Point based on CoM dynamics"""
        # ZMP = CoM projected onto ground plane with dynamic compensation
        # ZMP_x = CoM_x - (CoM_z - h) / g * CoM_acc_x
        # ZMP_y = CoM_y - (CoM_z - h) / g * CoM_acc_y
        
        g = 9.81  # Gravity constant
        h = 0.7   # Nominal CoM height (adjust based on actual robot)
        
        zmp_x = com_position.x - ((com_position.z - h) / g) * com_acceleration.x
        zmp_y = com_position.y - ((com_position.z - h) / g) * com_acceleration.y
        
        zmp = Point(x=zmp_x, y=zmp_y, z=0.0)  # ZMP is on ground plane (z=0)
        return zmp

    def analyze_com(self):
        """Analyze CoM trajectory and stability"""
        current_com = self.calculate_com()
        
        # Add to trajectory
        self.com_trajectory.append({
            'timestamp': self.get_clock().now(),
            'position': current_com
        })
        
        # Keep only recent trajectory points
        if len(self.com_trajectory) > self.trajectory_window:
            self.com_trajectory.pop(0)
        
        # Calculate metrics
        metrics = self.calculate_com_metrics()
        
        # Publish CoM trajectory visualization
        self.publish_com_trajectory()
        
        # Publish metrics
        self.publish_com_metrics(metrics)
        
        # Log stability information
        self.log_stability_info(metrics)

    def calculate_com_metrics(self):
        """Calculate CoM stability metrics"""
        if not self.com_trajectory:
            return {'stability_score': 0.0, 'deviation': 0.0, 'velocity': 0.0}
        
        # Calculate CoM deviation from nominal position
        x_positions = [p['position'].x for p in self.com_trajectory]
        y_positions = [p['position'].y for p in self.com_trajectory]
        z_positions = [p['position'].z for p in self.com_trajectory]
        
        avg_x = np.mean(x_positions) if x_positions else 0
        avg_y = np.mean(y_positions) if y_positions else 0
        avg_z = np.mean(z_positions) if z_positions else 0.7
        
        # Calculate deviation from center (should be minimal for stable walking)
        deviation = math.sqrt(avg_x**2 + avg_y**2)
        
        # Calculate velocity (change over time)
        if len(self.com_trajectory) > 1:
            dt = (self.com_trajectory[-1]['timestamp'] - self.com_trajectory[0]['timestamp']).nanoseconds / 1e9
            dx = self.com_trajectory[-1]['position'].x - self.com_trajectory[0]['position'].x
            dy = self.com_trajectory[-1]['position'].y - self.com_trajectory[0]['position'].y
            velocity = math.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
        else:
            velocity = 0
        
        # Calculate stability score (0-1 scale, 1 being most stable)
        # Lower deviation and velocity = higher stability
        deviation_penalty = min(deviation / 0.2, 1.0)  # 0.2m threshold
        velocity_penalty = min(velocity / 0.5, 1.0)    # 0.5 m/s threshold
        stability_score = max(0.0, 1.0 - (deviation_penalty + velocity_penalty) / 2.0)
        
        return {
            'stability_score': stability_score,
            'deviation': deviation,
            'velocity': velocity,
            'avg_x': avg_x,
            'avg_y': avg_y,
            'avg_z': avg_z,
            'trajectory_length': len(self.com_trajectory)
        }

    def create_trajectory_marker(self):
        """Create a marker for CoM trajectory visualization"""
        marker = Marker()
        marker.header.frame_id = "odom"  # or "map" depending on your setup
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "com_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Line width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Alpha
        
        return marker

    def publish_com_trajectory(self):
        """Publish CoM trajectory for visualization"""
        marker = self.trajectory_marker
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.points = []  # Clear previous points
        
        # Add points to the trajectory
        for point_data in self.com_trajectory:
            p = Point()
            p.x = point_data['position'].x
            p.y = point_data['position'].y
            p.z = point_data['position'].z
            marker.points.append(p)
        
        self.com_trajectory_pub.publish(marker)

    def publish_com_metrics(self, metrics):
        """Publish quantitative CoM metrics"""
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            metrics['stability_score'],
            metrics['deviation'],
            metrics['velocity'],
            metrics['avg_x'],
            metrics['avg_y'],
            metrics['avg_z'],
            metrics['trajectory_length']
        ]
        self.com_metrics_pub.publish(metrics_msg)

    def log_stability_info(self, metrics):
        """Log stability information"""
        stability_status = "STABLE" if metrics['stability_score'] > 0.7 else (
            "UNSTABLE" if metrics['stability_score'] < 0.4 else "CAUTION"
        )
        
        self.get_logger().info(
            f'CoM Status: {stability_status} | '
            f'Stability: {metrics["stability_score"]:.2f} | '
            f'Deviation: {metrics["deviation"]:.3f}m | '
            f'Velocity: {metrics["velocity"]:.3f}m/s'
        )

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Center of Mass Analyzer Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    analyzer = CenterOfMassAnalyzer()
    
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

### 2. Balance Validation with Physics Constraints

Create a balance validation system that leverages Gazebo physics:

```python
# scripts/physics_balance_validator.py
#!/usr/bin/env python3

"""
Physics-based balance validator for humanoid robots in Gazebo.
Validates balance by analyzing physics simulation results.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Wrench, Vector3
from std_msgs.msg import Float64MultiArray, String
from builtin_interfaces.msg import Duration
import numpy as np
import math


class PhysicsBalanceValidator(Node):
    def __init__(self):
        super().__init__('physics_balance_validator')
        
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
        self.balance_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/balance_metrics',
            10
        )
        
        self.balance_report_pub = self.create_publisher(
            String,
            '/balance_report',
            10
        )
        
        # Internal state
        self.current_joint_state = JointState()
        self.current_imu_data = None
        self.balance_history = []
        self.fall_threshold = 0.785  # ~45 degrees in radians
        self.stability_threshold = 5  # Number of consecutive stable readings
        
        # Physics analysis timer
        self.physics_analysis_timer = self.create_timer(0.05, self.analyze_balance)  # 20Hz
        
        # Parameters
        self.mass_distribution = 75.0  # Robot mass in kg (adjust to your robot)
        self.nominal_com_height = 0.8  # Nominal CoM height in meters
        self.support_polygon_margin = 0.05  # Safety margin in meters
        
        self.get_logger().info('Physics Balance Validator Initialized')

    def joint_state_callback(self, msg):
        """Store joint state data"""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Process IMU data for balance analysis"""
        self.current_imu_data = msg

    def analyze_balance(self):
        """Analyze balance using physics simulation data"""
        if not self.current_imu_data:
            return
            
        # Extract orientation from IMU
        orientation = self.current_imu_data.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Calculate angular velocities
        angular_vel = self.current_imu_data.angular_velocity
        angular_speed = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)
        
        # Calculate linear accelerations
        linear_acc = self.current_imu_data.linear_acceleration
        linear_acceleration_magnitude = math.sqrt(
            linear_acc.x**2 + linear_acc.y**2 + linear_acc.z**2
        )
        
        # Check balance criteria
        is_balanced = self.check_balance_criteria(roll, pitch, angular_speed)
        
        # Store in history
        self.balance_history.append({
            'timestamp': self.get_clock().now(),
            'roll': abs(roll),
            'pitch': abs(pitch),
            'angular_speed': angular_speed,
            'linear_acc': linear_acceleration_magnitude,
            'balanced': is_balanced
        })
        
        # Keep only recent history
        if len(self.balance_history) > 100:  # Keep last 5 seconds of 20Hz data
            self.balance_history.pop(0)
        
        # Calculate balance metrics
        metrics = self.calculate_balance_metrics()
        
        # Publish results
        self.publish_balance_metrics(metrics)
        self.publish_balance_report(metrics)
        
        # Log status
        self.log_balance_status(metrics)

    def check_balance_criteria(self, roll, pitch, angular_speed):
        """Check if the robot is maintaining balance"""
        # Primary balance check: tilt angles within safe limits
        tilt_angle = math.sqrt(roll**2 + pitch**2)
        
        # Secondary check: angular velocity within reasonable limits
        angular_velocity_ok = angular_speed < 1.0  # rad/s threshold
        
        # Combined criteria
        return (tilt_angle < self.fall_threshold and 
                abs(roll) < self.fall_threshold and 
                abs(pitch) < self.fall_threshold and
                angular_velocity_ok)

    def calculate_balance_metrics(self):
        """Calculate quantitative balance metrics"""
        if not self.balance_history:
            return {
                'balance_score': 0.0,
                'stability_duration': 0.0,
                'fall_risk': 1.0,
                'avg_roll': 0.0,
                'avg_pitch': 0.0,
                'max_tilt': 0.0,
                'avg_angular_velocity': 0.0
            }
        
        rolls = [b['roll'] for b in self.balance_history]
        pitches = [b['pitch'] for b in self.balance_history]
        angular_speeds = [b['angular_speed'] for b in self.balance_history]
        balanced_states = [b['balanced'] for b in self.balance_history]
        
        # Calculate averages
        avg_roll = np.mean(rolls)
        avg_pitch = np.mean(pitches)
        avg_angular_velocity = np.mean(angular_speeds)
        
        # Calculate maximum tilt
        max_rolls_and_pitches = [max(abs(r), abs(p)) for r, p in zip(rolls, pitches)]
        max_tilt = max(max_rolls_and_pitches) if max_rolls_and_pitches else 0.0
        
        # Calculate stability percentage
        stable_count = sum(1 for b in balanced_states if b)
        stability_percentage = (stable_count / len(balanced_states)) * 100 if balanced_states else 0.0
        
        # Calculate fall risk (inverse of stability)
        fall_risk = (100 - stability_percentage) / 100  # Normalize to 0-1
        
        # Calculate balance score (higher is better)
        stability_score = stability_percentage / 100  # Normalize to 0-1
        
        # Calculate stability duration (time spent in stable state)
        stable_indices = [i for i, b in enumerate(balanced_states) if b]
        if stable_indices:
            # Calculate longest continuous stable period
            max_stable_run = self.longest_continuous_stable_period(balanced_states)
            stability_duration = (max_stable_run / 20.0) if balanced_states else 0.0  # Convert from steps to seconds (20Hz)
        else:
            stability_duration = 0.0
        
        return {
            'balance_score': stability_score,
            'stability_duration': stability_duration,
            'fall_risk': fall_risk,
            'avg_roll': math.degrees(avg_roll),
            'avg_pitch': math.degrees(avg_pitch),
            'max_tilt': math.degrees(max_tilt),
            'avg_angular_velocity': avg_angular_velocity,
            'stability_percentage': stability_percentage,
            'total_samples': len(balanced_states)
        }

    def longest_continuous_stable_period(self, balanced_states):
        """Find the longest continuous period of stable states"""
        max_run = 0
        current_run = 0
        
        for state in balanced_states:
            if state:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run

    def publish_balance_metrics(self, metrics):
        """Publish quantitative balance metrics"""
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            metrics['balance_score'],
            metrics['stability_duration'],
            metrics['fall_risk'],
            metrics['avg_roll'],
            metrics['avg_pitch'],
            metrics['max_tilt'],
            metrics['avg_angular_velocity'],
            metrics['stability_percentage'],
            metrics['total_samples']
        ]
        self.balance_metrics_pub.publish(metrics_msg)

    def publish_balance_report(self, metrics):
        """Publish detailed balance report"""
        report_parts = []
        report_parts.append("=== PHYSICS BALANCE VALIDATION REPORT ===")
        report_parts.append(f"Analysis Time: {metrics['total_samples'] / 20.0:.1f}s (at 20Hz)")
        report_parts.append("")
        
        # Balance status
        stability_status = "STABLE" if metrics['balance_score'] > 0.8 else (
            "PRECARIOUS" if metrics['balance_score'] > 0.5 else "UNSTABLE"
        )
        report_parts.append(f"BALANCE STATUS: {stability_status}")
        report_parts.append(f"  Overall Balance Score: {metrics['balance_score']:.2f}")
        report_parts.append(f"  Stability Percentage: {metrics['stability_percentage']:.1f}%")
        report_parts.append(f"  Fall Risk: {metrics['fall_risk']:.2f}")
        report_parts.append("")
        
        # Orientation metrics
        report_parts.append("ORIENTATION METRICS:")
        report_parts.append(f"  Average Roll: {metrics['avg_roll']:.2f}°")
        report_parts.append(f"  Average Pitch: {metrics['avg_pitch']:.2f}°")
        report_parts.append(f"  Maximum Tilt: {metrics['max_tilt']:.2f}°")
        report_parts.append("")
        
        # Dynamic metrics
        report_parts.append("DYNAMIC METRICS:")
        report_parts.append(f"  Avg Angular Velocity: {metrics['avg_angular_velocity']:.2f} rad/s")
        report_parts.append(f"  Stability Duration: {metrics['stability_duration']:.2f}s")
        
        # Determine recommendation
        if metrics['balance_score'] > 0.9:
            recommendation = "BALANCE EXCELLENT - Continue current gait"
        elif metrics['balance_score'] > 0.7:
            recommendation = "BALANCE GOOD - Minor adjustments advised"
        elif metrics['balance_score'] > 0.5:
            recommendation = "BALANCE CAUTION - Consider gait modification"
        else:
            recommendation = "BALANCE POOR - STOP AND RECOVER BALANCE"
        
        report_parts.append("")
        report_parts.append(f"RECOMMENDATION: {recommendation}")
        
        report = "\n".join(report_parts)
        
        report_msg = String()
        report_msg.data = report
        self.balance_report_pub.publish(report_msg)

    def log_balance_status(self, metrics):
        """Log balance status to console"""
        stability_status = "STABLE" if metrics['balance_score'] > 0.8 else (
            "CAUTION" if metrics['balance_score'] > 0.5 else "UNSTABLE"
        )
        
        self.get_logger().info(
            f'Balance: {stability_status} | '
            f'Score: {metrics["balance_score"]:.2f} | '
            f'Angle: ({metrics["avg_roll"]:.1f}°, {metrics["avg_pitch"]:.1f}°) | '
            f'Max Tilt: {metrics["max_tilt"]:.1f}°'
        )

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

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Physics Balance Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = PhysicsBalanceValidator()
    
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

## Locomotion Pattern Validation

### 1. Walking Gait Validation

Create validators for specific locomotion patterns:

```python
# scripts/walking_gait_validator.py
#!/usr/bin/env python3

"""
Walking gait validator for humanoid robots in Gazebo simulation.
Validates walking patterns using physics simulation and kinematic analysis.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
import numpy as np
import math
from collections import deque


class WalkingGaitValidator(Node):
    def __init__(self):
        super().__init__('walking_gait_validator')
        
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
        self.gait_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/walking_metrics',
            10
        )
        
        self.gait_report_pub = self.create_publisher(
            String,
            '/walking_report',
            10
        )
        
        # Internal state
        self.joint_history = deque(maxlen=200)  # Store 10 seconds of 20Hz data
        self.foot_contact_history = deque(maxlen=200)
        self.step_events = []
        
        # Analysis timers
        self.gait_analysis_timer = self.create_timer(0.1, self.analyze_gait)
        self.step_detection_timer = self.create_timer(0.05, self.detect_steps)
        
        # Gait parameters
        self.nominal_step_length = 0.3  # meters
        self.nominal_step_height = 0.05  # meters
        self.nominal_step_duration = 0.8  # seconds
        self.nominal_walking_speed = 0.4  # m/s
        
        # Joint names for leg joints (adjust for your robot)
        self.left_leg_joints = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint'
        ]
        self.right_leg_joints = [
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        # State variables
        self.current_joint_state = JointState()
        self.current_imu_data = None
        self.current_odom_data = None
        self.last_step_time = None
        self.current_phase = "stance"  # "swing" or "stance"
        
        self.get_logger().info('Walking Gait Validator Initialized')

    def joint_state_callback(self, msg):
        """Store joint state for gait analysis"""
        self.joint_history.append({
            'timestamp': self.get_clock().now(),
            'data': msg
        })
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Store IMU data for balance validation"""
        self.current_imu_data = msg

    def odom_callback(self, msg):
        """Store odometry for speed calculation"""
        self.current_odom_data = msg

    def detect_steps(self):
        """Detect step events based on joint positions and velocity"""
        if len(self.joint_history) < 5:  # Need at least 5 data points
            return
        
        # In a real implementation, you would use:
        # - Inverse kinematics to determine foot positions
        # - Contact sensors on feet
        # - Ground contact detection
        
        # For simulation, we'll approximate step detection using joint velocities
        current_joints = self.joint_history[-1]['data']
        
        # Calculate average leg joint velocity
        left_leg_vel = self.calculate_leg_velocity(current_joints, 'left')
        right_leg_vel = self.calculate_leg_velocity(current_joints, 'right')
        
        # Detect potential step based on leg movement
        threshold = 0.5  # rad/s velocity threshold
        if left_leg_vel > threshold or right_leg_vel > threshold:
            current_time = self.get_clock().now()
            
            # Determine which leg is likely stepping
            leg = 'left' if left_leg_vel > right_leg_vel else 'right'
            
            # Only record step if enough time has passed since last step
            if self.last_step_time is None:
                step_event = {
                    'time': current_time,
                    'leg': leg,
                    'type': 'step_detected'
                }
                self.step_events.append(step_event)
                self.last_step_time = current_time
            else:
                time_since_last = (current_time - self.last_step_time).nanoseconds / 1e9
                if time_since_last > 0.3:  # At least 300ms between steps
                    step_event = {
                        'time': current_time,
                        'leg': leg,
                        'type': 'step_detected'
                    }
                    self.step_events.append(step_event)
                    self.last_step_time = current_time

    def calculate_leg_velocity(self, joint_state, leg_prefix):
        """Calculate characteristic velocity for a leg"""
        try:
            leg_joints = self.left_leg_joints if leg_prefix == 'left' else self.right_leg_joints
            total_velocity = 0.0
            valid_joints = 0
            
            for joint_name in leg_joints:
                if joint_name in joint_state.name:
                    idx = joint_state.name.index(joint_name)
                    if idx < len(joint_state.velocity):
                        total_velocity += abs(joint_state.velocity[idx])
                        valid_joints += 1
            
            return total_velocity / valid_joints if valid_joints > 0 else 0.0
        except Exception:
            return 0.0

    def analyze_gait(self):
        """Analyze walking gait patterns"""
        if not self.current_joint_state or not self.current_imu_data:
            return
        
        # Calculate gait metrics
        metrics = {}
        
        # 1. Step timing analysis
        metrics.update(self.analyze_step_timing())
        
        # 2. Balance during walking
        metrics.update(self.analyze_balance_during_walk())
        
        # 3. Speed and efficiency
        metrics.update(self.calculate_locomotion_metrics())
        
        # 4. Joint loading analysis
        metrics.update(self.analyze_joint_loading())
        
        # 5. Step quality metrics
        metrics.update(self.calculate_step_quality())
        
        # Publish results
        self.publish_gait_metrics(metrics)
        self.publish_gait_report(metrics)

    def analyze_step_timing(self):
        """Analyze timing characteristics of walking steps"""
        if len(self.step_events) < 2:
            return {
                'step_frequency': 0.0,
                'avg_step_duration': 0.0,
                'step_timing_consistency': 0.0,
                'stride_duration': 0.0
            }
        
        # Calculate step durations
        step_durations = []
        for i in range(1, len(self.step_events)):
            duration = (self.step_events[i]['time'] - self.step_events[i-1]['time']).nanoseconds / 1e9
            step_durations.append(duration)
        
        if not step_durations:
            return {
                'step_frequency': 0.0,
                'avg_step_duration': 0.0,
                'step_timing_consistency': 0.0,
                'stride_duration': 0.0
            }
        
        avg_duration = np.mean(step_durations)
        std_duration = np.std(step_durations)
        step_frequency = 1.0 / avg_duration if avg_duration > 0 else 0.0
        
        # Calculate timing consistency (lower std = more consistent)
        consistency = 1.0 - min(std_duration, 0.2) / 0.2  # Normalize to 0-1 scale
        consistency = max(0.0, consistency)  # Ensure positive
        
        # Stride duration (time for both legs to complete a cycle)
        # In double support phase, both feet are on ground
        stride_duration = (self.step_events[-1]['time'] - self.step_events[0]['time']).nanoseconds / 1e9
        stride_duration /= (len(self.step_events) // 2)  # Average stride duration
        
        return {
            'step_frequency': step_frequency,
            'avg_step_duration': avg_duration,
            'step_timing_consistency': consistency,
            'stride_duration': stride_duration,
            'std_step_duration': std_duration
        }

    def analyze_balance_during_walk(self):
        """Analyze balance during walking motion"""
        if not self.current_imu_data:
            return {
                'avg_roll_disp': 0.0,
                'avg_pitch_disp': 0.0,
                'balance_during_walk': 0.0
            }
        
        # Extract orientation from IMU
        orientation = self.current_imu_data.orientation
        roll, pitch, _ = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Calculate balance score during walking
        # During walking, some oscillation is normal, but extremes indicate instability
        max_acceptable_oscillation = math.radians(10)  # 10 degrees
        
        roll_score = max(0.0, 1.0 - abs(roll) / max_acceptable_oscillation)
        pitch_score = max(0.0, 1.0 - abs(pitch) / max_acceptable_oscillation)
        
        balance_score = (roll_score + pitch_score) / 2.0
        
        return {
            'avg_roll_disp': math.degrees(abs(roll)),
            'avg_pitch_disp': math.degrees(abs(pitch)),
            'balance_during_walk': balance_score
        }

    def calculate_locomotion_metrics(self):
        """Calculate locomotion efficiency metrics"""
        if not self.current_odom_data:
            return {
                'walking_speed': 0.0,
                'speed_efficiency': 0.0,
                'direction_accuracy': 0.0
            }
        
        # Extract linear velocity from odometry
        linear_velocity = self.current_odom_data.twist.twist.linear
        current_speed = math.sqrt(
            linear_velocity.x**2 + 
            linear_velocity.y**2 + 
            linear_velocity.z**2
        )
        
        # Calculate speed efficiency (ratio to nominal)
        speed_efficiency = current_speed / self.nominal_walking_speed if self.nominal_walking_speed > 0 else 0.0
        speed_efficiency = min(speed_efficiency, 1.0)  # Cap at 1.0
        
        # Calculate direction accuracy (how well we're moving forward)
        target_direction = [1.0, 0.0, 0.0]  # Assuming forward is X direction
        actual_direction = [
            linear_velocity.x, 
            linear_velocity.y, 
            linear_velocity.z
        ]
        
        if current_speed > 0.01:  # Avoid division by zero
            actual_direction_norm = [
                comp / current_speed for comp in actual_direction
            ]
            direction_dot_product = sum(a*b for a, b in zip(target_direction, actual_direction_norm))
            direction_accuracy = max(0.0, min(1.0, direction_dot_product))
        else:
            direction_accuracy = 1.0  # Stationary is in "correct" direction
        
        return {
            'walking_speed': current_speed,
            'speed_efficiency': speed_efficiency,
            'direction_accuracy': direction_accuracy
        }

    def analyze_joint_loading(self):
        """Analyze joint loading during walking"""
        if not self.current_joint_state.effort:
            return {
                'avg_joint_load': 0.0,
                'max_joint_load': 0.0,
                'loading_efficiency': 1.0
            }
        
        # Calculate average and maximum joint loading
        efforts = [effort for effort in self.current_joint_state.effort if abs(effort) < 1000]  # Filter outliers
        if not efforts:
            return {
                'avg_joint_load': 0.0,
                'max_joint_load': 0.0,
                'loading_efficiency': 1.0
            }
        
        avg_load = np.mean([abs(e) for e in efforts])
        max_load = np.max([abs(e) for e in efforts])
        
        # Calculate efficiency (lower load is more efficient)
        # Assume 50 N*m is acceptable maximum load
        efficiency = max(0.0, 1.0 - min(1.0, max_load / 50.0))
        
        return {
            'avg_joint_load': avg_load,
            'max_joint_load': max_load,
            'loading_efficiency': efficiency
        }

    def calculate_step_quality(self):
        """Calculate step quality metrics"""
        if len(self.step_events) < 2:
            return {
                'step_symmetry': 0.0,
                'step_length_consistency': 0.0,
                'gait_smoothness': 0.0
            }
        
        # For now, use simplified symmetry based on step timing
        # In a real implementation, you'd use actual step length measurements
        left_steps = [s for s in self.step_events if s['leg'] == 'left']
        right_steps = [s for s in self.step_events if s['leg'] == 'right']
        
        if left_steps and right_steps:
            symmetry = min(len(left_steps), len(right_steps)) / max(len(left_steps), len(right_steps))
        else:
            symmetry = 0.0
        
        # Step length consistency would be measured from actual positions
        # Here we use a proxy based on step timing regularity
        step_durations = []
        for i in range(1, len(self.step_events)):
            duration = (self.step_events[i]['time'] - self.step_events[i-1]['time']).nanoseconds / 1e9
            step_durations.append(duration)
        
        if step_durations:
            std_duration = np.std(step_durations)
            # Lower standard deviation means more consistent steps
            consistency = max(0.0, 1.0 - std_duration / 0.2)  # Normalize
        else:
            consistency = 1.0
        
        # Gait smoothness based on IMU readings
        if self.current_imu_data:
            angular_vel = self.current_imu_data.angular_velocity
            angular_magnitude = math.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)
            # Lower angular velocity means smoother gait
            smoothness = max(0.0, 1.0 - angular_magnitude / 0.5)
        else:
            smoothness = 1.0
        
        return {
            'step_symmetry': symmetry,
            'step_length_consistency': consistency,
            'gait_smoothness': smoothness
        }

    def publish_gait_metrics(self, metrics):
        """Publish quantitative gait metrics"""
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            metrics.get('step_frequency', 0.0),
            metrics.get('avg_step_duration', 0.0),
            metrics.get('step_timing_consistency', 0.0),
            metrics.get('balance_during_walk', 0.0),
            metrics.get('walking_speed', 0.0),
            metrics.get('speed_efficiency', 0.0),
            metrics.get('step_symmetry', 0.0),
            metrics.get('gait_smoothness', 0.0),
            metrics.get('loading_efficiency', 0.0)
        ]
        self.gait_metrics_pub.publish(metrics_msg)

    def publish_gait_report(self, metrics):
        """Publish detailed walking gait report"""
        report_parts = []
        report_parts.append("=== WALKING GAIT VALIDATION REPORT ===")
        
        # Step timing
        report_parts.append(f"\nSTEP TIMING:")
        report_parts.append(f"  Frequency: {metrics.get('step_frequency', 0.0):.2f} Hz")
        report_parts.append(f"  Duration: {metrics.get('avg_step_duration', 0.0):.2f}s")
        report_parts.append(f"  Consistency: {metrics.get('step_timing_consistency', 0.0):.2f}")
        
        # Balance during walk
        report_parts.append(f"\nBALANCE DURING WALK:")
        report_parts.append(f"  Stability Score: {metrics.get('balance_during_walk', 0.0):.2f}")
        
        # Locomotion
        report_parts.append(f"\nLOCOMOTION:")
        report_parts.append(f"  Speed: {metrics.get('walking_speed', 0.0):.2f} m/s")
        report_parts.append(f"  Efficiency: {metrics.get('speed_efficiency', 0.0):.2f}")
        
        # Quality metrics
        report_parts.append(f"\nGAIT QUALITY:")
        report_parts.append(f"  Symmetry: {metrics.get('step_symmetry', 0.0):.2f}")
        report_parts.append(f"  Smoothness: {metrics.get('gait_smoothness', 0.0):.2f}")
        report_parts.append(f"  Loading Efficiency: {metrics.get('loading_efficiency', 0.0):.2f}")
        
        # Overall assessment
        avg_score = np.mean([
            metrics.get('step_timing_consistency', 0.0),
            metrics.get('balance_during_walk', 0.0),
            metrics.get('speed_efficiency', 0.0),
            metrics.get('step_symmetry', 0.0),
            metrics.get('gait_smoothness', 0.0)
        ])
        
        if avg_score > 0.8:
            assessment = "EXCELLENT WALKING GAiT"
        elif avg_score > 0.6:
            assessment = "GOOD WALKING GAiT"
        elif avg_score > 0.4:
            assessment = "ACCEPTABLE WALKING GAiT"
        else:
            assessment = "POOR WALKING GAiT - REQUIRES ATTENTION"
        
        report_parts.append(f"\nOVERALL ASSESSMENT: {assessment}")
        
        report = "\n".join(report_parts)
        
        report_msg = String()
        report_msg.data = report
        self.gait_report_pub.publish(report_msg)
        
        self.get_logger().info(f'Walking gait assessment: {assessment} (score: {avg_score:.2f})')

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

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Walking Gait Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = WalkingGaitValidator()
    
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

## Advanced Physics Validation

### 1. Dynamic Stability Margin Analysis

Create advanced validation for dynamic stability:

```python
# scripts/dynamic_stability_analyzer.py
#!/usr/bin/env python3

"""
Dynamic stability margin analyzer for humanoid robots.
Uses physics simulation to calculate stability margins during locomotion.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point, Polygon
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker
import numpy as np
import math


class DynamicStabilityAnalyzer(Node):
    def __init__(self):
        super().__init__('dynamic_stability_analyzer')
        
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
        self.stability_margin_pub = self.create_publisher(
            Float64MultiArray,
            '/stability_margins',
            10
        )
        
        self.support_polygon_pub = self.create_publisher(
            Marker,
            '/support_polygon',
            10
        )
        
        self.zmp_pub = self.create_publisher(
            Marker,
            '/zmp_trajectory',
            10
        )
        
        self.stability_report_pub = self.create_publisher(
            Float64MultiArray,
            '/stability_report',
            10
        )
        
        # Internal state
        self.current_joint_state = JointState()
        self.current_imu_data = None
        self.zmp_trajectory = []
        self.support_polygons = []
        
        # Analysis parameters
        self.foot_size = {'length': 0.2, 'width': 0.1}  # meters
        self.mass = 75.0  # Robot mass in kg
        self.gravity = 9.81  # m/s²
        self.com_height_nominal = 0.8  # meters
        
        # Analysis timer
        self.analysis_timer = self.create_timer(0.02, self.analyze_stability)  # 50Hz
        
        self.get_logger().info('Dynamic Stability Analyzer Initialized')

    def joint_state_callback(self, msg):
        """Store joint state data"""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Store IMU data for CoM estimation"""
        self.current_imu_data = msg

    def analyze_stability(self):
        """Analyze dynamic stability during motion"""
        # Calculate Zero Moment Point (ZMP)
        zmp = self.calculate_zmp()
        
        # Calculate support polygon based on foot positions
        support_polygon = self.calculate_support_polygon()
        
        # Calculate stability margin
        stability_margin = self.calculate_stability_margin(zmp, support_polygon)
        
        # Add to trajectory
        self.zmp_trajectory.append({
            'timestamp': self.get_clock().now(),
            'point': zmp
        })
        
        # Keep only recent trajectory
        if len(self.zmp_trajectory) > 500:  # Keep last 10 seconds at 50Hz
            self.zmp_trajectory.pop(0)
        
        # Publish for visualization
        self.publish_support_polygon(support_polygon)
        self.publish_zmp_trajectory()
        
        # Calculate and publish quantitative metrics
        metrics = self.calculate_stability_metrics(stability_margin, zmp, support_polygon)
        self.publish_stability_metrics(metrics)

    def calculate_zmp(self):
        """Calculate Zero Moment Point from IMU and joint data"""
        if not self.current_imu_data:
            return Point(x=0, y=0, z=0)
        
        # Get CoM estimate from IMU orientation and joint positions
        # This is a simplified calculation - in practice, use forward kinematics
        com = self.estimate_com_position()
        
        # Get linear acceleration from IMU
        acc = self.current_imu_data.linear_acceleration
        
        # ZMP calculation: ZMP_x = CoM_x - (CoM_z - h) / g * CoM_acc_x
        zmp_x = com.x - ((com.z - self.com_height_nominal) / self.gravity) * acc.x
        zmp_y = com.y - ((com.z - self.com_height_nominal) / self.gravity) * acc.y
        
        return Point(x=zmp_x, y=zmp_y, z=0.0)

    def estimate_com_position(self):
        """Estimate CoM position from joint configuration"""
        if not self.current_imu_data:
            return Point(x=0, y=0, z=self.com_height_nominal)
        
        # Simplified CoM estimation based on IMU orientation
        # In practice, use full kinematic chain with link masses
        orientation = self.current_imu_data.orientation
        roll, pitch, _ = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Adjust CoM position based on lean angles
        com_x = math.tan(roll) * self.com_height_nominal
        com_y = math.tan(pitch) * self.com_height_nominal
        
        return Point(x=com_x, y=com_y, z=self.com_height_nominal)

    def calculate_support_polygon(self):
        """Calculate support polygon based on foot contact areas"""
        # This would normally use contact sensor data or kinematic foot positions
        # For simulation, we'll create a polygon based on estimated foot positions
        
        # Simplified: assume feet are at fixed positions relative to CoM
        left_foot_x = -0.1  # meters from center
        right_foot_x = 0.1
        foot_y_offset = 0.05  # meters from center line
        
        # Define support polygon as rectangle encompassing both feet
        polygon_points = [
            Point(x=left_foot_x - self.foot_size['length']/2, y=-foot_y_offset - self.foot_size['width']/2, z=0),
            Point(x=left_foot_x + self.foot_size['length']/2, y=-foot_y_offset - self.foot_size['width']/2, z=0),
            Point(x=right_foot_x + self.foot_size['length']/2, y=foot_y_offset + self.foot_size['width']/2, z=0),
            Point(x=right_foot_x - self.foot_size['length']/2, y=foot_y_offset + self.foot_size['width']/2, z=0)
        ]
        
        return polygon_points

    def calculate_stability_margin(self, zmp, support_polygon):
        """Calculate minimum distance from ZMP to edge of support polygon"""
        # Find minimum distance from ZMP to any edge of support polygon
        min_distance = float('inf')
        
        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]
            
            # Calculate distance from ZMP to edge
            distance = self.point_to_line_distance(zmp, p1, p2)
            min_distance = min(min_distance, distance)
        
        return min_distance

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from a point to a line segment"""
        # Vector from line_start to line_end
        line_vec = [line_end.x - line_start.x, line_end.y - line_start.y]
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        if line_len_sq == 0:
            # Line is actually a point
            return math.sqrt((point.x - line_start.x)**2 + (point.y - line_start.y)**2)
        
        # Vector from line_start to point
        point_vec = [point.x - line_start.x, point.y - line_start.y]
        
        # Dot product
        dot_product = point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]
        t = max(0, min(1, dot_product / line_len_sq))  # Clamp to line segment
        
        # Projection point
        projection = Point()
        projection.x = line_start.x + t * line_vec[0]
        projection.y = line_start.y + t * line_vec[1]
        
        # Distance from point to projection
        distance = math.sqrt(
            (point.x - projection.x)**2 + 
            (point.y - projection.y)**2
        )
        
        return distance

    def calculate_stability_metrics(self, stability_margin, zmp, support_polygon):
        """Calculate comprehensive stability metrics"""
        # Calculate support polygon area
        area = self.polygon_area(support_polygon)
        
        # Calculate ZMP position relative to polygon center
        poly_center = self.polygon_centroid(support_polygon)
        center_dist = math.sqrt((zmp.x - poly_center.x)**2 + (zmp.y - poly_center.y)**2)
        
        # Calculate stability score (higher = more stable)
        # Stability increases with margin and decreases with distance from center
        stability_score = max(0.0, min(1.0, stability_margin / 0.1))  # 0.1m = good margin
        centering_score = max(0.0, 1.0 - center_dist / (math.sqrt(area/math.pi) if area > 0 else 1.0))
        
        final_score = (stability_score * 0.7) + (centering_score * 0.3)  # Weighted combination
        
        return {
            'stability_margin': stability_margin,
            'stability_score': final_score,
            'zmp_center_distance': center_dist,
            'support_polygon_area': area,
            'zmp_x': zmp.x,
            'zmp_y': zmp.y,
            'com_x': self.estimate_com_position().x,
            'com_y': self.estimate_com_position().y
        }

    def polygon_area(self, points):
        """Calculate area of a polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        j = len(points) - 1
        
        for i in range(len(points)):
            area += (points[j].x + points[i].x) * (points[j].y - points[i].y)
            j = i
        
        return abs(area) / 2.0

    def polygon_centroid(self, points):
        """Calculate centroid of a polygon"""
        if len(points) < 1:
            return Point()
        
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        
        return Point(x=cx, y=cy, z=0)

    def publish_support_polygon(self, polygon_points):
        """Publish support polygon for visualization"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "stability"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.points = polygon_points[:]
        # Close the polygon
        if polygon_points:
            marker.points.append(polygon_points[0])
        
        self.support_polygon_pub.publish(marker)

    def publish_zmp_trajectory(self):
        """Publish ZMP trajectory for visualization"""
        if not self.zmp_trajectory:
            return
            
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "zmp"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        
        marker.points = [data['point'] for data in self.zmp_trajectory]
        
        self.zmp_pub.publish(marker)

    def publish_stability_metrics(self, metrics):
        """Publish quantitative stability metrics"""
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            metrics['stability_margin'],
            metrics['stability_score'],
            metrics['zmp_center_distance'],
            metrics['support_polygon_area'],
            metrics['zmp_x'],
            metrics['zmp_y'],
            metrics['com_x'],
            metrics['com_y']
        ]
        self.stability_margin_pub.publish(metrics_msg)
        
        # Also publish stability report
        report_msg = Float64MultiArray()
        report_msg.data = [metrics['stability_score']]
        self.stability_report_pub.publish(report_msg)
        
        # Log stability status
        status = "STABLE" if metrics['stability_score'] > 0.7 else (
            "UNSTABLE" if metrics['stability_score'] < 0.4 else "CAUTION"
        )
        self.get_logger().info(
            f'Stability: {status} | Score: {metrics["stability_score"]:.2f} | '
            f'Margin: {metrics["stability_margin"]:.3f}m'
        )

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

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Dynamic Stability Analyzer Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    analyzer = DynamicStabilityAnalyzer()
    
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

## Validation Testing Procedures

### 1. Comprehensive Validation Test Suite

Create a test suite that runs multiple validation scenarios:

```python
# launch/validation_test_suite.launch.py
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
    validation_scenario = LaunchConfiguration('scenario', default='walking_basic')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_validation')  # Adjust package name
    
    # Launch Gazebo with appropriate world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', f'{validation_scenario}_test.world']),
            'gui': 'true',
            'verbose': 'false'
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
    
    # Joint State Publisher (for non-controlled joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Physics-based validation nodes
    com_analyzer = Node(
        package='humanoid_validation',
        executable='com_analyzer',
        name='com_analyzer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    balance_validator = Node(
        package='humanoid_validation',
        executable='physics_balance_validator',
        name='physics_balance_validator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    walking_validator = Node(
        package='humanoid_validation',
        executable='walking_gait_validator',
        name='walking_gait_validator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    stability_analyzer = Node(
        package='humanoid_validation',
        executable='dynamic_stability_analyzer',
        name='dynamic_stability_analyzer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Robot controller for test execution
    robot_controller = Node(
        package='humanoid_control',
        executable='test_pattern_controller',
        name='test_pattern_controller',
        parameters=[{
            'use_sim_time': use_sim_time,
            'test_scenario': validation_scenario
        }],
        output='screen'
    )
    
    # RViz for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare('humanoid_validation'),
        'rviz',
        'validation.rviz'
    ])
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Result logger
    result_logger = Node(
        package='humanoid_validation',
        executable='results_logger',
        name='results_logger',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo
        gazebo,
        
        # Launch publishers
        TimerAction(period=1.0, actions=[robot_state_publisher]),
        TimerAction(period=2.0, actions=[joint_state_publisher]),
        
        # Launch validation nodes
        TimerAction(period=3.0, actions=[com_analyzer]),
        TimerAction(period=3.0, actions=[balance_validator]),
        TimerAction(period=3.0, actions=[walking_validator]),
        TimerAction(period=3.0, actions=[stability_analyzer]),
        TimerAction(period=4.0, actions=[robot_controller]),
        TimerAction(period=5.0, actions=[rviz]),
        TimerAction(period=5.0, actions=[result_logger]),
    ])
```

## Physics-Based Validation Reports

### 1. Automated Report Generation

Create a system for generating validation reports:

```python
# scripts/validation_reporter.py
#!/usr/bin/env python3

"""
Automated validation reporter for physics-based humanoid locomotion evaluation.
Aggregates data from multiple validation nodes into comprehensive reports.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class ValidationReporter(Node):
    def __init__(self):
        super().__init__('validation_reporter')
        
        # Subscribers for various validation metrics
        self.com_metrics_sub = self.create_subscription(
            Float64MultiArray, '/com_metrics', self.com_metrics_callback, 10
        )
        
        self.balance_metrics_sub = self.create_subscription(
            Float64MultiArray, '/balance_metrics', self.balance_metrics_callback, 10
        )
        
        self.gait_metrics_sub = self.create_subscription(
            Float64MultiArray, '/walking_metrics', self.gait_metrics_callback, 10
        )
        
        self.stability_metrics_sub = self.create_subscription(
            Float64MultiArray, '/stability_margins', self.stability_metrics_callback, 10
        )
        
        # Internal data stores
        self.com_data = []
        self.balance_data = []
        self.gait_data = []
        self.stability_data = []
        
        # Reporting timer
        self.reporting_timer = self.create_timer(5.0, self.generate_report)
        
        # Parameters
        self.report_interval = 30.0  # seconds between reports
        self.data_retention = 1000  # maximum data points to retain
        self.results_directory = "/tmp/humanoid_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(self.results_directory, exist_ok=True)
        
        self.get_logger().info(f'Validation Reporter Initialized. Results will be saved to: {self.results_directory}')

    def com_metrics_callback(self, msg):
        """Store center of mass metrics"""
        self.com_data.append({
            'timestamp': self.get_timestamp(),
            'values': list(msg.data)
        })
        self.trim_data_list(self.com_data)

    def balance_metrics_callback(self, msg):
        """Store balance metrics"""
        self.balance_data.append({
            'timestamp': self.get_timestamp(),
            'values': list(msg.data)
        })
        self.trim_data_list(self.balance_data)

    def gait_metrics_callback(self, msg):
        """Store gait metrics"""
        self.gait_data.append({
            'timestamp': self.get_timestamp(),
            'values': list(msg.data)
        })
        self.trim_data_list(self.gait_data)

    def stability_metrics_callback(self, msg):
        """Store stability metrics"""
        self.stability_data.append({
            'timestamp': self.get_timestamp(),
            'values': list(msg.data)
        })
        self.trim_data_list(self.stability_data)

    def get_timestamp(self):
        """Get current timestamp"""
        return self.get_clock().now().nanoseconds / 1e9

    def trim_data_list(self, data_list):
        """Trim data list to maximum retention size"""
        if len(data_list) > self.data_retention:
            data_list.pop(0)

    def generate_report(self):
        """Generate comprehensive validation report"""
        self.get_logger().info('Generating validation report...')
        
        # Calculate aggregate statistics
        stats = {}
        stats['com'] = self.calculate_statistics(self.com_data, [
            'stability_score', 'deviation', 'velocity', 'avg_x', 'avg_y', 'avg_z'
        ])
        
        stats['balance'] = self.calculate_statistics(self.balance_data, [
            'balance_score', 'stability_duration', 'fall_risk', 'avg_roll', 'avg_pitch', 'max_tilt'
        ])
        
        stats['gait'] = self.calculate_statistics(self.gait_data, [
            'step_frequency', 'avg_step_duration', 'step_timing_consistency', 
            'balance_during_walk', 'walking_speed', 'speed_efficiency'
        ])
        
        stats['stability'] = self.calculate_statistics(self.stability_data, [
            'stability_margin', 'stability_score', 'zmp_center_distance', 'support_polygon_area'
        ])
        
        # Generate summary
        summary = self.generate_summary(stats)
        
        # Save detailed data
        self.save_detailed_data(stats)
        
        # Save summary report
        self.save_summary_report(summary, stats)
        
        self.get_logger().info(f'Validation report saved to: {self.results_directory}')

    def calculate_statistics(self, data_list, labels):
        """Calculate statistics for a set of metrics"""
        if not data_list or len(data_list[0]['values']) == 0:
            return {}
        
        # Extract values for each metric
        metrics = {}
        num_values = len(data_list[0]['values'])
        
        for i in range(num_values):
            values = [entry['values'][i] for entry in data_list if i < len(entry['values'])]
            if values:
                metrics[f"{labels[i] if i < len(labels) else f'metric_{i}'}"] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return metrics

    def generate_summary(self, stats):
        """Generate high-level summary of validation results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_stability_score': self.calculate_overall_score(stats),
            'locomotion_efficiency': self.calculate_locomotion_efficiency(stats),
            'balance_quality': self.calculate_balance_quality(stats),
            'gait_performance': self.calculate_gait_performance(stats),
            'risk_assessment': self.calculate_risk_assessment(stats)
        }
        
        return summary

    def calculate_overall_score(self, stats):
        """Calculate overall stability score"""
        # Weighted average of key stability metrics
        stability_weighted = 0.0
        total_weight = 0.0
        
        if 'stability_score' in stats['balance']:
            stability_weighted += stats['balance']['stability_score']['mean'] * 0.4
            total_weight += 0.4
            
        if 'stability_score' in stats['stability']:
            stability_weighted += stats['stability']['stability_score']['mean'] * 0.3
            total_weight += 0.3
            
        if 'balance_during_walk' in stats['gait']:
            stability_weighted += stats['gait']['balance_during_walk']['mean'] * 0.3
            total_weight += 0.3
            
        return stability_weighted / total_weight if total_weight > 0 else 0.0

    def calculate_locomotion_efficiency(self, stats):
        """Calculate locomotion efficiency"""
        efficiency = 0.0
        total_weight = 0.0
        
        if 'speed_efficiency' in stats['gait']:
            efficiency += stats['gait']['speed_efficiency']['mean'] * 0.5
            total_weight += 0.5
            
        if 'loading_efficiency' in stats['gait']:
            efficiency += stats['gait']['loading_efficiency']['mean'] * 0.5
            total_weight += 0.5
            
        return efficiency / total_weight if total_weight > 0 else 0.0

    def calculate_balance_quality(self, stats):
        """Calculate balance quality"""
        quality = 0.0
        total_weight = 0.0
        
        if 'balance_score' in stats['balance']:
            quality += stats['balance']['balance_score']['mean'] * 0.4
            total_weight += 0.4
            
        if 'stability_score' in stats['com']:
            quality += stats['com']['stability_score']['mean'] * 0.3
            total_weight += 0.3
            
        if 'stability_score' in stats['stability']:
            quality += stats['stability']['stability_score']['mean'] * 0.3
            total_weight += 0.3
            
        return quality / total_weight if total_weight > 0 else 0.0

    def calculate_gait_performance(self, stats):
        """Calculate gait performance"""
        performance = 0.0
        total_weight = 0.0
        
        if 'step_timing_consistency' in stats['gait']:
            performance += stats['gait']['step_timing_consistency']['mean'] * 0.4
            total_weight += 0.4
            
        if 'gait_smoothness' in stats['gait']:
            performance += stats['gait']['gait_smoothness']['mean'] * 0.3
            total_weight += 0.3
            
        if 'step_symmetry' in stats['gait']:
            performance += stats['gait']['step_symmetry']['mean'] * 0.3
            total_weight += 0.3
            
        return performance / total_weight if total_weight > 0 else 0.0

    def calculate_risk_assessment(self, stats):
        """Calculate risk assessment"""
        risk_score = 0.0
        total_weight = 0.0
        
        if 'fall_risk' in stats['balance']:
            # Lower fall risk is better (invert)
            risk_score += (1.0 - stats['balance']['fall_risk']['mean']) * 0.5
            total_weight += 0.5
            
        if 'max_tilt' in stats['balance']:
            # Convert max tilt to risk (smaller max tilt means less risk)
            risk_score += max(0.0, 1.0 - min(1.0, stats['balance']['max_tilt']['mean'] / 15.0)) * 0.5
            total_weight += 0.5
            
        return risk_score / total_weight if total_weight > 0 else 0.0

    def save_detailed_data(self, stats):
        """Save detailed statistical data to JSON"""
        detailed_path = os.path.join(self.results_directory, 'detailed_stats.json')
        
        with open(detailed_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def save_summary_report(self, summary, stats):
        """Save summary report to file"""
        summary_path = os.path.join(self.results_directory, 'validation_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=== HUMANOID LOCOMOTION VALIDATION REPORT ===\n\n")
            f.write(f"Generated: {summary['timestamp']}\n\n")
            
            f.write(f"Overall Stability Score: {summary['overall_stability_score']:.3f}\n")
            f.write(f"Locomotion Efficiency: {summary['locomotion_efficiency']:.3f}\n")
            f.write(f"Balance Quality: {summary['balance_quality']:.3f}\n")
            f.write(f"Gait Performance: {summary['gait_performance']:.3f}\n")
            f.write(f"Risk Assessment: {summary['risk_assessment']:.3f}\n\n")
            
            # Determine overall rating
            avg_score = (summary['overall_stability_score'] + summary['locomotion_efficiency'] + 
                         summary['balance_quality'] + summary['gait_performance']) / 4.0
            
            if avg_score > 0.8:
                rating = "EXCELLENT"
            elif avg_score > 0.6:
                rating = "GOOD"
            elif avg_score > 0.4:
                rating = "FAIR"
            else:
                rating = "POOR"
                
            f.write(f"Overall Rating: {rating}\n\n")
            
            f.write("Detailed Statistics:\n")
            for category, metrics in stats.items():
                f.write(f"\n{category.upper()}:\n")
                for metric_name, values in metrics.items():
                    f.write(f"  {metric_name}: mean={values['mean']:.3f}, std={values['std']:.3f}\n")

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Validation Reporter Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    reporter = ValidationReporter()
    
    try:
        rclpy.spin(reporter)
    except KeyboardInterrupt:
        reporter.get_logger().info('Node interrupted by user')
    finally:
        reporter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Next Steps

With physics-based validation techniques established, you'll next document simulation debugging techniques to help troubleshoot and resolve issues that may arise during validation. The physics validation framework provides a solid foundation for ensuring your humanoid robot's locomotion is stable and realistic in the simulation environment.