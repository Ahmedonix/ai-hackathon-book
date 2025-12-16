---
sidebar_position: 9
---

# Humanoid Control: Joint States, Transforms, and TF2 Basics

## Overview

Controlling a humanoid robot requires precise management of joint states and spatial transformations. In ROS 2, the Transform Library (TF2) provides the tools to track and manipulate coordinate frames, while proper joint state management ensures coordinated movement of the robot's many degrees of freedom. This section covers the fundamentals of controlling humanoid robots, focusing on joint state coordination and spatial transformations.

## Joint States in Humanoid Robots

### Understanding Joint States

Humanoid robots have many joints that need to be coordinated for stable and purposeful movement. The `sensor_msgs/JointState` message is the standard way to represent the state of all joints in a robot system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math


class HumanoidJointController(Node):
    def __init__(self):
        super().__init__('humanoid_joint_controller')
        
        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50Hz
        
        # Store current joint states
        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}
        
        # Define humanoid joint names (simplified example)
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_joint',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_joint'
        ]
        
        # Initialize joint positions
        for joint_name in self.joint_names:
            self.current_positions[joint_name] = 0.0
            self.current_velocities[joint_name] = 0.0
            self.current_efforts[joint_name] = 0.0
        
        self.get_logger().info('Humanoid Joint Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update current joint states from sensor feedback"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.current_efforts[name] = msg.effort[i]
    
    def control_loop(self):
        """Main control loop for humanoid joint control"""
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Set joint names
        msg.name = self.joint_names
        
        # Calculate desired joint positions (example: walking gait)
        msg.position = []
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
        
        for joint_name in self.joint_names:
            # Example: different movement patterns for different joints
            if 'hip' in joint_name:
                # Hip joints move in coordination for walking
                phase = 0.0 if 'left' in joint_name else math.pi  # Opposite phase for balance
                desired_pos = 0.2 * math.sin(current_time * 2 + phase)
            elif 'knee' in joint_name:
                # Knee joints move to complement hip motion
                phase = 0.0 if 'left' in joint_name else math.pi
                desired_pos = -0.3 * math.sin(current_time * 2 + phase)
            elif 'ankle' in joint_name:
                # Ankle joints for balance
                desired_pos = 0.1 * math.sin(current_time * 2)
            elif 'shoulder' in joint_name:
                # Arm movement for balance
                phase = math.pi/2 if 'pitch' in joint_name else 0.0
                desired_pos = 0.1 * math.sin(current_time * 1.5 + phase)
            elif 'elbow' in joint_name:
                # Elbow movement coordinated with shoulders
                desired_pos = 0.2 * math.sin(current_time * 1.5)
            else:
                desired_pos = 0.0
            
            msg.position.append(desired_pos)
        
        # Set velocities and efforts (optional, depending on controller)
        msg.velocity = [0.0] * len(msg.position)  # Zero velocity commands
        msg.effort = [0.0] * len(msg.position)    # Zero effort commands
        
        # Publish joint commands
        self.joint_cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidJointController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Joint State Management Strategies

Humanoid robots require sophisticated joint management strategies:

1. **Inverse Kinematics (IK)**: Calculating joint angles to achieve desired end-effector positions
2. **Trajectory Planning**: Smooth interpolation between joint configurations
3. **Balance Control**: Adjusting joint positions to maintain stability

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
import math
import numpy as np


class HumanoidTrajectoryController(Node):
    def __init__(self):
        super().__init__('humanoid_trajectory_controller')
        
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Timer for trajectory control
        self.trajectory_timer = self.create_timer(0.02, self.trajectory_callback)
        
        # Define trajectory waypoints
        self.waypoints = [
            {'left_hip': 0.0, 'left_knee': 0.0, 'left_ankle': 0.0,
             'right_hip': 0.0, 'right_knee': 0.0, 'right_ankle': 0.0},
            {'left_hip': 0.2, 'left_knee': -0.4, 'left_ankle': 0.2,
             'right_hip': 0.0, 'right_knee': 0.0, 'right_ankle': 0.0},
            {'left_hip': 0.0, 'left_knee': 0.0, 'left_ankle': 0.0,
             'right_hip': 0.2, 'right_knee': -0.4, 'right_ankle': 0.2},
        ]
        
        self.current_waypoint = 0
        self.interpolation_time = 0.0
        self.total_interpolation_time = 2.0  # seconds per waypoint transition
        
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        self.get_logger().info('Humanoid Trajectory Controller initialized')
    
    def trajectory_callback(self):
        """Execute joint trajectory with smooth interpolation"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Get start and end waypoints
        start_waypoint = self.waypoints[self.current_waypoint % len(self.waypoints)]
        end_waypoint = self.waypoints[(self.current_waypoint + 1) % len(self.waypoints)]
        
        # Update interpolation time
        self.interpolation_time += 0.02  # timer period
        
        # Calculate interpolation factor (0 to 1)
        factor = min(self.interpolation_time / self.total_interpolation_time, 1.0)
        
        # Apply smooth interpolation (cosine interpolation for smoother motion)
        smooth_factor = 0.5 * (1 - math.cos(factor * math.pi))
        
        # Calculate interpolated joint positions
        msg.name = self.joint_names
        msg.position = []
        
        for joint_name in self.joint_names:
            # Remove '_joint' suffix to match waypoint keys
            joint_key = joint_name.replace('_joint', '')
            
            start_pos = start_waypoint.get(joint_key, 0.0)
            end_pos = end_waypoint.get(joint_key, 0.0)
            
            interpolated_pos = start_pos + (end_pos - start_pos) * smooth_factor
            msg.position.append(interpolated_pos)
        
        # Check if we've reached the end of this segment
        if factor >= 1.0:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.interpolation_time = 0.0
        
        self.joint_cmd_pub.publish(msg)
        
        self.get_logger().info(f'Executing trajectory: waypoint {self.current_waypoint}, factor {factor:.2f}')


def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidTrajectoryController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Understanding TF2 and Coordinate Transforms

### TF2 Overview

TF2 (Transform Library 2) is ROS 2's system for tracking coordinate frame transformations over time. For humanoid robots, TF2 is essential for:
- Tracking the position and orientation of each body part
- Computing where sensors are located relative to other frames
- Planning movements in world coordinates
- Maintaining spatial relationships as the robot moves

### Setting Up TF2 for Humanoid Robots

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
import tf2_ros
import math


class HumanoidTFBroadcaster(Node):
    def __init__(self):
        super().__init__('humanoid_tf_broadcaster')
        
        # Create transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Subscribe to joint states to update transforms
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        
        # Timer to broadcast transforms regularly
        self.broadcast_timer = self.create_timer(0.02, self.broadcast_transforms)
        
        # Store joint positions
        self.joint_positions = {}
        
        # Define humanoid kinematic chain (simplified)
        self.kinematic_chain = [
            ('base_link', 'torso', [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]),
            ('torso', 'left_hip', [0.0, 0.15, 0.0], [0.0, 0.0, 0.0]),
            ('torso', 'right_hip', [0.0, -0.15, 0.0], [0.0, 0.0, 0.0]),
            ('left_hip', 'left_knee', [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]),
            ('right_hip', 'right_knee', [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]),
            ('left_knee', 'left_ankle', [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]),
            ('right_knee', 'right_ankle', [0.0, 0.0, -0.3], [0.0, 0.0, 0.0]),
        ]
        
        self.get_logger().info('Humanoid TF Broadcaster initialized')
    
    def joint_callback(self, msg):
        """Update joint positions from JointState message"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
    
    def broadcast_transforms(self):
        """Broadcast all transforms in the kinematic chain"""
        # Base link is at origin
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        
        # Process kinematic chain
        for parent, child, translation, rotation in self.kinematic_chain:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]
            
            # Apply rotation (simplified - in real systems, this would involve 
            # more complex kinematic calculations)
            t.transform.rotation.x = rotation[0]
            t.transform.rotation.y = rotation[1]
            t.transform.rotation.z = rotation[2]
            t.transform.rotation.w = rotation[3]
            
            self.tf_broadcaster.sendTransform(t)
        
        # Special handling for joints that move (example: hip joints)
        # Update left hip position based on joint angle
        left_hip_angle = self.joint_positions.get('left_hip_joint', 0.0)
        right_hip_angle = self.joint_positions.get('right_hip_joint', 0.0)
        
        # For demonstration, create an additional transform showing joint effect
        self.broadcast_joint_transforms(left_hip_angle, right_hip_angle)
    
    def broadcast_joint_transforms(self, left_hip_angle, right_hip_angle):
        """Broadcast transforms that show joint movement"""
        # Example: left leg link affected by hip joint
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'left_hip'
        t.child_frame_id = 'left_leg_visual'
        
        # Apply transformation based on joint angle
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.1  # Move leg down
        
        # Convert angle to quaternion (simplified)
        cy = math.cos(left_hip_angle * 0.5)
        sy = math.sin(left_hip_angle * 0.5)
        cp = math.cos(0.0 * 0.5)  # No pitch
        sp = math.sin(0.0 * 0.5)
        cr = math.cos(0.0 * 0.5)  # No roll
        sr = math.sin(0.0 * 0.5)
        
        t.transform.rotation.w = cy * cp * cr + sy * sp * sr
        t.transform.rotation.x = cy * cp * sr - sy * sp * cr
        t.transform.rotation.y = sy * cp * sr + cy * sp * cr
        t.transform.rotation.z = sy * cp * cr - cy * sp * sr
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    broadcaster = HumanoidTFBroadcaster()
    
    try:
        rclpy.spin(broadcaster)
    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Transform Lookups and Conversions

TF2 allows looking up transforms between any two frames at any point in time:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import String
import tf2_ros
import math


class HumanoidTFListener(Node):
    def __init__(self):
        super().__init__('humanoid_tf_listener')
        
        # Create TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publisher for results
        self.result_pub = self.create_publisher(String, '/tf_results', 10)
        
        # Timer to periodically check transforms
        self.lookup_timer = self.create_timer(1.0, self.lookup_transforms)
        
        self.get_logger().info('Humanoid TF Listener initialized')
    
    def lookup_transforms(self):
        """Look up transforms between various humanoid frames"""
        try:
            # Get transform from base_link to left_foot
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'left_ankle', rclpy.time.Time())
            
            # Extract position information
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            
            # Calculate distance from base to left foot
            distance = math.sqrt(x*x + y*y + z*z)
            
            result_msg = String()
            result_msg.data = f'Left foot position: ({x:.2f}, {y:.2f}, {z:.2f}), distance: {distance:.2f}'
            self.result_pub.publish(result_msg)
            
            self.get_logger().info(result_msg.data)
            
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}')
        
        try:
            # Get transform from base_link to right_foot
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'right_ankle', rclpy.time.Time())
            
            # Use transform in computations
            # For example, check if robot is balanced
            left_to_center = self.calculate_support_polygon('left_ankle')
            right_to_center = self.calculate_support_polygon('right_ankle')
            
            if abs(left_to_center - right_to_center) < 0.1:  # Balanced if within 10cm
                self.get_logger().info('Robot is balanced')
            else:
                self.get_logger().info('Robot is unbalanced')
                
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not get right foot transform: {ex}')
    
    def calculate_support_polygon(self, foot_frame):
        """Calculate distance from center of mass to foot (simplified)"""
        try:
            # Get foot position relative to base (simplified as center of mass)
            transform = self.tf_buffer.lookup_transform(
                'base_link', foot_frame, rclpy.time.Time())
            
            # Return distance in x-y plane
            return math.sqrt(
                transform.transform.translation.x**2 + 
                transform.transform.translation.y**2
            )
        except tf2_ros.TransformException:
            return float('inf')


def main(args=None):
    rclpy.init(args=args)
    listener = HumanoidTFListener()
    
    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Humanoid Control: Balance and Stability

### Center of Mass (CoM) Control

For stable humanoid locomotion, managing the Center of Mass (CoM) is crucial:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point, Vector3, Twist
from std_msgs.msg import Float64
import tf2_ros
import math


class HumanoidBalanceController(Node):
    def __init__(self):
        super().__init__('humanoid_balance_controller')
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Publishers
        self.com_pub = self.create_publisher(Point, '/center_of_mass', 10)
        self.balance_cmd_pub = self.create_publisher(Twist, '/balance_correction', 10)
        
        # TF2 components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Timer for balance control
        self.balance_timer = self.create_timer(0.02, self.balance_control)
        
        # Robot state
        self.joint_positions = {}
        self.imu_orientation = None
        self.com_position = Point(x=0.0, y=0.0, z=0.8)  # Initial estimate
        
        # Balance control parameters
        self.com_threshold = 0.05  # 5cm threshold for balance correction
        self.ankle_stiffness = 200.0  # Nm/rad
        self.ankle_damping = 10.0    # Nm*s/rad
        
        self.get_logger().info('Humanoid Balance Controller initialized')
    
    def joint_callback(self, msg):
        """Update joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
    
    def imu_callback(self, msg):
        """Update IMU orientation"""
        self.imu_orientation = msg.orientation
    
    def estimate_com(self):
        """Estimate Center of Mass position (simplified model)"""
        # This is a simplified CoM estimation based on joint positions
        # In a real robot, you'd use more sophisticated models
        
        # Get positions of key links (should use TF2 for accurate positions)
        try:
            # Calculate CoM based on link positions and masses (simplified)
            # For now, we'll just return a position based on hip and torso angles
            left_hip_pos = self.joint_positions.get('left_hip_joint', 0.0)
            right_hip_pos = self.joint_positions.get('right_hip_joint', 0.0)
            
            # Simplified CoM calculation
            self.com_position.x = 0.0  # Assuming balanced around center
            self.com_position.y = (left_hip_pos - right_hip_pos) * 0.02  # Simplified
            self.com_position.z = 0.8  # Approximate height
            
            # Publish CoM for visualization
            self.com_pub.publish(self.com_position)
            
        except Exception as e:
            self.get_logger().warn(f'Error estimating CoM: {e}')
    
    def calculate_support_polygon(self):
        """Calculate the support polygon for balance"""
        # Get positions of feet
        try:
            left_foot_transform = self.tf_buffer.lookup_transform(
                'base_link', 'left_ankle', rclpy.time.Time())
            right_foot_transform = self.tf_buffer.lookup_transform(
                'base_link', 'right_ankle', rclpy.time.Time())
            
            # Support polygon is area between feet
            left_x = left_foot_transform.transform.translation.x
            left_y = left_foot_transform.transform.translation.y
            right_x = right_foot_transform.transform.translation.x
            right_y = right_foot_transform.transform.translation.y
            
            # Simplified: average of foot positions
            support_center_x = (left_x + right_x) / 2.0
            support_center_y = (left_y + right_y) / 2.0
            
            return Point(x=support_center_x, y=support_center_y, z=0.0)
            
        except tf2_ros.TransformException:
            return Point(x=0.0, y=0.0, z=0.0)
    
    def balance_control(self):
        """Main balance control loop"""
        # Estimate current CoM
        self.estimate_com()
        
        # Calculate support polygon center
        support_center = self.calculate_support_polygon()
        
        # Calculate CoM offset from support center
        com_offset_x = self.com_position.x - support_center.x
        com_offset_y = self.com_position.y - support_center.y
        com_offset_distance = math.sqrt(com_offset_x**2 + com_offset_y**2)
        
        # Generate balance correction command
        balance_cmd = Twist()
        
        if com_offset_distance > self.com_threshold:
            # Apply corrective torque/force
            # In a real system, this would adjust ankle angles, hip positions, etc.
            balance_cmd.angular.y = -com_offset_x * 0.5  # Correct X translation
            balance_cmd.angular.x = -com_offset_y * 0.5  # Correct Y translation
            
            self.get_logger().info(
                f'Balance correction applied: offset=({com_offset_x:.3f}, {com_offset_y:.3f})'
            )
        
        # Publish balance correction command
        self.balance_cmd_pub.publish(balance_cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidBalanceController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Integration Example: Complete Humanoid Controller

Here's a complete example that combines joint control, TF2 transforms, and balance control:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TransformStamped, Point
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import math


class IntegratedHumanoidController(Node):
    def __init__(self):
        super().__init__('integrated_humanoid_controller')
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Control timer
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        # Robot state
        self.joint_positions = {}
        self.imu_data = None
        
        # Define humanoid joint structure
        self.left_leg_joints = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint']
        self.right_leg_joints = ['right_hip_joint', 'right_knee_joint', 'right_ankle_joint']
        self.left_arm_joints = ['left_shoulder_joint', 'left_elbow_joint']
        self.right_arm_joints = ['right_shoulder_joint', 'right_elbow_joint']
        
        self.all_joints = (self.left_leg_joints + self.right_leg_joints + 
                          self.left_arm_joints + self.right_arm_joints)
        
        # Initialize all joint positions
        for joint in self.all_joints:
            self.joint_positions[joint] = 0.0
        
        self.get_logger().info('Integrated Humanoid Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update actual joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position) and name in self.joint_positions:
                self.joint_positions[name] = msg.position[i]
    
    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg
    
    def generate_standing_pose(self):
        """Generate a stable standing pose"""
        # Zero all joints for basic standing pose
        target_positions = {joint: 0.0 for joint in self.all_joints}
        
        # Slightly bend knees for stability (tripod stance)
        target_positions['left_knee_joint'] = -0.1
        target_positions['right_knee_joint'] = -0.1
        
        # Slightly bend ankles to center of pressure
        target_positions['left_ankle_joint'] = 0.05
        target_positions['right_ankle_joint'] = 0.05
        
        return target_positions
    
    def generate_walking_gait(self, time):
        """Generate walking gait pattern"""
        target_positions = {joint: 0.0 for joint in self.all_joints}
        
        # Walking pattern with phase differences
        step_phase = (time % 2.0) / 2.0  # 2 second step cycle
        
        # Hip movement
        target_positions['left_hip_joint'] = 0.2 * math.sin(math.pi * step_phase)
        target_positions['right_hip_joint'] = 0.2 * math.sin(math.pi * step_phase + math.pi)
        
        # Knee movement (coordinated with hips)
        target_positions['left_knee_joint'] = -0.1 - 0.3 * math.sin(math.pi * step_phase)
        target_positions['right_knee_joint'] = -0.1 - 0.3 * math.sin(math.pi * step_phase + math.pi)
        
        # Ankle movement for foot clearance and support
        target_positions['left_ankle_joint'] = 0.1 * math.sin(math.pi * step_phase * 2)
        target_positions['right_ankle_joint'] = 0.1 * math.sin(math.pi * step_phase * 2 + math.pi)
        
        # Arm movement for balance
        target_positions['left_shoulder_joint'] = 0.1 * math.sin(math.pi * step_phase + math.pi)
        target_positions['right_shoulder_joint'] = 0.1 * math.sin(math.pi * step_phase)
        
        return target_positions
    
    def control_loop(self):
        """Main integrated control loop"""
        # Get current time for gait pattern
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Generate target joint positions (for this example, use walking gait)
        target_positions = self.generate_walking_gait(current_time)
        
        # Create and publish joint state command
        joint_cmd = JointState()
        joint_cmd.header = Header()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'
        
        joint_cmd.name = list(target_positions.keys())
        joint_cmd.position = list(target_positions.values())
        
        self.joint_cmd_pub.publish(joint_cmd)
        
        # Broadcast transforms for current joint configuration
        self.broadcast_current_transforms()
        
        # Log progress
        self.get_logger().info(
            f'Control loop executed, time: {current_time:.2f}s, ' +
            f'left_knee: {target_positions["left_knee_joint"]:.3f}'
        )
    
    def broadcast_current_transforms(self):
        """Broadcast TF transforms based on current joint configuration"""
        current_time = self.get_clock().now().to_msg()
        
        # Base link (robot origin)
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.8  # Robot height
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        
        # Left leg transforms
        # Hip
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'left_hip'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.1
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        
        # Knee
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'left_hip'
        t.child_frame_id = 'left_knee'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.3  # Thigh length
        # Apply knee rotation based on joint angle
        angle = self.joint_positions.get('left_knee_joint', 0.0)
        cy = math.cos(angle / 2.0)
        sy = math.sin(angle / 2.0)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = sy
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)
        
        # Ankle
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'left_knee'
        t.child_frame_id = 'left_ankle'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.3  # Shin length
        # Apply ankle rotation
        angle = self.joint_positions.get('left_ankle_joint', 0.0)
        cy = math.cos(angle / 2.0)
        sy = math.sin(angle / 2.0)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = sy
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    controller = IntegratedHumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Move to safe standing pose before shutdown
        controller.get_logger().info('Shutting down - moving to safe pose...')
        # In a real implementation, you might send a safe pose command here
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid Control

### 1. Safety Considerations
- Always implement emergency stop procedures
- Set joint position, velocity, and effort limits
- Monitor for hardware limits and collisions

### 2. Performance Optimization
- Use appropriate control frequencies (typically 100-500Hz for joint control)
- Implement efficient transform lookups
- Consider using dedicated hardware for real-time control

### 3. Debugging and Visualization
- Publish joint state feedback for monitoring
- Use RViz to visualize the robot state and transforms
- Log important control parameters for analysis

### 4. Modular Design
- Separate high-level behavior from low-level control
- Use standardized interfaces between modules
- Implement proper error handling in each module

## Summary

Controlling humanoid robots in ROS 2 requires a solid understanding of:

1. **Joint State Management**: Properly managing and commanding multiple joints simultaneously
2. **TF2 Transformations**: Tracking coordinate frames and spatial relationships
3. **Balance and Stability**: Maintaining center of mass within support polygon
4. **Integration**: Coordinating multiple control systems for coherent behavior

With these concepts, you can develop sophisticated humanoid control systems that leverage ROS 2's communication infrastructure while maintaining stability and achieving desired behaviors. The TF2 system is particularly important as it allows you to reason about the robot's configuration in a spatial context, which is essential for tasks like walking, manipulation, and navigation.

The next section will provide hands-on exercises to practice implementing these humanoid control concepts.