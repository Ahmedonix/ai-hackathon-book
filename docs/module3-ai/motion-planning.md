---
title: Bipedal Motion Planning
description: Fundamentals of humanoid motion planning for bipedal locomotion
sidebar_position: 7
---

# Bipedal Humanoid Motion Planning

## Overview

Bipedal locomotion is one of the most challenging aspects of humanoid robotics. This chapter introduces the fundamentals of motion planning specifically for bipedal robots, covering the key concepts, algorithms, and techniques required to enable stable and efficient walking.

## Learning Objectives

- Understand the challenges of bipedal locomotion
- Learn the fundamentals of Zero Moment Point (ZMP) and Center of Mass (CoM) control
- Implement basic walking patterns and gait generation
- Understand balance control principles for humanoid robots
- Integrate motion planning with perception systems

## Challenges of Bipedal Locomotion

Bipedal locomotion is inherently unstable because the robot has only two points of contact with the ground. Unlike wheeled robots or quadrupeds, a bipedal robot must constantly adjust its center of mass to maintain balance.

### Key Challenges

1. **Dynamic Stability**: Maintaining balance while moving requires continuous adjustment of the body's center of mass
2. **Multi-DOF Control**: Humanoid robots typically have 20+ degrees of freedom, making coordination complex
3. **Terrain Adaptation**: The robot must adapt its gait to different surfaces and obstacles
4. **Real-time Performance**: Balance adjustments must happen within milliseconds to prevent falls
5. **Energy Efficiency**: Optimizing for energy consumption while maintaining stability

## Zero Moment Point (ZMP) Theory

The Zero Moment Point (ZMP) is a fundamental concept in bipedal locomotion. It represents a point on the ground where the sum of all moments caused by the ground reaction forces equals zero.

### ZMP Calculation

```python
import numpy as np

def calculate_zmp(ground_reaction_forces, cop_x, cop_y, fz):
    """
    Calculate ZMP from ground reaction forces
    Ground reaction forces: forces acting on the robot from the ground
    cop_x, cop_y: center of pressure coordinates
    fz: normal force component
    """
    if fz == 0:
        return cop_x, cop_y
    
    # ZMP is calculated as:
    # zmp_x = cop_x - (m * g * z_height) / fz
    # zmp_y = cop_y - (m * g * z_height) / fz
    # where z_height is the height of the CoM above the ground
    z_height = 0.5  # Example CoM height in meters
    g = 9.81  # Gravity constant
    
    # For simplicity, assuming no moment in y-direction for now
    zmp_x = cop_x - (ground_reaction_forces[2] * z_height) / fz
    zmp_y = cop_y - (ground_reaction_forces[2] * z_height) / fz

    return zmp_x, zmp_y
```

### ZMP Stability Criteria

For stable locomotion, the ZMP must remain within the support polygon (the area defined by the feet in contact with the ground).

## Center of Mass (CoM) Control

The Center of Mass (CoM) of a humanoid robot must be carefully controlled to maintain balance during walking.

### CoM Control Algorithm

```python
class CoMController:
    def __init__(self):
        # PID controller parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 0.1   # Integral gain
        self.kd = 0.5   # Derivative gain
        
        # Previous error for derivative calculation
        self.previous_error = 0.0
        self.integral_error = 0.0
        
    def compute_com_control(self, current_com, desired_com, dt):
        """
        Compute control output for CoM stabilization
        """
        # Calculate error
        error = desired_com - current_com
        
        # Update integral
        self.integral_error += error * dt
        
        # Calculate derivative
        derivative = (error - self.previous_error) / dt
        
        # Compute PID output
        output = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        
        # Update previous error
        self.previous_error = error
        
        return output
```

## Walking Pattern Generation

### Inverted Pendulum Model

The simplest model for bipedal walking is the Linear Inverted Pendulum Model (LIPM), where the robot is modeled as a point mass supported by a massless leg.

```python
import numpy as np

class LIPMController:
    def __init__(self, com_height=0.8, g=9.81):
        self.com_height = com_height  # Height of CoM in meters
        self.g = g  # Gravity constant
        self.omega = np.sqrt(g / com_height)  # Natural frequency of the pendulum
        
    def compute_support_position(self, current_com, current_com_velocity, t_step):
        """
        Compute the support position (next foot placement) based on LIPM
        """
        # Calculate where to place the next foot based on current CoM state
        # and the desired walking pattern
        com_x, com_y = current_com
        com_vx, com_vy = current_com_velocity
        
        # Time to next foot contact
        # This is a simplified version - in practice, this depends on step length and cadence
        t_contact = 0.8  # Time to next foot contact in seconds
        
        # Desired ZMP position (typically in the middle of the support foot)
        desired_zmp_x = com_x - com_vx / self.omega
        desired_zmp_y = com_y - com_vy / self.omega
        
        return desired_zmp_x, desired_zmp_y
```

### Gait Generation Algorithm

```python
class BipedalGaitGenerator:
    def __init__(self, step_length=0.3, step_height=0.05, step_time=0.6):
        self.step_length = step_length  # Distance per step in meters
        self.step_height = step_height  # Height of foot during swing phase
        self.step_time = step_time      # Time per step in seconds
        
    def generate_foot_trajectory(self, start_pos, end_pos, t):
        """
        Generate smooth foot trajectory from start to end position
        using a 5th order polynomial to ensure smooth velocity/acceleration
        """
        # Calculate trajectory based on time in gait cycle
        if t < 0 or t > self.step_time:
            return start_pos if t <= 0 else end_pos
            
        # Normalize time
        t_norm = t / self.step_time
        
        # 5th order polynomial for smooth trajectory
        # This ensures velocity and acceleration are zero at start/end
        poly_coeff = [
            0,                    # t^0
            0,                    # t^1
            10,                   # t^2
            -15,                  # t^3
            6,                    # t^4
            0                     # t^5
        ]
        
        # Compute polynomial value
        s = (poly_coeff[2] * t_norm**2 + 
             poly_coeff[3] * t_norm**3 + 
             poly_coeff[4] * t_norm**4)
        
        # Interpolate between start and end positions
        x = start_pos[0] + s * (end_pos[0] - start_pos[0])
        y = start_pos[1] + s * (end_pos[1] - start_pos[1])
        
        # Calculate vertical trajectory to lift foot during swing
        if t_norm < 0.5:
            z_lift = self.step_height * 4 * t_norm**2  # Parabolic lift
        else:
            z_lift = self.step_height * (4 * t_norm - 4 * t_norm**2)  # Parabolic lower
        
        return [x, y, z_lift]
    
    def generate_walking_pattern(self, num_steps, start_pos=[0, 0], step_direction=[1, 0]):
        """
        Generate a complete walking pattern for a number of steps
        """
        pattern = []
        
        # Starting foot position (initially right foot)
        left_foot = [start_pos[0], start_pos[1] + 0.1]  # 10cm apart initially
        right_foot = [start_pos[0], start_pos[1] - 0.1]
        
        for i in range(num_steps):
            # Determine stance foot based on step number
            is_left_stance = (i % 2) == 0
            
            if is_left_stance:
                # Left foot is stance, move right foot forward
                next_right_pos = [
                    right_foot[0] + step_direction[0] * self.step_length,
                    right_foot[1] + step_direction[1] * self.step_length
                ]
                # Add slight outward movement for stability
                next_right_pos[1] += 0.2 if step_direction[0] > 0 else -0.2
                
                pattern.append({
                    'time': i * self.step_time,
                    'left_foot': left_foot,
                    'right_foot': next_right_pos,
                    'stance_foot': 'left'
                })
                
                right_foot = next_right_pos
            else:
                # Right foot is stance, move left foot forward
                next_left_pos = [
                    left_foot[0] + step_direction[0] * self.step_length,
                    left_foot[1] + step_direction[1] * self.step_length
                ]
                # Add slight inward movement for stability
                next_left_pos[1] -= 0.2 if step_direction[0] > 0 else -0.2
                
                pattern.append({
                    'time': i * self.step_time,
                    'left_foot': next_left_pos,
                    'right_foot': right_foot,
                    'stance_foot': 'right'
                })
                
                left_foot = next_left_pos
        
        return pattern
```

## Balance Control

### Balance Feedback Control

```python
import numpy as np
from scipy import signal

class BalanceController:
    def __init__(self):
        # PID parameters for balance control
        self.roll_pid = {'kp': 1.0, 'ki': 0.05, 'kd': 0.1}
        self.pitch_pid = {'kp': 1.5, 'ki': 0.08, 'kd': 0.2}
        
        # Previous measurements for derivative calculation
        self.prev_roll_error = 0.0
        self.prev_pitch_error = 0.0
        self.int_roll_error = 0.0
        self.int_pitch_error = 0.0
        
    def compute_balance_adjustment(self, imu_data, desired_pose, dt):
        """
        Compute balance adjustments based on IMU data
        """
        # Extract roll and pitch from IMU data
        current_roll = imu_data['roll']
        current_pitch = imu_data['pitch']
        
        # Calculate errors
        roll_error = desired_pose['roll'] - current_roll
        pitch_error = desired_pose['pitch'] - current_pitch
        
        # Compute PID outputs for roll and pitch
        roll_output = self._pid_control(
            roll_error, self.prev_roll_error, self.int_roll_error,
            self.roll_pid, dt
        )
        
        pitch_output = self._pid_control(
            pitch_error, self.prev_pitch_error, self.int_pitch_error,
            self.pitch_pid, dt
        )
        
        # Update previous errors
        self.prev_roll_error = roll_error
        self.prev_pitch_error = pitch_error
        
        return {'roll_adjustment': roll_output, 'pitch_adjustment': pitch_output}
    
    def _pid_control(self, error, prev_error, integral, pid_params, dt):
        """
        Generic PID controller implementation
        """
        # Update integral
        integral += error * dt
        
        # Calculate derivative
        derivative = (error - prev_error) / dt if dt > 0 else 0
        
        # Compute output
        output = (pid_params['kp'] * error + 
                  pid_params['ki'] * integral + 
                  pid_params['kd'] * derivative)
        
        return output, integral
```

## ROS 2 Integration for Motion Planning

### Motion Planning Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planning_node')
        
        # Initialize controllers
        self.gait_generator = BipedalGaitGenerator()
        self.balance_controller = BalanceController()
        self.com_controller = CoMController()
        
        # Subscribers
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # Publishers
        self.joint_command_pub = self.create_publisher(Float64MultiArray, 'joint_group_position_controller/commands', 10)
        self.com_state_pub = self.create_publisher(Float64MultiArray, 'com_state', 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz
        
        # Robot state
        self.current_imu = None
        self.current_joints = None
        self.desired_velocity = Twist()
        self.foot_positions = {'left': [0, 0.1, 0], 'right': [0, -0.1, 0]}
        
        self.get_logger().info('Motion Planning Node initialized')
    
    def imu_callback(self, msg):
        self.current_imu = msg
    
    def joint_state_callback(self, msg):
        self.current_joints = msg
    
    def cmd_vel_callback(self, msg):
        self.desired_velocity = msg
    
    def control_loop(self):
        """
        Main control loop for motion planning and balance
        """
        if self.current_imu is None or self.current_joints is None:
            return
        
        # Get current CoM from kinematics (simplified)
        current_com = self.estimate_com_position(self.current_joints)
        current_com_velocity = self.estimate_com_velocity()
        
        # Generate walking pattern based on desired velocity
        walking_pattern = self.gait_generator.generate_walking_pattern(
            num_steps=1,  # Generate next step only
            start_pos=[0, 0],
            step_direction=[self.desired_velocity.linear.x, self.desired_velocity.linear.y]
        )
        
        # Calculate desired CoM position based on ZMP
        desired_com = self.calculate_desired_com(walking_pattern)
        
        # Control CoM to follow desired trajectory
        com_adjustment = self.com_controller.compute_com_control(
            current_com, desired_com, 0.01  # 10ms dt
        )
        
        # Calculate balance adjustments based on IMU
        imu_data = {
            'roll': self.current_imu.orientation.x,
            'pitch': self.current_imu.orientation.y
        }
        
        desired_pose = {'roll': 0.0, 'pitch': 0.0}  # Keep upright
        balance_adjustment = self.balance_controller.compute_balance_adjustment(
            imu_data, desired_pose, 0.01
        )
        
        # Combine adjustments and generate joint commands
        joint_commands = self.generate_joint_commands(
            com_adjustment, balance_adjustment, walking_pattern
        )
        
        # Publish joint commands
        joint_cmd_msg = Float64MultiArray()
        joint_cmd_msg.data = joint_commands
        self.joint_command_pub.publish(joint_cmd_msg)
        
        # Publish CoM state for monitoring
        com_msg = Float64MultiArray()
        com_msg.data = [current_com[0], current_com[1], current_com[2]]
        self.com_state_pub.publish(com_msg)
    
    def estimate_com_position(self, joint_state):
        """
        Estimate center of mass position from joint angles
        Simplified implementation - in reality this would use full kinematics
        """
        # This is a placeholder - a real implementation would use kinematic models
        # to compute the CoM based on joint positions
        return [0.0, 0.0, 0.8]  # Simplified: CoM at 0.8m height
    
    def calculate_desired_com(self, walking_pattern):
        """
        Calculate desired CoM position based on walking pattern
        """
        # Simplified implementation - in reality this would use ZMP-based planning
        if walking_pattern:
            step = walking_pattern[0]
            # Desired CoM position slightly in front of support foot
            support_foot = step['left_foot'] if step['stance_foot'] == 'left' else step['right_foot']
            return [support_foot[0] + 0.1, support_foot[1], 0.8]  # 10cm in front of support foot
        else:
            return [0.0, 0.0, 0.8]
    
    def generate_joint_commands(self, com_adjustment, balance_adjustment, walking_pattern):
        """
        Generate joint position commands based on all adjustments
        """
        # Placeholder implementation - in reality this would use inverse kinematics
        # to convert desired foot positions and balance adjustments to joint angles
        return [0.0] * 12  # 12 DOF humanoid (simplified)

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    
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

## Perception Integration

Motion planning must be integrated with perception systems to enable the robot to navigate safely in real environments.

### Obstacle Avoidance

```python
class ObstacleAvoidance:
    def __init__(self):
        self.safe_distance = 0.5  # Minimum safe distance in meters
        self.avoidance_threshold = 1.0  # Distance at which to start avoiding
        
    def adjust_walking_path(self, current_path, obstacles):
        """
        Adjust walking path based on detected obstacles
        """
        adjusted_path = []
        
        for step_pos in current_path:
            closest_obstacle_dist = float('inf')
            
            for obstacle in obstacles:
                dist = self.calculate_distance(step_pos, obstacle)
                closest_obstacle_dist = min(closest_obstacle_dist, dist)
            
            if closest_obstacle_dist < self.avoidance_threshold:
                # Adjust step position to avoid obstacle
                adjusted_pos = self.calculate_avoidance_position(step_pos, obstacles)
                adjusted_path.append(adjusted_pos)
            else:
                adjusted_path.append(step_pos)
        
        return adjusted_path
    
    def calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_avoidance_position(self, original_pos, obstacles):
        """
        Calculate safe position to avoid obstacles
        """
        # Simple implementation: move perpendicular to the direction of the obstacle
        if obstacles:
            # Find the closest obstacle and move away from it
            closest_obstacle = min(obstacles, key=lambda obs: self.calculate_distance(original_pos, obs))
            
            # Calculate direction vector from obstacle to original position
            direction = [original_pos[0] - closest_obstacle[0], original_pos[1] - closest_obstacle[1]]
            distance = np.sqrt(direction[0]**2 + direction[1]**2)
            
            if distance > 0:
                # Normalize direction
                direction[0] /= distance
                direction[1] /= distance
                
                # Move safe distance away from obstacle
                safe_pos = [
                    closest_obstacle[0] + direction[0] * self.safe_distance,
                    closest_obstacle[1] + direction[1] * self.safe_distance
                ]
                
                return safe_pos
        
        return original_pos
```

## Practical Exercise: Implementing Basic Walking

### Exercise Objective

Implement a simple walking pattern generator and integrate it with balance control.

### Prerequisites

- Basic understanding of ROS 2
- Access to a humanoid robot simulation environment (Gazebo)
- Understanding of joint control

### Exercise Steps

1. **Set Up the Environment**
   ```bash
   # Launch the humanoid robot simulation
   ros2 launch my_robot_bringup simulation.launch.py
   ```

2. **Implement the Gait Generator**
   ```python
   # Create gait_generator.py with the BipedalGaitGenerator class
   # Test the walking pattern generation
   python3 -c "
   from gait_generator import BipedalGaitGenerator
   generator = BipedalGaitGenerator()
   pattern = generator.generate_walking_pattern(5, [0, 0], [1, 0])
   print('Walking pattern:', pattern)
   "
   ```

3. **Integrate with Balance Control**
   - Combine the gait generator with the balance controller
   - Test in simulation with various walking speeds and directions

4. **Test Obstacle Avoidance**
   - Add simulated obstacles to the environment
   - Implement the obstacle avoidance logic
   - Verify the robot can navigate around obstacles

5. **Evaluate Performance**
   - Measure stability metrics (ZMP deviation, CoM variance)
   - Assess energy efficiency
   - Check if the robot can maintain balance under perturbations

## Summary

In this chapter, we've covered the fundamentals of bipedal humanoid motion planning:

1. **ZMP and CoM control**: The theoretical foundation for stable bipedal locomotion
2. **Walking pattern generation**: Algorithms to create stable walking motions
3. **Balance control**: Feedback control systems to maintain stability
4. **ROS 2 integration**: Implementing motion planning within the ROS 2 framework
5. **Perception integration**: Incorporating obstacle avoidance and environment awareness

These concepts form the foundation for creating stable, efficient walking patterns for humanoid robots. The next step is to combine these motion planning techniques with the AI perception systems from earlier modules to create robots that can walk autonomously in complex environments.