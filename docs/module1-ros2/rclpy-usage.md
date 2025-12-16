---
sidebar_position: 7
---

# rclpy Usage for ROS Control

## Overview

`rclpy` is the Python client library for ROS 2, providing the standard interface for developing ROS 2 nodes in Python. Understanding rclpy is essential for controlling robots in ROS 2, as it provides the tools to create publishers, subscribers, services, actions, and manage node lifecycles. This section will guide you through the core concepts of using rclpy for robot control applications.

## Core Concepts

### Node Management

The `Node` class is the fundamental building block of ROS 2 Python applications. Every node should inherit from this class to gain access to ROS 2 functionality:

```python
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        # Initialize publishers, subscribers, etc.
        self.get_logger().info('Robot controller initialized')
```

### Parameters in rclpy

Parameters allow for runtime configuration of nodes without code changes:

```python
class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Declare parameters with default values and descriptions
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('wheel_radius', 0.05)
        
        # Access parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
```

## Publishers and Subscribers for Robot Control

### Publishers

Publishers send messages to topics, enabling broadcast of robot state or commands:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class RobotCommandPublisher(Node):
    def __init__(self):
        super().__init__('robot_command_publisher')
        
        # Publisher for velocity commands (for mobile base)
        self.velocity_publisher = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
        )
        
        # Publisher for joint commands (for manipulator or humanoid)
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, 
            '/joint_commands', 
            10
        )
        
        # Publisher for joint states (for feedback)
        self.joint_state_publisher = self.create_publisher(
            JointState, 
            '/joint_states', 
            10
        )
        
        # Timer for publishing commands at a fixed rate
        self.timer = self.create_timer(0.05, self.publish_commands)  # 20Hz
        
        self.command_counter = 0
    
    def publish_commands(self):
        """Publish robot commands at regular intervals"""
        # Create and publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        vel_msg.angular.z = 0.2  # Turn at 0.2 rad/s
        self.velocity_publisher.publish(vel_msg)
        
        # Create and publish joint commands
        joint_msg = Float64MultiArray()
        joint_msg.data = [0.1, 0.2, 0.3, 0.4]  # Example joint positions
        self.joint_command_publisher.publish(joint_msg)
        
        self.get_logger().info(f'Published robot commands - counter: {self.command_counter}')
        self.command_counter += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = RobotCommandPublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscribers

Subscribers receive messages from topics, enabling the robot to react to sensor data or other nodes:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Twist

class RobotSensorSubscriber(Node):
    def __init__(self):
        super().__init__('robot_sensor_subscriber')
        
        # Subscriber for laser scan data (for navigation)
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        # Subscriber for IMU data (for balance/attitude)
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Subscriber for joint states (for monitoring)
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher to send commands based on sensor data
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Robot state variables
        self.min_laser_distance = float('inf')
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.get_logger().info('Sensor subscriber initialized')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        # Find minimum distance in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 50:len(msg.ranges)//2 + 50]
        if front_scan:
            self.min_laser_distance = min(x for x in front_scan if not (x != x or x > 10.0))
        
        # Implement obstacle avoidance
        self.avoid_obstacles()
    
    def imu_callback(self, msg):
        """Process IMU data for balance control"""
        # Extract orientation from quaternion
        import math
        q = msg.orientation
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q.w * q.y - q.z * q.x)
        # Use 90 degrees if out of range
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)
        
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def joint_state_callback(self, msg):
        """Process joint state data"""
        # Log joint positions
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                position = msg.position[i]
                # Process joint position as needed
                # For example, check for joint limits
                if abs(position) > 3.0:  # Example limit
                    self.get_logger().warn(f'Joint {name} near limit: {position}')
    
    def avoid_obstacles(self):
        """Simple obstacle avoidance behavior"""
        cmd_msg = Twist()
        
        if self.min_laser_distance < 0.5:  # Obstacle closer than 0.5m
            # Stop and turn
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5  # Turn right
        else:
            # Move forward safely
            cmd_msg.linear.x = 0.3
            cmd_msg.angular.z = 0.0
        
        self.cmd_vel_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    subscriber = RobotSensorSubscriber()
    
    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services for Robot Control

Services allow for request-response communication, perfect for configuration or on-demand actions:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.srv import SetBool, Trigger
from std_msgs.msg import Bool
import math

class RobotConfigurationService(Node):
    def __init__(self):
        super().__init__('robot_configuration_service')
        
        # Service to enable/disable robot
        self.enable_service = self.create_service(
            SetBool,
            'enable_robot',
            self.enable_robot_callback
        )
        
        # Service to reset robot state
        self.reset_service = self.create_service(
            Trigger,
            'reset_robot',
            self.reset_robot_callback
        )
        
        # Robot state
        self.robot_enabled = False
        self.robot_state_publisher = self.create_publisher(Bool, 'robot_enabled', 10)
        
        self.get_logger().info('Robot configuration service initialized')
    
    def enable_robot_callback(self, request, response):
        """Handle robot enable/disable requests"""
        self.robot_enabled = request.data
        
        if self.robot_enabled:
            self.get_logger().info('Robot enabled')
            response.message = 'Robot enabled successfully'
        else:
            self.get_logger().info('Robot disabled')
            response.message = 'Robot disabled'
        
        response.success = True
        
        # Publish state change
        state_msg = Bool()
        state_msg.data = self.robot_enabled
        self.robot_state_publisher.publish(state_msg)
        
        return response
    
    def reset_robot_callback(self, request, response):
        """Handle robot reset requests"""
        self.get_logger().info('Resetting robot state...')
        
        # Reset any internal state
        # For example: reset odometry, joint positions, etc.
        self.robot_enabled = True  # Default to enabled after reset
        
        response.success = True
        response.message = 'Robot reset successfully'
        
        return response

def main(args=None):
    rclpy.init(args=args)
    service = RobotConfigurationService()
    
    try:
        rclpy.spin(service)
    except KeyboardInterrupt:
        pass
    finally:
        service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions for Complex Robot Behaviors

Actions are ideal for long-running robot behaviors with feedback:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from example_interfaces.action import NavigateToPose
import time
import math

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        
        self.get_logger().info('Navigation action server initialized')
    
    def goal_callback(self, goal_request):
        """Accept or reject goal based on some criteria"""
        # Example: reject goals that are too far away
        distance = math.sqrt(
            (goal_request.pose.position.x - self.current_x)**2 +
            (goal_request.pose.position.y - self.current_y)**2
        )
        
        if distance > 10.0:  # Too far
            self.get_logger().info('Rejected goal: too far away')
            return GoalResponse.REJECT
        else:
            self.get_logger().info('Accepted goal')
            return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Accept or reject cancel request"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        """Execute the navigation goal"""
        self.get_logger().info('Executing navigation goal')
        
        # Get goal pose
        target_x = goal_handle.request.pose.position.x
        target_y = goal_handle.request.pose.position.y
        
        # Calculate distance to target
        distance_to_target = math.sqrt(
            (target_x - self.current_x)**2 + 
            (target_y - self.current_y)**2
        )
        
        # Initialize feedback
        feedback_msg = NavigateToPose.Feedback()
        feedback_msg.current_pose.position.x = self.current_x
        feedback_msg.current_pose.position.y = self.current_y
        
        # Navigate to target with feedback
        steps = int(distance_to_target / 0.1)  # Move in 0.1m increments
        for i in range(steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.pose = feedback_msg.current_pose
                self.get_logger().info('Navigation canceled')
                return result
            
            # Simulate movement
            progress = float(i + 1) / float(steps)
            self.current_x = self.current_x + (target_x - self.current_x) * progress
            self.current_y = self.current_y + (target_y - self.current_y) * progress
            
            # Update feedback
            feedback_msg.current_pose.position.x = self.current_x
            feedback_msg.current_pose.position.y = self.current_y
            feedback_msg.distance_to_goal = distance_to_target * (1.0 - progress)
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate time to move
            time.sleep(0.1)
            
            self.get_logger().info(f'Progress: {progress*100:.1f}%')
        
        # Complete successfully
        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.pose = feedback_msg.current_pose
        self.get_logger().info(f'Navigation completed. Final position: ({self.current_x:.2f}, {self.current_y:.2f})')
        
        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = NavigationActionServer()
    
    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Timers for Periodic Control

Timers are crucial for robot control as they enable periodic execution of control loops:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import math

class RobotControlLoop(Node):
    def __init__(self):
        super().__init__('robot_control_loop')
        
        # Publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_command_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Timer for main control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz
        self.emergency_stop_timer = self.create_timer(0.1, self.emergency_stop_check)  # 10Hz
        
        # Robot state variables
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]  # Example: 4 joints
        self.emergency_stop = False
        
        self.get_logger().info('Robot control loop initialized')
    
    def control_loop(self):
        """Main control loop running at 100Hz"""
        if not self.emergency_stop:
            # Example: sinusoidal velocity commands
            current_time = self.get_clock().now().nanoseconds / 1e9
            self.linear_velocity = 0.5 * math.sin(current_time)
            self.angular_velocity = 0.2 * math.cos(current_time * 0.5)
            
            # Update joint positions (example: oscillating motion)
            for i in range(len(self.joint_positions)):
                self.joint_positions[i] = 0.5 * math.sin(current_time + i)
            
            # Publish velocity command
            cmd_msg = Twist()
            cmd_msg.linear.x = self.linear_velocity
            cmd_msg.angular.z = self.angular_velocity
            self.cmd_vel_publisher.publish(cmd_msg)
            
            # Publish joint commands
            joint_msg = JointState()
            joint_msg.name = ['joint1', 'joint2', 'joint3', 'joint4']
            joint_msg.position = self.joint_positions
            self.joint_command_publisher.publish(joint_msg)
    
    def emergency_stop_check(self):
        """Check for emergency stop conditions"""
        # This could check for various conditions like:
        # - laser scan showing imminent collision
        # - joint limit violations
        # - communication timeouts
        # - manual emergency stop button
        
        # For example, check if velocity commands are too high
        if abs(self.linear_velocity) > 2.0 or abs(self.angular_velocity) > 1.0:
            self.get_logger().error('EMERGENCY STOP: Velocity limits exceeded')
            self.emergency_stop = True
        
        if self.emergency_stop:
            # Send zero velocity command
            stop_msg = Twist()
            self.cmd_vel_publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotControlLoop()
    
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

## Best Practices for Robot Control with rclpy

### 1. Proper Shutdown and Cleanup
Always properly clean up resources in your node:

```python
class RobotController(Node):
    def destroy_node(self):
        # Stop any ongoing motion
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        
        # Clean up any other resources
        super().destroy_node()
```

### 2. Use Appropriate QoS Settings
For robot control, use reliable QoS settings:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

self.publisher = self.create_publisher(JointState, 'joint_states', qos_profile)
```

### 3. Implement Safety Checks
Always implement safety checks in your control nodes:

```python
def control_loop(self):
    if not self.safety_check():
        self.emergency_stop()
        return
    
    # Proceed with control commands

def safety_check(self):
    # Check for any unsafe conditions
    return not self.emergency_stop
```

## Debugging and Logging

Proper logging is essential for debugging robot control systems:

```python
# Use different log levels appropriately
self.get_logger().debug('Detailed debugging info')
self.get_logger().info('Normal operational info')
self.get_logger().warn('Warning about potential issues')
self.get_logger().error('Error occurred')
self.get_logger().fatal('Fatal error that requires shutdown')
```

## Summary

The rclpy library provides all the necessary tools for robot control in ROS 2:

1. **Nodes**: The foundation for all ROS 2 functionality
2. **Publishers/Subscribers**: For real-time communication with sensors and actuators
3. **Services**: For configuration and on-demand operations
4. **Actions**: For complex, long-running behaviors
5. **Timers**: For implementing control loops
6. **Parameters**: For runtime configuration

With these tools, you can implement sophisticated robot control systems that are robust, safe, and maintainable. The next section will cover practical hands-on exercises to help you apply these concepts to real-world robot control challenges.