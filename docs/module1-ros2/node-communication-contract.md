# ROS 2 Node Communication Contract

## Overview

This document details the ROS 2 node communication patterns based on the contract defined in `contracts/ros2-communication.yaml`. Understanding these communication patterns is fundamental to building reliable humanoid robot systems.

## Communication Interfaces

### 1. JointState Message Interface

#### Topic: `/joint_states`
- **Type**: `sensor_msgs/JointState`
- **Direction**: Publisher from robot controller, Subscriber for perception nodes
- **Purpose**: Publish current joint positions, velocities, and efforts for the humanoid robot

**Implementation Example:**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.joint_names = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint']
        self.i = 0

    def timer_callback(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [math.sin(self.i/10), math.cos(self.i/10), math.sin(self.i/10), math.cos(self.i/10)]
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        self.publisher.publish(msg)
        self.i += 1
```

### 2. Twist Message Interface

#### Topic: `/cmd_vel`
- **Type**: `geometry_msgs/Twist`
- **Direction**: Subscriber to motion commands
- **Purpose**: Receive velocity commands for robot base movement

**Implementation Example:**
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Linear: {msg.linear.x}, Angular: {msg.angular.z}')
```

### 3. IMU Data Interface

#### Topic: `/imu/data`
- **Type**: `sensor_msgs/Imu`
- **Direction**: Publisher from simulation/hardware
- **Purpose**: Provide inertial measurement data for balance and orientation

### 4. Laser Scan Interface

#### Topic: `/scan`
- **Type**: `sensor_msgs/LaserScan`
- **Direction**: Publisher from simulated LiDAR
- **Purpose**: Provide distance measurements for obstacle detection

### 5. Camera Image Interface

#### Topic: `/camera/image_raw`
- **Type**: `sensor_msgs/Image`
- **Direction**: Publisher from simulated camera
- **Purpose**: Provide visual data for perception algorithms

## Service Interfaces

### 1. Robot Control Service
- **Service Name**: `/robot_control`
- **Type**: Custom service (defined in book)
- **Purpose**: Provide high-level control commands to the robot

### 2. Navigation Service
- **Service Name**: `/navigate_to_pose`
- **Type**: `nav_msgs/GetPlan` (or similar)
- **Purpose**: Request navigation to a specific pose

## Action Interfaces

### 1. Navigation Action
- **Action Name**: `/move_base`
- **Type**: `move_base_msgs/MoveBaseAction`
- **Purpose**: Execute navigation task with feedback and goal management

## Quality Requirements

### Performance
- Topic update rates as specified by sensor requirements
- Service response time < 100ms for critical commands
- Action feedback published at minimum 10Hz

### Reliability
- Nodes must handle disconnections gracefully
- Error states must be published via appropriate channels
- Recovery procedures documented for each interface

### Compatibility
- Interfaces compatible with ROS 2 Iron
- Message types aligned with standard ROS 2 packages
- Backward compatibility maintained where possible

## Testing the Communication Patterns

### 1. Publisher-Subscriber Test
```bash
# Terminal 1: Start the joint state publisher
ros2 run humanoid_robot_examples joint_state_publisher

# Terminal 2: Monitor the joint states
ros2 topic echo /joint_states
```

### 2. Service Call Test
```bash
# Terminal 1: Start the service server
ros2 run humanoid_robot_examples robot_control_server

# Terminal 2: Call the service
ros2 service call /robot_control humanoid_interfaces/RobotControl "command: 'move_forward'"
```

## Integration with Humanoid Control

In the context of humanoid robots, these communication patterns enable:

1. **Sensor Integration**: Aggregating data from multiple sensors (IMU, LiDAR, cameras)
2. **Actuator Control**: Controlling multiple joints and actuators through coordinated commands
3. **Behavior Coordination**: Implementing complex behaviors that require coordination between multiple subsystems
4. **AI Integration**: Enabling AI agents to receive sensor data and send commands to the physical robot

This communication contract provides the foundation for building sophisticated humanoid robot systems where different components can communicate reliably and efficiently.