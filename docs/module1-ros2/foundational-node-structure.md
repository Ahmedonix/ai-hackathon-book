# Foundational ROS 2 Node Structure and Communication Patterns

## Overview

This document outlines the foundational ROS 2 node structure and communication patterns that will be used throughout the Physical AI & Humanoid Robotics Book. Understanding these patterns is essential for developing humanoid robot applications.

## ROS 2 Node Architecture

### Basic Node Structure

A ROS 2 node typically consists of:
- Node class that inherits from `rclpy.Node`
- Publishers and subscribers for message passing
- Services and actions for synchronous/asynchronous communication
- Parameters for configuration
- Timers for periodic execution

### Simple Publisher Node Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Simple Subscriber Node Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Communication Patterns

### 1. Publisher-Subscriber (Topics)
- Asynchronous, many-to-many communication
- Used for continuous data streams (sensors, robot state)
- Quality of Service (QoS) settings allow tuning for performance vs. reliability

### 2. Service-Client
- Synchronous, request-response pattern
- Used for actions that require acknowledgment
- Request blocks until response is received

### 3. Action-Client
- Asynchronous, with feedback and goal tracking
- Used for long-running tasks with progress updates
- Supports preemption (canceling ongoing actions)

## Core Communication Interfaces

### Joint State Interface
- Topic: `/joint_states`
- Type: `sensor_msgs/msg/JointState`
- Purpose: Report current positions, velocities, and efforts of robot joints

### Command Velocity Interface
- Topic: `/cmd_vel`
- Type: `geometry_msgs/msg/Twist`
- Purpose: Send velocity commands to robot base controller

### TF2 Transform Interface
- Topic: `/tf` and `/tf_static`
- Type: `tf2_msgs/msg/TFMessage`
- Purpose: Represent coordinate transformations between robot frames

### Robot State Interface
- Topic: `/robot_state`
- Type: `nav_msgs/msg/Odometry`
- Purpose: Report robot position, orientation, and velocities

## Best Practices

### Node Design
- Each node should have a single, well-defined responsibility
- Use composition over inheritance when combining functionality
- Implement proper error handling and logging
- Consider resource usage and cleanup in destroy methods

### Naming Conventions
- Use descriptive, lowercase names for nodes and topics
- Use underscores to separate words
- Follow ROS 2 naming conventions for standard interfaces

### Resource Management
- Always call `destroy_node()` to free resources
- Use context managers when available
- Implement proper cleanup in signal handlers

## Error Handling

### Common Error Patterns
- Handle connection loss gracefully
- Implement retry mechanisms for critical communications
- Use timeouts appropriately
- Log errors with sufficient context for debugging

### Shutdown Handling
- Implement proper cleanup during node shutdown
- Ensure all resources are released
- Handle SIGINT and SIGTERM signals appropriately

## Next Steps

With this foundational ROS 2 node structure in place, we can build more complex humanoid robot applications by:
1. Creating specialized nodes for robot control
2. Implementing sensor processing nodes
3. Building behavior trees or state machines
4. Adding AI integration for higher-level capabilities

These patterns will be expanded upon in Module 1 of the book, where we'll build complete ROS 2 packages for humanoid robot applications.