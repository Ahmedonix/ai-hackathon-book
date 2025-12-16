---
sidebar_position: 4
---

# Topics, Services, and Actions Concepts with Examples

## Overview

In ROS 2, there are three primary ways that nodes can communicate with each other: topics (using the publish-subscribe pattern), services (using the client-server pattern), and actions (for long-running tasks with feedback). Understanding these communication patterns is essential for building effective humanoid robots that can coordinate between different components.

## Topics: Publish-Subscribe Communication

Topics enable one-way, asynchronous communication through a publish-subscribe pattern. Publishers send messages to a topic, and subscribers receive messages from the topic without either knowing about the other.

### Key Characteristics:
- **Asynchronous**: Publishers don't wait for responses
- **Many-to-many**: Multiple publishers can publish to one topic; multiple subscribers can subscribe to one topic
- **Unidirectional**: Data flows in one direction from publisher to subscriber
- **Best for**: Continuous data streams, sensor data, state information

### Example: Sensor Data Broadcasting
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(
            JointState, 
            '/joint_states', 
            10
        )
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10Hz
        self.joint_names = ['hip_joint', 'knee_joint', 'ankle_joint']

    def publish_sensor_data(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [random.uniform(-1.5, 1.5) for _ in self.joint_names]
        msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        msg.effort = [random.uniform(-10.0, 10.0) for _ in self.joint_names]
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published joint states: {msg.position}')

def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisher()
    
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

## Services: Request-Response Communication

Services enable synchronous, bidirectional communication using a client-server pattern. A client sends a request to a server and waits for a response.

### Key Characteristics:
- **Synchronous**: Client waits for the response
- **Bidirectional**: Request goes one way, response goes the other
- **One-to-one**: One client makes a request to one server
- **Best for**: Configuration, on-demand information, state queries

### Example: Robot Configuration Service
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool

class RobotConfigService(Node):
    def __init__(self):
        super().__init__('robot_config_service')
        self.srv = self.create_service(
            SetBool,
            'enable_robot',
            self.enable_robot_callback
        )
        self.robot_enabled = False

    def enable_robot_callback(self, request, response):
        self.robot_enabled = request.data
        if self.robot_enabled:
            self.get_logger().info('Robot enabled')
        else:
            self.get_logger().info('Robot disabled')
        
        response.success = True
        response.message = f'Robot {"enabled" if self.robot_enabled else "disabled"}'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotConfigService()

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

## Actions: Long-Running Tasks with Feedback

Actions are designed for long-running tasks that provide continuous feedback to the client. They combine aspects of both topics and services.

### Key Characteristics:
- **Long-running**: Designed for tasks that take time to complete
- **Feedback**: Continuous feedback during execution
- **Goal/result**: Send a goal, get a result when complete
- **Cancelable**: Clients can cancel goals
- **Best for**: Navigation, complex manipulations, tasks with progress tracking

### Example: Robot Navigation Action
```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci
import time

class NavigationActionServer(Node):

    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Using Fibonacci as an example - in practice, you'd define a NavigateToPose action
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self._goal_handle = None

    def goal_callback(self, goal_request):
        self.get_logger().info('Received navigation goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing navigation goal...')
        self._goal_handle = goal_handle

        # Simulate navigation progress
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # Check if cancel was requested
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation goal canceled')
                return Fibonacci.Result()

            # Simulate movement progress
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Navigation progress: {len(feedback_msg.sequence)} steps completed')
            
            # Simulate robot movement time
            time.sleep(0.5)

        # Complete successfully
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Navigation completed successfully')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = NavigationActionServer()

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

## Comparison of Communication Patterns

| Pattern | Communication | Timing | Use Cases | Example |
|---------|---------------|--------|-----------|---------|
| Topics | Unidirectional | Asynchronous | Continuous data | Sensor streams, robot state |
| Services | Bidirectional | Synchronous | On-demand requests | Get robot status, configure |
| Actions | Bidirectional | Asynchronous | Long-running tasks | Navigation, manipulation |

## When to Use Each Pattern

### Use Topics When:
- Broadcasting sensor data or robot state
- Continuous monitoring is needed
- Multiple subscribers need the same information
- Real-time performance is critical
- Data loss is acceptable (e.g., old sensor readings)

### Use Services When:
- Requesting specific information
- Configuration changes needed
- Simple, quick operations
- You must get a response before proceeding
- Error handling and validation are important

### Use Actions When:
- Long-running operations are involved
- Progress feedback is needed
- Tasks can be canceled
- Success/failure results are important
- Complex behaviors with multiple steps

## Quality of Service (QoS) Considerations

Each communication pattern supports Quality of Service settings that determine how messages are handled:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example: Reliable communication for safety-critical data
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Use in publisher or subscription
publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Best Practices

1. **Choose the right pattern**: Use topics for continuous data, services for requests, and actions for long operations
2. **Define clear interfaces**: Document message types, services, and action definitions
3. **Handle errors gracefully**: Include error handling in service and action callbacks
4. **Use appropriate QoS**: Consider reliability and performance requirements
5. **Design for modularity**: Break complex behaviors into smaller communicating nodes
6. **Include logging**: Add informative log messages for debugging

## Summary

In this section, we've explored the three main communication patterns in ROS 2:
- **Topics**: For publish-subscribe communication, ideal for continuous data streams
- **Services**: For request-response communication, great for on-demand operations
- **Actions**: For long-running tasks with feedback, perfect for complex behaviors

Understanding these patterns and when to use each is crucial for designing effective communication architectures in your humanoid robot system. The next section will dive into using the `rclpy` library for ROS control in Python.