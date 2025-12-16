---
sidebar_position: 14
---

# Module 1 Assessment: ROS 2 Fundamentals

## Overview

This assessment tests your understanding of the core concepts of ROS 2 covered in Module 1. Successfully completing this assessment demonstrates proficiency in ROS 2 architecture, communication patterns, and robot control fundamentals.

## Assessment Objectives

By completing this assessment, you will demonstrate your ability to:

1. Create and execute ROS 2 nodes with proper structure
2. Implement communication patterns (topics, services, actions)
3. Work with robot description formats (URDF/XACRO)
4. Manage parameters and launch multi-node systems
5. Apply AI integration techniques to robot control
6. Implement safety and stability controls

## Assessment Tasks

### Task 1: Node Structure and Communication (25 points)

Create a ROS 2 package with two nodes that communicate via topics:

1. Create a publisher node that publishes `String` messages containing the current timestamp and a counter value to the topic `/robot/status`
2. Create a subscriber node that receives these messages and logs them to the console
3. Implement a service server that can reset the counter in the publisher node
4. Use proper node structure with initialization, execution, and cleanup

**Requirements:**
- Nodes should be in a package named `assessment_nodes`
- Publisher should send messages at 1Hz
- Service should be named `/reset_counter`
- Both nodes should handle shutdown gracefully

### Task 2: Parameter Management (20 points)

Extend the nodes from Task 1 to include parameter management:

1. Add a parameter to the publisher that controls the publishing rate (default 1.0 Hz)
2. Add a parameter to the subscriber that controls the minimum counter value to log (default 0)
3. Add a parameter that sets the maximum counter value before resetting (default 100)
4. Create a launch file that starts both nodes with custom parameter values

### Task 3: Service Implementation (20 points)

Implement a service that performs a calculation based on robot state:

1. Create a service server that accepts two float values (robot position x and y)
2. Calculate the Euclidean distance from the origin (0,0)
3. Return the result along with the angle from the positive x-axis
4. Create a service client that calls this service with predefined coordinates

### Task 4: Action Implementation (20 points)

Design an action for robot navigation:

1. Define an action that accepts a goal pose (x, y, theta)
2. Implement a server that simulates reaching the goal with feedback on progress
3. Provide the client with real-time updates about distance to goal
4. Handle goal cancellation and goal preemption

### Task 5: Robot Description (15 points)

Create a URDF/XACRO model of a simple mobile robot:

1. Design a robot with a base link, two wheels, and a caster
2. Define the visual and collision properties for each link
3. Create proper joints between links with appropriate limits
4. Use XACRO macros to define the wheels and reuse them for both left and right

## Implementation Guidelines

### Project Structure
```
assessment_project/
├── src/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── assessment_nodes/
│   │   ├── assessment_nodes/
│   │   │   ├── publisher_node.py
│   │   │   ├── subscriber_node.py
│   │   │   ├── service_server.py
│   │   │   ├── service_client.py
│   │   │   ├── action_server.py
│   │   │   └── action_client.py
│   │   ├── launch/
│   │   │   └── assessment_launch.py
│   │   ├── urdf/
│   │   │   └── simple_mobile_robot.xacro
│   │   ├── config/
│   │   └── setup.py
```

### Code Standards

For all Python nodes:
- Include proper docstrings
- Follow PEP8 coding standards
- Handle exceptions appropriately
- Use meaningful variable and function names
- Include proper logging

### Testing Requirements

Test your implementation to ensure:
- All nodes run without errors
- Communication works as expected
- Parameters are properly managed
- Launch files execute correctly
- Robot model displays correctly in RViz

## Assessment Rubric

### Task 1: Node Structure and Communication (25 points)
- **Node structure**: Proper inheritance from Node, initialization, and cleanup (10 points)
- **Publisher implementation**: Correctly publishes messages at specified rate (5 points)
- **Subscriber implementation**: Correctly receives and processes messages (5 points)
- **Service implementation**: Properly resets counter when called (5 points)

### Task 2: Parameter Management (20 points)
- **Parameter declaration**: Properly declares and uses parameters (10 points)
- **Launch file**: Correctly sets custom parameter values (10 points)

### Task 3: Service Implementation (20 points)
- **Server implementation**: Calculates correct distance and angle (10 points)
- **Client implementation**: Correctly calls service and handles response (10 points)

### Task 4: Action Implementation (20 points)
- **Action server**: Properly implements goal handling, feedback, and results (10 points)
- **Action client**: Correctly sends goal and handles feedback/results (10 points)

### Task 5: Robot Description (15 points)
- **URDF structure**: Properly defined links and joints (8 points)
- **XACRO usage**: Effective use of macros and properties (7 points)

## Submission Requirements

Prepare the following for submission:
1. Complete source code in proper package structure
2. Launch files demonstrating all functionality
3. URDF/XACRO model file
4. Configuration files if used
5. README.md with instructions on how to build and run your solution

## Self-Assessment Checklist

Before submitting, ensure your solution:
- [ ] Builds without errors using `colcon build`
- [ ] Runs without errors
- [ ] Follows ROS 2 best practices for node structure
- [ ] Implements all required functionality
- [ ] Includes proper error handling
- [ ] Has a working launch file
- [ ] Displays robot model in RViz (for Task 5)
- [ ] Includes documentation and comments

## Example Implementation Hints

### Publisher Node Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import SetBool  # For reset service
from datetime import datetime


class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        
        # Declare parameters with defaults
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('max_counter', 100)
        
        # Get parameter values
        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_counter = self.get_parameter('max_counter').value
        
        # Initialize publisher
        self.publisher = self.create_publisher(String, '/robot/status', 10)
        
        # Initialize counter
        self.counter = 0
        
        # Create timer based on parameter
        self.timer = self.create_timer(1.0/self.publish_rate, self.publish_status)
        
        # Create service
        self.srv = self.create_service(SetBool, 'reset_counter', self.reset_counter_callback)
        
        self.get_logger().info('Status publisher initialized')

    def publish_status(self):
        # Create and populate message
        msg = String()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg.data = f'{timestamp} - Counter: {self.counter}'
        
        # Publish message
        self.publisher.publish(msg)
        
        # Increment counter with wrap-around
        self.counter = (self.counter + 1) % self.max_counter
        
        self.get_logger().info(f'Published: {msg.data}')

    def reset_counter_callback(self, request, response):
        self.counter = 0
        response.success = True
        response.message = 'Counter reset to 0'
        self.get_logger().info('Counter reset via service call')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = StatusPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Client Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  # You might need to create a custom service
import math


class DistanceCalculatorClient(Node):
    def __init__(self):
        super().__init__('distance_calculator_client')
        
        # Create client
        self.client = self.create_client(Trigger, 'calculate_distance')  # Replace with custom srv
        
        # Wait for service
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
    
    def call_service(self, x, y):
        # Create request
        request = Trigger.Request()  # Replace with your custom request type
        
        # For custom service, you would have x and y fields
        # request.x = x
        # request.y = y
        
        # Call service
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            result = future.result()
            self.get_logger().info(f'Distance: {result.distance}, Angle: {result.angle}')
        else:
            self.get_logger().error(f'Service call failed: {future.exception()}')
        
        return future.result() if future.result() is not None else None


def main(args=None):
    rclpy.init(args=args)
    client = DistanceCalculatorClient()
    
    # Call service with some coordinates
    client.call_service(3.0, 4.0)
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Resources for Completion

- ROS 2 Documentation: https://docs.ros.org/
- ROS 2 tutorials: https://docs.ros.org/en/rolling/Tutorials.html
- URDF tutorials: http://wiki.ros.org/urdf/Tutorials
- Action tutorials: https://docs.ros.org/en/rolling/Tutorials/Actions.html

## Extension Challenges (Optional)

For extra practice, consider these extensions:
1. Add logging capabilities to record robot status over time
2. Implement a dynamic parameter callback to change behavior at runtime
3. Create a custom message type for the service instead of using built-in types
4. Add visualization markers for the robot's planned path in RViz
5. Implement a state machine in the behavior manager to coordinate multiple actions

## Conclusion

Successfully completing this assessment demonstrates mastery of core ROS 2 concepts including node development, communication patterns, parameter management, and robot description. These skills form the foundation for advanced robotics development with ROS 2.

Take your time to understand each component thoroughly, as these concepts will be essential as you progress to modules covering simulation, AI integration, and advanced robotics applications.