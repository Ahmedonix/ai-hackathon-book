---
sidebar_position: 3
---

# Step-by-Step Guide for Building Python Nodes

## Overview

This guide will walk you through the process of creating ROS 2 nodes in Python. We'll cover the essential components and patterns you'll use to build nodes that can communicate with other parts of your humanoid robot system.

## Prerequisites

- ROS 2 Iron installed on your system
- Python 3.8 or higher
- Basic Python programming knowledge
- Understanding of ROS 2 concepts (covered in the previous section)

## Creating Your First Node

### 1. Basic Node Structure

Every ROS 2 Python node follows the same basic structure:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class MyRobotNode(Node):
    def __init__(self):
        super().__init__('my_robot_node')
        # Node initialization code goes here
        self.get_logger().info('MyRobotNode has been started')

def main(args=None):
    rclpy.init(args=args)
    
    node = MyRobotNode()
    
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

### 2. Understanding the Basic Structure

- **Import statements**: Import the necessary ROS 2 Python libraries
- **Node class**: Inherit from `rclpy.node.Node` to create a ROS 2 node
- **Constructor (`__init__`)**: Initialize the node with a name and set up components
- **Main function**: Initialize ROS 2, create the node, spin it, and handle cleanup
- **Execution block**: Standard Python pattern to execute code when the script is run directly

### 3. Running the Node

To run your node, save it as `my_robot_node.py` and execute:

```bash
python3 my_robot_node.py
```

## Creating Publisher Nodes

A publisher node sends messages to a topic. Here's how to create one:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Import the message type

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        # Create a publisher
        # Parameters: message type, topic name, queue size
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        
        # Create a timer to publish messages at regular intervals
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Understanding Publisher Components

- `create_publisher()`: Creates a publisher that sends messages of a specific type to a topic
- `create_timer()`: Creates a timer that calls the callback function at regular intervals
- `timer_callback()`: Function called at each timer interval to publish messages

## Creating Subscriber Nodes

A subscriber node receives messages from a topic:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Import the message type

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        # Create a subscription
        # Parameters: message type, topic name, callback function, queue size
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # This function is called whenever a message is received
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Understanding Subscriber Components

- `create_subscription()`: Creates a subscription that receives messages of a specific type from a topic
- `listener_callback()`: Function called whenever a message is received on the topic

## Creating Service Client and Server Nodes

### Service Server

A service server responds to requests:

```python
#!/usr/bin/env python3

from rclpy.node import Node
from example_interfaces.srv import AddTwoInts  # Import service type

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        # Create a service
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    
    minimal_service = MinimalService()
    
    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

A service client makes requests:

```python
#!/usr/bin/env python3

from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
import sys

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    
    minimal_client.get_logger().info(
        f'Result of add_two_ints: {response.sum}')
    
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Action Client and Server Nodes

### Action Server

An action server handles long-running goals:

```python
#!/usr/bin/env python3

from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class MinimalActionServer(Node):

    def __init__(self):
        super().__init__('minimal_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()
            
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
        
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result

def main(args=None):
    rclpy.init(args=args)
    
    action_server = MinimalActionServer()
    
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

## Package Structure

To create a complete ROS 2 package for your nodes, you need several additional files:

### package.xml
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_nodes</name>
  <version>0.0.0</version>
  <description>Package for my robot nodes</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>example_interfaces</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### setup.py
```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package for my robot nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher_node = my_robot_nodes.publisher_node:main',
            'subscriber_node = my_robot_nodes.subscriber_node:main',
        ],
    },
)
```

### setup.cfg
```
[develop]
script-dir=$base/lib/my_robot_nodes

[install]
install-scripts=$base/lib/my_robot_nodes
```

## Best Practices

1. **Error Handling**: Always include proper error handling, especially when initializing ROS 2
2. **Logging**: Use `self.get_logger().info/warn/error` for debugging and monitoring
3. **Resource Management**: Properly destroy nodes and shut down ROS 2 in the `finally` block
4. **Parameter Validation**: Validate inputs to prevent unexpected behavior
5. **Modularity**: Break complex nodes into smaller, focused nodes that communicate via topics
6. **Documentation**: Document your node's purpose, topics, services, and parameters
7. **Testing**: Write unit tests for your node's logic separate from ROS 2 communication

## Running Your Package

After building your package, you can run your nodes using:

```bash
ros2 run my_robot_nodes publisher_node
```

## Summary

In this guide, we've covered:
- Basic node structure and components
- Creating publisher and subscriber nodes
- Implementing service and action nodes
- Proper package structure and setup files
- Best practices for node development

With this knowledge, you can create nodes that form the foundation of your humanoid robot's communication system. The next section will cover robot description formats like URDF, which are essential for representing your humanoid robot's physical structure in ROS 2.