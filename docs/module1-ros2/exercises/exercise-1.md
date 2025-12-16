---
sidebar_position: 10
---

# Exercise 1: Building Basic ROS 2 Publisher and Subscriber Nodes

## Objective

In this exercise, you will create your first ROS 2 publisher and subscriber nodes in Python. You'll learn how to set up the basic structure of a ROS 2 node, create publishers and subscribers, and exchange messages between nodes.

## Prerequisites

Before starting this exercise, ensure you have:
- ROS 2 Iron installed on your system
- Python 3.8 or higher
- Basic Python programming knowledge
- Terminal/command line familiarity

## Step 1: Setting Up Your Workspace

First, create a new ROS 2 package for your exercise:

```bash
# Navigate to your ROS 2 workspace
cd ~/ros2_ws/src

# Create a new package for the exercise
ros2 pkg create --build-type ament_python basic_nodes_exercise --dependencies rclpy std_msgs geometry_msgs

# Navigate to the package directory
cd basic_nodes_exercise/basic_nodes_exercise
```

## Step 2: Creating a Publisher Node

Create a file named `simple_publisher.py`:

```python
#!/usr/bin/env python3

"""
Simple Publisher Node
This node publishes messages to a topic at regular intervals.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        
        # Create a publisher for String messages on the 'chatter' topic
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        
        # Set up a timer to publish messages every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Counter to track messages sent
        self.i = 0
        
        # Log that the publisher has been created
        self.get_logger().info('Simple Publisher node initialized')

    def timer_callback(self):
        """Callback function that executes on timer events"""
        # Create a new String message
        msg = String()
        msg.data = f'Hello World: {self.i}'
        
        # Publish the message
        self.publisher_.publish(msg)
        
        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')
        
        # Increment the counter
        self.i += 1


def main(args=None):
    """Main function that initializes and runs the node"""
    rclpy.init(args=args)

    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        simple_publisher.get_logger().info('Interrupted by user')
    finally:
        # Destroy the node explicitly
        simple_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 3: Creating a Subscriber Node

Create a file named `simple_subscriber.py`:

```python
#!/usr/bin/env python3

"""
Simple Subscriber Node
This node subscribes to messages from a topic and logs them.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')
        
        # Create a subscription to String messages on the 'chatter' topic
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        
        # Prevent unused variable warning
        self.subscription  # for some Python linters
        
        # Log that the subscriber has been created
        self.get_logger().info('Simple Subscriber node initialized')

    def listener_callback(self, msg):
        """Callback function that executes when a message is received"""
        # Log the received message
        self.get_logger().info(f'Subscriber heard: "{msg.data}"')


def main(args=None):
    """Main function that initializes and runs the node"""
    rclpy.init(args=args)

    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        simple_subscriber.get_logger().info('Interrupted by user')
    finally:
        # Destroy the node explicitly
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 4: Update setup.py

Edit the `setup.py` file in the package root:

```python
from setuptools import find_packages, setup

package_name = 'basic_nodes_exercise'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Basic publisher and subscriber exercise for ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = basic_nodes_exercise.simple_publisher:main',
            'simple_subscriber = basic_nodes_exercise.simple_subscriber:main',
        ],
    },
)
```

## Step 5: Build and Run Your Nodes

Return to your workspace root and build the package:

```bash
# Navigate back to the workspace root
cd ~/ros2_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build the workspace
colcon build --packages-select basic_nodes_exercise

# Source the newly built package
source install/setup.bash

# In a new terminal, run the publisher:
ros2 run basic_nodes_exercise simple_publisher

# In another terminal, run the subscriber:
ros2 run basic_nodes_exercise simple_subscriber
```

## Step 6: Verify Communication

In a third terminal, you can use ROS 2 command-line tools to inspect the communication:

```bash
# List all topics
ros2 topic list

# Echo messages on the chatter topic
ros2 topic echo /chatter std_msgs/msg/String

# Check the node graph
ros2 run rqt_graph rqt_graph
```

## Exercise Challenges

Once you've successfully run the basic publisher and subscriber, try these challenges:

### Challenge 1: Modify Message Content
- Modify the publisher to send different types of messages (numbers, timestamps, etc.)
- Change the publishing rate to different intervals

### Challenge 2: Multiple Subscribers
- Run multiple instances of the subscriber node
- Observe how all subscribers receive the same published messages

### Challenge 3: Custom Message
- Create a custom message type instead of using String
- Update both publisher and subscriber to use your custom message

## Solution for Challenge 1

Here's a modified publisher that sends timestamped messages:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime


class TimestampedPublisher(Node):
    def __init__(self):
        super().__init__('timestamped_publisher')
        
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        
        # Publish every 2 seconds
        timer_period = 2.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0
        
        self.get_logger().info('Timestamped Publisher initialized')

    def timer_callback(self):
        msg = String()
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        msg.data = f'[{current_time}] Message #{self.counter}'
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    publisher = TimestampedPublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Interrupted by user')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Key Learning Points

This exercise taught you:

1. **Node Structure**: How to properly structure a ROS 2 node with initialization and destruction
2. **Publishers**: How to create and use publishers to send messages
3. **Subscribers**: How to create and use subscribers to receive messages
4. **Timers**: How to use timers to execute callbacks at regular intervals
5. **Message Types**: How to work with standard message types like String
6. **Building and Running**: How to build and run ROS 2 packages
7. **Debugging**: How to use command-line tools to inspect the system

## Conclusion

You have successfully created and tested a basic publisher-subscriber pair in ROS 2. This fundamental pattern is the building block for more complex robot applications. In the next exercise, you'll explore different communication patterns and message types.

Continue to experiment with these concepts, as they form the foundation for more advanced ROS 2 development. The publisher-subscriber paradigm is essential for decoupling components in a distributed robotic system, allowing for flexible and robust architectures.