---
sidebar_position: 11
---

# Exercise 2: Exploring ROS 2 Communication Patterns

## Objective

In this exercise, you will explore the three main communication patterns in ROS 2: topics (publish/subscribe), services (request/reply), and actions (goals, feedback, and results). You'll create and test each pattern to understand when to use each one.

## Prerequisites

Before starting this exercise, ensure you have:
- Completed Exercise 1 (building basic nodes)
- ROS 2 Iron installed
- Basic Python programming knowledge
- Understanding of ROS 2 concepts

## Step 1: Service-Based Communication

Let's start by creating a service server and client. Services are ideal for request-reply interactions.

Create `math_service_server.py`:

```python
#!/usr/bin/env python3

"""
Math Service Server
This node provides mathematical operations as a service.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MathServiceServer(Node):
    def __init__(self):
        super().__init__('math_service_server')
        
        # Create a service
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )
        
        self.get_logger().info('Math Service Server initialized')

    def add_two_ints_callback(self, request, response):
        """Callback function for handling service requests"""
        # Perform the addition
        response.sum = request.a + request.b
        
        # Log the operation
        self.get_logger().info(
            f'Request: {request.a} + {request.b} = {response.sum}'
        )
        
        return response


def main(args=None):
    rclpy.init(args=args)
    service_server = MathServiceServer()

    try:
        rclpy.spin(service_server)
    except KeyboardInterrupt:
        service_server.get_logger().info('Interrupted by user')
    finally:
        service_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Create `math_service_client.py`:

```python
#!/usr/bin/env python3

"""
Math Service Client
This node calls the math service to perform additions.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MathServiceClient(Node):
    def __init__(self):
        super().__init__('math_service_client')
        
        # Create a client for the service
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.get_logger().info('Math Service Client initialized')

    def send_request(self, a, b):
        """Send a request to the service and return the future"""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        return self.client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    
    # Get command line arguments
    if len(sys.argv) != 3:
        print('Usage: python3 math_service_client.py <a> <b>')
        return 1
    
    try:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    except ValueError:
        print('Arguments must be integers')
        return 1
    
    client = MathServiceClient()
    
    # Send the request
    future = client.send_request(a, b)
    
    try:
        # Wait for the result
        rclpy.spin_until_future_complete(client, future)
        
        if future.result() is not None:
            result = future.result()
            print(f'Result of {a} + {b} = {result.sum}')
        else:
            print('Exception while calling service: %r' % future.exception())
            
    except KeyboardInterrupt:
        client.get_logger().info('Interrupted by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 2: Action-Based Communication

Now let's create an action server and client. Actions are ideal for long-running tasks with feedback.

Create `fibonacci_action_server.py`:

```python
#!/usr/bin/env python3

"""
Fibonacci Action Server
This node implements an action server that computes Fibonacci sequences.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        
        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )
        
        self.get_logger().info('Fibonacci Action Server initialized')

    def execute_callback(self, goal_handle):
        """Execute the goal and return the result"""
        self.get_logger().info('Executing goal...')
        
        # Initialize the Fibonacci sequence
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        # Publish initial feedback
        goal_handle.publish_feedback(feedback_msg)
        
        # Compute the Fibonacci sequence up to the requested order
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal was cancelled')
                goal_handle.canceled()
                return Fibonacci.Result()

            # Calculate the next Fibonacci number
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            
            # Log progress
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

        # Set success and return the result
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        
        self.get_logger().info(f'Result: {result.sequence}')
        return result


def main(args=None):
    rclpy.init(args=args)
    action_server = FibonacciActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Create `fibonacci_action_client.py`:

```python
#!/usr/bin/env python3

"""
Fibonacci Action Client
This node calls the Fibonacci action server.
"""

import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        
        # Create an action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        """Send a goal to the action server"""
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        # Create a goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        self.get_logger().info(f'Sending goal: order = {goal_msg.order}')
        
        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        # Add a callback to handle the result
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the response from the action server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected')
            return

        self.get_logger().info('Goal accepted')
        
        # Get the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        """Handle the final result from the action server"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        
        # Shutdown after receiving result
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    # Get command line argument for the order
    if len(sys.argv) != 2:
        print('Usage: python3 fibonacci_action_client.py <order>')
        return 1
    
    try:
        order = int(sys.argv[1])
    except ValueError:
        print('Order must be an integer')
        return 1
    
    action_client = FibonacciActionClient()
    
    # Send the goal
    action_client.send_goal(order)
    
    # Spin to process callbacks
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Interrupted by user')


if __name__ == '__main__':
    main()
```

## Step 3: Comparing Communication Patterns

Now let's create a comparison node that demonstrates the differences between these patterns.

Create `communication_comparison.py`:

```python
#!/usr/bin/env python3

"""
Communication Comparison Node
This node demonstrates the differences between topics, services, and actions.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci


class CommunicationComparison(Node):
    def __init__(self):
        super().__init__('communication_comparison')
        
        # Initialize different communication types
        self.init_topics()
        self.init_services()
        self.init_actions()
        
        # Create a timer to trigger different communications
        self.timer = self.create_timer(3.0, self.trigger_example)
        self.trigger_count = 0
        
        self.get_logger().info('Communication Comparison Node initialized')

    def init_topics(self):
        """Initialize topic publisher and subscriber"""
        # Publisher
        self.topic_publisher = self.create_publisher(
            String, 'comparison_topic', 10)
        
        # Subscriber
        self.topic_subscriber = self.create_subscription(
            String, 'comparison_topic', self.topic_callback, 10)
        
        self.get_logger().info('Topics initialized')

    def init_services(self):
        """Initialize service client"""
        self.service_client = self.create_client(
            AddTwoInts, 'comparison_add_service')
        
        # Service server (for completeness in this example)
        self.service_server = self.create_service(
            AddTwoInts, 'comparison_add_service', self.service_callback)
        
        self.get_logger().info('Services initialized')

    def init_actions(self):
        """Initialize action client"""
        self.action_client = ActionClient(
            self, Fibonacci, 'comparison_fibonacci')
        
        self.get_logger().info('Actions initialized')

    def topic_callback(self, msg):
        """Handle received topic messages"""
        self.get_logger().info(f'Topic received: {msg.data}')

    def service_callback(self, request, response):
        """Handle service requests"""
        response.sum = request.a + request.b
        self.get_logger().info(f'Service request: {request.a} + {request.b} = {response.sum}')
        return response

    def trigger_example(self):
        """Trigger examples of different communication patterns"""
        self.trigger_count += 1
        self.get_logger().info(f'\n=== Trigger #{self.trigger_count} ===')
        
        if self.trigger_count == 1:
            # Topic example
            self.demo_topic()
        elif self.trigger_count == 2:
            # Service example
            self.demo_service()
        elif self.trigger_count == 3:
            # Action example
            self.demo_action()
        else:
            # Reset counter
            self.trigger_count = 0

    def demo_topic(self):
        """Demonstrate topic communication"""
        self.get_logger().info('Demonstrating TOPIC communication...')
        msg = String()
        msg.data = f'Topic message #{self.get_clock().now().nanoseconds}'
        self.topic_publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

    def demo_service(self):
        """Demonstrate service communication"""
        self.get_logger().info('Demonstrating SERVICE communication...')
        
        # Wait for service to be available
        if not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available')
            return
        
        # Create and send request
        request = AddTwoInts.Request()
        request.a = self.trigger_count
        request.b = self.trigger_count * 2
        
        # Call service synchronously
        future = self.service_client.call_async(request)
        
        # For demonstration purposes, we'll create a simple callback
        future.add_done_callback(lambda f: self.service_result_callback(f, request))

    def service_result_callback(self, future, request):
        """Handle service result"""
        try:
            result = future.result()
            self.get_logger().info(
                f'Service result: {request.a} + {request.b} = {result.sum}'
            )
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def demo_action(self):
        """Demonstrate action communication"""
        self.get_logger().info('Demonstrating ACTION communication...')
        
        # Wait for action server to be available
        if not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action server not available')
            return
        
        # Create and send goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = 5  # Use small order for quick execution
        
        self.get_logger().info(f'Sending Fibonacci goal: order = {goal_msg.order}')
        
        # Send goal asynchronously
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.action_feedback_callback)
        
        send_goal_future.add_done_callback(self.action_goal_response_callback)

    def action_goal_response_callback(self, future):
        """Handle action goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Action goal was rejected')
            return

        self.get_logger().info('Action goal accepted, getting result...')
        
        # Get result asynchronously
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.action_result_callback)

    def action_feedback_callback(self, feedback_msg):
        """Handle action feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Action feedback: {feedback.sequence}')

    def action_result_callback(self, future):
        """Handle action result"""
        result = future.result().result
        self.get_logger().info(f'Action result: {result.sequence}')


def main(args=None):
    rclpy.init(args=args)
    node = CommunicationComparison()
    
    # Use multi-threaded executor to handle multiple callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 4: Package Configuration

Update your `setup.py` to include the new executables:

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
    description='Communication patterns in ROS 2 exercise',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = basic_nodes_exercise.simple_publisher:main',
            'simple_subscriber = basic_nodes_exercise.simple_subscriber:main',
            'math_service_server = basic_nodes_exercise.math_service_server:main',
            'math_service_client = basic_nodes_exercise.math_service_client:main',
            'fibonacci_action_server = basic_nodes_exercise.fibonacci_action_server:main',
            'fibonacci_action_client = basic_nodes_exercise.fibonacci_action_client:main',
            'communication_comparison = basic_nodes_exercise.communication_comparison:main',
        ],
    },
)
```

## Step 5: Build and Run the Examples

Build your package:

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select basic_nodes_exercise

# Source the setup files
source install/setup.bash
```

Now run each communication pattern separately, starting with the servers:

**Terminal 1 - Service Server:**
```bash
ros2 run basic_nodes_exercise math_service_server
```

**Terminal 2 - Service Client:**
```bash
ros2 run basic_nodes_exercise math_service_client 10 20
```

**Terminal 1 - Action Server:**
```bash
ros2 run basic_nodes_exercise fibonacci_action_server
```

**Terminal 2 - Action Client:**
```bash
ros2 run basic_nodes_exercise fibonacci_action_client 10
```

**Terminal 1 - Comparison Node:**
```bash
ros2 run basic_nodes_exercise communication_comparison
```

## Communication Patterns Comparison

### Topics (Publish/Subscribe)
- **Characteristics**: Asynchronous, many-to-many, one-way communication
- **Use Case**: Continuous data streams, sensor data, state information
- **Example**: Camera images, laser scans, robot pose
- **Advantages**: Loosely coupled, real-time, multiple subscribers
- **Disadvantages**: No guaranteed delivery, no acknowledgment

### Services (Request/Reply)
- **Characteristics**: Synchronous, one-to-one, bi-directional
- **Use Case**: Configuration requests, simple computations, state queries
- **Example**: Map saving, robot reset, parameter retrieval
- **Advantages**: Guaranteed response, synchronous operation
- **Disadvantages**: Blocking, not suitable for long operations

### Actions (Goals, Feedback, Result)
- **Characteristics**: Asynchronous, bi-directional, long-running operations
- **Use Case**: Navigation, manipulation, complex tasks with progress
- **Example**: Moving to a pose, grabbing an object, executing a trajectory
- **Advantages**: Progress feedback, cancellations, long-running tasks
- **Disadvantages**: More complex than topics/services

## Exercise Challenges

### Challenge 1: Create a Navigation Action
- Implement an action that simulates navigation to a goal pose
- Provide feedback on distance to goal and estimated time
- Return success/failure result

### Challenge 2: Service with Complex Types
- Create a service that accepts a complex message with multiple fields
- Implement both the server and client for this service

### Challenge 3: Compare Performance
- Measure the latency of each communication pattern
- Test with different message sizes
- Document findings about when to use each pattern

## Solution for Challenge 1

Here's an example of a navigation action:

```python
#!/usr/bin/env python3

"""
Navigation Action Server
Simulates navigation to a goal pose.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Pose
from nav_msgs.action import NavigateToPose  # You might need to create this


class DummyNavigateToPose:
    """Mock class to simulate NavigateToPose action"""
    class Goal:
        def __init__(self):
            self.pose = Pose()
    
    class Feedback:
        def __init__(self):
            self.distance_remaining = 0.0
            self.estimated_time_remaining = 0.0
    
    class Result:
        def __init__(self):
            self.result = True  # Success/Fail


class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        
        # Create an action server
        # Note: In a real scenario, you'd use the actual NavigateToPose action
        # For this example, we're using our dummy class
        self._action_server = ActionServer(
            self,
            DummyNavigateToPose,
            'navigate_to_pose',
            self.execute_callback
        )
        
        self.get_logger().info('Navigation Action Server initialized')

    def execute_callback(self, goal_handle):
        """Execute navigation goal."""
        self.get_logger().info('Executing navigation goal...')
        
        # Initialize feedback
        feedback_msg = DummyNavigateToPose.Feedback()
        feedback_msg.distance_remaining = 10.0  # Simulated distance
        feedback_msg.estimated_time_remaining = 60.0  # 60 seconds estimated
        
        # Simulate navigation progress
        for i in range(10):  # 10 steps simulation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = DummyNavigateToPose.Result()
                result.result = False
                return result

            # Update feedback
            feedback_msg.distance_remaining = 10.0 - (i * 1.0)
            feedback_msg.estimated_time_remaining = 60.0 - (i * 6.0)
            
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(
                f'Navigation progress: {10-feedback_msg.distance_remaining}/10m '
                f'Distance remaining: {feedback_msg.distance_remaining:.1f}m'
            )
            
            # Simulate navigation time
            time.sleep(0.5)

        # Complete successfully
        goal_handle.succeed()
        result = DummyNavigateToPose.Result()
        result.result = True
        
        self.get_logger().info('Navigation completed successfully')
        return result


def main(args=None):
    rclpy.init(args=args)
    action_server = NavigationActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Key Learning Points

This exercise taught you:

1. **Service Communication**: Request-reply pattern for synchronous operations
2. **Action Communication**: Goal-based pattern for long-running operations with feedback
3. **Pattern Selection**: When to use topics, services, or actions based on requirements
4. **Implementation Details**: How to create servers and clients for each pattern
5. **Asynchronous Programming**: Handling callbacks and futures in ROS 2

## Summary

In this exercise, you've explored all three main communication patterns in ROS 2:
- **Topics**: For continuous, asynchronous data streaming
- **Services**: For synchronous request-reply operations
- **Actions**: For long-running operations with feedback and cancellation

Understanding when to use each pattern is crucial for effective ROS 2 system design. As you advance in your ROS 2 journey, you'll find that well-chosen communication patterns make your systems more robust, efficient, and maintainable.

The next exercise will focus on robot description and URDF, followed by multi-node system integration.