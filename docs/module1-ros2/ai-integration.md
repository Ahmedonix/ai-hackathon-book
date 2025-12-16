---
sidebar_position: 8
---

# Integration of AI Agents with ROS 2 Nodes

## Overview

Integrating AI agents with ROS 2 nodes enables intelligent robot behavior by combining the robust communication infrastructure of ROS 2 with the decision-making capabilities of artificial intelligence. This section explores various approaches for connecting AI algorithms to ROS 2 systems, from simple rule-based agents to complex machine learning models.

## Architecture Patterns for AI Integration

### 1. Centralized AI Node Pattern

In this pattern, a dedicated AI node processes sensor data and generates actions:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import numpy as np
import math


class CentralizedAIAgent(Node):
    def __init__(self):
        super().__init__('centralized_ai_agent')
        
        # Subscribers for various sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        
        # Publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher for status updates
        self.status_pub = self.create_publisher(String, '/ai_status', 10)
        
        # Store sensor data
        self.laser_data = None
        self.camera_data = None
        self.joint_data = None
        
        # AI decision timer
        self.ai_timer = self.create_timer(0.1, self.make_decision)  # 10Hz
        
        self.get_logger().info('Centralized AI Agent initialized')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = np.array(msg.ranges)
        # Filter out invalid readings (inf, nan)
        self.laser_data = self.laser_data[~np.isinf(self.laser_data)]
        self.laser_data = self.laser_data[~np.isnan(self.laser_data)]
    
    def camera_callback(self, msg):
        """Process camera image data (simplified)"""
        # In a real implementation, you'd convert ROS image to numpy array
        # For this example, we'll just store dimensions
        self.camera_data = (msg.height, msg.width)
    
    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_data = {
            name: pos for name, pos in zip(msg.name, msg.position)
        }
    
    def make_decision(self):
        """Main AI decision-making function"""
        if self.laser_data is not None:
            # Simple navigation logic based on laser data
            cmd_vel = Twist()
            
            # Find minimum distance in front of robot (forward sector)
            front_scan = self.laser_data[len(self.laser_data)//2-50:len(self.laser_data)//2+50]
            if len(front_scan) > 0:
                min_front_dist = np.min(front_scan)
                
                if min_front_dist < 0.5:  # Obstacle too close
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = 0.5  # Turn right
                    status = "Avoiding obstacle"
                else:
                    cmd_vel.linear.x = 0.3
                    cmd_vel.angular.z = 0.0
                    status = "Moving forward"
            
            # Publish command and status
            self.cmd_vel_pub.publish(cmd_vel)
            status_msg = String()
            status_msg.data = status
            self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    ai_agent = CentralizedAIAgent()
    
    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Distributed AI Pattern

In this pattern, multiple specialized AI nodes handle different aspects of robot intelligence:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float64
import numpy as np


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Subscribe to sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Publish processed data
        self.obstacle_pub = self.create_publisher(Point, '/nearest_obstacle', 10)
        self.free_space_pub = self.create_publisher(Point, '/nearest_free_space', 10)
        
        self.get_logger().info('Perception node initialized')
    
    def laser_callback(self, msg):
        """Process laser scan to identify obstacles and free space"""
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Find valid readings (not inf or nan)
        valid_idx = np.isfinite(ranges)
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]
        
        if len(valid_ranges) == 0:
            return
        
        # Find nearest obstacle
        nearest_idx = np.argmin(valid_ranges)
        nearest_range = valid_ranges[nearest_idx]
        nearest_angle = valid_angles[nearest_idx]
        
        obstacle_point = Point()
        obstacle_point.x = nearest_range * math.cos(nearest_angle)
        obstacle_point.y = nearest_range * math.sin(nearest_angle)
        obstacle_point.z = 0.0
        self.obstacle_pub.publish(obstacle_point)


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        
        # Subscribe to processed sensor data
        self.obstacle_sub = self.create_subscription(
            Point, '/nearest_obstacle', self.obstacle_callback, 10)
        self.goal_sub = self.create_subscription(
            Point, '/navigation_goal', self.goal_callback, 10)
        
        # Publish navigation plan
        self.plan_pub = self.create_publisher(Twist, '/plan_cmd_vel', 10)
        
        # Current state
        self.nearest_obstacle = None
        self.navigation_goal = Point(x=10.0, y=0.0, z=0.0)  # Default goal
        self.current_plan = Twist()
        
        self.get_logger().info('Planning node initialized')
    
    def obstacle_callback(self, msg):
        """Process obstacle information"""
        self.nearest_obstacle = msg
        self.update_plan()
    
    def goal_callback(self, msg):
        """Process navigation goal"""
        self.navigation_goal = msg
        self.update_plan()
    
    def update_plan(self):
        """Update navigation plan based on current information"""
        if self.nearest_obstacle is None:
            return
            
        cmd_vel = Twist()
        
        # Simple obstacle avoidance and goal-seeking behavior
        obstacle_distance = math.sqrt(
            self.nearest_obstacle.x**2 + self.nearest_obstacle.y**2
        )
        
        if obstacle_distance < 0.5:  # Too close to obstacle
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5 if self.nearest_obstacle.y > 0 else -0.5
            self.get_logger().info(f'Avoiding obstacle at ({self.nearest_obstacle.x:.2f}, {self.nearest_obstacle.y:.2f})')
        else:
            # Head toward goal
            goal_distance = math.sqrt(
                self.navigation_goal.x**2 + self.navigation_goal.y**2
            )
            
            if goal_distance > 0.1:  # Not at goal yet
                cmd_vel.linear.x = 0.3
                cmd_vel.angular.z = 0.2 if self.navigation_goal.y > 0 else -0.2
            else:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
        
        self.current_plan = cmd_vel
        self.plan_pub.publish(cmd_vel)


class ExecutionNode(Node):
    def __init__(self):
        super().__init__('execution_node')
        
        # Subscribe to plan
        self.plan_sub = self.create_subscription(
            Twist, '/plan_cmd_vel', self.plan_callback, 10)
        
        # Publish to robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('Execution node initialized')
    
    def plan_callback(self, msg):
        """Execute the received plan"""
        # In a real system, you might add safety checks here
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    perception = PerceptionNode()
    planning = PlanningNode()
    execution = ExecutionNode()
    
    try:
        # Run all nodes
        rclpy.spin_until_future_complete(
            perception, 
            rclpy.executors.MultiThreadedExecutor()
        )
    except KeyboardInterrupt:
        pass
    finally:
        perception.destroy_node()
        planning.destroy_node()
        execution.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Machine Learning Integration

### 1. TensorFlow/PyTorch Model Integration

AI models can be integrated directly into ROS 2 nodes:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import cv2  # For image processing
from cv_bridge import CvBridge
import tensorflow as tf  # Example with TensorFlow


class MLNavigationNode(Node):
    def __init__(self):
        super().__init__('ml_navigation_node')
        
        # Initialize CV bridge for image processing
        self.bridge = CvBridge()
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ml_status', 10)
        
        # Initialize ML model
        self.load_model()
        
        # Store data for inference
        self.current_image = None
        self.current_laser = None
        
        # Timer for ML inference
        self.inference_timer = self.create_timer(0.2, self.run_inference)
        
        self.get_logger().info('ML Navigation Node initialized')
    
    def load_model(self):
        """Load pre-trained ML model for navigation"""
        try:
            # Load a pre-trained model (simplified example)
            # In practice, you'd load your trained model here
            # self.model = tf.keras.models.load_model('/path/to/your/model')
            
            # For this example, we'll create a dummy model
            self.model_loaded = True
            self.get_logger().info('ML model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            self.model_loaded = False
    
    def image_callback(self, msg):
        """Process camera image"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Resize image to model input size (example: 224x224)
            cv_image = cv2.resize(cv_image, (224, 224))
            
            # Normalize pixel values
            cv_image = cv_image.astype(np.float32) / 255.0
            
            # Store for inference
            self.current_image = np.expand_dims(cv_image, axis=0)  # Add batch dimension
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def laser_callback(self, msg):
        """Process laser scan data"""
        try:
            # Convert laser scan to fixed-size array for model input
            ranges = np.array(msg.ranges)
            
            # Handle infinite/nan values
            ranges[~np.isfinite(ranges)] = 10.0  # Set max range for invalid readings
            
            # Normalize to [0, 1] range (assuming max range is 10m)
            ranges = np.clip(ranges, 0, 10) / 10.0
            
            # Pad or truncate to fixed size (example: 360 points)
            if len(ranges) > 360:
                ranges = ranges[:360]
            elif len(ranges) < 360:
                ranges = np.pad(ranges, (0, 360 - len(ranges)), mode='edge')
            
            # Store for inference
            self.current_laser = np.expand_dims(ranges, axis=0)  # Add batch dimension
        except Exception as e:
            self.get_logger().error(f'Error processing laser scan: {e}')
    
    def run_inference(self):
        """Run ML inference to determine navigation action"""
        if not self.model_loaded or self.current_image is None or self.current_laser is None:
            return
        
        try:
            # In a real implementation, you would run inference:
            # model_input = np.concatenate([self.current_image, self.current_laser], axis=-1)
            # predictions = self.model.predict(model_input)
            
            # For this example, we'll simulate model output based on sensor data
            cmd_vel = self.simulate_model_output()
            
            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)
            
            # Publish status
            status_msg = String()
            status_msg.data = f'Moving: linear={cmd_vel.linear.x:.2f}, angular={cmd_vel.angular.z:.2f}'
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')
    
    def simulate_model_output(self):
        """Simulate model output based on sensor data"""
        cmd_vel = Twist()
        
        # Example: if front laser readings are low (obstacle detected), turn
        front_scan = self.current_laser[0][170:190]  # Front 20 readings
        if np.min(front_scan) < 0.3:  # Obstacle detected
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn right
        elif np.min(front_scan) < 0.7:  # Potential obstacle
            cmd_vel.linear.x = 0.2
            cmd_vel.angular.z = 0.1  # Slight turn
        else:  # Clear path
            cmd_vel.linear.x = 0.4
            cmd_vel.angular.z = 0.0
        
        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    ml_node = MLNavigationNode()
    
    try:
        rclpy.spin(ml_node)
    except KeyboardInterrupt:
        pass
    finally:
        ml_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Reinforcement Learning Integration

RL agents can be integrated for adaptive behavior:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import math


class RLNavigationAgent(Node):
    def __init__(self):
        super().__init__('rl_navigation_agent')
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # RL parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # State and action spaces (simplified)
        self.state = None
        self.previous_state = None
        self.action = None
        self.reward = 0.0
        
        # Q-table (simplified for example)
        # In practice, you'd use function approximation for large state spaces
        self.q_table = np.zeros((100, 4))  # 100 states, 4 actions
        
        # Robot position tracking
        self.current_x = 0.0
        self.current_y = 0.0
        self.target_x = 10.0  # Example target
        self.target_y = 0.0
        
        # RL timer
        self.rl_timer = self.create_timer(0.1, self.rl_step)
        
        self.get_logger().info('Reinforcement Learning Agent initialized')
    
    def laser_callback(self, msg):
        """Process laser scan to determine state"""
        # Simplified state representation: min distance in 3 sectors
        ranges = np.array(msg.ranges)
        ranges = ranges[np.isfinite(ranges)]  # Remove invalid readings
        
        if len(ranges) == 0:
            return
            
        # Divide scan into 3 sectors
        mid_idx = len(ranges) // 2
        front = np.min(ranges[mid_idx-20:mid_idx+20])
        left = np.min(ranges[:len(ranges)//3])
        right = np.min(ranges[2*len(ranges)//3:])
        
        # Discretize distances for state representation
        front_state = min(int(front * 10), 9)  # 0-9 for 0-1m to 0-9m
        left_state = min(int(left * 10), 9)
        right_state = min(int(right * 10), 9)
        
        # Simple state encoding
        self.state = front_state * 100 + left_state * 10 + right_state
    
    def odom_callback(self, msg):
        """Track robot position"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
    
    def discretize_action(self, action_idx):
        """Map action index to velocity commands"""
        actions = [
            (0.3, 0.0),   # Forward
            (0.1, 0.5),   # Turn right
            (0.1, -0.5),  # Turn left
            (0.0, 0.0),   # Stop
        ]
        return actions[action_idx]
    
    def calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0
        
        # Distance to target
        dist_to_target = math.sqrt(
            (self.current_x - self.target_x)**2 + 
            (self.current_y - self.target_y)**2
        )
        
        # Reward for getting closer to target
        reward += (10.0 - dist_to_target) / 10.0
        
        # Penalty for being too close to obstacles
        if self.state is not None:
            front_dist = (self.state // 100) / 10.0
            if front_dist < 0.3:
                reward -= 1.0  # Significant penalty for collision risk
        
        # Penalty for not moving
        # (This would need actual velocity data to be more accurate)
        
        return reward
    
    def select_action(self):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(4)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[self.state % 100])  # Simplified state access
    
    def update_q_table(self):
        """Update Q-table using Q-learning"""
        if self.previous_state is None or self.action is None:
            return
            
        # Calculate reward
        self.reward = self.calculate_reward()
        
        # Find best next action
        next_best_action = np.max(self.q_table[self.state % 100])
        
        # Update Q-value
        current_q = self.q_table[self.previous_state % 100][self.action]
        new_q = current_q + self.learning_rate * (
            self.reward + self.discount_factor * next_best_action - current_q
        )
        self.q_table[self.previous_state % 100][self.action] = new_q
    
    def rl_step(self):
        """Main RL step"""
        if self.state is None:
            return
        
        # Update Q-table if we have a previous state
        if self.previous_state is not None:
            self.update_q_table()
        
        # Select action
        self.action = self.select_action()
        
        # Execute action
        linear_vel, angular_vel = self.discretize_action(self.action)
        
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Update previous state
        self.previous_state = self.state


def main(args=None):
    rclpy.init(args=args)
    rl_agent = RLNavigationAgent()
    
    try:
        rclpy.spin(rl_agent)
    except KeyboardInterrupt:
        pass
    finally:
        rl_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Communication Patterns for AI Integration

### 1. Action-based AI Services

For complex AI tasks that take time to complete:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import NavigateToPose
from geometry_msgs.msg import Pose, Twist
import time
import math


class AINavigationActionServer(Node):
    def __init__(self):
        super().__init__('ai_navigation_action_server')
        
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'ai_navigate_to_pose',
            self.execute_callback
        )
        
        # Robot command publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('AI Navigation Action Server initialized')
    
    def execute_callback(self, goal_handle):
        """Execute navigation goal with AI planning"""
        self.get_logger().info('AI Navigation goal received')
        
        target_pose = goal_handle.request.pose
        feedback_msg = NavigateToPose.Feedback()
        
        # In a real implementation, this would call your AI planner
        # For this example, we'll simulate the navigation
        
        # Extract target coordinates
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Get current position (simplified)
        current_x, current_y = 0.0, 0.0  # In practice, get from odometry
        
        # Calculate distance
        total_distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        # Navigation simulation
        steps = int(total_distance / 0.1)  # 0.1m increments
        for i in range(steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.pose.position.x = current_x
                result.pose.position.y = current_y
                return result
            
            # Simulate AI planning and movement
            progress = float(i + 1) / float(steps)
            current_x = current_x + (target_x - current_x) * progress
            current_y = current_y + (target_y - current_y) * progress
            
            # Update feedback
            feedback_msg.current_pose.position.x = current_x
            feedback_msg.current_pose.position.y = current_y
            feedback_msg.distance_to_goal = total_distance * (1.0 - progress)
            
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate time for AI processing and movement
            time.sleep(0.1)
        
        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.pose = target_pose
        self.get_logger().info('AI Navigation completed successfully')
        
        return result


def main(args=None):
    rclpy.init(args=args)
    action_server = AINavigationActionServer()
    
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

## Best Practices for AI Integration

### 1. Data Pipeline Design

Design efficient data pipelines for AI models:

```python
import queue
import threading
from collections import deque

class AIDataPipeline:
    def __init__(self, max_buffer_size=10):
        self.sensor_buffer = deque(maxlen=max_buffer_size)
        self.ai_input_queue = queue.Queue(maxsize=max_buffer_size)
        self.ai_output_queue = queue.Queue(maxsize=max_buffer_size)
        
        # Threading for non-blocking processing
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def add_sensor_data(self, sensor_data):
        """Add sensor data to processing buffer"""
        self.sensor_buffer.append(sensor_data)
    
    def get_ai_command(self):
        """Get AI-generated command (non-blocking)"""
        try:
            return self.ai_output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_loop(self):
        """Background processing loop"""
        while True:
            if len(self.sensor_buffer) > 0:
                data = self.sensor_buffer.popleft()
                # Process with AI model
                ai_output = self.run_ai_model(data)
                if not self.ai_output_queue.full():
                    self.ai_output_queue.put(ai_output)
            time.sleep(0.01)  # Prevent busy waiting
```

### 2. Safety and Failsafe Mechanisms

Always implement safety mechanisms when integrating AI:

```python
class SafeAIIntegrationNode(Node):
    def __init__(self):
        super().__init__('safe_ai_integration')
        
        # Safety publisher
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        
        # Safety timer
        self.safety_timer = self.create_timer(0.05, self.safety_check)
        
        # Safe default command
        self.last_safe_cmd = Twist()
        
        self.get_logger().info('Safe AI Integration Node initialized')
    
    def safety_check(self):
        """Check for safety violations"""
        # Example: velocity limits
        if (abs(self.last_ai_cmd.linear.x) > 1.0 or 
            abs(self.last_ai_cmd.angular.z) > 1.0):
            self.emergency_stop()
            self.get_logger().error('Safety violation: velocity limits exceeded')
    
    def emergency_stop(self):
        """Emergency stop procedure"""
        # Send stop command
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        # Trigger emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)
```

## Performance Considerations

### 1. Computational Efficiency

Consider computational requirements when integrating AI:

```python
class EfficientAINode(Node):
    def __init__(self):
        super().__init__('efficient_ai_node')
        
        # Use lower frequency for heavy computations
        self.ai_timer = self.create_timer(0.5, self.limited_frequency_ai)  # 2Hz
        
        # Use higher frequency for light computations
        self.control_timer = self.create_timer(0.01, self.high_freq_control)  # 100Hz
    
    def limited_frequency_ai(self):
        """Run AI computations at lower frequency"""
        # Process sensor data with AI
        pass
    
    def high_freq_control(self):
        """Run control commands at high frequency"""
        # Execute latest AI command
        pass
```

## Summary

Integrating AI agents with ROS 2 nodes requires careful consideration of:

1. **Architecture**: Choose between centralized and distributed approaches
2. **Communication**: Use appropriate ROS 2 communication patterns (topics, services, actions)
3. **Safety**: Implement failsafes and emergency stops
4. **Performance**: Consider computational requirements
5. **Data Flow**: Design efficient data pipelines for AI processing

The integration of AI with ROS 2 opens up possibilities for sophisticated robot behaviors, from simple rule-based systems to complex learning algorithms. With proper design, AI agents can leverage the robust communication infrastructure of ROS 2 while providing intelligent decision-making capabilities.

The next section will provide hands-on exercises to practice implementing AI integration with ROS 2 nodes.