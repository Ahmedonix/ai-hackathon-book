# Building AI Pipelines for Humanoid Robotics

## Overview

AI pipelines in humanoid robotics integrate perception, decision-making, and action execution to enable autonomous behaviors. In this section, we'll explore how to build comprehensive AI pipelines that connect perception systems (from Isaac ROS) to navigation and control systems, creating intelligent humanoid behaviors.

## Understanding AI Pipeline Architecture

### 1. Core Components of Humanoid AI Pipelines

AI pipelines for humanoid robots typically consist of:

- **Perception Module**: Processes sensor data to understand the environment
- **World Modeling**: Creates internal representations of the environment
- **Planning**: Generates action sequences to achieve goals
- **Control**: Executes low-level commands to the robot
- **Learning**: Adapts behaviors based on experience
- **Execution Monitoring**: Tracks the success of planned actions

### 2. Pipeline Integration Points

The AI pipeline connects to various systems:

```
Sensors (LiDAR, Camera, IMU) 
     ↓
Perception → World Model → Planning → Control → Robot Actuators
     ↑                                         ↓
Simulation/Reality ←─────────────────────────────
```

## Building Perception-to-Action Pipelines

### 1. Perception Pipeline Integration

First, we'll create a node that integrates perception data from Isaac ROS with higher-level AI components:

```python
# scripts/perception_integrator.py
#!/usr/bin/env python3

"""
Perception Integrator Node for Humanoid Robotics AI Pipeline.
Combines Isaac ROS perception outputs with higher-level reasoning components.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from std_msgs.msg import String, Float64MultiArray
from vision_msgs.msg import Detection2DArray, Detection2D
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np
import cv2
from cv_bridge import CvBridge


class PerceptionIntegrator(Node):
    def __init__(self):
        super().__init__('perception_integrator')
        
        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('lidar_topic', '/scan')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('update_rate', 10.0)  # Hz
        
        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.image_topic = self.get_parameter('image_topic').value
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # Initialize internal state
        self.latest_image = None
        self.latest_lidar = None
        self.latest_imu = None
        self.latest_detections = None
        self.world_model = WorldModel()  # Defined later
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.lidar_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detection_topic,
            self.detection_callback,
            10
        )
        
        # Publishers for higher-level AI components
        self.semantic_map_pub = self.create_publisher(
            String,
            '/semantic_map',
            10
        )
        
        self.world_state_pub = self.create_publisher(
            String,
            '/world_state',
            10
        )
        
        self.ai_goals_pub = self.create_publisher(
            PoseStamped,
            '/ai_goals',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/perception_pipeline_status',
            10
        )
        
        # TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for processing and publishing integrated data
        self.processing_timer = self.create_timer(1.0/self.update_rate, self.process_and_publish)
        
        self.get_logger().info('Perception Integrator Node Initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        self.latest_image = msg
        # For performance, we'll process this in the timer callback

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.latest_lidar = msg

    def imu_callback(self, msg):
        """Process IMU data for orientation and balance"""
        self.latest_imu = msg

    def detection_callback(self, msg):
        """Process object detections"""
        self.latest_detections = msg

    def process_and_publish(self):
        """Process all perception inputs and publish integrated state"""
        try:
            # Update world model with latest sensor data
            if self.latest_image:
                self.process_visual_data(self.latest_image)
            
            if self.latest_lidar:
                self.process_lidar_data(self.latest_lidar)
            
            if self.latest_detections:
                self.process_detection_data(self.latest_detections)
            
            if self.latest_imu:
                self.process_imu_data(self.latest_imu)
            
            # Update world model
            self.world_model.update()
            
            # Generate semantic map from processed data
            semantic_map = self.generate_semantic_map()
            self.publish_semantic_map(semantic_map)
            
            # Generate world state summary
            world_state = self.generate_world_state()
            self.publish_world_state(world_state)
            
            # Identify potential goals based on perception
            goals = self.identify_goals_from_perception()
            if goals:
                for goal in goals:
                    self.ai_goals_pub.publish(goal)
            
            # Publish status
            status = String()
            status.data = f"Perception updated: Img={self.latest_image is not None}, " \
                         f"Lidar={self.latest_lidar is not None}, " \
                         f"Detections={len(self.latest_detections.detections) if self.latest_detections else 0}"
            self.status_pub.publish(status)
            
        except Exception as e:
            self.get_logger().error(f'Error in perception processing: {str(e)}')

    def process_visual_data(self, image_msg):
        """Process camera image for visual perception"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # This is where Isaac ROS perception would typically process
            # the image, e.g., for object detection, segmentation, etc.
            
            # For demonstration, we'll extract some basic features
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges (potential obstacles or features)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours (potential obstacles/objects)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert to world coordinates using camera intrinsics and extrinsics
                    # In a real implementation: use Isaac ROS stereo/depth data
                    # to get 3D position from 2D image coordinates
                    self.world_model.add_visual_feature([x, y, w, h, area])
                    
        except Exception as e:
            self.get_logger().error(f'Error processing visual data: {str(e)}')

    def process_lidar_data(self, lidar_msg):
        """Process LiDAR data for obstacle detection and mapping"""
        try:
            # Process LiDAR ranges
            ranges = np.array(lidar_msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]
            
            if len(valid_ranges) == 0:
                return
            
            # Calculate statistics
            min_range = np.min(valid_ranges)
            max_range = np.max(valid_ranges)
            mean_range = np.mean(valid_ranges)
            
            # Detect obstacles in specific directions
            angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(lidar_msg.ranges))
            
            # Front obstacle detection (forward 30 degrees) 
            front_idx = slice(
                int(len(angles)/2 - len(angles)*0.125),  # -30 degrees (15% of 360)
                int(len(angles)/2 + len(angles)*0.125)   # +30 degrees (15% of 360)
            )
            
            front_ranges = np.array(lidar_msg.ranges)[front_idx]
            front_valid = front_ranges[np.isfinite(front_ranges)]
            
            if len(front_valid) > 0:
                closest_front = np.min(front_valid)
                
                # If obstacle is close, mark it in world model
                if closest_front < 1.0:  # 1 meter threshold
                    self.world_model.add_obstacle(
                        position=self.calculate_relative_position(closest_front, angles[front_idx][np.argmin(front_valid)]),
                        distance=closest_front,
                        type="obstacle"
                    )
            
            # Add LiDAR data to world model
            self.world_model.add_lidar_data({
                'timestamp': lidar_msg.header.stamp,
                'min_range': float(min_range),
                'max_range': float(max_range),
                'mean_range': float(mean_range),
                'front_clear': closest_front > 1.0 if 'closest_front' in locals() else True
            })
            
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {str(e)}')

    def process_detection_data(self, detection_msg):
        """Process object detection results"""
        try:
            for detection in detection_msg.detections:
                # Process each detection
                if detection.results:
                    # Get the most confident result
                    best_result = max(detection.results, key=lambda x: x.hypothesis.score)
                    
                    # In a real implementation, we'd transform this from camera frame
                    # to robot/base frame to get the position in the world
                    # For now, we'll add it as-is
                    self.world_model.add_detected_object({
                        'class': best_result.hypothesis.class_id,
                        'confidence': best_result.hypothesis.score,
                        'bbox': detection.bbox,
                        'timestamp': detection_msg.header.stamp
                    })
                    
        except Exception as e:
            self.get_logger().error(f'Error processing detection data: {str(e)}')

    def process_imu_data(self, imu_msg):
        """Process IMU data for orientation and balance"""
        try:
            # Extract orientation
            orientation = imu_msg.orientation
            roll, pitch, yaw = self.quaternion_to_euler(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            
            # Extract angular velocity
            ang_vel = imu_msg.angular_velocity
            
            # Extract linear acceleration
            lin_acc = imu_msg.linear_acceleration
            
            # Add to world model
            self.world_model.update_robot_state({
                'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'angular_velocity': {'x': ang_vel.x, 'y': ang_vel.y, 'z': ang_vel.z},
                'linear_acceleration': {'x': lin_acc.x, 'y': lin_acc.y, 'z': lin_acc.z},
                'timestamp': imu_msg.header.stamp
            })
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {str(e)}')

    def calculate_relative_position(self, range_val, angle_val):
        """Calculate relative position from range and angle"""
        x = range_val * np.cos(angle_val)
        y = range_val * np.sin(angle_val)
        return [x, y, 0.0]  # Assume z=0 for ground-level obstacles

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def generate_semantic_map(self):
        """Generate semantic map from perception data"""
        # In a real implementation, this would create a detailed semantic map
        # Here we'll return a simplified representation
        semantic_map = {
            'timestamp': self.get_clock().now().to_msg(),
            'obstacles': self.world_model.get_obstacles(),
            'objects': self.world_model.get_detected_objects(),
            'free_spaces': self.world_model.get_free_spaces()
        }
        return semantic_map

    def generate_world_state(self):
        """Generate world state description"""
        # Create a textual description of the current world state
        world_state_parts = []
        
        obstacles = self.world_model.get_obstacles()
        if obstacles:
            world_state_parts.append(f"Obstacles: {len(obstacles)} detected")
        
        objects = self.world_model.get_detected_objects()
        if objects:
            object_classes = [obj['class'] for obj in objects]
            unique_classes = set(object_classes)
            world_state_parts.append(f"Objects: {', '.join(unique_classes)}")
        
        robot_state = self.world_model.get_robot_state()
        if robot_state:
            pitch = robot_state['orientation']['pitch']
            if abs(pitch) > 0.3:  # Significant pitch angle
                world_state_parts.append(f"Robot tilted: {pitch:.2f} rad")
        
        if not world_state_parts:
            world_state_parts.append("World state: Clear")
        
        return " | ".join(world_state_parts)

    def identify_goals_from_perception(self):
        """Identify potential navigation goals from perception data"""
        goals = []
        
        # Example: If a specific object class is detected, set it as a goal
        objects = self.world_model.get_detected_objects()
        
        for obj in objects:
            if obj['class'] == 'person' and obj['confidence'] > 0.7:
                # Create a goal to navigate toward the person
                goal = PoseStamped()
                goal.header.frame_id = 'map'  # We'd need proper TF to convert this
                goal.header.stamp = self.get_clock().now().to_msg()
                
                # In a real implementation: transform object position to map frame
                # For now, we'll set a dummy position
                goal.pose.position.x = 1.0
                goal.pose.position.y = 1.0
                goal.pose.position.z = 0.0
                
                goals.append(goal)
        
        return goals

    def publish_semantic_map(self, semantic_map):
        """Publish semantic map"""
        # For simplicity, we'll publish as a string
        # In real implementation, use a proper semantic map message
        map_str = String()
        map_str.data = f"Obstacles: {len(semantic_map['obstacles'])}, " \
                      f"Objects: {len(semantic_map['objects'])}, " \
                      f"Free spaces: {len(semantic_map['free_spaces'])}"
        self.semantic_map_pub.publish(map_str)

    def publish_world_state(self, world_state_str):
        """Publish world state"""
        state_msg = String()
        state_msg.data = world_state_str
        self.world_state_pub.publish(state_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Perception Integrator Node Shutting Down')
        super().destroy_node()


class WorldModel:
    """
    Simple world model to maintain state about the environment.
    In a real implementation, this would be much more sophisticated.
    """
    
    def __init__(self):
        self.obstacles = []
        self.detected_objects = []
        self.visual_features = []
        self.lidar_data = []
        self.robot_state = {}
        self.free_spaces = []
        
        # Time windows for temporal reasoning
        self.temporal_window = 5.0  # seconds
        
    def update(self):
        """Update the world model based on temporal constraints"""
        current_time = time.time()
        
        # Remove old data that's outside our temporal window
        self.obstacles = [obs for obs in self.obstacles 
                         if (current_time - obs['timestamp']) < self.temporal_window]
        self.detected_objects = [obj for obj in self.detected_objects 
                                if (current_time - obj['timestamp'].nanoseconds/1e9) < self.temporal_window]
    
    def add_obstacle(self, position, distance, type="obstacle"):
        """Add an obstacle to the world model"""
        obstacle = {
            'position': position,
            'distance': distance,
            'type': type,
            'timestamp': time.time()
        }
        self.obstacles.append(obstacle)
    
    def add_detected_object(self, obj_dict):
        """Add a detected object to the world model"""
        obj_dict['timestamp'] = time.time()
        self.detected_objects.append(obj_dict)
    
    def add_visual_feature(self, feature_data):
        """Add a visual feature to the world model"""
        self.visual_features.append({
            'data': feature_data,
            'timestamp': time.time()
        })
    
    def add_lidar_data(self, lidar_dict):
        """Add LiDAR data to the world model"""
        self.lidar_data.append(lidar_dict)
    
    def update_robot_state(self, state_dict):
        """Update the robot's state in the world model"""
        self.robot_state.update(state_dict)
    
    def get_obstacles(self):
        """Get current obstacles"""
        return self.obstacles
    
    def get_detected_objects(self):
        """Get currently detected objects"""
        return self.detected_objects
    
    def get_robot_state(self):
        """Get current robot state"""
        return self.robot_state
    
    def get_free_spaces(self):
        """Get estimated free spaces (simplified)"""
        return self.free_spaces


def main(args=None):
    rclpy.init(args=args)
    
    integrator = PerceptionIntegrator()
    
    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        integrator.get_logger().info('Node interrupted by user')
    finally:
        integrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Creating the Planning Pipeline

### 1. AI Planning Node

Create a planning node that takes the integrated perception data and generates plans:

```python
# scripts/ai_planner.py
#!/usr/bin/env python3

"""
AI Planning Node for Humanoid Robotics.
Creates action plans based on perception data and goals.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from builtin_interfaces.msg import Duration
import json
import time
from enum import Enum


class PlanningState(Enum):
    IDLE = 1
    RECEIVING_GOALS = 2
    PLANNING = 3
    EXECUTING = 4
    RECOVERY = 5


class AIPlannerNode(Node):
    def __init__(self):
        super().__init__('ai_planner_node')
        
        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('planning_frequency', 5.0)  # Hz
        self.declare_parameter('max_plan_length', 100)     # Waypoints
        self.declare_parameter('replanning_threshold', 0.5)  # meters
        
        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.planning_frequency = self.get_parameter('planning_frequency').value
        self.max_plan_length = self.get_parameter('max_plan_length').value
        self.replanning_threshold = self.get_parameter('replanning_threshold').value
        
        # Internal state
        self.current_state = PlanningState.IDLE
        self.current_goal = None
        self.current_plan = None
        self.world_state = {}
        self.last_replan_time = 0
        self.active_plan_timestamp = None
        
        # Subscriptions
        self.world_state_sub = self.create_subscription(
            String,
            '/world_state',
            self.world_state_callback,
            10
        )
        
        self.semantic_map_sub = self.create_subscription(
            String,
            '/semantic_map',
            self.semantic_map_callback,
            10
        )
        
        self.ai_goals_sub = self.create_subscription(
            PoseStamped,
            '/ai_goals',
            self.ai_goal_callback,
            10
        )
        
        self.current_pose_sub = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.current_pose_callback,
            10
        )
        
        # Publishers
        self.plan_publisher = self.create_publisher(
            Path,
            '/ai_plan',
            10
        )
        
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        self.plan_status_publisher = self.create_publisher(
            String,
            '/plan_status',
            10
        )
        
        # Planning timer
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.plan_cycle)
        
        self.get_logger().info('AI Planner Node Initialized')

    def world_state_callback(self, msg):
        """Update world state from perception integrator"""
        try:
            # Parse world state from string representation
            self.world_state = self.parse_world_state(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error parsing world state: {str(e)}')

    def semantic_map_callback(self, msg):
        """Update semantic map information"""
        # In a real implementation, this would parse a semantic map message
        # For now, we'll just store the string
        self.last_semantic_map = msg.data

    def ai_goal_callback(self, msg):
        """Process AI goals from perception system"""
        self.current_goal = msg
        self.current_state = PlanningState.PLANNING
        self.get_logger().info(f'New AI goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def current_pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg

    def parse_world_state(self, world_state_str):
        """Parse the world state string into structured data"""
        # This is a simplified parser - in a real system, we'd have structured messages
        state_parts = world_state_str.split(' | ')
        world_state = {}
        
        for part in state_parts:
            if ':' in part:
                key, value = part.split(':', 1)
                world_state[key.strip()] = value.strip()
        
        return world_state

    def plan_cycle(self):
        """Main planning cycle"""
        try:
            if self.current_state == PlanningState.IDLE:
                # Wait for goals
                pass
            elif self.current_state == PlanningState.PLANNING:
                if self.current_goal and self.current_pose:
                    self.generate_plan()
                    self.current_state = PlanningState.EXECUTING
            elif self.current_state == PlanningState.EXECUTING:
                # Monitor plan execution
                if self.should_replan():
                    self.current_state = PlanningState.PLANNING
            elif self.current_state == PlanningState.RECOVERY:
                # Execute recovery behavior
                self.execute_recovery()
        except Exception as e:
            self.get_logger().error(f'Error in planning cycle: {str(e)}')

    def generate_plan(self):
        """Generate a plan to reach the current goal"""
        if not self.current_goal or not self.current_pose:
            return None
        
        try:
            # In a real implementation, this would connect to a path planner
            # For now, we'll create a simple straight-line plan
            plan = Path()
            plan.header.frame_id = 'map'
            plan.header.stamp = self.get_clock().now().to_msg()
            
            # Calculate straight-line path
            start_pos = self.current_pose.pose.position
            goal_pos = self.current_goal.pose.position
            
            # In a real implementation, we'd use a proper path planning algorithm
            # like A*, RRT, or NavFn that considers the humanoid's constraints
            path_points = self.calculate_straight_line_path(
                [start_pos.x, start_pos.y], 
                [goal_pos.x, goal_pos.y]
            )
            
            for point in path_points:
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = point[0]
                pose_stamped.pose.position.y = point[1]
                pose_stamped.pose.position.z = 0.0
                # Add simple orientation towards next point
                plan.poses.append(pose_stamped)
            
            # Publish the plan
            self.plan_publisher.publish(plan)
            self.current_plan = plan
            self.active_plan_timestamp = self.get_clock().now()
            
            # Publish plan status
            status_msg = String()
            status_msg.data = f"PLAN_GENERATED: {len(plan.poses)} waypoints to goal ({goal_pos.x:.2f}, {goal_pos.y:.2f})"
            self.plan_status_publisher.publish(status_msg)
            
            self.get_logger().info(f'Plan generated with {len(plan.poses)} waypoints')
            
        except Exception as e:
            self.get_logger().error(f'Error generating plan: {str(e)}')

    def calculate_straight_line_path(self, start, goal, resolution=0.1):
        """Calculate a simple straight-line path for demonstration"""
        import math
        
        path = []
        dist = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        
        if dist < resolution:
            return [goal]
        
        steps = int(dist / resolution)
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append([x, y])
        
        return path

    def should_replan(self):
        """Determine if the current plan needs to be replanned"""
        if not self.current_plan or not self.current_pose:
            return True
        
        # Check if it's been too long since the last replan
        current_time = self.get_clock().now()
        time_since_replan = (current_time - self.last_replan_time).nanoseconds / 1e9
        if time_since_replan > 5.0:  # Replan every 5 seconds at most
            return True
        
        # Check if world state has significantly changed
        if 'Obstacles' in self.world_state:
            obstacle_count = self.parse_obstacle_count(self.world_state['Obstacles'])
            if obstacle_count > 5:  # If suddenly many obstacles appeared
                return True
        
        # Check if we're significantly off the planned path
        if len(self.current_plan.poses) > 1:
            next_waypoint = self.current_plan.poses[0].pose.position
            current_pos = self.current_pose.pose.position
            dist_to_waypoint = math.sqrt(
                (next_waypoint.x - current_pos.x)**2 + 
                (next_waypoint.y - current_pos.y)**2
            )
            
            if dist_to_waypoint > self.replanning_threshold:
                return True
        
        return False

    def parse_obstacle_count(self, obstacle_str):
        """Parse number of obstacles from world state string"""
        # Simple parsing: extract number from string like "Obstacles: 5 detected"
        import re
        numbers = re.findall(r'\d+', obstacle_str)
        return int(numbers[0]) if numbers else 0

    def execute_recovery(self):
        """Execute recovery behavior when in recovery state"""
        # Stop the robot temporarily
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)
        
        # Attempt to detect escape route
        if self.find_escape_route():
            self.current_state = PlanningState.PLANNING
        else:
            # If no escape route found, stay in recovery
            time.sleep(1.0)  # Wait before trying again

    def find_escape_route(self):
        """Look for an escape route in the current situation"""
        # In a real implementation, this would analyze the semantic map
        # and world state to find a way out of an obstacle situation
        # For now, return True to exit recovery
        return True

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('AI Planner Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    planner = AIPlannerNode()
    
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        planner.get_logger().info('Node interrupted by user')
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Creating the Control Pipeline

### 1. Behavior Tree Integration

Create a node that implements behavior tree functionality for humanoid decision making:

```python
# scripts/behavior_tree_executor.py
#!/usr/bin/env python3

"""
Behavior Tree Executor for Humanoid Robotics.
Implements behavior tree logic for decision making and task execution.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Imu
from builtin_interfaces.msg import Duration
from enum import Enum
import time


class NodeStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2


class NodeType(Enum):
    DECORATOR = 1
    CONDITION = 2
    ACTION = 3


class BehaviorTreeNode:
    """Base class for all behavior tree nodes"""
    
    def __init__(self, name):
        self.name = name
        self.status = NodeStatus.RUNNING
        self.children = []
        self.node_type = None
    
    def tick(self):
        """Execute the node's behavior"""
        raise NotImplementedError
    
    def add_child(self, child):
        """Add a child node"""
        self.children.append(child)
    
    def reset(self):
        """Reset the node state"""
        self.status = NodeStatus.RUNNING


class SequenceNode(BehaviorTreeNode):
    """Sequence node - executes children in order until one fails"""
    
    def __init__(self, name):
        super().__init__(name)
        self.current_child_idx = 0
        self.node_type = NodeType.ACTION
    
    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child = self.children[i]
            child_status = child.tick()
            
            if child_status == NodeStatus.FAILURE:
                self.current_child_idx = 0
                self.status = NodeStatus.FAILURE
                return self.status
            
            elif child_status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            
            elif child_status == NodeStatus.SUCCESS:
                # Continue to next child
                self.current_child_idx = i + 1
        
        # If all children succeeded
        self.current_child_idx = 0
        self.status = NodeStatus.SUCCESS
        return self.status


class SelectorNode(BehaviorTreeNode):
    """Selector node - executes children in order until one succeeds"""
    
    def __init__(self, name):
        super().__init__(name)
        self.current_child_idx = 0
        self.node_type = NodeType.ACTION
    
    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child = self.children[i]
            child_status = child.tick()
            
            if child_status == NodeStatus.SUCCESS:
                self.current_child_idx = 0
                self.status = NodeStatus.SUCCESS
                return self.status
            
            elif child_status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return self.status
            
            elif child_status == NodeStatus.FAILURE:
                # Try next child
                self.current_child_idx = i + 1
        
        # If all children failed
        self.current_child_idx = 0
        self.status = NodeStatus.FAILURE
        return self.status


class MoveToWaypointAction(BehaviorTreeNode):
    """Action node for moving to a specific waypoint"""
    
    def __init__(self, name, target_pose):
        super().__init__(name)
        self.target_pose = target_pose
        self.node_type = NodeType.ACTION
        self.moving = False
        self.arrived = False
        self.timeout = 30.0  # seconds
        self.start_time = None
    
    def tick(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # In real implementation, this would publish movement commands
        # and monitor actual robot position
        # For simulation, we'll just return success after a delay
        
        if not self.moving:
            # Start moving
            self.moving = True
            
            # In real implementation:
            # - Publish navigation goal
            # - Monitor for arrival
            # - Handle obstacles
            return NodeStatus.RUNNING
        
        # Check if arrived (simplified)
        current_time = time.time()
        if current_time - self.start_time > 5.0:  # Simulate 5 seconds to reach waypoint
            self.arrived = True
            return NodeStatus.SUCCESS
        
        # Check for timeout
        if current_time - self.start_time > self.timeout:
            return NodeStatus.FAILURE
        
        return NodeStatus.RUNNING


class CheckObstaclesCondition(BehaviorTreeNode):
    """Condition node to check for obstacles ahead"""
    
    def __init__(self, name, threshold=0.5):
        super().__init__(name)
        self.threshold = threshold
        self.node_type = NodeType.CONDITION
        self.obstacle_detected = False
    
    def tick(self):
        # In real implementation, this would check sensor data for obstacles
        # For simulation, we'll use a flag that's updated elsewhere
        if self.obstacle_detected:
            return NodeStatus.FAILURE  # Condition failed - obstacles detected
        else:
            return NodeStatus.SUCCESS  # Condition passed - no obstacles


class BehaviorTreeExecutorNode(Node):
    """
    Behavior Tree Executor Node.
    Runs behavior trees for humanoid robot decision making.
    """
    
    def __init__(self):
        super().__init__('behavior_tree_executor')
        
        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('tree_frequency', 10.0)  # Hz
        self.declare_parameter('default_behavior_tree', 'explore_and_avoid')
        
        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.tree_frequency = self.get_parameter('tree_frequency').value
        self.default_behavior = self.get_parameter('default_behavior_tree').value
        
        # Subscriptions
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.new_goal_sub = self.create_subscription(
            PoseStamped,
            '/new_goal',
            self.new_goal_callback,
            10
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        self.bt_status_pub = self.create_publisher(
            String,
            '/behavior_tree_status',
            10
        )
        
        # State variables
        self.current_behavior_tree = None
        self.lidar_data = None
        self.imu_data = None
        self.current_goal = None
        
        # Initialize behavior trees
        self.initialize_behavior_trees()
        
        # Create execution timer
        self.execution_timer = self.create_timer(1.0/self.tree_frequency, self.execute_behavior_tree)
        
        self.get_logger().info('Behavior Tree Executor Node Initialized')

    def initialize_behavior_trees(self):
        """Initialize different behavior trees for various tasks"""
        # Create exploration tree with obstacle avoidance
        explore_tree = SequenceNode("explore_with_avoidance")
        
        # Add obstacle check
        obstacle_check = CheckObstaclesCondition("check_front_obstacles", threshold=0.8)
        explore_tree.add_child(obstacle_check)
        
        # If no obstacles, move forward
        move_forward = MoveToWaypointAction("move_forward", 
                                          PoseStamped().pose)  # Placeholder
        explore_tree.add_child(move_forward)
        
        # Also create other trees for different behaviors
        navigate_tree = SequenceNode("navigate_to_goal")
        goal_check = CheckObstaclesCondition("check_path_to_goal", threshold=1.0)
        navigate_to_goal = MoveToWaypointAction("navigate_to_goal", 
                                              PoseStamped().pose)  # Placeholder
        
        navigate_tree.add_child(goal_check)
        navigate_tree.add_child(navigate_to_goal)
        
        # Store trees
        self.behavior_trees = {
            'explore': explore_tree,
            'navigate': navigate_tree
        }
        
        # Set default tree
        self.current_behavior_tree = self.behavior_trees['explore']

    def lidar_callback(self, msg):
        """Update lidar data and obstacle conditions"""
        self.lidar_data = msg
        
        # Update obstacle detection for all relevant nodes
        front_obstacles = self.check_front_obstacles(msg)
        
        # Update any obstacle check nodes
        for node in self.find_nodes_by_type(CheckObstaclesCondition):
            if "front" in node.name.lower():
                node.obstacle_detected = front_obstacles

    def imu_callback(self, msg):
        """Update IMU data"""
        self.imu_data = msg

    def new_goal_callback(self, msg):
        """Handle new navigation goal"""
        self.current_goal = msg
        # Switch to navigation behavior tree
        self.current_behavior_tree = self.behavior_trees['navigate']
        
        # Update the goal for the navigation action
        for node in self.find_nodes_by_type(MoveToWaypointAction):
            if "goal" in node.name.lower():
                node.target_pose = self.current_goal.pose
                node.arrived = False
                node.moving = False
                node.current_time = None

    def check_front_obstacles(self, lidar_msg):
        """Check for obstacles in the front sector of LiDAR data"""
        if not lidar_msg.ranges:
            return False
        
        # Check front 30 degrees of the scan
        num_beams = len(lidar_msg.ranges)
        front_start = int(num_beams / 2 - num_beams * 0.125)  # -30 degrees (15% of 360)
        front_end = int(num_beams / 2 + num_beams * 0.125)   # +30 degrees (15% of 360)
        
        front_ranges = lidar_msg.ranges[max(0, front_start):min(num_beams, front_end)]
        front_valid = [r for r in front_ranges if not (math.isnan(r) or math.isinf(r))]
        
        if front_valid:
            closest_front = min(front_valid)
            return closest_front < 0.8  # If obstacle is closer than 0.8m
        
        return False

    def find_nodes_by_type(self, node_type):
        """Find all nodes of a specific type in the current behavior tree"""
        def find_in_tree(node, target_type, found_nodes):
            if isinstance(node, target_type):
                found_nodes.append(node)
            for child in getattr(node, 'children', []):
                find_in_tree(child, target_type, found_nodes)
        
        nodes = []
        find_in_tree(self.current_behavior_tree, node_type, nodes)
        return nodes

    def execute_behavior_tree(self):
        """Execute the current behavior tree"""
        if not self.current_behavior_tree:
            return
        
        try:
            # Tick the behavior tree
            status = self.current_behavior_tree.tick()
            
            # Publish status
            status_msg = String()
            status_msg.data = f"BT_STATUS: {self.current_behavior_tree.name} - {status.name}"
            self.bt_status_pub.publish(status_msg)
            
            # Log status periodically
            if int(self.get_clock().now().nanoseconds / 1e9) % 5 == 0:  # Every 5 seconds
                self.get_logger().info(f'Behavior Tree Status: {status_msg.data}')
            
        except Exception as e:
            self.get_logger().error(f'Error executing behavior tree: {str(e)}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Behavior Tree Executor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    executor = BehaviorTreeExecutorNode()
    
    try:
        rclpy.spin(executor)
    except KeyboardInterrupt:
        executor.get_logger().info('Node interrupted by user')
    finally:
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Creating the Complete AI Pipeline Launch File

### 1. Comprehensive AI Pipeline Launch

Create a launch file that brings together all components:

```python
# launch/ai_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace', default='')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_ai_pipeline')
    
    # Perception integrator node
    perception_integrator = Node(
        package='humanoid_ai_pipeline',
        executable='perception_integrator',
        name='perception_integrator',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'image_topic': '/camera/image_raw'},
            {'lidar_topic': '/scan'},
            {'imu_topic': '/imu'},
            {'detection_topic': '/detections'},
            {'update_rate': 10.0}
        ],
        remappings=[
            ('/camera/image_raw', '/simple_humanoid/camera/image_rect_color'),
            ('/scan', '/simple_humanoid/scan'),
            ('/imu', '/simple_humanoid/imu'),
        ],
        output='screen'
    )
    
    # AI planner node
    ai_planner = Node(
        package='humanoid_ai_pipeline',
        executable='ai_planner',
        name='ai_planner',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'planning_frequency': 5.0},
            {'max_plan_length': 50},
            {'replanning_threshold': 0.5}
        ],
        output='screen'
    )
    
    # Behavior tree executor
    behavior_tree_executor = Node(
        package='humanoid_ai_pipeline',
        executable='behavior_tree_executor',
        name='behavior_tree_executor',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'tree_frequency': 10.0},
            {'default_behavior_tree': 'explore_with_avoidance'}
        ],
        output='screen'
    )
    
    # Gazebo simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', 'pipeline_test.sdf']),
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_simple_robot'),
                'urdf',
                'advanced_humanoid.urdf'
            ])}
        ],
        output='screen'
    )
    
    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', 
            PathJoinSubstitution([
                FindPackageShare('humanoid_ai_pipeline'),
                'rviz',
                'ai_pipeline.rviz'
            ])
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo
        gazebo,
        
        # Robot state publisher after delay
        TimerAction(
            period=2.0,
            actions=[robot_state_publisher]
        ),
        
        # Launch perception after robot is loaded
        TimerAction(
            period=4.0,
            actions=[perception_integrator]
        ),
        
        # Launch AI planner after perception is ready
        TimerAction(
            period=6.0,
            actions=[ai_planner]
        ),
        
        # Launch behavior tree executor last
        TimerAction(
            period=8.0,
            actions=[behavior_tree_executor]
        ),
        
        # Launch RViz for visualization
        TimerAction(
            period=10.0,
            actions=[rviz]
        ),
    ])
```

## Testing the Complete AI Pipeline

### 1. Build and Launch the Pipeline

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build the AI pipeline package
colcon build --packages-select humanoid_ai_pipeline

# Source the workspace
source install/setup.bash

# Launch the complete AI pipeline
ros2 launch humanoid_ai_pipeline ai_pipeline.launch.py
```

### 2. Monitor the Pipeline

```bash
# Terminal 1: Monitor perception status
ros2 topic echo /perception_pipeline_status

# Terminal 2: Monitor AI planner status
ros2 topic echo /plan_status

# Terminal 3: Monitor behavior tree status
ros2 topic echo /behavior_tree_status

# Terminal 4: Monitor navigation plans
ros2 topic echo /ai_plan
```

### 3. Send Goals to Test the Pipeline

```bash
# Send a navigation goal to test the complete pipeline
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

## Pipeline Optimization Strategies

### 1. Performance Optimization

```yaml
# config/pipeline_optimization.yaml
perception_integrator:
  ros__parameters:
    update_rate: 15.0  # Balance between responsiveness and performance
    image_topic_queue_size: 5  # Limit image buffering
    lidar_topic_queue_size: 10  # Moderate for LiDAR
    process_only_when_needed: true  # Only process when actively navigating

ai_planner:
  ros__parameters:
    planning_frequency: 2.0  # Lower for less frequent replanning
    plan_cache_size: 5  # Cache recent plans
    replanning_delay: 0.5  # Avoid oscillating replanning

behavior_tree_executor:
  ros__parameters:
    tree_frequency: 20.0  # Higher for responsive behavior
    node_timeout: 30.0  # Time limit for node execution
    parallel_execution: true  # Execute non-conflicting nodes in parallel
```

### 2. Data Flow Optimization

Optimize data flow between pipeline components:

```python
# Optimized data structure for pipeline communication
class OptimizedDataContainer:
    """Optimized data container to minimize data copying between pipeline stages"""
    
    def __init__(self):
        self.timestamp = None
        self.sensor_data = {
            'lidar': None,
            'camera': None,
            'imu': None
        }
        self.perception_results = {
            'objects': [],
            'obstacles': [],
            'features': []
        }
        self.world_state = {
            'map': None,
            'robot_pose': None,
            'environment': {}
        }
        self.planning_data = {
            'current_plan': None,
            'goal': None,
            'status': 'idle'
        }
```

## Next Steps

With the complete AI pipeline implemented, you now have a foundational system that can:
- Integrate multiple sensor inputs from Isaac ROS
- Create meaningful world representations
- Generate plans based on perception data
- Execute behaviors using behavior trees

This pipeline serves as the cognitive brain of your humanoid robot in simulation, forming the basis for more advanced AI behaviors and learning algorithms that you'll implement in later modules.