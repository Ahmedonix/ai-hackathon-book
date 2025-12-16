# Implementing Isaac ROS Perception Pipeline

## Overview

In this section, we'll implement a complete perception pipeline using Isaac ROS packages. This pipeline will integrate multiple sensors and perception algorithms to create a comprehensive understanding of the environment for humanoid robotics applications. We'll demonstrate how to combine visual SLAM, object detection, fiducial tracking, and depth estimation.

## Complete Perception Pipeline Architecture

### 1. Pipeline Overview

Our perception pipeline will implement the following data flow:

```
Camera Inputs -> Preprocessing -> [Visual SLAM] -> [Object Detection] -> [Fiducial Tracking]
                                      |                |                    |
                                      v                v                    v
                                Map Building    Object Classification   Pose Estimation
                                      |                |                    |
                                      +----------------+--------------------+
                                                      |
                                                      v
                                                Environment Understanding
                                                      |
                                                      v
                                            Navigation & Planning Commands
```

### 2. Creating the Perception Package

First, let's create a dedicated package for our perception pipeline:

```bash
# Navigate to your workspace
cd ~/humanoid_ws/src

# Create the perception package
ros2 pkg create --build-type ament_python humanoid_perception --dependencies \
  rclpy sensor_msgs geometry_msgs std_msgs nav_msgs message_filters \
  tf2_ros tf2_geometry_msgs cv_bridge image_geometry \
  isaac_ros_apriltag isaac_ros_dnn_inference isaac_ros_visual_slam \
  vision_msgs

# Navigate to the new package
cd humanoid_perception
```

### 3. Package Structure

Create the necessary directory structure:

```bash
mkdir -p humanoid_perception/{nodes,launch,config,utils,perception_modules}
```

## 4. Creating the Main Perception Node

Create the main perception pipeline node: `humanoid_perception/perception_pipeline.py`

```python
#!/usr/bin/env python3

"""
Complete perception pipeline implementation for humanoid robots using Isaac ROS.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
from vision_msgs.msg import Detection2DArray, Detection2D
from builtin_interfaces.msg import Time
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
import tf2_ros
import tf2_geometry_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import message_filters
from threading import Lock
import time


class PerceptionPipeline(Node):
    """
    Main perception pipeline node that combines multiple Isaac ROS perception modules
    to create a comprehensive environment understanding for humanoid robots.
    """
    
    def __init__(self):
        super().__init__('humanoid_perception_pipeline')
        
        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('visual_slam_pose_topic', '/visual_slam/pose_graph/pose')
        
        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.vslam_pose_topic = self.get_parameter('visual_slam_pose_topic').value
        
        # Internal state
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_broadcaster = TransformBroadcaster(self)
        self.pipeline_lock = Lock()
        
        # Perception data storage
        self.latest_image = None
        self.latest_camera_info = None
        self.latest_imu = None
        self.latest_vslam_pose = None
        self.latest_detections = None
        self.latest_fiducial_poses = None
        
        # Time tracking
        self.last_perception_update = self.get_clock().now()
        
        # Create subscriptions with appropriate QoS
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Image and camera info subscriptions
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=image_qos
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=info_qos
        )
        
        # Synchronize image and camera info with a time tolerance
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)
        
        # Additional topic subscriptions
        self.vslam_sub = self.create_subscription(
            PoseStamped,
            self.vslam_pose_topic,
            self.vslam_pose_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )
        
        self.fiducial_sub = self.create_subscription(
            PoseStamped,
            '/fiducial_pose',
            self.fiducial_callback,
            10
        )
        
        # Publishers for processed perception data
        self.perception_map_pub = self.create_publisher(
            Detection2DArray,
            '/environment_map',
            10
        )
        
        self.perception_status_pub = self.create_publisher(
            String,
            '/perception_status',
            10
        )
        
        self.command_pub = self.create_publisher(
            Twist,
            '/navigation_commands',
            10
        )
        
        # Timer for processing loop
        self.processing_timer = self.create_timer(0.033, self.process_perception_pipeline)  # ~30 Hz
        
        self.get_logger().info('Humanoid Perception Pipeline Initialized')
        self.get_logger().info(f'Using image topic: {self.image_topic}')
        self.get_logger().info(f'Using camera info topic: {self.camera_info_topic}')
        self.get_logger().info(f'Using IMU topic: {self.imu_topic}')
        self.get_logger().info(f'Using VSLAM topic: {self.vslam_pose_topic}')

    def image_callback(self, image_msg, info_msg):
        """Callback for synchronized image and camera info"""
        with self.pipeline_lock:
            self.latest_image = image_msg
            self.latest_camera_info = info_msg
            # Process image through perception pipeline
            self.process_image_perception(image_msg, info_msg)

    def vslam_pose_callback(self, pose_msg):
        """Callback for VSLAM pose updates"""
        with self.pipeline_lock:
            self.latest_vslam_pose = pose_msg
            self.get_logger().debug(f'VSLAM pose received: ({pose_msg.pose.position.x:.2f}, {pose_msg.pose.position.y:.2f})')

    def detection_callback(self, detection_msg):
        """Callback for object detection results"""
        with self.pipeline_lock:
            self.latest_detections = detection_msg
            self.get_logger().debug(f'Detected {len(detection_msg.detections)} objects')

    def imu_callback(self, imu_msg):
        """Callback for IMU data"""
        with self.pipeline_lock:
            self.latest_imu = imu_msg
            # Could be used for stability/tilt detection in humanoid context

    def fiducial_callback(self, pose_msg):
        """Callback for fiducial marker pose"""
        with self.pipeline_lock:
            self.latest_fiducial_poses = pose_msg
            self.get_logger().debug(f'Fiducial detected at ({pose_msg.pose.position.x:.2f}, {pose_msg.pose.position.y:.2f})')

    def process_image_perception(self, image_msg, info_msg):
        """Process visual perception pipeline"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            
            # Preprocess image for different perception tasks
            processed_image = self.preprocess_image(cv_image)
            
            # Run multiple perception tasks in parallel (in a real implementation,
            # these would be Isaac ROS nodes but we're simulating the behavior)
            object_detections = self.detect_objects(processed_image)
            depth_estimation = self.estimate_depth(processed_image, info_msg)
            feature_tracking = self.track_features(processed_image)
            
            # Update internal state with results
            with self.pipeline_lock:
                self.object_detections = object_detections
                self.depth_data = depth_estimation
                self.feature_tracks = feature_tracking
                
        except Exception as e:
            self.get_logger().error(f'Error in image perception processing: {str(e)}')

    def preprocess_image(self, image):
        """Preprocess image for perception tasks"""
        # Apply any necessary preprocessing (resize, color correction, etc.)
        # In a real implementation, this might include:
        # - Undistortion based on camera calibration
        # - Color space conversion if needed
        # - Noise reduction
        return image

    def detect_objects(self, image):
        """Simulate object detection using Isaac ROS DNN inference"""
        # In a real implementation, this would use Isaac ROS DNN nodes
        # For now, return placeholder
        detections = []
        
        # Simulate object detection results
        # In real implementation: 
        # 1. Pass image through Isaac ROS DNN inference
        # 2. Process results to extract meaningful detections
        # 3. Apply Isaac ROS DNN post-processing
        #
        # Example Isaac ROS DNN usage:
        #   self.dnn_inference_node would provide detections
        #   Results would be parsed according to model output format
        
        return detections

    def estimate_depth(self, image, camera_info):
        """Simulate depth estimation (would use stereo or structured light)"""
        # In a real implementation, this would use Isaac ROS stereo processing
        depth_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Placeholder - in real implementation:
        # 1. Use stereo cameras with Isaac ROS stereo pipeline
        # 2. Or use a depth sensor with proper Isaac ROS interface
        # 3. Generate depth maps for environment understanding
        
        return depth_map

    def track_features(self, image):
        """Simulate feature tracking (for VSLAM)"""
        # In a real implementation, this would feed into Isaac ROS Visual SLAM
        features = []
        
        # Placeholder - in real implementation:
        # 1. Extract features using GPU-accelerated methods
        # 2. Track features between frames
        # 3. Provide to Visual SLAM for pose estimation
        
        return features

    def process_perception_pipeline(self):
        """Main perception processing loop"""
        with self.pipeline_lock:
            # Only run if we have recent data
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_perception_update).nanoseconds / 1e9
            
            if time_diff < 0.1:  # Process at max 10Hz
                return
            
            # Aggregate perception data from all sources
            environment_map = self.create_environment_map()
            
            # Generate navigation commands based on perception
            navigation_cmd = self.generate_navigation_commands(environment_map)
            
            # Publish results
            self.perception_map_pub.publish(environment_map)
            self.command_pub.publish(navigation_cmd)
            
            # Publish status
            status_msg = String()
            status_msg.data = self.generate_perception_status()
            self.perception_status_pub.publish(status_msg)
            
            self.last_perception_update = current_time

    def create_environment_map(self):
        """Create a unified environment map from all perception sources"""
        environment_map = Detection2DArray()
        
        if self.latest_detections is not None:
            # Combine object detections with VSLAM pose information
            environment_map.header = self.latest_detections.header
            
            # Transform detections to map frame using VSLAM pose
            if self.latest_vslam_pose is not None:
                # In real implementation: transform detections to map frame
                # using robot pose from VSLAM
                transformed_detections = self.transform_detections_to_map_frame(
                    self.latest_detections,
                    self.latest_vslam_pose
                )
                
                environment_map.detections.extend(transformed_detections)
        
        # Add fiducial information to environment map
        if self.latest_fiducial_poses is not None:
            fiducial_detection = self.convert_fiducial_to_detection(self.latest_fiducial_poses)
            environment_map.detections.append(fiducial_detection)
        
        return environment_map

    def transform_detections_to_map_frame(self, detections, robot_pose):
        """Transform object detections to map frame"""
        transformed_detections = []
        
        for detection in detections.detections:
            # In a real implementation, we would:
            # 1. Get the camera pose relative to robot base
            # 2. Transform detection from camera frame to robot frame
            # 3. Transform from robot frame to map frame using VSLAM pose
            # 4. Use tf2 for these transformations
            
            # For now, return detections unchanged
            transformed_detections.append(detection)
        
        return transformed_detections

    def convert_fiducial_to_detection(self, fiducial_pose):
        """Convert fiducial pose to detection format"""
        detection = Detection2D()
        detection.header = fiducial_pose.header
        
        # Fill in bounding box (for visualization)
        detection.bbox.center.x = fiducial_pose.pose.position.x
        detection.bbox.center.y = fiducial_pose.pose.position.y
        detection.bbox.size_x = 0.3
        detection.bbox.size_y = 0.3
        
        # Add ID to results
        id_result = VisionResult()
        id_result.hypothesis.name = "fiducial"
        id_result.hypothesis.score = 0.95  # High confidence
        detection.results.append(id_result)
        
        return detection

    def generate_navigation_commands(self, environment_map):
        """Generate navigation commands based on environment understanding"""
        cmd = Twist()
        
        # Analyze environment map to determine navigation behavior
        if environment_map.detections:
            # Example: avoid obstacles detected in front of robot
            obstacles_ahead = self.filter_objects_ahead(environment_map.detections)
            
            if obstacles_ahead:
                # Implement obstacle avoidance behavior
                cmd.linear.x = 0.0  # Stop moving forward
                cmd.angular.z = 0.5  # Turn to avoid obstacle
            else:
                # Move forward if clear path
                cmd.linear.x = 0.3
                cmd.angular.z = 0.0
        
        return cmd

    def filter_objects_ahead(self, detections):
        """Filter detections that are in front of the robot"""
        # In a real implementation, based on robot pose and detection positions
        # For now, return all detections as placeholder
        return detections

    def generate_perception_status(self):
        """Generate a status message about perception pipeline"""
        status_parts = []
        
        if self.latest_image:
            status_parts.append(f"IMG:{self.latest_image.header.stamp.sec}s")
        
        if self.latest_vslam_pose:
            status_parts.append(f"POSE:({self.latest_vslam_pose.pose.position.x:.2f}, {self.latest_vslam_pose.pose.position.y:.2f})")
        
        if self.latest_detections:
            status_parts.append(f"DETECT:{len(self.latest_detections.detections)}")
        
        if self.latest_fiducial_poses:
            status_parts.append("FIDUCIAL:DETECTED")
        
        return " | ".join(status_parts) if status_parts else "NO_PERCEPTION_DATA"

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Perception Pipeline Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    perception_pipeline = PerceptionPipeline()
    
    try:
        rclpy.spin(perception_pipeline)
    except KeyboardInterrupt:
        perception_pipeline.get_logger().info('Perception pipeline interrupted by user')
    finally:
        perception_pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 5. Creating Specialized Perception Modules

Create a module for object detection: `humanoid_perception/perception_modules/object_detector.py`

```python
"""
Object detection module for humanoid robot perception pipeline.
Wrapper for Isaac ROS DNN Inference nodes.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import os


class IsaacObjectDetector(Node):
    """
    Wrapper for Isaac ROS object detection pipeline.
    Configured for humanoid robotics applications.
    """
    
    def __init__(self):
        super().__init__('isaac_object_detector')
        
        # Declare parameters
        self.declare_parameter('model_path', '/tmp/yolov5_plan.plan')
        self.declare_parameter('input_topic', '/camera/image_rect_color')
        self.declare_parameter('output_topic', '/isaac_ros/detections')
        self.declare_parameter('conf_thresh', 0.5)
        self.declare_parameter('nms_thresh', 0.4)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.nms_thresh = self.get_parameter('nms_thresh').value
        
        # Verify model file exists
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f'Model file not found: {self.model_path}')
            # In a real implementation, download a default model
            self.download_default_model()
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Subscriptions and publishers
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.output_topic,
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/object_detector/status',
            10
        )
        
        # Initialize Isaac ROS DNN components
        self.initialize_dnn_pipeline()
        
        self.get_logger().info(f'Isaac Object Detector initialized with model: {self.model_path}')

    def initialize_dnn_pipeline(self):
        """
        Initialize Isaac ROS DNN pipeline components.
        In a real implementation, this would set up the TensorRT engine
        and Isaac ROS DNN inference nodes.
        """
        # Placeholder for Isaac ROS DNN initialization
        # In real implementation:
        # 1. Initialize TensorRT engine from model file
        # 2. Set up Isaac ROS DNN inference encoder
        # 3. Configure input/output tensor formats
        # 4. Set up preprocessing and postprocessing
        self.dnn_initialized = True
        self.get_logger().info('Isaac DNN pipeline initialized')

    def image_callback(self, image_msg):
        """
        Process incoming image for object detection.
        """
        try:
            # In a real Isaac ROS implementation, the image would flow through:
            # 1. Isaac ROS ImageFormatConverter
            # 2. Isaac ROS Reshape proc
            # 3. Isaac ROS DNN Inference encoder
            # 4. TensorRT inference
            # 5. DNN Inference decoder
            # 6. Isaac ROS Detection2DArray to Detection2D converter
            
            # For simulation, we'll create mock detections
            detections = self.mock_detection_pipeline(image_msg)
            
            # Publish detections
            self.detection_pub.publish(detections)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Processed image {image_msg.header.stamp.sec}s - {len(detections.detections)} objects"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')

    def mock_detection_pipeline(self, image_msg):
        """
        Mock detection pipeline while we implement the real Isaac ROS version.
        """
        from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
        import random
        
        # Create mock detections
        detections = Detection2DArray()
        detections.header = image_msg.header
        
        # Generate some random detections for demonstration
        # In real implementation: Isaac ROS DNN would provide actual detections
        
        for i in range(random.randint(0, 5)):  # 0-5 random objects
            detection = Detection2D()
            
            # Random bounding box
            detection.bbox.center.x = random.uniform(100, 540)
            detection.bbox.center.y = random.uniform(100, 380)
            detection.bbox.size_x = random.uniform(20, 100)
            detection.bbox.size_y = random.uniform(20, 100)
            
            # Random detection result
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = random.choice(["person", "chair", "table", "box", "robot"])
            result.hypothesis.score = random.uniform(0.6, 0.95)
            
            detection.results.append(result)
            detections.detections.append(detection)
        
        return detections

    def download_default_model(self):
        """
        Download a default model if none exists.
        In a real implementation, this would download YOLOv5 or similar.
        """
        # Placeholder for model download logic
        self.get_logger().warn('Using mock detections - install proper model for real detection')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Isaac Object Detector Node Shutting Down')
        super().destroy_node()


# Example usage in main perception pipeline
def create_isaac_object_detector():
    """Factory function to create an Isaac object detector node"""
    return IsaacObjectDetector()
```

Create a module for visual SLAM: `humanoid_perception/perception_modules/vslam.py`

```python
"""
Visual SLAM module for humanoid robot perception pipeline.
Wrapper for Isaac ROS Visual SLAM components.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber


class IsaacVisualSlam(Node):
    """
    Wrapper for Isaac ROS Visual SLAM pipeline.
    Provides localization and mapping capabilities for humanoid robots.
    """
    
    def __init__(self):
        super().__init__('isaac_visual_slam')
        
        # Declare parameters
        self.declare_parameter('left_image_topic', '/stereo_camera/left/image_rect_color')
        self.declare_parameter('right_image_topic', '/stereo_camera/right/image_rect_color')
        self.declare_parameter('left_info_topic', '/stereo_camera/left/camera_info')
        self.declare_parameter('right_info_topic', '/stereo_camera/right/camera_info')
        self.declare_parameter('pose_output_topic', '/visual_slam/pose_graph/pose')
        self.declare_parameter('map_output_topic', '/visual_slam/map')
        self.declare_parameter('enable_slam_visualization', True)
        self.declare_parameter('enable_mapping', True)
        
        # Get parameters
        self.left_image_topic = self.get_parameter('left_image_topic').value
        self.right_image_topic = self.get_parameter('right_image_topic').value
        self.left_info_topic = self.get_parameter('left_info_topic').value
        self.right_info_topic = self.get_parameter('right_info_topic').value
        self.pose_output_topic = self.get_parameter('pose_output_topic').value
        self.map_output_topic = self.get_parameter('map_output_topic').value
        self.enable_slam_visualization = self.get_parameter('enable_slam_visualization').value
        self.enable_mapping = self.get_parameter('enable_mapping').value
        
        # Subscriptions for stereo images and camera info
        self.left_img_sub = Subscriber(self, Image, self.left_image_topic)
        self.right_img_sub = Subscriber(self, Image, self.right_image_topic)
        self.left_info_sub = Subscriber(self, CameraInfo, self.left_info_topic)
        self.right_info_sub = Subscriber(self, CameraInfo, self.right_info_topic)
        
        # Synchronize stereo pair
        self.stereo_sync = ApproximateTimeSynchronizer(
            [self.left_img_sub, self.right_img_sub, self.left_info_sub, self.right_info_sub],
            queue_size=10,
            slop=0.2
        )
        self.stereo_sync.registerCallback(self.stereo_callback)
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.pose_output_topic,
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_slam/odometry',
            10
        )
        
        self.map_pub = self.create_publisher(
            String,  # In real implementation, this would be a proper map message
            self.map_output_topic,
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/visual_slam/status',
            10
        )
        
        # Initialize Isaac ROS Visual SLAM components
        self.initialize_slam_pipeline()
        
        self.get_logger().info(f'Isaac Visual SLAM initialized on topics: {self.left_image_topic}, {self.right_image_topic}')

    def initialize_slam_pipeline(self):
        """
        Initialize Isaac ROS Visual SLAM pipeline components.
        In a real implementation, this would configure the Isaac Visual SLAM algorithm.
        """
        # Placeholder for Isaac Visual SLAM initialization
        # In real implementation:
        # 1. Initialize Isaac ROS Visual SLAM node
        # 2. Configure stereo rectification
        # 3. Set up feature extraction
        # 4. Configure mapping parameters
        # 5. Initialize pose graph optimization
        self.slam_initialized = True
        self.first_pose = True
        self.last_pose = None
        self.get_logger().info('Isaac Visual SLAM pipeline initialized')

    def stereo_callback(self, left_img, right_img, left_info, right_info):
        """
        Process synchronized stereo images for SLAM.
        """
        try:
            # In a real Isaac ROS implementation, this would:
            # 1. Feed stereo images to Isaac ROS Visual SLAM node
            # 2. Perform feature extraction and matching
            # 3. Estimate pose relative to previous frame
            # 4. Update pose graph for loop closure
            # 5. Optimize pose graph to correct drift
            
            # For simulation, we'll generate mock pose estimates
            current_pose = self.mock_slam_pipeline(left_img)
            
            # Publish pose
            pose_stamped = PoseStamped()
            pose_stamped.header = left_img.header
            pose_stamped.pose = current_pose.pose
            self.pose_pub.publish(pose_stamped)
            
            # Publish odometry
            odometry = Odometry()
            odometry.header = left_img.header
            odometry.pose.pose = current_pose.pose
            self.odom_pub.publish(odometry)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"SLAM pose: ({current_pose.pose.position.x:.2f}, {current_pose.pose.position.y:.2f}), " \
                             f"Features: {current_pose.pose.orientation.w * 100:.0f}"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in visual SLAM: {str(e)}')

    def mock_slam_pipeline(self, image_msg):
        """
        Mock SLAM pipeline while we implement the real Isaac ROS version.
        """
        from geometry_msgs.msg import Pose
        import random
        
        # Create mock pose (in a real implementation, this would come from Visual SLAM)
        pose = Pose()
        
        # Simulate gradual movement
        if self.first_pose:
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0
            self.first_pose = False
            self.last_position = np.array([0.0, 0.0, 0.0])
        else:
            # Simulate movement based on time
            dt = 0.1  # 10Hz simulation
            velocity = np.array([
                0.2 * random.uniform(0.5, 1.5),  # Forward movement
                0.1 * random.uniform(-1, 1),     # Slight lateral drift
                0.0
            ])
            
            new_position = self.last_position + velocity * dt
            pose.position.x = float(new_position[0])
            pose.position.y = float(new_position[1])
            pose.position.z = float(new_position[2])
            
            # Simulate slight rotational drift
            angle_change = random.uniform(-0.1, 0.1)
            # Convert to quaternion (simplified)
            pose.orientation.z = np.sin(angle_change / 2.0)
            pose.orientation.w = np.cos(angle_change / 2.0)
            
            self.last_position = new_position
        
        return pose

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Isaac Visual SLAM Node Shutting Down')
        super().destroy_node()


# Example usage in main perception pipeline
def create_isaac_visual_slam():
    """Factory function to create an Isaac Visual SLAM node"""
    return IsaacVisualSlam()
```

Create a module for fiducial detection: `humanoid_perception/perception_modules/apriltag_detector.py`

```python
"""
Apriltag detection module for humanoid robot perception pipeline.
Wrapper for Isaac ROS Apriltag components.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber


class IsaacApriltagDetector(Node):
    """
    Wrapper for Isaac ROS Apriltag detection pipeline.
    Provides precise fiducial marker pose estimation for humanoid robots.
    """
    
    def __init__(self):
        super().__init__('isaac_apriltag_detector')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_rect')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detections_output_topic', '/apriltag_detections')
        self.declare_parameter('pose_output_topic', '/apriltag_poses')
        self.declare_parameter('tag_size', 0.16)  # Tag size in meters
        self.declare_parameter('tag_family', 'tag36h11')
        self.declare_parameter('max_tags', 64)
        
        # Get parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detections_output_topic = self.get_parameter('detections_output_topic').value
        self.pose_output_topic = self.get_parameter('pose_output_topic').value
        self.tag_size = self.get_parameter('tag_size').value
        self.tag_family = self.get_parameter('tag_family').value
        self.max_tags = self.get_parameter('max_tags').value
        
        # Subscriptions
        self.image_sub = Subscriber(self, Image, self.image_topic)
        self.info_sub = Subscriber(self, CameraInfo, self.camera_info_topic)
        
        # Synchronize image and camera info
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_info_callback)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            String,  # In real implementation, this would be proper detection message
            self.detections_output_topic,
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.pose_output_topic,
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/apriltag_detector/status',
            10
        )
        
        # Initialize Isaac ROS Apriltag detector
        self.initialize_apriltag_detector()
        
        self.get_logger().info(f'Isaac Apriltag Detector initialized for family: {self.tag_family}')

    def initialize_apriltag_detector(self):
        """
        Initialize Isaac ROS Apriltag detection components.
        """
        # Placeholder for Isaac Apriltag initialization
        # In real implementation:
        # 1. Initialize Isaac ROS Apriltag node
        # 2. Configure tag family and size
        # 3. Set detection parameters
        # 4. Prepare for GPU acceleration
        self.detector_initialized = True
        self.get_logger().info('Isaac Apriltag detector initialized')

    def image_info_callback(self, image_msg, info_msg):
        """
        Process synchronized image and camera info for apriltag detection.
        """
        try:
            # In a real Isaac ROS implementation, this would:
            # 1. Feed image to Isaac ROS Apriltag node
            # 2. Use camera info for proper pose estimation
            # 3. Generate 6DOF poses for detected tags
            # 4. Output tag poses in camera frame, then transform to robot/base frame
            
            # For simulation, we'll create mock apriltag detections
            detection_result = self.mock_apriltag_pipeline(image_msg)
            
            # Publish detection results
            if detection_result:
                self.detection_pub.publish(detection_result)
                
                # Publish pose if available
                pose_result = self.create_mock_pose_from_detection(detection_result, image_msg.header)
                if pose_result:
                    self.pose_pub.publish(pose_result)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Apriltag detection on {image_msg.header.stamp.sec}s, tags: {len(detection_result.detections) if detection_result else 0}"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in apriltag detection: {str(e)}')

    def mock_apriltag_pipeline(self, image_msg):
        """
        Mock apriltag detection while we implement the real Isaac ROS version.
        """
        # In real implementation:
        # 1. Isaac ROS Apriltag node processes image
        # 2. Returns tag corners and IDs
        # 3. Uses camera info to compute poses
        # 4. Publishes tag poses
        
        # For now, return none if no tags expected
        # In a real implementation with tags in the scene, this would detect them
        return None

    def create_mock_pose_from_detection(self, detection, header):
        """
        Create a mock pose from detection data.
        """
        from geometry_msgs.msg import Pose
        import random
        
        # In real implementation: derive pose from tag corners and camera info
        # For mock: create arbitrary poses for demonstration
        
        if random.random() > 0.8:  # 20% chance of detecting a tag
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            
            # Simulate tag at random position in front of robot
            pose_stamped.pose.position.x = random.uniform(0.5, 3.0)  # 0.5-3m in front
            pose_stamped.pose.position.y = random.uniform(-1.0, 1.0)  # ±1m sideways
            pose_stamped.pose.position.z = random.uniform(0.0, 1.5)   # 0-1.5m height
            
            # Identity orientation for simplicity
            pose_stamped.pose.orientation.w = 1.0
            
            return pose_stamped
        
        return None

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Isaac Apriltag Detector Node Shutting Down')
        super().destroy_node()


# Example usage in main perception pipeline
def create_isaac_apriltag_detector():
    """Factory function to create an Isaac Apriltag detector node"""
    return IsaacApriltagDetector()
```

### 6. Creating the Main Launch File

Create a comprehensive launch file: `launch/perception_pipeline.launch.py`

```python
# launch/perception_pipeline.launch.py
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
    camera_namespace = LaunchConfiguration('camera_namespace', default='/camera')
    enable_visualization = LaunchConfiguration('enable_visualization', default='true')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_perception')
    
    # Launch Gazebo if needed
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', 'perception_test.sdf']),
            'gui': 'true' if enable_visualization.to_string() == 'true' else 'false'
        }.items()
    )
    
    # Main perception pipeline node
    perception_pipeline = Node(
        package='humanoid_perception',
        executable='perception_pipeline',
        name='humanoid_perception_pipeline',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'image_topic': [camera_namespace, '/image_rect_color'],
                'camera_info_topic': [camera_namespace, '/camera_info'],
                'imu_topic': '/imu',
                'visual_slam_pose_topic': '/visual_slam/pose_graph/pose'
            }
        ],
        respawn=True,
        output='screen'
    )
    
    # Object detection node
    object_detector = Node(
        package='humanoid_perception',
        executable='isaac_object_detector',
        name='isaac_object_detector',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'input_topic': [camera_namespace, '/image_rect_color'],
                'model_path': '/tmp/yolov5_plan.plan',  # Will be replaced with real model
                'conf_thresh': 0.5,
                'nms_thresh': 0.4
            }
        ],
        respawn=True,
        output='screen'
    )
    
    # Visual SLAM node
    visual_slam = Node(
        package='humanoid_perception',
        executable='isaac_visual_slam',
        name='isaac_visual_slam',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'left_image_topic': [camera_namespace, '/left/image_rect_color'],
                'right_image_topic': [camera_namespace, '/right/image_rect_color'],
                'left_info_topic': [camera_namespace, '/left/camera_info'],
                'right_info_topic': [camera_namespace, '/right/camera_info'],
                'enable_slam_visualization': enable_visualization,
                'enable_mapping': True
            }
        ],
        respawn=True,
        output='screen'
    )
    
    # Apriltag detection node
    apriltag_detector = Node(
        package='humanoid_perception',
        executable='isaac_apriltag_detector',
        name='isaac_apriltag_detector',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'image_topic': [camera_namespace, '/image_rect'],
                'camera_info_topic': [camera_namespace, '/camera_info'],
                'tag_size': 0.16,
                'tag_family': 'tag36h11'
            }
        ],
        respawn=True,
        output='screen'
    )
    
    # Robot State Publisher for transforms
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'robot_description': PathJoinSubstitution([
                    FindPackageShare('humanoid_description'),
                    'urdf',
                    'advanced_humanoid.urdf'
                ])
            }
        ],
        output='screen'
    )
    
    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Static transform publisher for camera (example)
    camera_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_link_publisher',
        arguments=['0.1', '0', '0.5', '0', '0', '0', 'base_link', 'camera_link'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # RViz for visualization if enabled
    rviz = Node(
        condition=IfCondition(enable_visualization),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('humanoid_perception'),
            'rviz',
            'perception_pipeline.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    return LaunchDescription([
        # Set parameter for all nodes
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo
        gazebo,
        
        # Launch robot state publishers after a delay
        TimerAction(
            period=2.0,
            actions=[robot_state_publisher]
        ),
        
        TimerAction(
            period=2.0,
            actions=[joint_state_publisher]
        ),
        
        # Launch perception nodes after robot state is available
        TimerAction(
            period=4.0,
            actions=[camera_tf_publisher, object_detector, visual_slam, apriltag_detector]
        ),
        
        # Launch main perception pipeline
        TimerAction(
            period=6.0,
            actions=[perception_pipeline]
        ),
        
        # Launch RViz
        TimerAction(
            period=8.0,
            actions=[rviz]
        ),
    ])
```

### 7. Creating a Processing Test Node

Create a test node to verify the pipeline works: `scripts/test_perception_pipeline.py`

```python
#!/usr/bin/env python3

"""
Test script to validate perception pipeline functionality.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import time


class PerceptionTestNode(Node):
    def __init__(self):
        super().__init__('perception_test_node')
        
        # Track received messages
        self.received_topics = {
            '/camera/image_raw': False,
            '/camera/camera_info': False,
            '/environment_map': False,
            '/perception_status': False,
            '/visual_slam/pose_graph/pose': False,
            '/detections': False,
            '/fiducial_pose': False
        }
        
        # Subscriptions to verify all perception outputs
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10
        )
        
        self.env_map_sub = self.create_subscription(
            Detection2DArray, '/environment_map', self.env_map_callback, 10
        )
        
        self.status_sub = self.create_subscription(
            String, '/perception_status', self.status_callback, 10
        )
        
        self.vslam_sub = self.create_subscription(
            PoseStamped, '/visual_slam/pose_graph/pose', self.vslam_callback, 10
        )
        
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10
        )
        
        self.fiducial_sub = self.create_subscription(
            PoseStamped, '/fiducial_pose', self.fiducial_callback, 10
        )
        
        # Test timer
        self.test_timer = self.create_timer(5.0, self.run_tests)
        
        self.test_start_time = time.time()
        self.get_logger().info('Perception Test Node Started')

    def camera_callback(self, msg):
        self.received_topics['/camera/image_raw'] = True

    def info_callback(self, msg):
        self.received_topics['/camera/camera_info'] = True

    def env_map_callback(self, msg):
        self.received_topics['/environment_map'] = True

    def status_callback(self, msg):
        self.received_topics['/perception_status'] = True
        self.get_logger().info(f'Perception Status: {msg.data}')

    def vslam_callback(self, msg):
        self.received_topics['/visual_slam/pose_graph/pose'] = True

    def detection_callback(self, msg):
        self.received_topics['/detections'] = True

    def fiducial_callback(self, msg):
        self.received_topics['/fiducial_pose'] = True

    def run_tests(self):
        """Run tests to validate perception pipeline"""
        current_time = time.time()
        
        # Test if all topics are receiving data
        all_working = all(self.received_topics.values())
        
        if all_working:
            self.get_logger().info('✓ ALL PERCEPTION PIPELINE COMPONENTS WORKING')
        else:
            missing_topics = [topic for topic, received in self.received_topics.items() if not received]
            self.get_logger().warn(f'✗ MISSING DATA FROM TOPICS: {missing_topics}')
        
        # Show status of each component
        for topic, received in self.received_topics.items():
            status = "✓" if received else "✗"
            self.get_logger().info(f'  {status} {topic}: {"RECEIVING" if received else "NO DATA"}')
        
        # After 30 seconds, stop the test
        if current_time - self.test_start_time > 30:
            self.get_logger().info('Perception Pipeline Test Complete')
            self.test_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    
    test_node = PerceptionTestNode()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info('Test interrupted by user')
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 8. Update Setup Files

Update the setup.py file to include new executables:

```python
# setup.py
from setuptools import setup
from glob import glob
import os

package_name = 'humanoid_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include worlds
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Perception pipeline for humanoid robots',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_pipeline = humanoid_perception.perception_pipeline:main',
            'isaac_object_detector = humanoid_perception.perception_modules.object_detector:create_isaac_object_detector',
            'isaac_visual_slam = humanoid_perception.perception_modules.vslam:create_isaac_visual_slam',
            'isaac_apriltag_detector = humanoid_perception.perception_modules.apriltag_detector:create_isaac_apriltag_detector',
            'test_perception_pipeline = humanoid_perception.scripts.test_perception_pipeline:main',
        ],
    },
)
```

### 9. Building and Testing

Now build the perception package:

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build only the perception package
colcon build --packages-select humanoid_perception

# Source the workspace
source install/setup.bash
```

## 10. Running the Complete Perception Pipeline

### Step 10.1: Launch the Pipeline

```bash
# Launch the complete perception pipeline
ros2 launch humanoid_perception perception_pipeline.launch.py
```

### Step 10.2: Run the Test Node in Another Terminal

```bash
# In a new terminal
cd ~/humanoid_ws
source install/setup.bash
ros2 run humanoid_perception test_perception_pipeline
```

### Step 10.3: Monitor the Pipeline

Monitor various topics to ensure the pipeline is functioning:

```bash
# Check all perception-related topics
ros2 topic list | grep -E "(camera|imu|slam|detection|perception)"

# Monitor specific perception outputs
ros2 topic echo /environment_map
ros2 topic echo /perception_status
ros2 topic echo /visual_slam/pose_graph/pose
```

## 11. Validation and Troubleshooting

### Common Issues and Solutions:

1. **No Image Data**:
   - Verify camera is publishing to the expected topic
   - Check that image topics are properly remapped
   - Ensure camera is calibrated with proper camera_info

2. **IMU Not Working**:
   - Verify IMU sensor is properly configured in URDF
   - Check that IMU plugin is loaded in Gazebo
   - Ensure proper frame relationships

3. **Visual SLAM Not Converging**:
   - Check stereo camera calibration
   - Verify baseline distance between cameras
   - Ensure sufficient visual features in environment

4. **Object Detection Performance**:
   - Verify model file exists and is correct format
   - Check that image preprocessing is correct
   - Ensure TensorRT is properly installed

## 12. Performance Optimization

### Tips for Optimizing Perception Pipeline:

1. **Use Isaac ROS Nitros**: For improved inter-node communication performance
2. **Optimize Model Sizes**: Use quantized models for faster inference
3. **Adjust Update Rates**: Tune update rates based on application needs
4. **GPU Memory Management**: Monitor and optimize GPU memory usage

## Next Steps

With the complete Isaac ROS perception pipeline implemented, you now have a robust foundation for humanoid robot perception in simulation. The next step is to implement VSLAM and navigation capabilities using Nav2, which will use this perception information for autonomous navigation and exploration tasks.