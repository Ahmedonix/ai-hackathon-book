---
title: VSLAM Perception Pipeline
description: Building Visual Simultaneous Localization and Mapping pipelines for humanoid robots
sidebar_position: 4
---

# VSLAM Perception Pipeline

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is a critical component for humanoid robots operating in unknown environments. This chapter explores how to build robust VSLAM pipelines that enable humanoid robots to understand their position and map their surroundings using visual sensors like cameras.

## Learning Objectives

- Understand the fundamentals of VSLAM algorithms
- Implement visual odometry components for pose estimation
- Create mapping systems for environment representation
- Integrate VSLAM with navigation systems
- Optimize VSLAM for humanoid robot constraints

## VSLAM Fundamentals

### What is VSLAM?

VSLAM (Visual Simultaneous Localization and Mapping) combines computer vision and robotics to allow robots to build a map of an unknown environment while simultaneously determining their position within that map using visual data from cameras.

### Key Challenges for Humanoid Robots

1. **Computational Constraints**: Humanoid robots often have limited computational resources compared to wheeled robots
2. **Dynamic Motion**: Humanoid locomotion creates vibrations and dynamic motion that affects visual data
3. **Sensor Position**: Camera placement on humanoid robots may have limited field of view
4. **Real-time Processing**: Humanoid robots need real-time localization for balance and navigation

## Visual Odometry

### Feature Detection and Matching

```python
import cv2
import numpy as np
from collections import deque

class FeatureTracker:
    def __init__(self):
        # Use ORB features for efficiency on embedded systems
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Store feature points over time
        self.keypoints_history = deque(maxlen=10)
        self.descriptors_history = deque(maxlen=10)
    
    def detect_and_match(self, current_frame):
        """
        Detect features in current frame and match with previous frame
        """
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(current_frame, None)
        
        if len(self.descriptors_history) == 0:
            # First frame, just store features
            self.keypoints_history.append(keypoints)
            self.descriptors_history.append(descriptors)
            return [], []
        
        # Match with previous frame
        previous_descriptors = self.descriptors_history[-1]
        previous_keypoints = self.keypoints_history[-1]
        
        if descriptors is not None and previous_descriptors is not None:
            matches = self.matcher.match(descriptors, previous_descriptors)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract matched points
            current_points = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            previous_points = np.float32([previous_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            return current_points, previous_points
        
        return [], []

class VisualOdometry:
    def __init__(self, focal_length, principal_point):
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.feature_tracker = FeatureTracker()
        
        # Initial pose
        self.current_pose = np.eye(4)
        self.position_history = [self.current_pose[:3, 3]]
    
    def estimate_motion(self, current_frame):
        """
        Estimate motion between current and previous frame
        """
        current_points, previous_points = self.feature_tracker.detect_and_match(current_frame)
        
        if len(current_points) >= 8:
            # Compute essential matrix
            E, mask = cv2.findEssentialMat(
                current_points, 
                previous_points, 
                focal=self.focal_length, 
                pp=self.principal_point, 
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is not None:
                # Recover relative pose
                _, R, t, _ = cv2.recoverPose(
                    E, 
                    current_points, 
                    previous_points, 
                    focal=self.focal_length, 
                    pp=self.principal_point
                )
                
                # Create transformation matrix
                transformation = np.eye(4)
                transformation[:3, :3] = R
                transformation[:3, 3] = t.ravel()
                
                # Update current pose
                self.current_pose = self.current_pose @ np.linalg.inv(transformation)
                
                # Store position for history
                self.position_history.append(self.current_pose[:3, 3])
                
                return self.current_pose
        else:
            print("Not enough features to estimate motion")
        
        return self.current_pose
```

### Direct Methods vs. Feature-Based Methods

```python
class DirectMethodVO:
    """
    Direct method for visual odometry - works on pixel intensities directly
    """
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.min_gradients = 100  # Minimum gradient for reliable tracking
    
    def compute_intensity_gradients(self, image):
        """Compute gradients for tracking reliability"""
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(dx**2 + dy**2)
    
    def estimate_motion_direct(self, prev_image, curr_image, initial_pose=np.eye(4)):
        """
        Estimate motion using direct method
        """
        # Convert to grayscale if needed
        if len(prev_image.shape) == 3:
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_image
            curr_gray = curr_image
        
        # Compute gradients
        gradients = self.compute_intensity_gradients(prev_gray)
        
        # Select pixels with high gradients
        high_gradient_mask = gradients > self.min_gradients
        y_coords, x_coords = np.where(high_gradient_mask)
        selected_pixels = list(zip(x_coords, y_coords))
        
        # Limit number of pixels to process for computational efficiency
        if len(selected_pixels) > 1000:
            selected_pixels = np.random.choice(
                selected_pixels, size=1000, replace=False
            )
        
        # Initialize pose estimate
        pose = initial_pose.copy()
        
        # Iteratively optimize pose using Gauss-Newton
        for iteration in range(self.max_iterations):
            # Compute Jacobian and residuals
            jacobian, residuals = self.compute_jacobian_residuals(
                selected_pixels, prev_gray, curr_gray, pose
            )
            
            # Solve for pose update using least squares
            if jacobian.shape[0] >= jacobian.shape[1]:
                try:
                    update = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
                    
                    # Update pose
                    pose = self.update_pose(pose, update)
                except np.linalg.LinAlgError:
                    print("Singular matrix encountered in direct method")
                    break
            else:
                break
        
        return pose
    
    def compute_jacobian_residuals(self, pixels, prev_img, curr_img, pose):
        """
        Compute Jacobian matrix and residuals for least squares optimization
        """
        # This is a simplified implementation
        # In practice, this would be more complex involving camera projection
        residuals = np.zeros(len(pixels))
        jacobian = np.zeros((len(pixels), 6))  # [x, y, z, rx, ry, rz]
        
        for i, (x, y) in enumerate(pixels):
            # For each pixel, compute intensity difference and Jacobian
            # This is a placeholder - actual implementation would be more complex
            try:
                # Get intensity from previous image
                prev_intensity = prev_img[y, x]
                
                # Project to current frame using pose
                # Simplified projection
                projected_x = x + pose[0, 3]  # Translation in x
                projected_y = y + pose[1, 3]  # Translation in y
                
                # Bounds check
                if 0 <= int(projected_y) < curr_img.shape[0] and 0 <= int(projected_x) < curr_img.shape[1]:
                    curr_intensity = curr_img[int(projected_y), int(projected_x)]
                    residuals[i] = curr_intensity - prev_intensity
                else:
                    residuals[i] = 0
            except IndexError:
                residuals[i] = 0
        
        return jacobian, residuals
    
    def update_pose(self, pose, update):
        """
        Update pose using exponential map
        """
        # Create skew-symmetric matrix from rotation vector
        dt = update[:3]  # Translation update
        drot = update[3:]  # Rotation update (as axis-angle)
        
        # Convert axis-angle to rotation matrix
        angle = np.linalg.norm(drot)
        if angle > 1e-8:
            axis = drot / angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        else:
            dR = np.eye(3)
        
        # Create transformation matrix
        dT = np.eye(4)
        dT[:3, :3] = dR
        dT[:3, 3] = dt
        
        # Update pose: T_new = T_old * dT
        return pose @ dT
```

## Mapping Components

### Keyframe-Based Mapping

```python
class KeyframeMapper:
    def __init__(self):
        self.keyframes = []
        self.map_points = []
        self.last_keyframe_pose = None
        self.keyframe_threshold = 0.5  # Minimum translation to add keyframe
        
    def should_add_keyframe(self, current_pose):
        """Determine if current pose is different enough to warrant a new keyframe"""
        if self.last_keyframe_pose is None:
            return True
        
        # Calculate distance from last keyframe
        translation_diff = np.linalg.norm(
            current_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
        )
        
        # Calculate rotation difference
        rotation_diff = self.rotation_distance(
            current_pose[:3, :3], 
            self.last_keyframe_pose[:3, :3]
        )
        
        return (translation_diff > self.keyframe_threshold or 
                rotation_diff > 0.1)  # 0.1 radian threshold
    
    def rotation_distance(self, R1, R2):
        """Calculate rotation distance between two rotation matrices"""
        R_rel = R1 @ R2.T
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle
    
    def add_keyframe(self, image, pose, features):
        """Add a new keyframe to the map"""
        keyframe = {
            'image': image,
            'pose': pose.copy(),
            'features': features,
            'id': len(self.keyframes)
        }
        
        self.keyframes.append(keyframe)
        self.last_keyframe_pose = pose
        
        return keyframe['id']
    
    def triangulate_points(self, keyframe1, keyframe2, matches):
        """Triangulate 3D points from matched features between two keyframes"""
        # Camera intrinsic parameters (example values)
        K = np.array([
            [500, 0, 320],  # fx, 0, cx
            [0, 500, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ])
        
        # Get matched points
        points1 = np.float32([keyframe1['features']['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1, 2)
        points2 = np.float32([keyframe2['features']['keypoints'][m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Normalize points
        points1_norm = cv2.undistortPoints(np.expand_dims(points1, axis=1), K, None)
        points2_norm = cv2.undistortPoints(np.expand_dims(points2, axis=1), K, None)
        
        # Get transformation between keyframes
        T1 = keyframe1['pose']
        T2 = keyframe2['pose']
        
        # Create projection matrices
        P1 = K @ T1[:3, :4]  # Projection matrix for first keyframe
        P2 = K @ T2[:3, :4]  # Projection matrix for second keyframe
        
        # Triangulate points
        points4D = cv2.triangulatePoints(P1, P2, points1_norm.reshape(-1, 2).T, points2_norm.reshape(-1, 2).T)
        
        # Convert from homogeneous coordinates
        points3D = (points4D[:3] / points4D[3]).T
        
        return points3D
```

### Map Management

```python
class MapManager:
    def __init__(self):
        self.points = {}  # 3D map points
        self.keyframes = {}  # Keyframes
        self.point_observations = {}  # Track which keyframes observe which points
        
    def add_point(self, point_id, coordinates, descriptors=None):
        """Add a 3D point to the map"""
        self.points[point_id] = {
            'coordinates': coordinates,
            'descriptors': descriptors,
            'observations': []  # List of (keyframe_id, feature_index) tuples
        }
    
    def add_observation(self, point_id, keyframe_id, feature_index):
        """Record that a point was observed in a specific keyframe"""
        if point_id not in self.point_observations:
            self.point_observations[point_id] = []
        
        self.point_observations[point_id].append((keyframe_id, feature_index))
        
        # Also update the point's observation list
        if point_id in self.points:
            self.points[point_id]['observations'].append((keyframe_id, feature_index))
    
    def optimize_map(self):
        """Run bundle adjustment to optimize map and camera poses"""
        # This would typically use a library like Ceres Solver or g2o
        # For this example, we'll outline the approach:
        
        # 1. Formulate the bundle adjustment problem
        # 2. Optimize both 3D points and camera poses simultaneously
        # 3. Minimize reprojection error
        pass
    
    def remove_outliers(self, reprojection_threshold=5.0):
        """Remove points with high reprojection error"""
        # Calculate reprojection error for each 3D point
        for point_id, point_data in list(self.points.items()):
            total_error = 0
            num_observations = len(point_data['observations'])
            
            if num_observations == 0:
                continue
                
            for keyframe_id, feature_idx in point_data['observations']:
                if keyframe_id in self.keyframes:
                    keyframe = self.keyframes[keyframe_id]
                    
                    # Project 3D point to 2D
                    K = keyframe['camera_matrix']
                    pose = keyframe['pose']
                    
                    # Transform point to camera coordinates
                    point_cam = pose @ np.append(point_data['coordinates'], 1)
                    
                    # Project to image plane
                    if point_cam[2] != 0:
                        x = K[0, 0] * point_cam[0] / point_cam[2] + K[0, 2]
                        y = K[1, 1] * point_cam[1] / point_cam[2] + K[1, 2]
                        
                        # Calculate reprojection error
                        observed_x = keyframe['features']['keypoints'][feature_idx].pt[0]
                        observed_y = keyframe['features']['keypoints'][feature_idx].pt[1]
                        
                        error = np.sqrt((x - observed_x)**2 + (y - observed_y)**2)
                        total_error += error
            
            avg_error = total_error / num_observations if num_observations > 0 else 0
            
            # Remove if error is too high
            if avg_error > reprojection_threshold:
                del self.points[point_id]
```

## Integration with Navigation

### Path Planning Based on VSLAM Map

```python
import heapq

class VSLAMNavigator:
    def __init__(self, map_manager):
        self.map_manager = map_manager
        self.grid_resolution = 0.1  # 10cm per grid cell
        self.grid_size = 100  # 10m x 10m grid
    
    def create_occupancy_grid(self):
        """Convert VSLAM 3D points to 2D occupancy grid"""
        # Initialize grid with unknown state
        grid = np.full((self.grid_size, self.grid_size), 0.5)  # 0.5 = unknown
        
        # Convert 3D points to grid coordinates
        for point_id, point_data in self.map_manager.points.items():
            x, y, z = point_data['coordinates']
            
            # Convert to grid coordinates
            grid_x = int((x + self.grid_size * self.grid_resolution / 2) / self.grid_resolution)
            grid_y = int((y + self.grid_size * self.grid_resolution / 2) / self.grid_resolution)
            
            # Mark as occupied if within grid bounds
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_x, grid_y] = 0.9  # Occupied
        
        return grid
    
    def find_path(self, start_pos, goal_pos):
        """Find path using A* algorithm on the occupancy grid"""
        grid = self.create_occupancy_grid()
        
        # Convert positions to grid coordinates
        start_grid = (
            int((start_pos[0] + self.grid_size * self.grid_resolution / 2) / self.grid_resolution),
            int((start_pos[1] + self.grid_size * self.grid_resolution / 2) / self.grid_resolution)
        )
        
        goal_grid = (
            int((goal_pos[0] + self.grid_size * self.grid_resolution / 2) / self.grid_resolution),
            int((goal_pos[1] + self.grid_size * self.grid_resolution / 2) / self.grid_resolution)
        )
        
        # A* algorithm implementation
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # Priority queue for A*
        pq = [(0, start_grid)]
        came_from = {}
        cost_so_far = {start_grid: 0}
        
        while pq:
            current_priority, current = heapq.heappop(pq)
            
            if current == goal_grid:
                break
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                next_node = (current[0] + dx, current[1] + dy)
                
                # Check if next node is within grid bounds
                if (0 <= next_node[0] < self.grid_size and 
                    0 <= next_node[1] < self.grid_size):
                    
                    # Check if cell is not occupied
                    if grid[next_node] < 0.7:  # Consider < 0.7 as free space
                        # Calculate movement cost (diagonal movement costs more)
                        if dx == 0 or dy == 0:
                            move_cost = 1.0
                        else:
                            move_cost = 1.414  # sqrt(2)
                        
                        new_cost = cost_so_far[current] + move_cost
                        
                        if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                            cost_so_far[next_node] = new_cost
                            priority = new_cost + heuristic(goal_grid, next_node)
                            heapq.heappush(pq, (priority, next_node))
                            came_from[next_node] = current
        
        # Reconstruct path
        path = []
        current = goal_grid
        while current != start_grid:
            path.append((
                current[0] * self.grid_resolution - self.grid_size * self.grid_resolution / 2,
                current[1] * self.grid_resolution - self.grid_size * self.grid_resolution / 2
            ))
            if current in came_from:
                current = came_from[current]
            else:
                # No path found
                return []
        
        path.append((
            start_grid[0] * self.grid_resolution - self.grid_size * self.grid_resolution / 2,
            start_grid[1] * self.grid_resolution - self.grid_size * self.grid_resolution / 2
        ))
        
        # Reverse the path to go from start to goal
        return path[::-1]
```

## Optimization for Humanoid Robots

### Handling Motion-Induced Blur

```python
class MotionCompensatedVO:
    def __init__(self, imu_threshold=0.5):
        self.imu_threshold = imu_threshold
        self.imu_data_buffer = deque(maxlen=5)
        
    def compensate_for_motion(self, image, imu_data):
        """
        Compensate for robot motion when capturing images
        """
        # Add current IMU data
        self.imu_data_buffer.append(imu_data)
        
        # Calculate recent motion statistics
        if len(self.imu_data_buffer) > 1:
            recent_angular_velocities = [data['angular_velocity'] for data in self.imu_data_buffer]
            avg_angular_vel = np.mean(recent_angular_velocities, axis=0)
            
            # Threshold to determine if motion compensation is needed
            motion_magnitude = np.linalg.norm(avg_angular_vel)
            
            if motion_magnitude > self.imu_threshold:
                # Apply motion compensation to image
                # This is a simplified approach - in practice, this would require
                # more sophisticated image processing techniques
                return self.apply_motion_compensation(image, avg_angular_vel)
        
        return image
    
    def apply_motion_compensation(self, image, angular_velocity):
        """
        Apply compensation based on angular velocity
        """
        # Calculate compensation matrix based on angular velocity
        # This is a simplified representation
        dt = 1.0 / 30.0  # Assuming 30 FPS
        
        # Calculate rotation due to angular velocity
        rotation = angular_velocity * dt
        
        # Create transformation matrix for compensation
        # Simplified for demonstration - real implementation would be more complex
        R_comp = np.array([
            [1, -rotation[2], rotation[1]],
            [rotation[2], 1, -rotation[0]],
            [-rotation[1], rotation[0], 1]
        ])
        
        # Apply transformation to image (for demonstration, using simple warp)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Convert rotation matrix to warp matrix
        M = np.eye(3)
        M[:2, :2] = R_comp[:2, :2]
        M[:2, 2] = -R_comp[:2, :2] @ np.array(center) + center
        
        # Apply warp
        compensated_image = cv2.warpPerspective(
            image, M[:2], (w, h), 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return compensated_image
```

## ROS 2 Integration

### VSLAM Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster

class VSLAMNode(Node):
    def __init__(self):
        super().__init__('vslam_node')
        
        # Initialize VSLAM components
        self.visual_odometry = VisualOdometry(focal_length=500, principal_point=(320, 240))
        self.keyframe_mapper = KeyframeMapper()
        self.map_manager = MapManager()
        self.navigator = VSLAMNavigator(self.map_manager)
        self.motion_compensator = MotionCompensatedVO()
        
        # ROS 2 components
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'vslam/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, 'vslam/map', 10)
        
        # Internal state
        self.prev_image = None
        self.imu_data = None
        
        self.get_logger().info('VSLAM Node initialized')
    
    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Apply motion compensation if IMU data is available
            if self.imu_data:
                cv_image = self.motion_compensator.compensate_for_motion(cv_image, self.imu_data)
            
            # Perform visual odometry
            current_pose = self.visual_odometry.estimate_motion(cv_image)
            
            # Check if we should add a keyframe
            if self.keyframe_mapper.should_add_keyframe(current_pose):
                # Extract features for the new keyframe
                gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                kp, desc = self.visual_odometry.feature_tracker.detector.detectAndCompute(gray_img, None)
                
                features = {'keypoints': kp, 'descriptors': desc}
                self.keyframe_mapper.add_keyframe(cv_image, current_pose, features)
                
                # Update map manager with new keyframe
                self.map_manager.keyframes[len(self.map_manager.keyframes)] = {
                    'pose': current_pose,
                    'camera_matrix': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
                }
            
            # Publish estimated pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = current_pose[0, 3]
            pose_msg.pose.position.y = current_pose[1, 3]
            pose_msg.pose.position.z = current_pose[2, 3]
            
            # Convert rotation matrix to quaternion
            rotation = current_pose[:3, :3]
            qw = np.sqrt(1 + rotation[0,0] + rotation[1,1] + rotation[2,2]) / 2
            qx = (rotation[2,1] - rotation[1,2]) / (4 * qw)
            qy = (rotation[0,2] - rotation[2,0]) / (4 * qw)
            qz = (rotation[1,0] - rotation[0,1]) / (4 * qw)
            
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            
            self.pose_pub.publish(pose_msg)
            
            # Broadcast transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'camera'
            t.transform.translation.x = current_pose[0, 3]
            t.transform.translation.y = current_pose[1, 3]
            t.transform.translation.z = current_pose[2, 3]
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def imu_callback(self, msg):
        """Process IMU data for motion compensation"""
        self.imu_data = {
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        }

def main(args=None):
    rclpy.init(args=args)
    node = VSLAMNode()
    
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

## Performance Optimization

### Multi-threading for Real-time Performance

```python
import threading
import queue
from collections import deque

class ThreadedVSLAM:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = True
        
        # Initialize VSLAM components
        self.visual_odometry = VisualOdometry(focal_length=500, principal_point=(320, 240))
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()
    
    def start_processing(self):
        """Start the frame processing loop"""
        self.processing_thread.start()
    
    def process_frames(self):
        """Process frames in a separate thread"""
        while self.running:
            try:
                # Get frame from input queue
                frame = self.input_queue.get(timeout=1.0)
                
                # Process with VSLAM
                pose = self.visual_odometry.estimate_motion(frame)
                
                # Put result in result queue
                self.result_queue.put(pose)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in processing thread: {e}')
    
    def submit_frame(self, frame):
        """Submit a frame for processing"""
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame if queue is full
            pass
    
    def get_latest_pose(self):
        """Get the most recent pose estimate"""
        latest_pose = None
        try:
            # Try to get all available poses and keep the latest
            while True:
                latest_pose = self.result_queue.get_nowait()
        except queue.Empty:
            pass
        
        return latest_pose
    
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.processing_thread.join()
```

## Practical Exercise: Implementing a Basic VSLAM Pipeline

### Exercise Objective

Implement a basic VSLAM pipeline with visual odometry and mapping components, then integrate it with ROS 2.

### Prerequisites

- Basic understanding of computer vision
- ROS 2 environment
- Camera sensor
- Understanding of 3D transformations

### Exercise Steps

1. **Set up the development environment**:
   ```bash
   # Create workspace
   mkdir -p ~/vslam_ws/src
   cd ~/vslam_ws/src
   git clone [your_robot_name]_vslam_pkg
   cd ~/vslam_ws
   colcon build
   source install/setup.bash
   ```

2. **Implement the Visual Odometry component**:
   ```python
   # Create visual_odometry.py
   from vslam_components import FeatureTracker, VisualOdometry
   
   # Test the visual odometry with sample images
   vo = VisualOdometry(focal_length=500, principal_point=(320, 240))
   
   # Load sample images (you'll need a sequence of images for testing)
   import cv2
   img1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
   img2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)
   
   # Estimate motion
   pose1 = vo.estimate_motion(img1)
   pose2 = vo.estimate_motion(img2)
   
   print(f"Estimated poses: {pose1}, {pose2}")
   ```

3. **Add mapping functionality**:
   ```python
   # Test keyframe-based mapping
   from vslam_components import KeyframeMapper
   
   mapper = KeyframeMapper()
   
   # Simulate adding keyframes over time
   for i in range(10):
       # Simulate pose (in practice, this would come from VO)
       pose = np.eye(4)
       pose[0, 3] = i * 0.2  # Move 20cm forward each step
       
       if mapper.should_add_keyframe(pose):
           # For this example, we'll use a simple feature placeholder
           dummy_features = {'keypoints': [], 'descriptors': []}
           mapper.add_keyframe(f"image_{i}", pose, dummy_features)
   
   print(f"Added {len(mapper.keyframes)} keyframes")
   ```

4. **Integrate with ROS 2**:
   ```python
   # Create launch file for VSLAM node
   # vslam_launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   
   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='[your_robot_name]_vslam_pkg',
               executable='vslam_node',
               name='vslam_node',
               parameters=[
                   {'camera_topic': '/camera/image_raw'},
                   {'imu_topic': '/imu/data'},
                   {'publish_rate': 30.0}
               ],
               output='screen'
           )
       ])
   ```

5. **Test with simulation**:
   ```bash
   # Launch the robot simulation
   ros2 launch [your_robot_name]_bringup simulation.launch.py
   
   # In another terminal, launch VSLAM
   ros2 launch [your_robot_name]_vslam_pkg vslam_launch.py
   
   # Monitor the output
   ros2 topic echo /vslam/pose
   ```

6. **Evaluate performance**:
   - Check estimation accuracy against ground truth if available
   - Monitor computational performance (FPS, CPU usage)
   - Test in various lighting conditions and environments

## Summary

In this chapter, we've covered the essential elements of building a VSLAM perception pipeline for humanoid robots:

1. **Visual Odometry**: Feature detection, matching, and pose estimation
2. **Mapping**: Keyframe-based mapping and map management
3. **Integration**: Connecting VSLAM with navigation systems
4. **Optimization**: Handling humanoid-specific challenges and performance optimization
5. **ROS 2 Integration**: Implementing the pipeline within the ROS 2 framework

VSLAM systems enable humanoid robots to operate autonomously in unknown environments by providing them with spatial awareness and the ability to navigate effectively. The combination of visual perception, mapping, and navigation creates a foundation for more advanced autonomous capabilities.