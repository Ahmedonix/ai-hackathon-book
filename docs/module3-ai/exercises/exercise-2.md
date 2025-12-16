---
title: Perception Pipeline Exercise
description: Practical hands-on exercise for creating perception pipelines in Isaac Sim
sidebar_position: 9
---

# Perception Pipeline Exercise

## Overview

In this hands-on exercise, you'll build a complete perception pipeline for humanoid robotics in Isaac Sim. You'll learn to create vision-based perception systems that allow robots to understand their environment, detect objects, and make decisions based on visual data.

## Learning Objectives

- Build a complete perception pipeline using Isaac Sim sensors
- Implement object detection and segmentation algorithms
- Create a semantic mapping system
- Integrate perception with robot control systems
- Evaluate perception pipeline performance

## Prerequisites

- Completed Isaac Sim setup exercise
- Understanding of basic computer vision concepts
- Familiarity with Python programming
- NVIDIA Isaac Sim installed and configured

## Exercise Setup

### Step 1: Initialize the Environment

First, let's set up a workspace for our perception pipeline:

```bash
# Create directory for perception exercises
mkdir -p ~/isaac_sim_projects/perception_exercises
cd ~/isaac_sim_projects/perception_exercises
```

### Step 2: Create the Base Scene

For this exercise, we'll create a scene with various objects that our perception system needs to detect:

```python
# perception_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.semantics import add_semantics
from omni.isaac.synthetic_utils import visualize
from pxr import Gf

def create_perception_scene():
    """Create a scene with various objects for perception testing"""
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Create ground plane
    ground_prim = create_prim(
        prim_path="/World/GroundPlane",
        prim_type="Plane",
        position=[0, 0, 0],
        scale=[10, 10, 1]
    )
    
    # Add semantic schema to ground
    add_semantics(ground_prim, "Ground")
    
    # Add lighting
    create_prim(
        prim_path="/World/DomeLight",
        prim_type="DomeLight",
        attributes={"color": (0.1, 0.1, 0.1), "intensity": 3000}
    )
    
    create_prim(
        prim_path="/World/KeyLight",
        prim_type="DistantLight",
        position=[5, 5, 10],
        attributes={"color": (1.0, 1.0, 0.95), "intensity": 400}
    )
    
    # Add various objects for perception
    # Cube object
    cube_prim = create_prim(
        prim_path="/World/Cube",
        prim_type="Cube",
        position=[2, 1, 0.5],
        scale=[0.5, 0.5, 0.5],
        color=[1.0, 0.0, 0.0]  # Red
    )
    add_semantics(cube_prim, "Obstacle")
    
    # Sphere object
    sphere_prim = create_prim(
        prim_path="/World/Sphere",
        prim_type="Sphere",
        position=[0, 2, 0.5],
        scale=[0.4, 0.4, 0.4],
        color=[0.0, 1.0, 0.0]  # Green
    )
    add_semantics(sphere_prim, "Target")
    
    # Cylinder object
    cylinder_prim = create_prim(
        prim_path="/World/Cylinder",
        prim_type="Cylinder",
        position=[1, -1, 0.5],
        scale=[0.3, 0.3, 1.0],
        color=[0.0, 0.0, 1.0]  # Blue
    )
    add_semantics(cylinder_prim, "Obstacle")
    
    # Add a humanoid robot
    # Note: Update this path to your actual humanoid robot USD file
    robot_prim = add_reference_to_stage(
        usd_path="path/to/your/humanoid_robot.usd",
        prim_path="/World/Humanoid"
    )
    
    # Set initial camera view
    set_camera_view(eye=[5, 5, 5], target=[0, 0, 1])
    
    print("Perception scene created with labeled objects")
    
    # Play the simulation briefly to see the scene
    world.play()
    for i in range(60):  # Run for 1 second at 60 FPS
        world.step(render=True)
    world.stop()
    
    return world

if __name__ == "__main__":
    create_perception_scene()
```

## Sensor Configuration

### Step 3: Add Advanced Sensors

Now, let's configure the sensors needed for our perception system:

```python
# perception_sensors.py
from omni.isaac.sensor import Camera
from omni.isaac.core.sensors import Imu
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.range_sensor import _range_sensor
import numpy as np

class PerceptionSensors:
    def __init__(self):
        self.rgb_camera = None
        self.depth_camera = None
        self.semantic_camera = None
        self.lidar = None
        self.imu = None
        
        # Initialize synthetic data helper for semantic segmentation
        self.sd_helper = SyntheticDataHelper(['rgb', 'depth', 'semantic'])
        
    def setup_cameras(self, robot_prim_path):
        """Set up RGB, depth, and semantic cameras"""
        
        # RGB camera for the robot
        self.rgb_camera = Camera(
            prim_path=f"{robot_prim_path}/head/rgb_camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.1, 0, 0.1],  # Slightly in front of head
        )
        
        # Depth camera
        self.depth_camera = Camera(
            prim_path=f"{robot_prim_path}/head/depth_camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.1, 0.05, 0.1],  # Slightly offset from RGB
        )
        
        # Semantic segmentation camera
        self.semantic_camera = Camera(
            prim_path=f"{robot_prim_path}/head/semantic_camera",
            frequency=30,
            resolution=(640, 480),
            position=[0.1, -0.05, 0.1],  # Slightly offset from RGB
        )
        
        # Enable semantic API for the semantic camera
        from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(self.semantic_camera._sensor.prim)
        
    def setup_lidar(self, robot_prim_path):
        """Set up LIDAR for 3D perception"""
        lidar_interface = _range_sensor.acquire_range_sensor_interface()
        
        # Create LIDAR sensor
        self.lidar = lidar_interface.new_lidar(
            prim_path=f"{robot_prim_path}/torso/lidar",
            translation=(0.0, 0.0, 0.7),  # Position at torso level
            orientation=(1.0, 0.0, 0.0, 0.0),
            name="robot_lidar",
            # LIDAR parameters
            m_sensor_horizontal_fov_min=-90,
            m_sensor_horizontal_fov_max=90,
            m_sensor_vertical_fov_min=-5,
            m_sensor_vertical_fov_max=5,
            m_sensor_horizontal_resolution=1,
            m_sensor_vertical_resolution=1,
            m_sensor_range_min=0.1,
            m_sensor_range_max=25.0
        )
    
    def setup_imu(self, robot_prim_path):
        """Set up IMU for motion detection"""
        self.imu = Imu(
            prim_path=f"{robot_prim_path}/torso/imu",
            frequency=100,
            position=[0, 0, 0.5]  # At torso level
        )
```

## Perception Pipeline Implementation

### Step 4: Implement Object Detection

Let's create a simple object detection module:

```python
# object_detector.py
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class ObjectDetector:
    def __init__(self, config=None):
        self.config = config or {}
        self.known_objects = {
            'red_cube': {'color': [255, 0, 0], 'label': 'Obstacle'},
            'green_sphere': {'color': [0, 255, 0], 'label': 'Target'},
            'blue_cylinder': {'color': [0, 0, 255], 'label': 'Obstacle'}
        }
        
    def detect_by_color(self, rgb_image, mask=None):
        """Simple color-based object detection"""
        if mask is not None:
            rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        detections = []
        
        # For each known object type
        for obj_name, obj_info in self.known_objects.items():
            # Convert object color to HSV
            obj_color_bgr = [obj_info['color'][2], obj_info['color'][1], obj_info['color'][0]]
            obj_hsv = cv2.cvtColor(np.uint8([[obj_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Define color range (adjust tolerance as needed)
            lower = np.array([obj_hsv[0] - 10, 50, 50])
            upper = np.array([obj_hsv[0] + 10, 255, 255])
            
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detection = {
                        'class': obj_info['label'],
                        'confidence': 0.9,  # For this simple detector
                        'bbox': [x, y, x+w, y+h],
                        'center': [x + w//2, y + h//2],
                        'area': area
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_by_template_matching(self, rgb_image, template_images):
        """Object detection using template matching"""
        detections = []
        
        for obj_name, template in template_images.items():
            # Perform template matching
            result = cv2.matchTemplate(rgb_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where matching exceeds threshold
            locations = np.where(result >= 0.7)  # Threshold
            
            for pt in zip(*locations[::-1]):
                detection = {
                    'class': obj_name,
                    'confidence': float(result[pt[1], pt[0]]),
                    'bbox': [pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]],
                    'center': [pt[0] + template.shape[1]//2, pt[1] + template.shape[0]//2]
                }
                detections.append(detection)
        
        return detections
```

### Step 5: Implement Semantic Segmentation

```python
# semantic_segmenter.py
import numpy as np
from scipy import ndimage
import cv2

class SemanticSegmenter:
    def __init__(self):
        self.semantic_labels = {
            0: 'background',
            1: 'ground',
            2: 'obstacle',
            3: 'target',
            4: 'robot'
        }
        
    def process_semantic_image(self, semantic_data):
        """Process semantic segmentation data from Isaac Sim"""
        # Semantic data comes as a 2D array with unique IDs for each object
        # We'll map these to meaningful labels
        
        # Create segmentation mask
        segmentation_mask = np.zeros_like(semantic_data, dtype=np.uint8)
        
        # Map semantic IDs to labels
        unique_ids = np.unique(semantic_data)
        
        for unique_id in unique_ids:
            if unique_id in [1, 2, 3]:  # Ground, obstacle, or target
                segmentation_mask[semantic_data == unique_id] = unique_id
        
        return segmentation_mask
    
    def get_object_masks(self, semantic_data):
        """Extract individual object masks from semantic data"""
        unique_ids = np.unique(semantic_data)
        object_masks = {}
        
        for uid in unique_ids:
            if uid != 0:  # Skip background
                mask = semantic_data == uid
                object_masks[uid] = mask
        
        return object_masks
    
    def get_object_bounding_boxes(self, object_masks):
        """Compute bounding boxes for each object in the segmentation"""
        bboxes = {}
        
        for obj_id, mask in object_masks.items():
            # Find contours to get bounding boxes
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get the bounding box of the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                bboxes[obj_id] = {
                    'bbox': [x, y, x+w, y+h],
                    'center': [x + w//2, y + h//2],
                    'area': w * h
                }
        
        return bboxes
```

### Step 6: Create the Main Perception Pipeline

```python
# perception_pipeline.py
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: List[int]  # [x, y]
    area: float

class PerceptionPipeline:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.semantic_segmenter = SemanticSegmenter()
        
        # For tracking objects over time
        self.object_trackers = {}
        self.tracked_objects = {}
        
    def process_frame(self, rgb_image, depth_image, semantic_image, lidar_data):
        """
        Process a single frame through the perception pipeline
        """
        results = {}
        
        # 1. Object detection from RGB
        rgb_detections = self.object_detector.detect_by_color(rgb_image)
        results['rgb_detections'] = [Detection(**det) for det in rgb_detections]
        
        # 2. Semantic segmentation
        segmentation_mask = self.semantic_segmenter.process_semantic_image(semantic_image)
        object_masks = self.semantic_segmenter.get_object_masks(semantic_image)
        object_bboxes = self.semantic_segmenter.get_object_bounding_boxes(object_masks)
        
        results['segmentation'] = {
            'mask': segmentation_mask,
            'objects': object_bboxes
        }
        
        # 3. Depth processing - estimate distances
        if depth_image is not None:
            results['depth_analysis'] = self.analyze_depth(depth_image, object_bboxes)
        
        # 4. LIDAR processing
        if lidar_data is not None:
            results['lidar_analysis'] = self.analyze_lidar(lidar_data)
        
        # 5. Object tracking (simplified)
        results['tracked_objects'] = self.update_tracking(rgb_detections)
        
        return results
    
    def analyze_depth(self, depth_image, object_bboxes):
        """Analyze depth information for detected objects"""
        depth_analysis = {}
        
        for obj_id, bbox_data in object_bboxes.items():
            bbox = bbox_data['bbox']
            
            # Get depth values within the bounding box
            obj_depth = depth_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Calculate average depth (distance to object)
            valid_depths = obj_depth[obj_depth > 0]  # Filter invalid depth values
            avg_depth = np.mean(valid_depths) if len(valid_depths) > 0 else float('inf')
            
            depth_analysis[obj_id] = {
                'avg_distance': avg_depth,
                'min_distance': np.min(valid_depths) if len(valid_depths) > 0 else float('inf'),
                'max_distance': np.max(valid_depths) if len(valid_depths) > 0 else float('inf')
            }
        
        return depth_analysis
    
    def analyze_lidar(self, lidar_data):
        """Analyze LIDAR data for object detection"""
        # This is a simplified implementation
        # In a real system, you would implement more sophisticated clustering
        # and object detection algorithms
        
        # Detect obstacles from LIDAR
        min_distance = np.min(lidar_data) if lidar_data.size > 0 else float('inf')
        obstacle_detected = min_distance < 2.0  # Obstacle within 2 meters
        
        return {
            'min_distance': min_distance,
            'obstacle_detected': obstacle_detected,
            'free_directions': self.find_free_directions(lidar_data)
        }
    
    def find_free_directions(self, lidar_data):
        """Find directions with the most free space"""
        # Simple implementation: look for the direction with maximum range
        if len(lidar_data) == 0:
            return []
        
        # Divide LIDAR readings into sectors (e.g., 8 sectors = 45° each)
        num_sectors = 8
        sector_size = len(lidar_data) // num_sectors
        sectors = []
        
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = min(start_idx + sector_size, len(lidar_data))
            sector_avg = np.mean(lidar_data[start_idx:end_idx])
            sectors.append({
                'sector_id': i,
                'avg_distance': sector_avg,
                'start_angle': i * (360 / num_sectors),
                'end_angle': (i + 1) * (360 / num_sectors)
            })
        
        # Sort sectors by distance (largest first)
        sectors.sort(key=lambda x: x['avg_distance'], reverse=True)
        
        return sectors
    
    def update_tracking(self, new_detections):
        """Update object tracking"""
        # Simple tracking based on overlap of bounding boxes
        for det in new_detections:
            det_bbox = det['bbox']
            det_center = det['center']
            
            # Find if this detection matches any existing tracked object
            matched = False
            for obj_id, obj_info in self.tracked_objects.items():
                last_bbox = obj_info['last_detection']['bbox']
                
                # Check for overlap (IoU > 0.3)
                iou = self.calculate_iou(det_bbox, last_bbox)
                
                if iou > 0.3:
                    # Update tracked object
                    self.tracked_objects[obj_id]['last_detection'] = det
                    self.tracked_objects[obj_id]['trajectory'].append(det_center)
                    matched = True
                    break
            
            # If no match found, create a new tracked object
            if not matched:
                new_obj_id = len(self.tracked_objects)
                self.tracked_objects[new_obj_id] = {
                    'last_detection': det,
                    'trajectory': [det_center],
                    'type': det['class']
                }
        
        return self.tracked_objects
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas of both boxes
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0
        
        return iou
```

## Integration with Robot Control

### Step 7: Create Perception-to-Control Interface

```python
# perception_control_interface.py
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class PerceptionControlInterface:
    def __init__(self):
        self.safe_distance = 0.8  # meters
        self.target_approach_distance = 1.5  # meters for approaching targets
        
    def make_navigation_decision(self, perception_results):
        """
        Make navigation decisions based on perception results
        """
        # Get depth analysis for obstacles
        depth_analysis = perception_results.get('depth_analysis', {})
        lidar_analysis = perception_results.get('lidar_analysis', {})
        
        # Check for obstacles
        obstacles = []
        for obj_id, depth_info in depth_analysis.items():
            if depth_info['avg_distance'] < self.safe_distance:
                obstacles.append({
                    'id': obj_id,
                    'distance': depth_info['avg_distance']
                })
        
        # If obstacles are detected via LIDAR
        if lidar_analysis.get('obstacle_detected', False):
            # Find safest direction
            free_dirs = lidar_analysis.get('free_directions', [])
            if free_dirs:
                # Choose direction with maximum free space
                safest_dir = free_dirs[0]
                return self.avoid_obstacle(safest_dir)
        
        # Check for targets to approach
        detections = perception_results.get('rgb_detections', [])
        for det in detections:
            if det.class_name == 'Target' and det.confidence > 0.7:
                # Approach the target
                return self.approach_target(det)
        
        # Default: move forward carefully
        return self.move_forward()
    
    def avoid_obstacle(self, direction_info):
        """Generate command to avoid obstacle"""
        cmd_vel = Twist()
        
        # Turn toward the safest direction
        center_angle = (direction_info['start_angle'] + direction_info['end_angle']) / 2
        
        # Convert angle to angular velocity (-1 to 1)
        # Assuming forward direction is 0°
        angular_vel = np.radians(center_angle) / 2.0
        angular_vel = np.clip(angular_vel, -1.0, 1.0)
        
        cmd_vel.angular.z = angular_vel
        cmd_vel.linear.x = 0.2  # Move slowly when avoiding obstacles
        
        return cmd_vel
    
    def approach_target(self, target_detection):
        """Generate command to approach a target"""
        cmd_vel = Twist()
        
        # For simplicity, we'll move toward the center of the target
        # in the image. In practice, you'd use depth information to 
        # determine 3D position
        
        # Calculate how far the target is from the center of the image
        img_width = 640  # Assuming 640x480 image
        target_center_x = target_detection.center[0]
        center_deviation = (target_center_x - img_width/2) / (img_width/2)  # Normalize to [-1, 1]
        
        # Adjust angular velocity to align with target
        cmd_vel.angular.z = -center_deviation * 0.5  # Negative for correct direction
        
        # Move forward if centered on target
        if abs(center_deviation) < 0.2:
            cmd_vel.linear.x = 0.3
        else:
            cmd_vel.linear.x = 0.1  # Move slowly when adjusting orientation
        
        return cmd_vel
    
    def move_forward(self):
        """Generate command for default forward motion"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3
        cmd_vel.angular.z = 0.0
        return cmd_vel
```

## Complete Integration Example

### Step 8: Create the Complete Perception System

```python
# complete_perception_system.py
#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.articulations import Articulation
from perception_pipeline import PerceptionPipeline
from perception_control_interface import PerceptionControlInterface
from perception_sensors import PerceptionSensors

def run_perception_pipeline():
    """Run the complete perception pipeline in Isaac Sim"""
    
    print("Initializing perception pipeline...")
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Add the scene from our first script
    # Create ground plane
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.semantics import add_semantics
    
    create_prim(
        prim_path="/World/GroundPlane",
        prim_type="Plane",
        position=[0, 0, 0],
        scale=[10, 10, 1]
    )
    
    create_prim(
        prim_path="/World/DomeLight",
        prim_type="DomeLight",
        attributes={"color": (0.1, 0.1, 0.1), "intensity": 3000}
    )
    
    # Add sample objects
    cube_prim = create_prim(
        prim_path="/World/Cube",
        prim_type="Cube",
        position=[2, 1, 0.5],
        scale=[0.5, 0.5, 0.5],
        color=[1.0, 0.0, 0.0]
    )
    add_semantics(cube_prim, "Obstacle")
    
    sphere_prim = create_prim(
        prim_path="/World/Sphere",
        prim_type="Sphere",
        position=[0, 2, 0.5],
        scale=[0.4, 0.4, 0.4],
        color=[0.0, 1.0, 0.0]
    )
    add_semantics(sphere_prim, "Target")
    
    # Add humanoid robot
    add_reference_to_stage(
        usd_path="path/to/your/humanoid_robot.usd",
        prim_path="/World/Humanoid"
    )
    
    # Initialize perception components
    perception_pipeline = PerceptionPipeline()
    control_interface = PerceptionControlInterface()
    sensors = PerceptionSensors()
    
    # Set up sensors on the robot
    # Note: You need to make sure your robot has the required prim paths
    sensors.setup_cameras("/World/Humanoid")
    sensors.setup_lidar("/World/Humanoid")
    sensors.setup_imu("/World/Humanoid")
    
    # Add robot to world
    robot = world.scene.add(
        Articulation(
            prim_path="/World/Humanoid",
            name="humanoid_robot",
            position=[0, 0, 1.0]
        )
    )
    
    # Set camera view
    set_camera_view(eye=[5, 5, 5], target=[0, 0, 1])
    
    print("Starting perception pipeline...")
    
    # Play the simulation
    world.play()
    
    # Run perception pipeline for 100 steps
    for step in range(100):
        # Step the physics simulation
        world.step(render=True)
        
        # Get sensor data
        try:
            rgb_data = sensors.rgb_camera.get_rgb()
            depth_data = sensors.depth_camera.get_depth()
            semantic_data = sensors.semantic_camera.get_semantic_segmentation()
            lidar_data = sensors.lidar.get_linear_depth_data()
        except Exception as e:
            print(f"Error getting sensor data: {e}")
            continue
        
        if step % 30 == 0:  # Print status every second
            print(f"Processing frame {step}")
        
        # Process frame through perception pipeline
        perception_results = perception_pipeline.process_frame(
            rgb_data, depth_data, semantic_data, lidar_data
        )
        
        # Make navigation decision based on perception
        cmd_vel = control_interface.make_navigation_decision(perception_results)
        
        # In a real implementation, we would send cmd_vel to the robot
        # For this exercise, we'll just print the decision
        if step % 30 == 0:
            print(f"Navigation command: linear.x={cmd_vel.linear.x:.2f}, angular.z={cmd_vel.angular.z:.2f}")
        
        # Show processed image with detections
        if step % 10 == 0:  # Show every 10th frame
            vis_image = rgb_data.copy()
            
            # Draw detections on image
            for det in perception_results.get('rgb_detections', []):
                if det.confidence > 0.7:  # Only show confident detections
                    pt1 = (det.bbox[0], det.bbox[1])
                    pt2 = (det.bbox[2], det.bbox[3])
                    color = (0, 255, 0) if det.class_name == 'Target' else (0, 0, 255)
                    cv2.rectangle(vis_image, pt1, pt2, color, 2)
                    cv2.putText(vis_image, f"{det.class_name} {det.confidence:.2f}", 
                               (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display the image
            cv2.imshow('Perception Output', vis_image[:, :, [2, 1, 0]])  # Convert RGB to BGR for OpenCV
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cv2.destroyAllWindows()
    
    world.stop()
    print("Perception pipeline completed!")

if __name__ == "__main__":
    run_perception_pipeline()
```

## Exercise Tasks

### Step 9: Complete Implementation Tasks

Complete the following tasks to finish the perception pipeline exercise:

1. **Implement the Perception Pipeline**:
   - Create all the Python files we've defined
   - Connect them to your Isaac Sim humanoid robot
   - Ensure sensors are properly configured

2. **Test Object Detection**:
   - Add different colored objects to your scene
   - Verify that the color-based detector works
   - Test with both static and moving objects

3. **Integrate with Robot Control**:
   - Connect perception outputs to robot motion
   - Test obstacle avoidance behavior
   - Implement target approach behavior

4. **Evaluate Performance**:
   - Measure detection accuracy
   - Check real-time performance (FPS)
   - Verify that perception-guided navigation works

## Evaluation Metrics

### Step 10: Performance Evaluation

Implement metrics to evaluate your perception system:

```python
# perception_evaluation.py
import numpy as np
import matplotlib.pyplot as plt

class PerceptionEvaluator:
    def __init__(self):
        self.detection_accuracy = []
        self.processing_times = []
        self.false_positives = []
        self.missed_detections = []
        
    def evaluate_detection(self, detected_bboxes, ground_truth_bboxes, iou_threshold=0.5):
        """Evaluate detection performance against ground truth"""
        # Calculate true positives, false positives, false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        detected_copy = detected_bboxes.copy()
        
        # For each ground truth, find the best matching detection
        for gt_bbox in ground_truth_bboxes:
            best_iou = 0
            best_match_idx = -1
            
            for i, det_bbox in enumerate(detected_copy):
                iou = self.calculate_iou(gt_bbox, det_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match_idx = i
                    break
            
            if best_match_idx != -1:
                true_positives += 1
                detected_copy.pop(best_match_idx)  # Remove matched detection
            else:
                false_negatives += 1
        
        # Remaining detections are false positives
        false_positives = len(detected_copy)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0
        
        return iou
    
    def plot_performance(self):
        """Plot performance metrics over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot processing times
        ax1.plot(self.processing_times)
        ax1.set_title('Perception Processing Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Time (ms)')
        
        # Plot detection accuracy
        ax2.plot(self.detection_accuracy)
        ax2.set_title('Detection Accuracy Over Time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()

# Example usage:
def evaluate_perception_system():
    evaluator = PerceptionEvaluator()
    
    # Example ground truth and detections for evaluation
    ground_truth_bboxes = [
        [100, 100, 200, 200],  # [x1, y1, x2, y2]
        [300, 150, 400, 250]
    ]
    
    detected_bboxes = [
        [102, 105, 198, 195],
        [295, 148, 405, 255],
        [500, 300, 550, 350]  # False positive
    ]
    
    results = evaluator.evaluate_detection(detected_bboxes, ground_truth_bboxes)
    print(f"Evaluation Results: {results}")
```

## Challenge: Extend the Perception System

As an additional challenge, try to:

1. Add more sophisticated object detection using deep learning
2. Implement SLAM (Simultaneous Localization and Mapping)
3. Add more sensor types (thermal, multispectral)
4. Create a more complex navigation task

## Summary

In this exercise, you've built a complete perception pipeline for humanoid robots in Isaac Sim:

1. **Scene Setup**: Created a test environment with labeled objects
2. **Sensor Integration**: Configured RGB, depth, semantic, and LIDAR sensors
3. **Object Detection**: Implemented color-based and template-based detection
4. **Semantic Segmentation**: Processed semantic labels for object identification
5. **Perception Control**: Connected perception to robot navigation
6. **Evaluation**: Implemented metrics to assess performance

This perception system forms the foundation for more advanced autonomous behaviors in humanoid robots, enabling them to understand and interact with their environment effectively.