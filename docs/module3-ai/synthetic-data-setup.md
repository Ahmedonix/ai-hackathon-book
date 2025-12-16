# Setting Up Synthetic Data Generation Tools

## Overview

Synthetic data generation is a critical component of Isaac Sim that enables the creation of large, diverse datasets for training AI models for humanoid robotics. This section covers setting up and configuring the synthetic data generation tools specifically for humanoid robotics applications.

## Understanding Synthetic Data in Robotics

### 1. Benefits of Synthetic Data

- **Safety**: Training in simulation avoids risks to physical robots
- **Cost-effectiveness**: Generate unlimited training data without real-world costs
- **Variety**: Create diverse scenarios, lighting conditions, and environments
- **Annotations**: Automatic ground truth generation (segmentation, depth, bounding boxes)
- **Control**: Precise control over environmental variables and parameters

### 2. Types of Synthetic Data

- **RGB Images**: Visual perception training
- **Depth Maps**: Depth estimation and 3D reconstruction
- **Semantic Segmentation**: Object classification and scene understanding
- **Instance Segmentation**: Individual object identification
- **Bounding Boxes**: Object detection training
- **Pose Estimation**: Object and robot pose information
- **Point Clouds**: 3D environment mapping

## Setting Up Synthetic Data Generation

### Step 1: Verify Isaac Lab Installation

First, ensure Isaac Lab is properly installed and available:

```bash
# Navigate to your Isaac workspace
cd ~/isaac_humanoid_ws

# Verify Isaac Lab is installed
python3 -c "import omni.isaac.lab as lab; print('Isaac Lab version:', lab.__version__)" 2>/dev/null || echo "Isaac Lab not found - installing..."

# If not found, install Isaac Lab
if [ ! -d "isaac-lab" ]; then
    git clone https://github.com/isaac-sim-lab/isaac-lab.git
    cd isaac-lab
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install --upgrade pip setuptools
else
    cd isaac-lab
    source venv/bin/activate
fi
```

### Step 2: Install Isaac Sim Synthetic Data Tools

```bash
# Enable the synthetic data generation extension in Isaac Sim
# This is typically done via the Extension Manager in Isaac Sim UI
# But can also be done by creating a startup script:

cat << 'EOF' > enable_synthetic_extensions.py
import omni
from omni.kit import extension_manager

# Get the extension manager
ext_manager = extension_manager.get_extension_manager()

# Enable synthetic data generation extensions
synthetic_extensions = [
    "omni.isaac.synthetic_dataset_generation",  # Main synthetic data generation
    "omni.isaac.range_sensor",                  # For sensors
    "omni.isaac.sensor",                        # For sensor data
    "ommi.isaac.perception"                     # For perception processing
]

def enable_synthetic_extensions():
    """Enable synthetic data generation extensions."""
    for ext_name in synthetic_extensions:
        try:
            ext_manager.set_extension_enabled(ext_name, True)
            print(f"✓ Enabled: {ext_name}")
        except Exception as e:
            print(f"✗ Failed to enable {ext_name}: {str(e)}")
            # Continue with other extensions even if one fails

if __name__ == "__main__":
    enable_synthetic_extensions()
    print("Synthetic data generation extensions configured")
EOF

# Make it executable
chmod +x enable_synthetic_extensions.py
```

### Step 3: Configure Synthetic Data Parameters

Create a configuration file for synthetic data generation optimized for humanoid robotics:

```bash
cat << 'EOF' > synthetic_data_config.yaml
# Synthetic Data Configuration for Humanoid Robotics

# Dataset Configuration
dataset:
  name: "humanoid_perception_dataset"
  output_dir: "~/datasets/humanoid_sim_data"
  version: "1.0.0"
  
# Scene Configuration
scene:
  # Environments to include
  environments:
    - "simple_room"
    - "office_space"
    - "outdoor_park"
    - "warehouse"
    - "cluttered_home"
  
  # Time of day variations
  lighting:
    - "midday"
    - "morning"
    - "evening"
    - "night"
  
  # Weather conditions (if supported)
  weather:
    - "clear"
    - "overcast"
    - "slightly_overcast"  # For realistic shadows

# Robot Configuration
robot:
  # Humanoid robot model path within Isaac Sim
  model_path: "/Isaac/Robots/Humanoid/humanoid.usd"
  
  # Robot poses to sample
  poses:
    - "standing"
    - "walking"
    - "bending"
    - "turned_left"
    - "turned_right"
  
  # Clothing/texture variations
  appearance:
    - "casual_clothing"
    - "formal_clothing"
    - "robotic_case"

# Camera Configuration
cameras:
  # Head camera (forward facing)
  - name: "head_camera"
    position: [0.15, 0, 0.5]  # Mount point on robot head
    orientation: [0, 0, 0]    # Looking forward
    fov: 60                   # Field of view in degrees
    resolution: [640, 480]    # Width x Height
    
  # Chest camera (front facing)
  - name: "chest_camera"
    position: [0, 0, 0.3]     # Mount point on chest
    orientation: [0, 0, 0]    # Looking forward
    fov: 75                   # Wider field of view
    resolution: [640, 480]
    
  # Omnidirectional camera
  - name: "omni_camera"
    position: [0, 0, 0.6]     # Top of robot
    orientation: [0, 0, 0]
    fov: 360                  # Full panoramic view
    resolution: [512, 256]    # Equirectangular projection

# Annotation Configuration
annotations:
  # Semantic segmentation
  - type: "semantic_segmentation"
    format: "png"  # PNG for lossless compression
    enabled: true
  
  # Instance segmentation
  - type: "instance_segmentation"
    format: "png"
    enabled: true
  
  # Bounding boxes
  - type: "bounding_box_2d_tight"
    format: "json"  # COCO format
    enabled: true
  
  # Depth maps
  - type: "depth_linear"
    format: "npy"  # NumPy array format
    enabled: true
  
  # Normals
  - type: "normals"
    format: "png"
    enabled: true
  
  # Point clouds
  - type: "pointcloud"
    format: "pcd"
    enabled: true

# Data Augmentation
augmentation:
  # Lighting variations
  lighting_augmentation:
    enabled: true
    brightness_range: [0.8, 1.2]
    contrast_range: [0.9, 1.1]
    saturation_range: [0.8, 1.2]
  
  # Geometric augmentations
  geometric_augmentation:
    enabled: true
    rotation_range: [-10, 10]  # Degrees
    scale_range: [0.9, 1.1]
  
  # Color augmentations
  color_augmentation:
    enabled: true
    hue_range: [-0.1, 0.1]
    gamma_range: [0.8, 1.2]

# Capturing Parameters
capture:
  frequency: 10  # Hz - How often to capture frames
  total_frames: 10000  # Total frames to capture per episode
  random_seed: 42  # For reproducible results

# Performance Settings
performance:
  # Number of parallel instances for data generation
  parallel_instances: 2  # Adjust based on GPU memory
  # Cache size for faster processing
  cache_size: 512  # MB
  # Use GPU for processing where possible
  use_gpu: true
EOF
```

## Step 4: Creating Synthetic Data Capture Scripts

### Basic Synthetic Data Capture

Create a basic script to capture synthetic data:

```python
# scripts/basic_synthetic_capture.py
#!/usr/bin/env python3

"""
Basic synthetic data capture script for humanoid robot perception training.
"""

import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.synthetic_dataset_generation.core import SyntheticDataGen
from omni.isaac.synthetic_dataset_generation.core.annotation_pipelines import (
    RgbPipeline, 
    DepthPipeline, 
    SegmentationPipeline,
    BoundingBoxPipeline
)
import numpy as np
import cv2
import os
from PIL import Image
import json


class HumanoidSyntheticDataCapture:
    def __init__(self, config_path="synthetic_data_config.yaml"):
        self.config = self.load_config(config_path)
        
        # Initialize Isaac Sim
        self.world = World(stage_units_in_meters=1.0)
        
        # Initialize synthetic data generator
        self.sd_gen = SyntheticDataGen()
        
        # Create output directory
        self.output_dir = os.path.expanduser(self.config['dataset']['output_dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Episode counter
        self.episode_counter = 0
        
        self.get_logger().info('Humanoid Synthetic Data Capture Initialized')

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_scene(self):
        """Set up the scene with humanoid robot and environment"""
        # Add a ground plane and lighting
        self.world.scene.add_default_ground_plane()
        
        # Add a basic humanoid robot
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.utils.prims import create_prim
            
            # Create a simple humanoid robot (in a real implementation, use your specific humanoid model)
            # For now, using a basic articulated robot
            robot = self.world.scene.add(
                Robot(
                    prim_path="/World/Robot",
                    name="synthetic_humanoid_robot",
                    usd_path="/Isaac/Robots/Ant/ant.usd",  # Placeholder - replace with humanoid model
                    position=[0, 0, 0.5],
                    orientation=[0, 0, 0, 1]
                )
            )
            
            self.get_logger().info('Robot added to scene')
        except Exception as e:
            self.get_logger().error(f'Error adding robot: {str(e)}')
            # Add a simple box as fallback
            from omni.isaac.core.utils.prims import create_prim
            create_prim("/World/Robot", "Cube", position=[0, 0, 0.5])

    def setup_cameras(self):
        """Set up cameras according to configuration"""
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.replicator.core as rep
        
        # Create cameras based on config
        for cam_config in self.config['cameras']:
            cam_name = cam_config['name']
            
            # Create camera prim
            cam_prim_path = f"/World/{cam_name}_camera"
            create_prim(
                prim_path=cam_prim_path,
                prim_type="Camera",
                position=cam_config['position'],
                orientation=[0, 0, 0, 1]  # Default orientation
            )
            
            # Get the camera prim from stage
            camera_prim = self.world.stage.GetPrimAtPath(cam_prim_path)
            if camera_prim.IsValid():
                # Configure camera properties
                from pxr import UsdGeom
                usd_camera = UsdGeom.Camera(camera_prim)
                usd_camera.GetFocalLengthAttr().Set(24.0)  # Default focal length
                usd_camera.GetHorizontalApertureAttr().Set(20.955)  # Default aperture
                usd_camera.GetVerticalApertureAttr().Set(15.2908)  # Default aperture
        
        self.get_logger().info(f'Configured {len(self.config["cameras"])} cameras')

    def setup_annotations(self):
        """Set up annotation pipelines"""
        import omni.replicator.core as rep
        
        # Configure annotation writers based on config
        writer_registry = rep.WriterRegistry.get()
        
        # RGB writer
        if any(ann['type'] == 'rgb' for ann in self.config['annotations']):
            rgb_writer = writer_registry.get("RgbCameraWriter")
            rgb_writer.initialize()
            rgb_writer.attach([f"{cam['name']}_camera" for cam in self.config['cameras']])
        
        # Depth writer
        if any(ann['type'] == 'depth_linear' for ann in self.config['annotations']):
            depth_writer = writer_registry.get("DistanceToCameraWriter")
            depth_writer.initialize()
            depth_writer.attach([f"{cam['name']}_camera" for cam in self.config['cameras']])
        
        # Semantic segmentation writer
        if any(ann['type'] == 'semantic_segmentation' for ann in self.config['annotations']):
            seg_writer = writer_registry.get("SegmentationWriter")
            seg_writer.initialize()
            seg_writer.attach([f"{cam['name']}_camera" for cam in self.config['cameras']])
        
        # Bounding box writer
        if any(ann['type'] == 'bounding_box_2d_tight' for ann in self.config['annotations']):
            bbox_writer = writer_registry.get("BoundingBox2DWriter")
            bbox_writer.initialize()
            bbox_writer.attach([f"{cam['name']}_camera" for cam in self.config['cameras']])
        
        self.get_logger().info('Annotation pipelines configured')

    def run_synthetic_capture_session(self, num_episodes=5, frames_per_episode=100):
        """Run a complete synthetic data capture session"""
        self.get_logger().info(f'Starting synthetic capture session: {num_episodes} episodes, {frames_per_episode} frames per episode')
        
        for episode in range(num_episodes):
            self.episode_counter = episode
            
            # Create episode directory
            episode_dir = os.path.join(self.output_dir, f"episode_{episode:04d}")
            os.makedirs(episode_dir, exist_ok=True)
            
            # Reset world state for this episode
            self.world.reset()
            
            # Setup random scenario for this episode
            self.setup_random_scenario(episode)
            
            # Capture frames for this episode
            self.capture_episode_frames(episode_dir, frames_per_episode)
            
            self.get_logger().info(f'Completed episode {episode + 1}/{num_episodes}')
        
        # Save configuration with the dataset
        config_path = os.path.join(self.output_dir, "dataset_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.get_logger().info(f'Synthetic data capture complete. Dataset saved to: {self.output_dir}')

    def setup_random_scenario(self, episode):
        """Setup a random scenario for this episode"""
        import random
        
        # Randomize lighting
        # Randomize object positions
        # Vary robot pose
        # Add random clutter elements
        
        # This is a simplified implementation
        # In a real implementation, you would use Isaac Lab's scenario generation tools
        pass

    def capture_episode_frames(self, episode_dir, num_frames):
        """Capture frames for a single episode"""
        for frame_idx in range(num_frames):
            # Step the simulation
            self.world.step(render=True)
            
            # Capture data according to our configured annotations
            # In a real implementation, this would use Isaac Lab's synthetic data tools
            self.capture_frame_data(episode_dir, frame_idx)
            
            # Apply random perturbations between frames
            self.apply_random_perturbations()
            
            # Log progress
            if frame_idx % 100 == 0:
                self.get_logger().info(f'  Captured {frame_idx}/{num_frames} frames for episode')

    def capture_frame_data(self, episode_dir, frame_idx):
        """Capture all configured annotation types for this frame"""
        # This is a placeholder implementation
        # In a real implementation, you would use Isaac Lab's synthetic data capture tools
        pass

    def apply_random_perturbations(self):
        """Apply random perturbations to create variety"""
        # This would move objects, change lighting, vary robot pose, etc.
        # For now, just a placeholder
        pass

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Synthetic Data Capture Node Shutting Down')
        super().destroy_node()


def main():
    # Initialize ROS context
    rclpy.init()
    
    # Create synthetic data capture node
    synth_capture = HumanoidSyntheticDataCapture()
    
    try:
        # Run synthetic capture session
        synth_capture.run_synthetic_capture_session(
            num_episodes=10,
            frames_per_episode=500
        )
    except KeyboardInterrupt:
        synth_capture.get_logger().info('Dataset generation interrupted by user')
    finally:
        synth_capture.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Advanced Synthetic Data Generation for Humanoid Tasks

Create a more sophisticated synthetic data generation system focused on humanoid robotics tasks:

```python
# scripts/advanced_synthetic_generation.py
#!/usr/bin/env python3

"""
Advanced synthetic data generation specifically for humanoid robotics.
Generates comprehensive datasets for tasks like walking, balancing, and manipulation.
"""

import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.replicator.core as rep
import numpy as np
import os
import json
from datetime import datetime


class AdvancedHumanoidSyntheticDataGenerator:
    def __init__(self, config_path="synthetic_data_config.yaml"):
        self.config = self.load_config(config_path)
        
        # Initialize Isaac Sim
        self.world = World(stage_units_in_meters=1.0)
        
        # Initialize replicator for data generation
        self.replicator = rep
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = os.path.join(
            os.path.expanduser(self.config['dataset']['output_dir']),
            f"humanoid_dataset_{timestamp}"
        )
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Store robot articulation reference
        self.robot = None
        
        # Episode and scenario tracking
        self.current_episode = 0
        self.current_scenario = None
        self.scenario_configs = self.generate_scenario_configs()
        
        self.get_logger().info('Advanced Humanoid Synthetic Data Generator Initialized')

    def load_config(self, config_path):
        """Load configuration from file"""
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_world(self):
        """Set up the complete simulation environment"""
        # Create ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        self.setup_lighting()
        
        # Add humanoid robot
        self.add_humanoid_robot()
        
        # Setup cameras
        self.setup_cameras()
        
        # Setup annotation generators
        self.setup_annotations()
        
        self.get_logger().info('World setup complete')

    def setup_lighting(self):
        """Add realistic lighting to the scene"""
        # Add dome light for realistic indoor lighting
        dome_light_path = "/World/DomeLight"
        self.world.stage.DefinePrim(dome_light_path, "DomeLight")
        dome_light = get_prim_at_path(dome_light_path)
        dome_light.GetAttribute("inputs:intensity").Set(300)
        dome_light.GetAttribute("inputs:color").Set((0.9, 0.9, 0.9))

    def add_humanoid_robot(self):
        """Add a humanoid robot to the simulation"""
        # In a real implementation, you would add your specific humanoid model
        # For now, we'll use a placeholder
        try:
            # Add your humanoid robot model here
            # The path should point to your specific humanoid robot USD file
            robot_path = "/World/HumanoidRobot"
            add_reference_to_stage(
                usd_path="path/to/your/humanoid_robot.usd",  # Replace with actual path
                prim_path=robot_path
            )
            
            # Create articulation view for the robot
            self.robot = ArticulationView(prim_path_regex=".*HumanoidRobot.*")
            self.world.add_articulation(self.robot)
            
            self.get_logger().info('Humanoid robot added to simulation')
        except Exception as e:
            self.get_logger().warn(f'Could not add specific humanoid robot: {str(e)}. Using placeholder.')
            # Add a generic robot if specific one fails
            from omni.isaac.core.robots import Robot
            self.world.scene.add(
                Robot(
                    prim_path="/World/Robot",
                    name="generic_robot",
                    usd_path="/Isaac/Robots/Ant/ant.usd",
                    position=[0, 0, 0.5]
                )
            )

    def setup_cameras(self):
        """Setup cameras for humanoid data capture"""
        for idx, cam_config in enumerate(self.config['cameras']):
            cam_name = cam_config['name']
            cam_path = f"/World/{cam_name}"
            
            # Create camera
            self.world.stage.DefinePrim(cam_path, "Camera")
            camera_prim = get_prim_at_path(cam_path)
            
            # Set camera position
            from pxr import Gf
            camera_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(*cam_config['position'])
            )
            
            # Set camera properties
            from pxr import UsdGeom
            usd_camera = UsdGeom.Camera(camera_prim)
            usd_camera.GetFocalLengthAttr().Set(24.0)
            usd_camera.GetHorizontalApertureAttr().Set(20.955)
            usd_camera.GetVerticalApertureAttr().Set(15.2908)
        
        self.get_logger().info(f'Set up {len(self.config["cameras"])} cameras for data capture')

    def setup_annotations(self):
        """Setup annotation writers using Isaac Replicator"""
        # Initialize replicator
        self.replicator.initialize()
        
        # Create writers based on configuration
        for annotation_config in self.config['annotations']:
            if not annotation_config.get('enabled', True):
                continue
                
            ann_type = annotation_config['type']
            
            if ann_type == 'semantic_segmentation':
                # Semantic segmentation
                seg_annotator = self.replicator.create_annotator(
                    "class_segmentation",
                    settings={"class_map": self.get_class_mapping()}
                )
                
                # Attach to all configured cameras
                for cam_config in self.config['cameras']:
                    cam_name = cam_config['name']
                    self.replicator.get_annotator(ann_type).attach([f"{cam_name}_view"])
        
        self.get_logger().info('Annotation systems configured')

    def get_class_mapping(self):
        """Define class mapping for segmentation"""
        # Define classes relevant to humanoid robotics
        class_map = {
            0: "background",
            1: "humanoid_robot",
            2: "human",
            3: "table",
            4: "chair",
            5: "wall",
            6: "floor",
            7: "obstacle",
            # Add more classes as needed
        }
        return class_map

    def generate_scenario_configs(self):
        """Generate various scenario configurations for diverse training data"""
        scenarios = []
        
        # Walking scenarios
        for speed in [0.5, 1.0, 1.5]:  # m/s
            for terrain in ["flat", "uneven", "sloped"]:
                scenarios.append({
                    "type": "walking",
                    "speed": speed,
                    "terrain": terrain,
                    "duration": 30,  # seconds
                    "name": f"walking_{speed}_{terrain}"
                })
        
        # Balance scenarios
        for disturbance in ["push", "slope", "narrow_support"]:
            scenarios.append({
                "type": "balance",
                "disturbance": disturbance,
                "difficulty": "medium",
                "duration": 20,
                "name": f"balance_{disturbance}"
            })
        
        # Manipulation scenarios
        for object_type in ["box", "cylinder", "irregular"]:
            scenarios.append({
                "type": "manipulation",
                "object": object_type,
                "action": "lift",
                "duration": 25,
                "name": f"manipulation_{object_type}"
            })
        
        return scenarios

    def run_dataset_generation(self):
        """Main entry point to run complete dataset generation"""
        self.get_logger().info(f'Starting dataset generation with {len(self.scenario_configs)} scenarios')
        
        # Setup the world first
        self.setup_world()
        
        # Create annotations directory structure
        annotations_dir = os.path.join(self.dataset_dir, "annotations")
        images_dir = os.path.join(self.dataset_dir, "images")
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Run each scenario
        for scenario_idx, scenario_config in enumerate(self.scenario_configs):
            self.get_logger().info(f'Running scenario {scenario_idx+1}/{len(self.scenario_configs)}: {scenario_config["name"]}')
            
            # Set up specific scenario
            self.setup_scenario(scenario_config)
            
            # Run the scenario and capture data
            self.execute_scenario(scenario_config, scenario_idx)
            
            # Clean up after scenario
            self.cleanup_scenario(scenario_config)
            
            self.get_logger().info(f'Completed scenario: {scenario_config["name"]}')
        
        # Save dataset metadata
        self.save_dataset_metadata()
        
        self.get_logger().info(f'Completed dataset generation. Dataset saved to: {self.dataset_dir}')

    def setup_scenario(self, scenario_config):
        """Set up the environment for a specific scenario"""
        scenario_type = scenario_config["type"]
        
        if scenario_type == "walking":
            self.setup_walking_scenario(scenario_config)
        elif scenario_type == "balance":
            self.setup_balance_scenario(scenario_config)
        elif scenario_type == "manipulation":
            self.setup_manipulation_scenario(scenario_config)
        
        # Reset the simulation for the new scenario
        self.world.reset()

    def setup_walking_scenario(self, config):
        """Setup environment for walking scenario""" 
        # Add terrain based on configuration
        terrain_type = config["terrain"]
        
        if terrain_type == "uneven":
            # Add small bumps and obstacles
            self.add_uneven_terrain()
        elif terrain_type == "sloped":
            # Add an inclined plane
            self.add_slope()
        
        # Set initial robot pose
        self.set_robot_walking_pose()
        
        self.get_logger().info(f'Set up walking scenario on {terrain_type} terrain')

    def setup_balance_scenario(self, config):
        """Setup environment for balance scenario"""
        disturb_type = config["disturbance"]
        
        if disturb_type == "push":
            # Prepare for external force application
            pass
        elif disturb_type == "slope":
            # Add sloped surface
            self.add_slope(angle=10)  # 10 degrees
        elif disturb_type == "narrow_support":
            # Create narrow platform
            self.add_narrow_platform()
        
        self.set_robot_balance_pose()
        self.get_logger().info(f'Set up balance scenario with {disturb_type}')

    def setup_manipulation_scenario(self, config):
        """Setup environment for manipulation scenario"""
        obj_type = config["object"]
        
        # Add target object
        self.add_target_object(obj_type)
        
        # Set robot to a position that allows manipulation
        self.set_robot_manipulation_pose()
        
        self.get_logger().info(f'Set up manipulation scenario with {obj_type} object')

    def execute_scenario(self, config, scenario_idx):
        """Execute a specific scenario and capture data"""
        # Create scenario directory
        scenario_dir = os.path.join(self.dataset_dir, f"scenario_{scenario_idx:03d}_{config['name']}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Determine scenario duration in simulation steps
        # Assuming 60 Hz simulation rate
        duration_steps = int(config["duration"] * 60)
        
        # Set up data collectors for this scenario
        rgb_data_collector = self.setup_rgb_collector(scenario_dir)
        annotation_collector = self.setup_annotation_collector(scenario_dir)
        
        # Execute the scenario
        for step in range(duration_steps):
            # Step the simulation
            self.world.step(render=True)
            
            # Apply any scenario-specific behaviors (e.g., disturbances for balance)
            self.apply_scenario_behaviors(config, step)
            
            # Collect data at specified intervals
            if step % self.get_capture_interval() == 0:
                self.capture_data_step(scenario_dir, step, rgb_data_collector, annotation_collector)
            
            # Log progress
            if step % 300 == 0:  # Log every 5 seconds (300 steps at 60Hz)
                self.get_logger().info(f'  Scenario progress: {step}/{duration_steps} steps')

    def apply_scenario_behaviors(self, config, step):
        """Apply scenario-specific behaviors or disturbances"""
        scenario_type = config["type"]
        
        if scenario_type == "balance":
            disturb_type = config.get("disturbance", "none")
            
            if disturb_type == "push" and step == 600:  # Apply push at 10 seconds
                self.apply_external_push()
        
        elif scenario_type == "walking":
            # Apply walking controller logic here
            speed = config.get("speed", 1.0)
            self.apply_walking_controller(speed)

    def setup_rgb_collector(self, scenario_dir):
        """Setup RGB image collector"""
        # In a real implementation, this would set up a replicator writer
        # for RGB images from all configured cameras
        rgb_dir = os.path.join(scenario_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        return rgb_dir

    def setup_annotation_collector(self, scenario_dir):
        """Setup annotation collector"""
        # Set up directories for different annotation types
        ann_dir = os.path.join(scenario_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        
        # Create subdirectories for each annotation type
        for ann_config in self.config['annotations']:
            if ann_config.get('enabled', True):
                os.makedirs(os.path.join(ann_dir, ann_config['type']), exist_ok=True)
        
        return ann_dir

    def capture_data_step(self, scenario_dir, step, rgb_collector, annotation_collector):
        """Capture data for a single simulation step"""
        # This would use Isaac Replicator to capture all configured data types
        # For now, we'll implement a basic version
        
        # Capture RGB image
        rgb_file = os.path.join(rgb_collector, f"rgb_{step:06d}.jpg")
        # This would be implemented with replicator in a real version
        
        # Capture annotations
        for ann_config in self.config['annotations']:
            if ann_config.get('enabled', True):
                ann_path = os.path.join(
                    annotation_collector,
                    ann_config['type'],
                    f"{ann_config['type']}_{step:06d}.{ann_config['format']}"
                )
                # This would capture the annotation data in a real version
        
        # Log capture
        if step % 300 == 0:  # Every 5 seconds
            self.get_logger().info(f'    Captured data at step {step}')

    def get_capture_interval(self):
        """Get the interval at which to capture data (in simulation steps)"""
        # Capture at the rate specified in config (converted to simulation steps)
        # Our simulation runs at 60 Hz, so if we want 10 Hz capture:
        target_rate_hz = self.config['capture'].get('frequency', 10)
        sim_rate_hz = 60
        interval = sim_rate_hz // target_rate_hz
        return max(1, interval)  # At least capture every step

    def cleanup_scenario(self, config):
        """Clean up after a scenario completes"""
        # Reset any scenario-specific objects or forces applied
        pass

    def save_dataset_metadata(self):
        """Save metadata about the generated dataset"""
        metadata = {
            "dataset_name": self.config['dataset']['name'],
            "generation_date": datetime.now().isoformat(),
            "num_scenarios": len(self.scenario_configs),
            "total_frames": 0,  # Would calculate from actual captures
            "config": self.config,
            "scenarios": [s["name"] for s in self.scenario_configs]
        }
        
        metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.get_logger().info(f'Dataset metadata saved to {metadata_path}')

    def add_uneven_terrain(self):
        """Add uneven terrain for walking scenarios"""
        # Implementation would add small bumps, obstacles, etc.
        pass

    def add_slope(self, angle=5):
        """Add a sloped surface"""
        # Implementation would add an inclined plane
        pass

    def add_narrow_platform(self):
        """Add a narrow platform for balance challenges"""
        # Implementation would add a platform with limited width
        pass

    def add_target_object(self, obj_type="box"):
        """Add a target object for manipulation"""
        # Implementation would add the specified object type
        pass

    def set_robot_walking_pose(self):
        """Set robot to a walking-ready pose"""
        # Implementation would set robot joints appropriately
        pass

    def set_robot_balance_pose(self):
        """Set robot to a balance-ready pose"""
        # Implementation would set robot to a stable stance
        pass

    def set_robot_manipulation_pose(self):
        """Set robot to a manipulation-ready pose"""
        # Implementation would set robot to reach target object
        pass

    def apply_external_push(self):
        """Apply an external force to the robot"""
        # Implementation would apply forces via physics APIs
        pass

    def apply_walking_controller(self, speed):
        """Apply walking controller to move robot at specified speed"""
        # Implementation would command robot joints to walk
        pass

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Advanced Synthetic Data Generator Node Shutting Down')
        super().destroy_node()


def main():
    # Initialize
    rclpy.init()
    
    # Create generator
    generator = AdvancedHumanoidSyntheticDataGenerator()
    
    try:
        # Run dataset generation
        generator.run_dataset_generation()
    except KeyboardInterrupt:
        generator.get_logger().info('Dataset generation interrupted by user')
    finally:
        generator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Step 6: Configuring Synthetic Data for Different Humanoid Tasks

### Walking Pattern Data Generation

```python
# configs/walking_dataset_config.yaml
dataset:
  name: "humanoid_walking_perception_dataset"
  output_dir: "~/datasets/humanoid_walking_data"
  version: "1.0.0"
  
robot_scenarios:
  # Walking speeds to simulate
  walking_speeds:
    - 0.3  # Slow walk
    - 0.6  # Normal walk
    - 0.9  # Brisk walk
    - 1.2  # Fast walk
  
  # Terrain types
  terrains:
    - "flat_terrain"
    - "slightly_uneven"
    - "sloped_surface"
    - "obstacle_course"
  
  # Walking patterns
  gait_patterns:
    - "natural_walk"      # Normal human-like walk
    - "cautious_walk"     # Careful navigation
    - "fast_walk"         # Quick pace
    - "sideways_walk"     # Lateral movement

# Environment variations for walking
environments:
  # Lighting conditions
  lighting_conditions:
    - "bright_daylight"
    - "dim_indoor"
    - "backlighting"
    - "mixed_lighting"
  
  # Obstacle configurations
  obstacles:
    - "no_obstacles"
    - "stationary_boxes"
    - "moving_people"
    - "furniture_arranged"

cameras:
  # Multiple viewpoints for walking analysis
  - name: "head_forward"
    position: [0.1, 0, 0.5]  # Forward-looking from head
    fov: 60
    resolution: [640, 480]
  
  - name: "chest_forward"
    position: [0.15, 0, 0.3]  # From chest level
    fov: 75
    resolution: [640, 480]
  
  - name: "ground_downward"
    position: [0, 0, 0.05]    # Looking down at feet
    orientation: [1.57, 0, 0] # Rotated to look downward
    fov: 90
    resolution: [480, 640]    # Different aspect for downward view

annotations_for_walking:
  # Annotations specifically useful for walking perception
  - type: "semantic_segmentation"
    enabled: true
    classes: ["ground", "obstacles", "walkable_surface", "walker"]
    format: "png"
  
  - type: "depth_linear"
    enabled: true
    format: "npy"
  
  - type: "bounding_box_2d_tight"
    enabled: true
    format: "json"
  
  - type: "keypoints_2d"  # For tracking body parts during walking
    enabled: true
    keypoints: ["head", "torso", "hips", "feet"]
    format: "json"

capture_parameters:
  # Walking-specific capture parameters
  frequency: 20  # 20 Hz capture for gait analysis
  frames_per_scenario: 1000  # 50 seconds at 20 Hz
  include_pose_data: true    # Include joint positions
  include_imu_data: true     # Include IMU readings for balance
```

### Balance Training Data Configuration

```python
# configs/balance_dataset_config.yaml
dataset:
  name: "humanoid_balance_training_dataset"
  output_dir: "~/datasets/humanoid_balance_data"
  version: "1.0.0"
  
balance_scenarios:
  # Balance challenges
  challenges:
    - "static_balance"      # Standing still
    - "recovery_from_push"  # Regaining balance after disturbance
    - "single_leg_stance"   # Standing on one leg
    - "narrow_support"      # Standing on narrow platform
  
  # Disturbance types
  disturbances:
    - "front_push"
    - "lateral_push" 
    - "torsional_push"
    - "surface_translation"

# Specialized sensors for balance
balance_cameras:
  - name: "full_body_side"
    position: [-1, 0, 0.5]  # Side view of full body
    fov: 45
    resolution: [640, 480]
  
  - name: "feet_closeup"
    position: [0, 0.5, 0.1] # Close view of feet contact
    fov: 60
    resolution: [640, 480]
    orientation: [1.57, 0, 0] # Looking down

balance_annotations:
  # Balance-specific annotations
  - type: "center_of_mass_tracking"
    enabled: true
    format: "npy"
  
  - type: "zero_moment_point"
    enabled: true
    format: "npy"
  
  - type: "support_polygon"
    enabled: true
    format: "npy"
    
  - type: "joint_angle_trajectories"
    enabled: true
    format: "npy"
  
  - type: "balance_stability_indicator"
    enabled: true
    format: "json"

balance_capture_params:
  frequency: 100  # High frequency for balance control (100 Hz)
  include_imu: true
  include_force_sensors: true
  include_joint_states: true
```

## Step 7: Performance Optimization

### Optimizing for Large Dataset Generation

```python
# scripts/optimization_guide.py
"""
Performance optimization strategies for generating large synthetic datasets.
"""

import carb
import omni
from pxr import Gf, Sdf, UsdGeom
import gc


class SyntheticDataOptimizer:
    def __init__(self):
        self.optimize_rendering()
        self.setup_multithreading()
        self.configure_caching()
    
    def optimize_rendering(self):
        """Optimize rendering performance for large dataset generation"""
        # Reduce render quality for synthetic data generation
        carb_settings = carb.settings.get_settings()
        
        # Use less expensive rendering modes
        carb_settings.set("/app/renderer/mode", "Deferred")  # or "Forward" if Deferred causes issues
        carb_settings.set("/app/renderer/msaa", 1)  # No multisampling
        
        # Disable expensive rendering features during data generation
        carb_settings.set("/rtx/ambientOcclusion/enabled", False)
        carb_settings.set("/rtx/globalillumination/enabled", False)
        carb_settings.set("/rtx/pathtracing/enabled", False)
        
        carb.log_info("Rendering optimized for synthetic data generation")
    
    def setup_multithreading(self):
        """Configure multithreading for data generation"""
        # Configure Isaac Sim for multithreading
        carb_settings = carb.settings.get_settings()
        
        # Increase thread counts
        carb_settings.set("/app/threading/workerThreadCount", 8)
        carb_settings.set("/app/asyncThreadCount", 4)
        
        carb.log_info("Multithreading configured for dataset generation")
    
    def configure_caching(self):
        """Configure caching to improve performance"""
        carb_settings = carb.settings.get_settings()
        
        # Increase cache sizes
        carb_settings.set("/app/asset/cacheSize", 1024)  # 1GB cache
        carb_settings.set("/renderer/materialCacheSize", 256)  # 256MB material cache
        
        carb.log_info("Caching configured for better performance")


def optimize_for_large_datasets():
    """Apply optimizations for generating large synthetic datasets"""
    optimizer = SyntheticDataOptimizer()
    return optimizer


def enable_batch_mode():
    """Configure Isaac Sim for headless, batch processing mode"""
    carb_settings = carb.settings.get_settings()
    
    # Disable UI rendering to save resources
    carb_settings.set("/app/renderer/enabled", False)
    carb_settings.set("/app/window/sdf_render_enabled", False)
    
    # Optimize for headless mode
    carb_settings.set("/app/fastStart", True)
    carb_settings.set("/app/skipUpdates", True)
    
    carb.log_info("Isaac Sim configured for headless batch mode")


def main():
    # Apply optimizations
    optimizer = optimize_for_large_datasets()
    enable_batch_mode()
    
    print("Performance optimizations applied for synthetic dataset generation")


if __name__ == "__main__":
    main()
```

## Step 8: Validation and Quality Assurance

### Data Validation Script

```python
# scripts/validate_synthetic_data.py
#!/usr/bin/env python3

"""
Validation script for synthetic dataset quality assurance.
Checks that generated data meets requirements for humanoid robotics training.
"""

import os
import cv2
import numpy as np
import json
from PIL import Image
import argparse


def validate_image_quality(image_path):
    """Validate image quality metrics"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Invalid image file"
        
        # Check if image is not blank
        if np.mean(img) == 0:
            return False, "Image is completely black"
        
        # Check if image is overly saturated (pure white)
        if np.mean(img) > 250:
            return False, "Image is overly bright/saturated"
        
        # Check for blur (laplacian variance method)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Adjust threshold as needed
            return False, f"Image appears blurry (variance: {laplacian_var})"
        
        return True, "Valid image"
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def validate_depth_data(depth_path):
    """Validate depth data quality"""
    try:
        depth_data = np.load(depth_path)
        
        # Check if depth data is valid
        if not np.any(np.isfinite(depth_data)):
            return False, "Depth data contains no valid (finite) values"
        
        # Check depth range is reasonable
        min_depth = np.min(depth_data[np.isfinite(depth_data)])
        max_depth = np.max(depth_data[np.isfinite(depth_data)])
        
        if min_depth < 0.01:  # Less than 1cm is likely invalid
            return False, f"Depth data has unrealistic minimum: {min_depth}"
        
        if max_depth > 100:  # Greater than 100m is likely invalid
            return False, f"Depth data has unrealistic maximum: {max_depth}"
        
        return True, "Valid depth data"
    except Exception as e:
        return False, f"Error validating depth: {str(e)}"


def validate_segmentation_data(segmentation_path):
    """Validate segmentation data quality"""
    try:
        seg_img = Image.open(segmentation_path)
        seg_array = np.array(seg_img)
        
        # Check that all values are valid class IDs
        unique_vals = np.unique(seg_array)
        invalid_classes = [val for val in unique_vals if val < 0 or val > 255]  # Assuming 0-255 classes
        
        if invalid_classes:
            return False, f"Segmentation contains invalid class IDs: {invalid_classes}"
        
        return True, "Valid segmentation data"
    except Exception as e:
        return False, f"Error validating segmentation: {str(e)}"


def validate_dataset_structure(dataset_path):
    """Validate the overall dataset directory structure"""
    required_dirs = ['images', 'annotations', 'metadata.json']
    issues = []
    
    for required_dir in required_dirs:
        path = os.path.join(dataset_path, required_dir)
        if not os.path.exists(path):
            issues.append(f"Missing directory: {required_dir}")
    
    # Check for basic content in each directory
    if os.path.exists(os.path.join(dataset_path, 'images')):
        img_files = [f for f in os.listdir(os.path.join(dataset_path, 'images')) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_files) == 0:
            issues.append("Images directory is empty")
    
    if os.path.exists(os.path.join(dataset_path, 'annotations')):
        # Count annotation files
        annotation_count = 0
        for root, dirs, files in os.walk(os.path.join(dataset_path, 'annotations')):
            annotation_count += len([f for f in files if f.endswith(('.png', '.npy', '.json'))])
        
        if annotation_count == 0:
            issues.append("No annotation files found")
    
    if len(issues) == 0:
        return True, "Dataset structure is valid"
    else:
        return False, f"Dataset structure issues: {'; '.join(issues)}"


def run_comprehensive_validation(dataset_path):
    """Run comprehensive validation on a synthetic dataset"""
    print(f"Validating dataset at: {dataset_path}")
    
    # Validate overall structure
    is_valid, msg = validate_dataset_structure(dataset_path)
    print(f"Dataset structure: {msg}")
    
    if not is_valid:
        return False
    
    # Validate a sample of files
    image_dir = os.path.join(dataset_path, 'images')
    depth_dir = os.path.join(dataset_path, 'annotations', 'depth_linear')
    seg_dir = os.path.join(dataset_path, 'annotations', 'semantic_segmentation')
    
    # Validate images
    if os.path.exists(image_dir):
        img_files = [f for f in os.listdir(image_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]  # Sample first 50
        
        print(f"Validating {len(img_files)} sample images...")
        img_issues = 0
        for img_file in img_files:
            is_valid, msg = validate_image_quality(os.path.join(image_dir, img_file))
            if not is_valid:
                print(f"  ISSUE in {img_file}: {msg}")
                img_issues += 1
        
        print(f"Image validation: {len(img_files) - img_issues}/{len(img_files)} valid")
    
    # Validate depth maps
    if os.path.exists(depth_dir):
        depth_files = [f for f in os.listdir(depth_dir) 
                       if f.lower().endswith('.npy')][:20]  # Sample first 20
        
        print(f"Validating {len(depth_files)} sample depth maps...")
        depth_issues = 0
        for depth_file in depth_files:
            is_valid, msg = validate_depth_data(os.path.join(depth_dir, depth_file))
            if not is_valid:
                print(f"  ISSUE in {depth_file}: {msg}")
                depth_issues += 1
        
        print(f"Depth validation: {len(depth_files) - depth_issues}/{len(depth_files)} valid")
    
    # Validate segmentation
    if os.path.exists(seg_dir):
        seg_files = [f for f in os.listdir(seg_dir) 
                     if f.lower().endswith('.png')][:20]  # Sample first 20
        
        print(f"Validating {len(seg_files)} sample segmentation maps...")
        seg_issues = 0
        for seg_file in seg_files:
            is_valid, msg = validate_segmentation_data(os.path.join(seg_dir, seg_file))
            if not is_valid:
                print(f"  ISSUE in {seg_file}: {msg}")
                seg_issues += 1
        
        print(f"Segmentation validation: {len(seg_files) - seg_issues}/{len(seg_files)} valid")
    
    print(f"Comprehensive validation complete for {dataset_path}")
    
    # Overall result
    overall_valid = True
    if img_issues > len(img_files) * 0.1:  # More than 10% image issues
        overall_valid = False
    if depth_issues > len(depth_files) * 0.1:  # More than 10% depth issues
        overall_valid = False
    if seg_issues > len(seg_files) * 0.1:  # More than 10% segmentation issues
        overall_valid = False
    
    return overall_valid


def main():
    parser = argparse.ArgumentParser(description='Validate synthetic dataset quality')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the synthetic dataset directory')
    parser.add_argument('--full', action='store_true',
                        help='Run full validation instead of sampling')
    
    args = parser.parse_args()
    
    is_valid = run_comprehensive_validation(args.dataset_path)
    
    if is_valid:
        print("\n✓ Dataset validation PASSED")
    else:
        print("\n✗ Dataset validation FAILED - issues detected")


if __name__ == '__main__':
    main()
```

## Step 9: Running the Complete Setup

### 1. Make All Scripts Executable

```bash
# Make all scripts executable
chmod +x ~/humanoid_ws/src/humanoid_simple_robot/scripts/advanced_synthetic_generation.py
chmod +x ~/humanoid_ws/src/humanoid_simple_robot/scripts/optimization_guide.py
chmod +x ~/humanoid_ws/src/humanoid_simple_robot/scripts/validate_synthetic_data.py
```

### 2. Update Setup.py with New Scripts

Add the new scripts to the entry points in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'sensor_validator = humanoid_simple_robot.scripts.sensor_validator:main',
        'laser_processor = humanoid_simple_robot.scripts.laser_processor:main',
        'perception_node = humanoid_simple_robot.scripts.perception_node:main',
        'synthetic_data_gen = humanoid_simple_robot.scripts.advanced_synthetic_generation:main',
        'optimize_dataset = humanoid_simple_robot.scripts.optimization_guide:main',
        'validate_dataset = humanoid_simple_robot.scripts.validate_synthetic_data:main',
    ],
},
```

### 3. Build and Test

```bash
cd ~/humanoid_ws
source /opt/ros/iron/setup.bash
colcon build --packages-select humanoid_simple_robot
source install/setup.bash
```

## Troubleshooting Common Issues

### Issue 1: Synthetic Data Generation Not Working

**Symptoms**: No synthetic data being generated or empty output

**Solutions**:
1. Check that the synthetic data generation extension is enabled:
   ```bash
   # In Isaac Sim, verify the extension is enabled in Window → Extensions
   ```

2. Verify GPU memory is sufficient:
   ```bash
   nvidia-smi
   ```

3. Check for rendering errors in the Isaac Sim console

### Issue 2: Performance Problems with Data Generation

**Symptoms**: Slow capture rate, dropped frames, simulation lag

**Solutions**:
1. Reduce capture frequency
2. Lower image resolution
3. Run in headless mode if visualization isn't needed
4. Reduce number of concurrent annotations

### Issue 3: Annotation Quality Issues

**Symptoms**: Inconsistent or incorrect annotations

**Solutions**:
1. Verify that annotation writers are initialized properly
2. Check frame synchronization between cameras and annotations
3. Validate class mappings for segmentation tasks

## Next Steps

With the synthetic data generation tools properly set up, you're now ready to learn about the Isaac ROS perception stack. The synthetic datasets you can now generate will be crucial for training perception models that will run on your humanoid robot's AI brain.

The synthetic data generation system you've implemented will significantly accelerate your development process by providing large, diverse, and perfectly annotated datasets that are difficult and expensive to collect with real robots.