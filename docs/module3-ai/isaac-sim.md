---
title: Isaac Sim Project Setup
description: Creating Isaac Sim projects with sensors and scenes for humanoid robotics
sidebar_position: 3
---

# Creating Isaac Sim Projects with Sensors and Scenes

## Overview

NVIDIA Isaac Sim is a powerful, GPU-accelerated simulation environment for robotics development. This chapter covers how to set up Isaac Sim projects specifically for humanoid robotics, including the creation of scenes, integration of various sensors, and preparation for AI training and testing.

## Learning Objectives

- Set up an Isaac Sim project with proper scene configuration
- Integrate various sensors for humanoid robots
- Create realistic environments for training and testing
- Understand the workflow for sim-to-real transfer
- Implement synthetic data generation pipelines

## Isaac Sim Introduction

Isaac Sim is built on NVIDIA's Omniverse platform and provides a physically accurate simulation environment for robotics. Key features include:

- PhysX physics engine for accurate simulation
- Physically-based rendering with RTX ray tracing
- Integration with NVIDIA's AI and simulation tools
- Support for synthetic data generation
- Direct integration with ROS 2 and ROS 1

## Installing and Setting Up Isaac Sim

### Prerequisites

- NVIDIA RTX GPU with 8GB+ VRAM (RTX 4070 Ti+ recommended)
- NVIDIA GPU driver supporting CUDA 11.8+
- Isaac Sim license (free for academic and personal use)
- Understanding of USD (Universal Scene Description) format

### First Launch and Configuration

1. **Download and install Isaac Sim**:
   - Visit NVIDIA Developer website and download Isaac Sim
   - Run the installer and follow the setup process
   - Activate your license

2. **Verify Installation**:
   ```bash
   # Check Isaac Sim installation
   cd /path/to/isaac_sim
   ./isaac-sim.sh
   ```

3. **Configure for Robotics**:
   - Open Isaac Sim
   - Go to Window → Extensions → Isaac Examples
   - Install robotics-specific extensions
   - Set up workspace for humanoid projects

## Creating Your First Scene

### Scene Structure Overview

Isaac Sim uses USD (Universal Scene Description) files to represent scenes. A typical humanoid robotics scene includes:

- Environment (floors, walls, obstacles)
- Humanoid robot model
- Sensors and actuators
- Lighting and materials
- Physics properties

### Creating a Basic Scene

1. **Open Isaac Sim** and create a new scene:
   - File → New Scene
   - Save as a new USD file (e.g., `humanoid_lab.usd`)

2. **Add basic environment**:
   ```python
   # Using Isaac Sim's Python API
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.core.utils.stage import add_reference_to_stage
   
   # Create a ground plane
   create_prim(
       prim_path="/World/ground_plane",
       prim_type="Plane",
       position=[0, 0, 0],
       scale=[10, 10, 1]
   )
   
   # Add a simple obstacle
   create_prim(
       prim_path="/World/obstacle",
       prim_type="Cube",
       position=[2, 0, 0.5],
       scale=[0.5, 0.5, 1.0]
   )
   ```

3. **Import or create humanoid model**:
   ```python
   # Import your humanoid robot model (URDF or USD)
   add_reference_to_stage(
       usd_path="path/to/humanoid.usd",  # or load URDF
       prim_path="/World/Humanoid"
   )
   ```

4. **Set up lighting**:
   ```python
   # Create dome light for realistic environment
   create_prim(
       prim_path="/World/DomeLight",
       prim_type="DomeLight",
       attributes={"color": (0.1, 0.1, 0.1), "intensity": 3000}
   )
   
   # Add key light
   create_prim(
       prim_path="/World/KeyLight",
       prim_type="DistantLight",
       position=[0, 0, 10],
       attributes={"color": (1.0, 1.0, 0.95), "intensity": 400}
   )
   ```

## Sensor Integration

### Camera Sensors

Camera sensors are essential for vision-based perception and navigation tasks.

```python
from omni.isaac.sensor import Camera

# Create RGB camera
camera = Camera(
    prim_path="/World/Humanoid/head/camera",
    frequency=30,  # Hz
    resolution=(640, 480)
)

# Get camera data
rgb_data = camera.get_rgb()
depth_data = camera.get_depth()
```

### LiDAR Sensors

LiDAR sensors provide accurate depth information for navigation and obstacle detection.

```python
from omni.isaac.range_sensor import _range_sensor

# Create LiDAR sensor
lidar_interface = _range_sensor.acquire_range_sensor_interface()
lidar_path = "/World/Humanoid/lidar"

# Create LiDAR sensor with parameters
lidar_config = {
    "rotation_frequency": 10,  # 10 Hz
    "rows": 32,                # Number of vertical beams
    "columns": 1024,           # Number of horizontal points
    "horizontal_fov": 360,     # Horizontal field of view in degrees
    "vertical_fov": 45,        # Vertical field of view in degrees
    "min_range": 0.1,          # Minimum range in meters
    "max_range": 25.0          # Maximum range in meters
}

# Apply configuration
range_sensor = lidar_interface.new_range_sensor(
    lidar_path,
    128,    # width
    128,    # height
    25.0,   # depth max distance
    0.1,    # depth min distance
    0.8,    # clipping range
    1.0,    # horizontal_fov
    1.0,    # vertical_fov
    30.0,   # velodyne rotation rate
    _range_sensor.RangeSensorPattern.LI_32_PATTERN,
    _range_sensor.RangeSensorPositionOnUsd.X,
    _range_sensor.RangeSensorPositionOnUsd.Z,
    0.0,    # horizontal offset
    1.0     # vertical offset
)
```

### IMU Sensors

IMU sensors provide orientation and acceleration data critical for balance control.

```python
from omni.isaac.core.sensors import Imu

# Create IMU sensor
imu_sensor = Imu(
    prim_path="/World/Humanoid/imu",
    frequency=100,  # 100 Hz
    position=[0, 0, 0.8],  # Position at CoM height
    orientation=[0, 0, 0, 1]  # Default orientation
)

# Get IMU data
linear_acceleration = imu_sensor.get_linear_acceleration()
angular_velocity = imu_sensor.get_angular_velocity()
orientation = imu_sensor.get_orientation()
```

### Force/Torque Sensors

Force/torque sensors are essential for interaction and manipulation tasks.

```python
from omni.isaac.core.utils.prims import get_prim_at_path

# Create force/torque sensor on joint
def create_force_torque_sensor(prim_path, joint_path):
    # Get the joint prim
    joint_prim = get_prim_at_path(joint_path)
    
    # Attach force/torque sensor
    # This would typically be done through USD properties
    joint_prim.GetAttribute("physics:enableCollision").Set(False)
    
    # The actual force/torque data would be obtained through physics callbacks
    # in the simulation loop
```

## Scene Creation for Training

### Randomization

For robust AI training, scenes should be randomized to improve generalization:

```python
import numpy as np
from omni.isaac.core.utils.prims import create_prim

def create_randomized_scene():
    """Create a randomized scene for training"""
    
    # Randomize lighting conditions
    light_intensity = np.random.uniform(1000, 5000)
    create_prim(
        prim_path="/World/DomeLight",
        prim_type="DomeLight",
        attributes={"color": (0.1, 0.1, 0.1), "intensity": light_intensity}
    )
    
    # Randomize objects
    for i in range(5):
        position = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5]
        scale = [np.random.uniform(0.3, 1.0)] * 3
        create_prim(
            prim_path=f"/World/RandomBox_{i}",
            prim_type="Cube",
            position=position,
            scale=scale
        )
    
    # Randomize floor texture
    # This would involve applying different materials to the ground plane
```

### Synthetic Data Generation

Isaac Sim excels at generating synthetic data for AI training:

```python
import cv2
import numpy as np
from omni.isaac.sensor import Camera

def generate_synthetic_dataset(camera, num_frames=1000):
    """Generate synthetic dataset for training vision models"""
    
    for i in range(num_frames):
        # Step simulation
        world.step(render=True)
        
        # Capture RGB image
        rgb = camera.get_rgb()
        
        # Capture semantic segmentation
        semantic = camera.get_semantic_segmentation()
        
        # Capture depth
        depth = camera.get_depth()
        
        # Save data
        cv2.imwrite(f"dataset/rgb_{i:05d}.png", rgb)
        cv2.imwrite(f"dataset/depth_{i:05d}.png", depth)
        
        # Save semantic segmentation with mappings
        np.save(f"dataset/semantic_{i:05d}.npy", semantic)
        
        # Add random robot pose for variety
        random_pos = [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0.8]
        humanoid.set_world_positon(random_pos)
```

## Advanced Scene Features

### Physics Configuration

Proper physics configuration is crucial for realistic humanoid simulation:

```python
from pxr import PhysicsSchemaTools
from omni.physx.scripts import physicsUtils

# Set physics properties for humanoid
def configure_humanoid_physics(prim_path):
    # Create rigid body for each link
    for link_name in ["torso", "left_leg", "right_leg", "left_arm", "right_arm", "head"]:
        link_path = f"{prim_path}/{link_name}"
        
        # Create rigid body
        physicsUtils.add_rigid_body(
            prim_path=link_path,
            mass=1.0,
            linear_damping=0.05,
            angular_damping=0.05
        )
        
        # Add collision shapes
        physicsUtils.add_sphere_geometry(
            prim_path=link_path,
            radius=0.1
        )
    
    # Configure joint properties
    for joint_name in ["hip_joint", "knee_joint", "ankle_joint"]:
        joint_path = f"{prim_path}/{joint_name}"
        
        # Configure joint limits, stiffness, damping
        # This would typically be loaded from URDF
```

### Animation and Motion Capture

For more realistic humanoid behavior, you can import motion capture data:

```python
from omni.anim import AnimUtils

def load_motion_capture_animation(usd_path, anim_file):
    """Load motion capture data to animate humanoid"""
    
    # Import animation data
    anim_data = AnimUtils.load_motion(anim_file)
    
    # Apply animation to humanoid joints
    for frame_idx, frame_data in enumerate(anim_data):
        for joint_name, joint_angles in frame_data.items():
            joint_path = f"/World/Humanoid/{joint_name}"
            # Set joint positions for this frame
            # This involves setting joint transforms at each frame
```

## Practical Exercise: Creating a Humanoid Training Scene

### Exercise Objective

Create a complete Isaac Sim scene with humanoid robot, sensors, and randomized environment for AI training.

### Prerequisites

- Isaac Sim installation
- Humanoid robot model in USD or URDF format
- Basic understanding of USD and Python scripting

### Exercise Steps

1. **Set up the workspace**:
   ```bash
   mkdir -p ~/isaac_sim_workspace/humanoid_training
   cd ~/isaac_sim_workspace/humanoid_training
   ```

2. **Create the scene script**:
   ```python
   # humanoid_scene.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.sensor import Camera
   from omni.isaac.range_sensor import _range_sensor
   from omni.isaac.core.sensors import Imu
   import numpy as np
   
   # Initialize world
   world = World(stage_units_in_meters=1.0)
   
   # Create ground plane
   create_prim(
       prim_path="/World/ground_plane",
       prim_type="Plane",
       position=[0, 0, 0],
       scale=[10, 10, 1]
   )
   
   # Add humanoid model
   add_reference_to_stage(
       usd_path="path/to/humanoid.usd",  # Replace with your model path
       prim_path="/World/Humanoid"
   )
   
   # Add lighting
   create_prim(
       prim_path="/World/DomeLight",
       prim_type="DomeLight",
       attributes={"color": (0.1, 0.1, 0.1), "intensity": 3000}
   )
   
   # Add camera sensor
   camera = Camera(
       prim_path="/World/Humanoid/head/camera",
       frequency=30,
       resolution=(640, 480)
   )
   
   # Add IMU sensor
   imu_sensor = Imu(
       prim_path="/World/Humanoid/imu",
       frequency=100,
       position=[0, 0, 0.8]
   )
   
   # Randomize environment
   for i in range(5):
       position = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5]
       scale = [np.random.uniform(0.3, 1.0)] * 3
       create_prim(
           prim_path=f"/World/Obstacle_{i}",
           prim_type="Cube",
           position=position,
           scale=scale
       )
   
   # Play the simulation
   world.play()
   
   # Run simulation for a period to test sensors
   for i in range(1000):
       world.step(render=True)
       
       # Check sensor data
       if i % 30 == 0:  # Every second (at 30 Hz)
           rgb_data = camera.get_rgb()
           print(f"Frame {i}: RGB shape {rgb_data.shape}")
   
   # Stop simulation
   world.stop()
   ```

3. **Run the simulation**:
   ```bash
   python3 humanoid_scene.py
   ```

4. **Customize the scene**:
   - Add more complex environments
   - Include different types of obstacles
   - Add dynamic elements (moving objects)
   - Implement more advanced sensor configurations

5. **Test the scene**:
   - Verify all sensors are providing data
   - Check physics behavior is realistic
   - Ensure the humanoid can be controlled properly

## Sim-to-Real Transfer Considerations

### Domain Randomization

To improve sim-to-real transfer, implement domain randomization:

```python
def apply_domain_randomization():
    """Apply various randomizations to improve sim-to-real transfer"""
    
    # Randomize physics properties
    physics_params = {
        "gravity_magnitude": np.random.uniform(9.7, 9.9),  # Earth's gravity varies slightly
        "ground_friction": np.random.uniform(0.5, 1.5),    # Different surface frictions
        "robot_mass_variance": np.random.uniform(0.95, 1.05)  # Small mass changes
    }
    
    # Randomize sensor noise
    sensor_params = {
        "camera_noise": np.random.uniform(0.001, 0.01),
        "imu_noise": [np.random.uniform(0.001, 0.01), np.random.uniform(0.001, 0.01), np.random.uniform(0.001, 0.01)],
        "lidar_noise": np.random.uniform(0.01, 0.1)
    }
    
    return physics_params, sensor_params
```

## Summary

In this chapter, we've covered the essential steps for creating Isaac Sim projects with sensors and scenes for humanoid robotics:

1. **Scene Setup**: Creating environments with proper physics and lighting
2. **Sensor Integration**: Adding cameras, LiDAR, IMU, and other sensors
3. **Synthetic Data Generation**: Creating datasets for AI training
4. **Advanced Features**: Physics configuration and animation
5. **Sim-to-Real Transfer**: Techniques to improve real-world performance

These capabilities make Isaac Sim a powerful tool for developing and testing humanoid robots before deployment in the real world. The combination of realistic physics simulation and sensor modeling enables comprehensive testing of robot algorithms in a safe, controlled environment.