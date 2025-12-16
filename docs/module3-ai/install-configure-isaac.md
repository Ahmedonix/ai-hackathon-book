# Installing and Configuring NVIDIA Isaac Sim Environment

## Overview

This section provides a detailed, step-by-step guide to installing and configuring the NVIDIA Isaac Sim environment specifically for humanoid robotics applications. We'll cover both the Omniverse Launcher approach and Docker-based installation, along with configuration for optimal performance.

## Prerequisites

Before installing Isaac Sim, ensure your system meets the following requirements:

### Hardware Prerequisites
- NVIDIA GPU with RTX 4070 Ti or better (or equivalent professional GPU)
- At least 32GB RAM (64GB recommended for complex scenes)
- Intel i7 / Ryzen 7 or better CPU
- At least 100GB available disk space
- Ubuntu 22.04 LTS

### Software Prerequisites
- NVIDIA proprietary drivers (version 535 or newer)
- CUDA toolkit (12.0 or newer)
- Python 3.10 (for ROS 2 Iron compatibility)
- Docker (if using container approach)

## Installation Methods

### Method 1: Omniverse Launcher Installation (Recommended)

This is the most straightforward approach, though it requires an NVIDIA Developer account.

#### Step 1: Install Omniverse Launcher

Download the Omniverse Launcher from the NVIDIA Developer portal:

```bash
# Download the Omniverse Launcher AppImage
cd ~/Downloads
wget https://developer.nvidia.com/omniverse-launcher-linux-x86_64

# Make executable
chmod +x omniverse-launcher-linux-x86_64.AppImage

# Run the installer
./omniverse-launcher-linux-x86_64.AppImage
```

This will install the Omniverse Launcher application to your system.

#### Step 2: Configure NVIDIA Developer Account

1. Launch the Omniverse Launcher
2. Sign in with your NVIDIA Developer account (register if you don't have one)
3. In the Launcher, go to the "Assets" tab
4. Search for "Isaac Sim"
5. Click "Add" to install Isaac Sim

#### Step 3: Initial Isaac Sim Configuration

Once installed, launch Isaac Sim and perform initial configuration:

```bash
# The Isaac Sim application should now be available in your applications
# Launch it from the application menu or command line:
~/.local/share/ov/pkg/isaac_sim-4.0.0/python.sh --exec "omni.isaac.kit" -- --exec-path=standalone
```

### Method 2: Docker Installation (Alternative)

For systems where the Omniverse Launcher approach doesn't work well, or for containerized deployments:

#### Step 1: Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
sudo apt update
sudo apt install docker.io

# Add your user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### Step 2: Pull Isaac Sim Docker Image

```bash
# Pull the Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Verify the image was pulled
docker images | grep isaac-sim
```

#### Step 3: Create Isaac Sim Launch Script

Create a script to launch Isaac Sim in Docker:

```bash
# Create a directory for Isaac Sim scripts
mkdir -p ~/isaac_sim_docker
cd ~/isaac_sim_docker

cat << 'EOF' > launch_isaac_sim_docker.sh
#!/bin/bash

# Isaac Sim Docker Launch Script
# For optimal performance, ensure X11 forwarding is properly configured

# Check if NVIDIA GPU is available
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected!"
    exit 1
fi

# Set environment variables
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics"

# Ensure X11 forwarding works if running with display
if [ -n "$DISPLAY" ]; then
    # Mount X11 Unix socket
    X11_MOUNT="--volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
    DISPLAY_ENV="--env=DISPLAY"
    XAUTH_FILE="$HOME/.Xauthority"
    
    # Copy Xauthority if it exists
    if [ -f "$XAUTH_FILE" ]; then
        XAUTH_MOUNT="--volume=$XAUTH_FILE:/root/.Xauthority:rw"
    fi
else
    # Running headless
    echo "Running Isaac Sim in headless mode"
    X11_MOUNT=""
    DISPLAY_ENV=""
    XAUTH_MOUNT=""
fi

# Create workspace directory if it doesn't exist
WORKSPACE_DIR="${HOME}/isaac_sim_workspace"
mkdir -p "${WORKSPACE_DIR}"

# Launch the container
docker run --rm -it \
    --gpus all \
    --network=host \
    $DISPLAY_ENV \
    $X11_MOUNT \
    $XAUTH_MOUNT \
    --volume="${WORKSPACE_DIR}:/isaac-sim/workspace:rw" \
    --volume="${HOME}/.nvidia-omniverse:/root/.nvidia-omniverse:rw" \
    nvcr.io/nvidia/isaac-sim:4.0.0
EOF

chmod +x launch_isaac_sim_docker.sh
```

#### Step 4: Run Isaac Sim in Docker

```bash
# First, configure X11 access (if running with display)
xhost +local:docker

# Launch Isaac Sim
./launch_isaac_sim_docker.sh
```

### Method 3: Manual Installation (For Advanced Users)

If you prefer a more hands-on approach or need to customize components:

#### Step 1: Install Isaac Sim Dependencies

```bash
# Install required system packages
sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-pip \
    libgl1-mesa-glx libgl1-mesa-dri libgomp1 \
    curl wget git unzip

# Install Python packages
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 2: Download Isaac Sim

```bash
# Create installation directory
mkdir -p ~/nvidia-isaac-sim
cd ~/nvidia-isaac-sim

# Download Isaac Sim (requires NVIDIA Developer account credentials)
# This would typically be done through the Omniverse Launcher
# But for direct download, visit https://developer.nvidia.com/isaac-sim
```

## Configuration for Humanoid Robotics

### Step 1: Create Isaac Sim Workspace

```bash
# Create workspace for humanoid robotics projects
mkdir -p ~/isaac_humanoid_ws
cd ~/isaac_humanoid_ws

# Create basic directory structure
mkdir -p {scenes,robots,assets,scripts,data}

# Create configuration file for humanoid robotics
cat << 'EOF' > humanoid_robot_config.py
"""Configuration file for Isaac Sim with humanoid robotics focus."""

import carb
import omni

# Physics settings optimized for humanoid robots
PHYSICS_SETTINGS = {
    "solver_type": "LGS",  # LGS for better stability with articulated figures
    "position_iterations": 16,  # Higher for humanoid stability
    "velocity_iterations": 8,   # Higher for humanoid stability
    "max_depenetration_velocity": 10.0,
    "friction_model": "cone_friction",
    "enable_gpu_physics": True,  # Enable if GPU computation available
    "gpu_max_particles": 10000,
    "gpu_max_triangle_count": 5000000,
    "gpu_max_contact_pairs": 1000000,
    "gpu_max_projections": 50,
    "gpu_max_corrections": 25,
    "gpu_max_refinements": 1,
    "gpu_max_separations": 1000
}

# Rendering settings for humanoid simulation
RENDERING_SETTINGS = {
    "render_resolution_width": 1280,
    "render_resolution_height": 720,
    "render_resolution_screen_scale": 0.8,
    "renderer": "RaytracedLightMap",  # For high-quality rendering
    "use_fxaa": True,
    "use_motion_blur": False,  # Disable for clearer perception
    "max_gpu_cache_size": 1024,  # MB
    "max_cpu_cache_percent": 10.0  # Percent of available RAM
}

# Synthetic data settings for humanoid robotics
SYNTHETIC_DATA_SETTINGS = {
    "enable_rgb": True,
    "enable_depth": True,
    "enable_segmentation": True,
    "enable_bounding_boxes": True,
    "capture_frequency": 30,  # Hz
    "data_format": "png",  # Format for captured images
    "annotation_format": "coco"  # Annotation format
}

def apply_humanoid_robotics_config():
    """Apply configuration settings optimized for humanoid robotics."""
    settings = carb.settings.get_settings()
    
    # Apply physics settings
    settings.set("/physics/solverType", PHYSICS_SETTINGS["solver_type"])
    settings.set("/physics/solverPositionIterationCount", PHYSICS_SETTINGS["position_iterations"])
    settings.set("/physics/solverVelocityIterationCount", PHYSICS_SETTINGS["velocity_iterations"])
    settings.set("/physics/maxDepenetrationVelocity", PHYSICS_SETTINGS["max_depenetration_velocity"])
    settings.set("/physics/frictionModel", PHYSICS_SETTINGS["friction_model"])
    
    # Apply rendering settings
    settings.set("/app/renderer/resolution/width", RENDERING_SETTINGS["render_resolution_width"])
    settings.set("/app/renderer/resolution/height", RENDERING_SETTINGS["render_resolution_height"])
    settings.set("/app/renderer/resolution/screenScale", RENDERING_SETTINGS["render_resolution_screen_scale"])
    settings.set("/app/renderer/enabled", True)
    settings.set("/app/renderer/msaa", 1)
    
    # Apply synthetic data settings (these might apply differently in Isaac Sim)
    carb.log_info("Applied Isaac Sim configuration for humanoid robotics")

if __name__ == "__main__":
    apply_humanoid_robotics_config()
EOF
```

### Step 2: Configure Isaac Sim for High-Performance Robotics

Create a custom Isaac Sim configuration profile for humanoid robotics:

```bash
# Create the configuration directory if using the standard installation
mkdir -p ~/.ov/pkg/isaac_sim-4.0.0/config

cat << 'EOF' > ~/.ov/pkg/isaac_sim-4.0.0/config/humanoid_robotics_config.json
{
  "app": {
    "name": "Isaac Sim - Humanoid Robotics",
    "renderer": "RaytracedLightMap",
    "width": 1280,
    "height": 720,
    "fps_limit": 60,
    "vsync": 0,
    "window_width": 1280,
    "window_height": 720
  },
  "exts": {
    "dotnetbrowser.enabled": false,
    "omni.kit.window.property": {
      "collapsed": true
    },
    "omni.kit.window.viewport": {
      "docked": true,
      "visible": true
    },
    "omni.isaac.ros2_bridge": {
      "enabled": true
    },
    "omni.isaac.synthetic_dataset_generation": {
      "enabled": true
    },
    "omni.isaac.range_sensor": {
      "enabled": true
    }, 
    "omni.isaac.sensor": {
      "enabled": true
    }
  },
  "settings": {
    "persistent": {
      "/app/renderer/enabled": true,
      "/app/window/sdf_render_enabled": true,
      "/rtx/fx_gamma": 2.2,
      "/rtx/ray_tracing": {
        "enable": true,
        "minGpuCount": 1
      },
      "/physics/solverType": 0,
      "/physics/solverPositionIterationCount": 16,
      "/physics/solverVelocityIterationCount": 8,
      "/physics/frictionModel": 1,
      "/app/performBackgroundUpdate": true,
      "/app/showDeveloperMode": true
    }
  }
}
EOF
```

### Step 3: Set Up Isaac Sim Extensions

Enable essential extensions for humanoid robotics development:

```bash
# Create an extensions configuration script
cat << 'EOF' > setup_extensions.py
import omni
from omni.kit import extension_manager

# Get the extension manager
ext_manager = extension_manager.get_extension_manager()

# List of essential extensions for humanoid robotics
essential_extensions = [
    "omni.isaac.ros2_bridge",
    "omni.isaac.synthetic_dataset_generation",
    "omni.isaac.range_sensor",
    "omni.isaac.sensor",
    "omni.isaac.perception",
    "omni.isaac.utils",
    "omni.graph.core",
    "omni.timeline"
]

def enable_extensions():
    """Enable essential Isaac Sim extensions for humanoid robotics."""
    for ext_name in essential_extensions:
        try:
            ext_manager.set_extension_enabled(ext_name, True)
            print(f"Enabled: {ext_name}")
        except Exception as e:
            print(f"Failed to enable {ext_name}: {str(e)}")

def disable_non_essential_extensions():
    """Disable extensions that may impact performance."""
    performance_impacting_extensions = [
        "omni.kit.window.content_browser",
        "omni.kit.window.file",
        "omni.kit.window.material_picker"
    ]
    
    for ext_name in performance_impacting_extensions:
        try:
            ext_manager.set_extension_enabled(ext_name, False)
            print(f"Disabled for performance: {ext_name}")
        except Exception as e:
            print(f"Failed to disable {ext_name}: {str(e)}")

if __name__ == "__main__":
    enable_extensions()
    disable_non_essential_extensions()
    print("Isaac Sim extensions configured for humanoid robotics")
EOF
```

## Optimizing System for Isaac Sim

### Step 1: GPU Memory Optimization

For optimal performance with humanoid robotics applications:

```bash
# Create a script to optimize GPU memory usage
cat << 'EOF' > optimize_gpu.sh
#!/bin/bash

# Isaac Sim GPU Memory Optimization Script

echo "Optimizing system for Isaac Sim with humanoid robotics..."

# Set GPU compute mode to allow multiple applications
sudo nvidia-smi -i 0 -c EXCLUSIVE_THREAD  # Or EXCLUSIVE_PROCESS

# Set persistence mode for consistent performance
sudo nvidia-smi -pm 1

# Check current GPU memory
echo "Current GPU memory status:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Create system configuration for Isaac Sim
echo '# Isaac Sim Configuration' | sudo tee -a /etc/environment
echo 'export NVIDIA_PRIME_RENDER_OFFLOAD=1' | sudo tee -a /etc/environment
echo 'export __NV_PRIME_RENDER_OFFLOAD=1' | sudo tee -a /etc/environment
echo 'export __VK_LAYER_NV_optimus=1' | sudo tee -a /etc/environment
echo 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json' | sudo tee -a /etc/environment

echo "GPU optimization complete. Please reboot for environment changes to take effect."
EOF

chmod +x optimize_gpu.sh
```

### Step 2: System Performance Tuning

```bash
# System tuning for Isaac Sim
cat << 'EOF' > system_tuning.sh
#!/bin/bash

# System performance tuning for Isaac Sim
echo "Applying system performance tuning for Isaac Sim..."

# Increase shared memory size (needed for large robot models)
echo "kernel.shmmax = 134217728" | sudo tee -a /etc/sysctl.conf
echo "kernel.shmall = 32768" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p

# Increase file watches (for Isaac Sim file monitoring)
echo "fs.inotify.max_user_watches = 524288" | sudo tee -a /etc/sysctl.conf
echo "fs.inotify.max_user_instances = 256" | sudo tee -a /etc/sysctl.conf

# Apply file watch changes
sudo sysctl -p

# Create systemd service to set real-time priority (optional)
cat << SYSTEMD_SERVICE | sudo tee /etc/systemd/system/isaac-sim.service
[Unit]
Description=Isaac Sim Optimized Runtime
After=graphical-session.target

[Service]
Type=simple
User=$USER
ExecStartPre=/bin/sleep 10
ExecStart=/usr/bin/env __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only vblank_mode=0 /home/$USER/.local/share/ov/pkg/isaac_sim-4.0.0/isaac-sim.glibc2.27.engine.linux-x86_64.cuda11.8.0.Release.AppImage
Environment=QT_QUICK_BACKEND=software
Restart=always

[Install]
WantedBy=graphical-session.target
SYSTEMD_SERVICE

echo "System tuning applied. Changes will take effect after reboot."
EOF

chmod +x system_tuning.sh
```

## Verification and Testing

### Step 1: Basic Functionality Test

Create a simple test script to verify Isaac Sim is properly installed and configured:

```python
# test_isaac_installation.py
import omni
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
import carb


def test_basic_functionality():
    """Test basic Isaac Sim functionality."""
    print("Testing Isaac Sim installation...")
    
    # Initialize world
    world = World(stage_units_in_meters=1.0)
    
    # Add a simple robot (using a basic articulated robot as example)
    try:
        # Create a simple robot - in a real implementation, you'd use your humanoid model
        world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="test_robot",
                usd_path="/Isaac/Robots/Ant/ant.usd",  # Use Ant as simple test robot
                position=[0, 0, 0.5],
                orientation=[0, 0, 0, 1]
            )
        )
        print("✓ Successfully added test robot")
    except Exception as e:
        print(f"✗ Failed to add test robot: {str(e)}")
        return False
    
    # Step the world to ensure physics initializes correctly
    try:
        world.reset()
        for _ in range(10):
            world.step(render=True)
        print("✓ Successfully stepped the physics simulation")
    except Exception as e:
        print(f"✗ Failed to step physics simulation: {str(e)}")
        return False
    
    print("✓ Isaac Sim basic functionality test PASSED")
    return True


def test_ros_bridge():
    """Test ROS bridge functionality."""
    print("\nTesting ROS 2 bridge...")
    
    try:
        # Try importing ROS bridge components
        from omni.isaac.ros2_bridge import ROS2Bridge
        
        print("✓ Successfully imported ROS 2 bridge components")
        
        # Test if ROS is available
        import rclpy
        print("✓ Successfully imported rclpy")
        
        return True
    except ImportError as e:
        print(f"✗ ROS 2 bridge test FAILED: {str(e)}")
        print("  This might be expected if ROS 2 is not sourced or Isaac ROS bridge is not installed properly")
        return False
    except Exception as e:
        print(f"✗ ROS 2 bridge test ERROR: {str(e)}")
        return False


def test_synthetic_data():
    """Test synthetic data generation capability."""
    print("\nTesting synthetic data generation...")
    
    try:
        # Try importing synthetic dataset generation components
        from omni.isaac.synthetic_dataset_generation.core import DatasetGenerator
        
        print("✓ Successfully imported synthetic data components")
        return True
    except ImportError:
        print("? Synthetic data components not available (this may be okay depending on installation)")
        return True  # Not critical for basic functionality
    except Exception as e:
        print(f"✗ Synthetic data test ERROR: {str(e)}")
        return False


def main():
    """Run all installation tests."""
    carb.log_info("Starting Isaac Sim installation verification")
    
    all_tests_passed = True
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test ROS bridge (optional but recommended)
    if not test_ros_bridge():
        print("! ROS bridge test failed, but continuing...")
    
    # Test synthetic data (optional but recommended)
    if not test_synthetic_data():
        print("! Synthetic data test failed, but continuing...")
    
    if all_tests_passed:
        print("\n✓ All tests PASSED - Isaac Sim is properly installed and configured!")
        print("You're ready to proceed with humanoid robotics simulation.")
    else:
        print("\n✗ Some tests FAILED - Please review the installation steps above.")
        print("Refer to the Isaac Sim documentation for troubleshooting.")


if __name__ == "__main__":
    main()
```

### Step 2: Running the Verification Test

```bash
# Run the installation verification
cd ~/isaac_humanoid_ws
python3 test_isaac_installation.py
```

## Troubleshooting Common Installation Issues

### Issue 1: Isaac Sim Won't Launch

**Symptoms**: Isaac Sim fails to start or crashes immediately

**Solutions**:
1. Check GPU driver installation:
   ```bash
   nvidia-smi
   glxinfo | grep "OpenGL renderer"
   ```

2. Verify sufficient disk space:
   ```bash
   df -h ~/.local/share/ov/pkg/
   ```

3. Check permissions:
   ```bash
   ls -la ~/.local/share/ov/pkg/isaac_sim-*
   ```

### Issue 2: GPU Memory Errors

**Symptoms**: Isaac Sim crashes with GPU memory errors

**Solutions**:
1. Reduce rendering resolution in Isaac Sim settings
2. Close other GPU-intensive applications
3. Update to the latest NVIDIA drivers

### Issue 3: ROS Bridge Not Working

**Symptoms**: Cannot connect Isaac Sim to ROS 2

**Solutions**:
1. Ensure ROS 2 is properly sourced:
   ```bash
   source /opt/ros/iron/setup.bash
   printenv | grep ROS
   ```

2. Check Isaac Sim extension:
   - Open Isaac Sim
   - Go to Window → Extensions
   - Search for "ROS 2 Bridge" and enable it

3. Verify Isaac ROS bridge installation:
   ```bash
   # Check if Isaac ROS bridge is installed
   dpkg -l | grep ros-iron-isaac-ros
   ```

## Performance Optimization for Humanoid Robotics

### Step 1: Physics Settings for Humanoid Models

Humanoid robots require special physics consideration for stable simulation:

```bash
# Create humanoid-specific physics configuration
cat << 'EOF' > ~/isaac_humanoid_ws/humanoid_physics_config.py
"""Physics configuration optimized for humanoid robotics in Isaac Sim."""

import carb
from omni.isaac.core.utils import nucleus

def apply_humanoid_physics_config():
    """Apply physics settings optimized for humanoid robots."""
    settings = carb.settings.get_settings()
    
    # Physics solver settings for articulated humanoids
    settings.set("/physics/solverType", "LGS")  # LGS solver for stability
    settings.set("/physics/solverPositionIterationCount", 16)  # Higher for humanoid stability
    settings.set("/physics/solverVelocityIterationCount", 8)   # Higher for humanoid stability
    
    # Friction settings for humanoid feet stability
    settings.set("/physics/frictionModel", "cone_friction")
    
    # Contact settings for accurate foot-ground contact
    settings.set("/physics/contactSurfaceLayer", 0.001)
    settings.set("/physics/defaultContactOffset", 0.001)
    settings.set("/physics/defaultRestOffset", 0.0)
    
    # Enable CCD for fast-moving humanoid parts
    settings.set("/physics/enableCCD", True)
    settings.set("/physics/maxCCDIterations", 10)
    
    # Mass scaling for humanoid proportions
    settings.set("/physics/globalMaxPenetration", 0.001)
    settings.set("/physics/globalMaxVelocity", 10.0)
    settings.set("/physics/globalMaxAngularVelocity", 64.0)
    
    carb.log_info("Applied physics configuration for humanoid robotics")


def setup_humanoid_robot_materials():
    """Set up appropriate physical materials for humanoid robots."""
    from omini.isaac.core.materials import PhysicsMaterial
    
    # Create materials with appropriate properties for humanoid robot parts
    robot_body_material = PhysicsMaterial(
        prim_path="/World/RobotBodyMaterial",
        static_friction=0.7,   # Higher for stable stance
        dynamic_friction=0.5,  # Sufficient for movement
        restitution=0.1        # Low bounce for stable contact
    )
    
    robot_foot_material = PhysicsMaterial(
        prim_path="/World/FootMaterial",
        static_friction=0.9,   # High friction for stable walking
        dynamic_friction=0.8,  # High dynamic friction for traction
        restitution=0.05       # Very low bounce
    )
    
    return robot_body_material, robot_foot_material


if __name__ == "__main__":
    apply_humanoid_physics_config()
    body_mat, foot_mat = setup_humanoid_robot_materials()
    print("Humanoid-specific physics configuration applied")
EOF
```

## Next Steps

With Isaac Sim successfully installed and configured, you're now ready to set up the synthetic data generation tools. The synthetic data generation capabilities will allow you to create large training datasets for your humanoid robot's perception and AI systems, which is essential for effective robot development in Module 3.

The installation process you've completed provides the foundation for all advanced perception and AI capabilities that will be implemented in the remainder of Module 3.