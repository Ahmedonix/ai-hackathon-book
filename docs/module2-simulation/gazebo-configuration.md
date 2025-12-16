# Configuring Gazebo Simulation Environment on Ubuntu 22.04

## Overview

This guide provides detailed instructions for configuring your Gazebo simulation environment specifically on Ubuntu 22.04. Proper configuration is essential for achieving optimal performance and stability when simulating humanoid robots.

## Pre-Installation Configuration

### 1. System Preparation

Before installing Gazebo, ensure your Ubuntu 22.04 system is properly configured:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install build-essential cmake pkg-config -y
sudo apt install python3-dev python3-pip -y
```

### 2. Graphics Driver Configuration

For optimal Gazebo performance, ensure your graphics drivers are properly installed:

```bash
# Check current graphics configuration
lspci | grep -E "VGA|3D"

# For NVIDIA GPUs (recommended for Isaac Sim):
# Follow official NVIDIA driver installation instructions
sudo apt install nvidia-driver-535  # or latest stable version

# For AMD GPUs:
sudo apt install mesa-vulkan-drivers xserver-xorg-video-amdgpu

# For Intel GPUs:
sudo apt install mesa-vulkan-drivers intel-media-va-driver
```

After installing graphics drivers, reboot your system:

```bash
sudo reboot
```

## Gazebo Garden Installation and Configuration

### 1. Add OSRF Repository

```bash
# Install software properties common
sudo apt install software-properties-common -y

# Add the OSRF repository
sudo add-apt-repository "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main"

# Add the GPG key
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update package list
sudo apt update
```

### 2. Install Gazebo Garden

```bash
# Install Gazebo
sudo apt install gazebo -y

# Install additional tools
sudo apt install gz-tools* -y
```

### 3. Install ROS 2 Iron Gazebo Packages

```bash
sudo apt install ros-iron-gazebo-ros ros-iron-gazebo-plugins ros-iron-gazebo-dev -y
```

## Environment Configuration

### 1. Set Up Environment Variables

Add the following to your `~/.bashrc` file:

```bash
# Gazebo Environment Variables
export GZ_VERSION=garden
export GZ_SIM_RESOURCE_PATH=$HOME/.gazebo/worlds:/usr/share/gazebo/worlds
export GZ_SIM_MODEL_PATH=$HOME/.gazebo/models:/usr/share/gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-${GZ_VERSION}/plugins
export GZ_SIM_MEDIA_PATH=/usr/share/gazebo/media

# For compatibility during migration
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:${HOME}/.gazebo/models:/usr/share/gazebo/models
export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:${HOME}/.gazebo:${HOME}/.gazebo/worlds:/usr/share/gazebo/worlds
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/usr/lib/x86_64-linux-gnu/gazebo-${GZ_VERSION}/plugins
```

Apply the changes:

```bash
source ~/.bashrc
```

### 2. Create Gazebo Configuration Directory

```bash
mkdir -p ~/.gazebo
mkdir -p ~/.gazebo/models
mkdir -p ~/.gazebo/worlds
mkdir -p ~/.gazebo/plugins
```

### 3. Configure Gazebo Settings

Create or edit `~/.gazebo/config`:

```xml
<gazebo>
  <record_path>~/.gazebo/captures</record_path>
  <local_resource_path>~/.gazebo</local_resource_path>
  <save_on_exit>true</save_on_exit>
</gazebo>
```

## Performance Configuration

### 1. Graphics Configuration

For optimal rendering performance, configure your graphics settings:

```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"
glxinfo | grep "direct rendering"

# If direct rendering is not enabled, install additional packages:
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri -y
```

### 2. Physics Configuration

Optimize physics settings for humanoid simulation by creating a custom physics configuration in your world files:

```xml
<sdf version='1.7'>
  <world name='humanoid_world'>
    <!-- Physics engine configuration -->
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_contacts>20</max_contacts>
      
      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Your world content here -->
  </world>
</sdf>
```

### 3. System Optimization

For better performance with complex humanoid models:

```bash
# Increase shared memory size (important for large models)
echo "kernel.shmmax=134217728" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optional: Add your user to the dialout group for hardware access
sudo usermod -a -G dialout $USER
```

## ROS 2 Integration Configuration

### 1. Verify ROS 2 Gazebo Bridge

Test the ROS 2 Gazebo bridge functionality:

```bash
# Source ROS 2 Iron
source /opt/ros/iron/setup.bash

# Test basic Gazebo services
gz service --service /gazebo/worlds

# If using ROS 2 Iron, also verify the ROS 2 Gazebo plugins:
pkg-config --cflags --libs gazebo_ros
```

### 2. Test Gazebo-ROS Integration

Create a simple launch file to test the integration:

```python
# test_gazebo_launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Start Gazebo with an empty world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', 'empty.sdf'],
        output='screen'
    )

    return LaunchDescription([
        gazebo
    ])
```

Test the launch file:

```bash
source /opt/ros/iron/setup.bash
python3 test_gazebo_launch.py
```

## Troubleshooting Ubuntu 22.04 Specific Issues

### 1. Common Installation Issues

If you encounter issues during installation:

```bash
# Clean up any partial installations
sudo apt remove gazebo* gz-tools* --purge
sudo apt autoremove

# Clear APT cache
sudo apt clean
sudo apt autoclean

# Reinstall with verbose output for debugging
sudo apt install gazebo -y -v
```

### 2. Permission Issues

If you encounter permission errors:

```bash
# Ensure proper ownership of Gazebo directories
sudo chown -R $USER:$USER ~/.gazebo/
chmod -R 755 ~/.gazebo/
```

### 3. Rendering Issues

If you experience rendering issues on Ubuntu 22.04:

```bash
# Try running with software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gazebo

# Or try with Wayland disabled (if using Ubuntu desktop)
export GDK_BACKEND=x11
gazebo
```

## Verification

### 1. Complete System Check

Run a complete verification to ensure everything is configured properly:

```bash
# Check Gazebo installation
gz --version

# Check ROS 2 environment
source /opt/ros/iron/setup.bash
echo $AMENT_PREFIX_PATH

# Verify Gazebo can be launched
gz sim --headless-rendering --verbose

# Check environment variables
echo $GZ_SIM_RESOURCE_PATH
echo $GAZEBO_MODEL_PATH
```

### 2. Integration Test

Run a complete integration test with a simple model:

```bash
# Create a test world file
cat << EOF > ~/.gazebo/worlds/test_humanoid.world
<sdf version="1.7">
  <world name="test_humanoid">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
EOF

# Launch Gazebo with the test world
gz sim -r ~/.gazebo/worlds/test_humanoid.world
```

## Next Steps

With your Gazebo simulation environment properly configured on Ubuntu 22.04, you're now ready to import your URDF robot model into Gazebo. The next step is to learn how to import and configure your humanoid robot model in the simulation environment.

Your properly configured environment will provide a stable and performant foundation for all the simulation work you'll do in Module 2.