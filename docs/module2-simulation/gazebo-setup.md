# Gazebo Simulation Environment Setup

## Overview

Gazebo is a powerful 3D simulation tool that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. In this module, we'll set up Gazebo to simulate our humanoid robot in various environments and scenarios.

## System Requirements

Before installing Gazebo, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 22.04 LTS (required for ROS 2 Iron compatibility)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Dedicated GPU with OpenGL 3.3+ support (required for rendering)
- **CPU**: Multi-core processor with good performance
- **Storage**: 10GB free space minimum

## Installation Steps

### 1. Install System Dependencies

First, update your package list and install required dependencies:

```bash
sudo apt update
sudo apt install wget gnupg lsb-release
```

### 2. Add Gazebo Repository

Add the Gazebo repository to your system:

```bash
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
```

### 3. Install Gazebo Signing Key

```bash
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```

### 4. Install Gazebo Garden

At the time of writing, Gazebo Garden is the recommended version to use with ROS 2 Iron:

```bash
sudo apt update
sudo apt install gazebo
```

### 5. Install ROS 2 Gazebo Packages

Install the ROS 2 packages for Gazebo integration:

```bash
sudo apt install ros-iron-gazebo-ros ros-iron-gazebo-plugins ros-iron-gazebo-dev
```

### 6. Install Additional Simulation Tools

Install additional tools that will be useful for humanoid simulation:

```bash
sudo apt install ros-iron-joint-state-publisher ros-iron-robot-state-publisher ros-iron-xacro
```

## Verification

### 1. Test Gazebo Installation

Run Gazebo to verify it's properly installed:

```bash
gazebo
```

You should see the Gazebo GUI with a default empty world. If the GUI doesn't start properly, you may have GPU issues which are covered in the troubleshooting section.

### 2. Test ROS 2 Integration

In a separate terminal, verify that ROS 2 can communicate with Gazebo:

```bash
source /opt/ros/iron/setup.bash
gz service --service /gazebo/resource_paths/get
```

This command should return the list of resource paths used by Gazebo.

### 3. Launch Simple Gazebo with ROS 2

Test the basic integration by launching Gazebo through ROS 2:

```bash
source /opt/ros/iron/setup.bash
ros2 launch gazebo_ros empty_world.launch.py
```

This should open Gazebo with an empty world and show ROS 2 topics being published.

## Configuration Files

### 1. Gazebo Environment Configuration

Create or edit your `.bashrc` file to set Gazebo environment variables:

```bash
# Add to ~/.bashrc
export GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:~/.gazebo/models
export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:~/.gazebo
```

### 2. Verify Configuration

After adding these to your `.bashrc`, source the file:

```bash
source ~/.bashrc
```

## Troubleshooting Common Issues

### 1. Gazebo GUI Does Not Launch

If Gazebo fails to start with a GUI error, especially on systems without a dedicated GPU:

```bash
# Try software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gazebo
```

### 2. GPU Issues

If you encounter rendering errors or crashes:

- Ensure your GPU drivers are up to date
- Install proper OpenGL libraries: `sudo apt install mesa-utils`
- Check OpenGL support: `glxinfo | grep "OpenGL version"`

### 3. ROS 2 Communication Issues

If ROS 2 nodes can't communicate with Gazebo:

- Ensure you're using compatible versions (ROS 2 Iron with Gazebo Garden)
- Check that both systems are using the same RMW implementation
- Verify that no other ROS 2 processes are interfering

## Performance Optimization

### 1. GPU Acceleration

For optimal performance with humanoid simulation:
- Use a dedicated GPU with good OpenGL support
- Ensure proper GPU drivers are installed
- Consider reducing visual quality for faster physics simulation

### 2. Real-time Performance

To maintain real-time simulation:
- Monitor CPU usage and adjust physics steps if needed
- Reduce the complexity of the simulated scene if performance is poor
- Consider using simplified meshes for collision detection

## Next Steps

Now that Gazebo is installed and configured, you'll next learn how to import your URDF robot model into Gazebo in the "URDF Import" section. This will allow you to simulate your humanoid robot with realistic physics in the Gazebo environment.

The Gazebo simulation environment provides the foundation for testing all the robot behaviors you'll develop in this book, making it a critical component of your development workflow.