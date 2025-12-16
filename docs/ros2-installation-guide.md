# ROS 2 Iron Installation and Verification Guide

This guide provides instructions for installing and verifying ROS 2 Iron on your system. ROS 2 Iron (Irrwaddy) is the latest long-term support (LTS) release of ROS 2, which will be used throughout this book for building humanoid robot applications.

## System Requirements

Before installing ROS 2 Iron, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **RAM**: At least 8GB (16GB recommended)
- **Disk Space**: At least 5GB of free space
- **Python**: Version 3.10 or 3.11

## Installation on Ubuntu 22.04

Follow these steps to install ROS 2 Iron on Ubuntu 22.04:

1. **Set up the ROS 2 package repository**:
   ```bash
   sudo apt update && sudo apt install -y software-properties-common
   sudo add-apt-repository universe
   sudo apt update && sudo apt install curl -y
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
   
   # Add the repository to your sources list
   sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list'
   ```

2. **Install ROS 2 Iron packages**:
   ```bash
   sudo apt update
   sudo apt install ros-iron-desktop
   sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
   sudo rosdep init || true
   rosdep update
   ```

3. **Set up the ROS 2 environment**:
   ```bash
   source /opt/ros/iron/setup.bash
   # To make the environment permanent, add the following to your ~/.bashrc:
   echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
   ```

4. **Install additional packages needed for the book**:
   ```bash
   sudo apt install ros-iron-gazebo-ros ros-iron-gazebo-plugins ros-iron-gazebo-dev
   sudo apt install ros-iron-navigation2 ros-iron-nav2-bringup
   sudo apt install ros-iron-isaac-ros-*  # for perception pipelines
   ```

## Installation on Windows with WSL2

For Windows users, we recommend using Windows Subsystem for Linux (WSL2) with Ubuntu 22.04:

1. **Install WSL2 with Ubuntu 22.04**:
   - Open PowerShell as Administrator and run: `wsl --install -d Ubuntu-22.04`
   - Follow the prompts to complete the installation

2. **Follow the Ubuntu installation steps** from the previous section inside your WSL2 terminal

## Verification Steps

After installation, verify that ROS 2 Iron is correctly installed by running these commands:

1. **Check the ROS 2 version**:
   ```bash
   ros2 --version
   ```
   You should see output similar to `ros2 foxy` or the appropriate version.

2. **Source the ROS 2 environment**:
   ```bash
   source /opt/ros/iron/setup.bash
   ```

3. **Test basic ROS 2 functionality**:
   ```bash
   # In one terminal, run a publisher:
   source /opt/ros/iron/setup.bash
   ros2 run demo_nodes_cpp talker
   
   # In another terminal, run a subscriber:
   source /opt/ros/iron/setup.bash
   ros2 run demo_nodes_cpp listener
   ```
   You should see messages being published by the talker and received by the listener.

4. **Check available ROS 2 commands**:
   ```bash
   ros2
   ```
   This should display a list of available ROS 2 commands.

5. **Check available packages**:
   ```bash
   ros2 pkg list | grep -i ros
   ```

## Troubleshooting Common Issues

### Issue: Cannot find package `ros-iron-desktop`
**Solution**: Verify your Ubuntu version and that you've added the correct repository. Make sure your system clock is accurate, as repository verification requires correct time.

### Issue: `rosdep init` or `rosdep update` fails
**Solution**: This is often a network connectivity issue. Try:
   ```bash
   sudo apt install python3-rosdep
   sudo rosdep init
   rosdep update
   ```

### Issue: Command not found after sourcing setup.bash
**Solution**: Make sure you've added the source command to your `~/.bashrc` file and restarted your terminal or run `source ~/.bashrc`.

## Next Steps

Once you have successfully installed and verified your ROS 2 Iron installation, you can proceed with:

1. Creating your first ROS 2 workspace (covered in Module 1)
2. Running the example ROS 2 packages to understand the publisher-subscriber pattern
3. Moving on to Module 1 of this book to learn ROS 2 fundamentals

## Additional Resources

- [Official ROS 2 Iron Installation Guide](https://docs.ros.org/en/iron/Installation.html)
- [ROS 2 Tutorials](https://docs.ros.org/en/iron/Tutorials.html)
- [ROS 2 Index](https://index.ros.org/doc/ros2/)