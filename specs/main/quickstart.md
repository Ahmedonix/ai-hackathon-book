# Quick Start Guide: Physical AI & Humanoid Robotics Book

## Overview
This guide provides a quick path to get started with the Physical AI & Humanoid Robotics book project. It covers the essential setup steps and the first module to help you begin your journey into humanoid robotics.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space minimum
- **GPU**: NVIDIA GPU with RTX 4070 Ti or better recommended (for Isaac Sim)
- **Processor**: Multi-core processor with good performance

### Software Requirements
- **ROS 2**: Iron Irrwaddy (installation guide in Module 1)
- **Docker**: For containerized development environments
- **Git**: For version control
- **Python 3.10 or 3.11**: Required for ROS 2 Iron
- **Node.js & npm**: For Docusaurus documentation site

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-hackathon-book.git
cd ai-hackathon-book
```

### 2. Install ROS 2 Iron (Ubuntu 22.04)
```bash
# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -

# Add ROS 2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Iron packages
sudo apt update
sudo apt install ros-iron-desktop
sudo apt install python3-colcon-common-extensions

# Source ROS 2 setup
source /opt/ros/iron/setup.bash
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
```

### 3. Set Up the Documentation Environment
```bash
# Navigate to the documentation directory
cd docs  # if it exists, or create one with Docusaurus

# Install Node.js if not already installed
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docusaurus
npm init docusaurus@latest ai-hackathon-book classic
cd ai-hackathon-book

# Install additional dependencies for robotics documentation
npm install @docusaurus/module-type-aliases @docusaurus/types
```

### 4. Set Up Simulation Environment
```bash
# Install Gazebo Garden (or compatible version)
sudo apt install ros-iron-gazebo-*
sudo apt install gazebo

# For Isaac Sim (optional, requires NVIDIA GPU)
# Follow NVIDIA's installation guide for Isaac Sim
# This typically involves downloading from NVIDIA Developer site
```

## Getting Started with Module 1: The Robotic Nervous System (ROS 2)

### 1. Create a ROS 2 Workspace
```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build workspace (initially empty)
colcon build --symlink-install
source install/setup.bash
```

### 2. Create Your First ROS 2 Package
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python humanoid_robot_bringup
```

### 3. Run the First Example
The book includes a simple publisher/subscriber example in Module 1. To run it:

```bash
# Navigate to your workspace
cd ~/ros2_ws
source install/setup.bash

# Run the publisher
ros2 run demo_nodes_py talker

# In a new terminal, run the subscriber
cd ~/ros2_ws
source install/setup.bash
ros2 run demo_nodes_py listener
```

## Running the Documentation Site

### 1. Start the Docusaurus Development Server
```bash
cd ai-hackathon-book  # documentation directory
npm start
```

### 2. Access the Documentation
Open your browser and navigate to `http://localhost:3000` to view the documentation.

## Building the Book Content

### 1. Add Module Content
Each module is organized as a separate section in the documentation:

```text
docs/
├── module1-ros2/
│   ├── index.md
│   ├── architecture.md
│   ├── nodes.md
│   └── exercises/
├── module2-simulation/
│   ├── index.md
│   └── ...
├── module3-ai/
│   ├── index.md
│   └── ...
└── module4-vla/
    ├── index.md
    └── ...
```

### 2. Create a New Chapter
```bash
# Add new content to the appropriate module directory
touch docs/module1-ros2/my-new-chapter.md

# Format with MDX for interactive content
---
title: My New Chapter
description: Learning ROS 2 concepts
sidebar_position: 3
---

import Component from '@site/src/components/Component';

# My New Chapter

This chapter covers...

<Component />

## Section 1

Some content here...
```

## Testing Your Setup

### 1. Verify ROS 2 Installation
```bash
# Check ROS 2 version
ros2 --version

# Verify core commands work
ros2 topic list
ros2 service list
```

### 2. Test Example Code
Each module includes runnable examples. To test:

```bash
# Navigate to examples directory (after following book instructions)
cd ~/ros2_ws/src/humanoid_examples
colcon build --packages-select example_publisher example_subscriber
source install/setup.bash

# Run an example
ros2 run example_publisher publisher_node
```

### 3. Run Documentation Tests
```bash
# Build documentation
npm run build

# Check for formatting issues
npm run serve  # Serve locally to preview
```

## Troubleshooting

### Common Issues

1. **ROS 2 Commands Not Found**: Ensure you've sourced the ROS 2 setup.bash file
   ```bash
   source /opt/ros/iron/setup.bash
   ```

2. **Python Package Issues**: Make sure you're using Python 3.10 or 3.11
   ```bash
   python3 --version
   ```

3. **Documentation Build Errors**: Check for MDX syntax errors in the content files

### Getting Help

- Check the book's troubleshooting sections in each module
- Visit the ROS 2 community forums for general ROS 2 questions
- Join the project's GitHub discussions for book-specific issues

## Next Steps

1. **Start with Module 1**: Begin with the ROS 2 fundamentals to establish your foundation
2. **Follow the Exercises**: Complete the hands-on exercises to reinforce learning
3. **Progress Sequentially**: Each module builds upon the previous one
4. **Experiment**: Modify the examples to deepen your understanding
5. **Join the Community**: Share your progress and ask questions in the project's community channels