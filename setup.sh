#!/bin/bash

# Setup script for Physical AI & Humanoid Robotics Book development environment
# This script sets up the necessary tools and dependencies to work with the book content

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up development environment for Physical AI & Humanoid Robotics Book..."

# Check if running on Windows with WSL
if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
    echo "WSL detected. Some features may need manual setup."
fi

# Check if running on Ubuntu 22.04
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$NAME" != "Ubuntu" || "$VERSION_ID" != "22.04" ]]; then
        echo "Warning: This script is designed for Ubuntu 22.04. You are running $NAME $VERSION_ID"
    fi
else
    echo "Warning: Not running on Ubuntu. Some features may not work correctly."
fi

# Function to install ROS 2 Iron
install_ros2() {
    echo "Installing ROS 2 Iron..."

    # Add ROS 2 GPG key and repository
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install curl -y
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -

    # Add ROS 2 repository
    if ! grep -q "packages.ros.org/ros2/ubuntu" /etc/apt/sources.list.d/ros2.list; then
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    fi

    # Install ROS 2 Iron packages
    sudo apt update
    sudo apt install -y ros-iron-desktop
    sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-vcstool
    sudo rosdep init || true
    rosdep update

    # Add ROS 2 setup to bashrc if not already present
    if ! grep -q "source /opt/ros/iron/setup.bash" ~/.bashrc; then
        echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
    fi

    echo "ROS 2 Iron installed successfully!"
}

# Function to install Node.js and Docusaurus
install_docusaurus() {
    echo "Installing Node.js and Docusaurus..."

    # Install Node.js 18.x
    if ! command -v node &> /dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        echo "Node.js already installed: $(node --version)"
    fi

    # Install Docusaurus globally
    npm install -g @docusaurus/core@latest

    echo "Node.js and Docusaurus installed successfully!"
}

# Function to set up the workspace
setup_workspace() {
    echo "Setting up the development workspace..."

    # Create ROS 2 workspace if it doesn't exist
    if [ ! -d ~/ros2_ws ]; then
        mkdir -p ~/ros2_ws/src
        echo "ROS 2 workspace created at ~/ros2_ws"
    else
        echo "ROS 2 workspace already exists at ~/ros2_ws"
    fi

    # Install project dependencies
    if [ -f package.json ]; then
        npm install
        echo "Project dependencies installed."
    else
        echo "package.json not found in current directory."
    fi

    echo "Development workspace set up successfully!"
}

# Function to install additional simulation tools
install_simulation() {
    echo "Installing simulation tools..."

    # Install Gazebo Garden
    sudo apt install -y ros-iron-gazebo-ros ros-iron-gazebo-plugins ros-iron-gazebo-dev

    # Install additional useful packages
    sudo apt install -y python3-pip python3-rosinstall

    echo "Simulation tools installed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ros2)
            install_ros2
            shift
            ;;
        --docusaurus)
            install_docusaurus
            shift
            ;;
        --workspace)
            setup_workspace
            shift
            ;;
        --simulation)
            install_simulation
            shift
            ;;
        --all)
            install_ros2
            install_docusaurus
            setup_workspace
            install_simulation
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --ros2         Install only ROS 2 Iron"
            echo "  --docusaurus   Install only Docusaurus and Node.js"
            echo "  --workspace    Set up only the workspace"
            echo "  --simulation   Install only simulation tools"
            echo "  --all          Install everything (default)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no arguments provided, run all setup
if [[ $# -eq 0 ]]; then
    install_ros2
    install_docusaurus
    setup_workspace
    install_simulation
fi

echo "Setup complete! Please run 'source ~/.bashrc' or restart your terminal."
echo "To start the documentation server, run 'npm start' in the project directory."