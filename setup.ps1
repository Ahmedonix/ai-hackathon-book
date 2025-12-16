# PowerShell setup script for Physical AI & Humanoid Robotics Book development environment
# This script sets up the necessary tools and dependencies to work with the book content

Write-Host "Setting up development environment for Physical AI & Humanoid Robotics Book..."

# Check if running on Windows Subsystem for Linux (WSL)
if (Test-Path "HKLM:\SYSTEM\CurrentControlSet\Control\DeviceClasses\{133b9b82-10e7-45a2-8306-d550e6e30c01}") {
    Write-Host "WSL detected. Some features may need manual setup."
}

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to install Node.js and Docusaurus
function Install-Docusaurus {
    Write-Host "Installing Node.js and Docusaurus..."

    if (-not (Test-Command node)) {
        Write-Host "Node.js not found. Please install Node.js from https://nodejs.org/ or using a package manager."
        Write-Host "For Windows, you can use Chocolatey: choco install nodejs"
        Write-Host "Or use NVM: nvm install node"
    } else {
        Write-Host "Node.js already installed: $(node --version)"
    }

    # Install Docusaurus globally
    npm install -g @docusaurus/core@latest

    Write-Host "Node.js and Docusaurus installed successfully!"
}

# Function to set up the workspace
function Setup-Workspace {
    Write-Host "Setting up the development workspace..."

    # Create ROS 2 workspace if it doesn't exist
    if (-not (Test-Path "$env:USERPROFILE\ros2_ws")) {
        New-Item -ItemType Directory -Path "$env:USERPROFILE\ros2_ws\src" -Force
        Write-Host "ROS 2 workspace created at $env:USERPROFILE\ros2_ws"
    } else {
        Write-Host "ROS 2 workspace already exists at $env:USERPROFILE\ros2_ws"
    }

    # Install project dependencies
    if (Test-Path "package.json") {
        npm install
        Write-Host "Project dependencies installed."
    } else {
        Write-Host "package.json not found in current directory."
    }

    Write-Host "Development workspace set up successfully!"
}

# Main execution
try {
    Install-Docusaurus
    Setup-Workspace
    
    Write-Host "Setup complete!"
    Write-Host "To start the documentation server, run 'npm start' in the project directory."
} 
catch {
    Write-Error "An error occurred during setup: $_"
}