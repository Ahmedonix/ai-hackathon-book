# Importing URDF Robot Model into Gazebo Environment

## Overview

In this section, we'll learn how to import your URDF robot model into the Gazebo simulation environment. This process involves converting your robot description into a format that Gazebo can understand and simulate with realistic physics.

## Prerequisites

Before importing your URDF robot into Gazebo, ensure you have:

1. A properly defined URDF file for your humanoid robot (like the simple_humanoid.urdf from Module 1)
2. Gazebo and ROS 2 properly installed and configured
3. The `ros-iron-gazebo-ros` and `ros-iron-gazebo-plugins` packages installed

## Understanding URDF for Gazebo

### 1. Required Gazebo-Specific Elements

Your URDF file needs specific elements for proper Gazebo integration:

```xml
<!-- Gazebo plugin for controlling the robot -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/simple_humanoid</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>

<!-- Gazebo materials for visualization -->
<gazebo reference="base_link">
  <material>Gazebo/Grey</material>
</gazebo>
```

### 2. Physics Properties in URDF

For proper physics simulation, your URDF must include:

- **Inertial properties**: Mass, center of mass, and inertia tensor for each link
- **Collision properties**: Collision geometry for physics calculations
- **Visual properties**: How the robot appears in the simulation

## Method 1: Direct URDF Loading

### 1. Prepare Your URDF File

First, ensure your URDF file includes the necessary Gazebo-specific elements. Here's an example extension to the simple humanoid model:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

  <!-- [Previous URDF content remains the same] -->

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <robot_namespace>/simple_humanoid</robot_namespace>
      <robot_param>robot_description</robot_param>
      <actuator_config_type>individual</actuator_config_type>
    </plugin>
  </gazebo>

  <!-- Gazebo materials for visualization -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Add materials for all other links -->
  <!-- ... -->

</robot>
```

### 2. Launch with Direct URDF Loading

Create a launch file to load your URDF directly into Gazebo:

```python
# launch/humanoid_spawn.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    urdf_package = LaunchConfiguration('urdf_package', default='humanoid_description')
    urdf_file = LaunchConfiguration('urdf_file', default='simple_humanoid.urdf')
    
    # Get URDF file path
    urdf_path = PathJoinSubstitution([
        FindPackageShare(urdf_package),
        'urdf',
        urdf_file
    ])
    
    # Get robot description content
    robot_description_content = Command(['xacro ', urdf_path])
    
    # Set parameters for ROS 2
    params = {
        'use_sim_time': use_sim_time,
        'robot_description': robot_description_content,
    }
    
    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[params]
    )
    
    # Gazebo server and client nodes
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ]),
        launch_arguments={
            'verbose': 'false',
            'pause_on_start': 'false',
        }.items()
    )
    
    # Spawn entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0', '-y', '0', '-z', '1.0'  # Spawn 1m above ground
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Set parameter
        SetParameter(name='use_sim_time', value=use_sim_time),
        # Launch Gazebo
        gazebo,
        # Robot State Publisher
        robot_state_publisher,
        # Spawn the robot
        spawn_entity
    ])
```

### 3. Execute the Launch File

```bash
# Source ROS 2 Iron
source /opt/ros/iron/setup.bash

# Navigate to your workspace
cd ~/ros2_ws

# Build your packages
colcon build --packages-select humanoid_description  # or your package name

# Source the workspace
source install/setup.bash

# Launch the robot in Gazebo
ros2 launch humanoid_description humanoid_spawn.launch.py
```

## Method 2: SDF Conversion

### 1. Convert URDF to SDF

Gazebo natively uses SDF (Simulation Description Format), so you can convert your URDF:

```bash
# Convert URDF to SDF
gz sdf -p /path/to/your/simple_humanoid.urdf > simple_humanoid.sdf
```

### 2. Direct SDF Loading

You can also directly load SDF files into Gazebo:

```python
# Alternative launch file using SDF
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ])
    )
    
    # Spawn the robot from SDF
    spawn_robot = ExecuteProcess(
        cmd=['gz', 'service', '-s', '/world/default/create',
             '--reqtype', 'gz.msgs.EntityFactory',
             '--req', f'sdf_filename:=simple_humanoid.sdf,name:=simple_humanoid'],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo,
        spawn_robot
    ])
```

## Troubleshooting Common Import Issues

### 1. Robot Not Loading

If your robot doesn't appear in Gazebo:

- Check the ROS logs for errors: `ros2 run rqt_console rqt_console`
- Verify the URDF file is valid: `check_urdf path/to/your.urdf`
- Ensure the robot is spawning at the right location (try different coordinates)
- Check that all mesh files referenced in the URDF exist

### 2. Physics Issues

If your robot behaves unexpectedly:

- Verify inertial properties are correctly defined
- Check that mass values are realistic (too low can cause floating, too high can cause sinking)
- Ensure collision geometries match visual geometries appropriately

### 3. Joint Control Issues

If you have troubles controlling joints:

```bash
# Check available topics
ros2 topic list | grep joint

# Check if joint state publisher is working
ros2 topic echo /joint_states

# Verify controller manager is running
ros2 service list | grep controller_manager
```

## Best Practices for URDF in Gazebo

### 1. Simplified Collisions

For better performance, use simplified collision geometries:

```xml
<!-- In your URDF file -->
<link name="left_knee">
  <visual>
    <!-- Detailed visual representation -->
    <geometry>
      <mesh filename="package://humanoid_description/meshes/complex_knee.dae"/>
    </geometry>
  </visual>
  <collision>
    <!-- Simplified collision geometry -->
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>
```

### 2. Proper Inertial Values

Ensure your robot's inertial values are physically plausible:

- Total mass should match a real humanoid robot (50-100kg typical)
- Center of mass should be in the torso region
- Inertia values should follow the parallel axis theorem

### 3. Fixed Joints for Visual Elements

Use fixed joints for visual elements that don't need physics:

```xml
<joint name="lidar_mount_joint" type="fixed">
  <parent link="head"/>
  <child link="lidar_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>
```

## Verifying the Import

### 1. Visual Verification

- The robot should appear in Gazebo at the specified location
- All links should be visible and properly connected
- No parts should be floating or disconnected

### 2. Physics Verification

- The robot should be affected by gravity appropriately
- Links shouldn't pass through each other
- The robot should rest stably on the ground plane

### 3. Joint Verification

```bash
# Check joint states
ros2 topic echo /joint_states

# Check TF tree
ros2 run tf2_tools view_frames
```

## Next Steps

Once your URDF robot is successfully imported into Gazebo, you'll need to configure physics properties for realistic humanoid simulation. The next section will cover gravity, collisions, and joints in detail.

Properly importing your robot model is a critical first step in the simulation process, as all subsequent physics, sensor, and control operations depend on having an accurate model in the simulation environment.