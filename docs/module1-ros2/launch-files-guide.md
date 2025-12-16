# Launch System for Humanoid Robot Basic Functionality

This document describes the foundational launch files for multi-node ROS 2 systems, which will be used throughout the Physical AI & Humanoid Robotics Book. Launch files allow you to start multiple nodes with a single command, making it easier to run complex humanoid robot applications.

## Basic Multi-Node Launch File

Here is a basic launch file that starts several essential nodes for the humanoid robot:

```xml
<launch>
  <!-- Launch file for basic humanoid robot functionality -->
  
  <!-- Arguments -->
  <arg name="use_sim_time" default="false" description="Use simulation time"/>
  
  <!-- Robot State Publisher Node -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    <!-- Load the URDF from a parameter server -->
    <param name="robot_description" textfile="$(find-pkg-share humanoid_bringup)/urdf/simple_humanoid.urdf"/>
  </node>
  
  <!-- Joint State Publisher Node -->
  <node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
  
  <!-- Example: Humanoid Status Publisher (from our examples package) -->
  <node pkg="humanoid_robot_examples" exec="talker" name="humanoid_status_publisher">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
  
  <!-- Example: Humanoid Status Subscriber (from our examples package) -->
  <node pkg="humanoid_robot_examples" exec="listener" name="humanoid_status_subscriber" output="screen">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
  
  <!-- TF2 Static Transform Publisher for base_link to odom -->
  <node pkg="tf2_ros" exec="static_transform_publisher" name="base_to_odom_publisher" 
        args="0 0 0 0 0 0 base_link odom">
  </node>
  
</launch>
```

## Python Launch File Alternative

For more complex launch scenarios, you can use Python launch files:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('humanoid_bringup')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Define nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': open(
                os.path.join(pkg_share, 'urdf', 'simple_humanoid.urdf')
            ).read()}
        ]
    )
    
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    humanoid_status_publisher = Node(
        package='humanoid_robot_examples',
        executable='talker',
        name='humanoid_status_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    humanoid_status_subscriber = Node(
        package='humanoid_robot_examples',
        executable='listener',
        name='humanoid_status_subscriber',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_odom_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'odom']
    )
    
    # Return the launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use simulation time if true'
        ),
        robot_state_publisher,
        joint_state_publisher,
        humanoid_status_publisher,
        humanoid_status_subscriber,
        tf_publisher
    ])
```

## Advanced Launch File for Simulation

For simulation scenarios (Module 2), you might have a more complex launch file:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch Gazebo with the humanoid robot
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/gazebo.launch.py'
        ])
    )
    
    # Spawn the humanoid robot in Gazebo
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
    
    # Robot State Publisher for simulation
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': open(
                os.path.join(get_package_share_directory('humanoid_bringup'), 
                           'urdf', 'simple_humanoid.urdf')
            ).read()}
        ]
    )
    
    return LaunchDescription([
        gazebo,
        spawn_entity,
        robot_state_publisher
    ])
```

## Launch File Best Practices

### 1. Modularity
- Break complex systems into smaller, reusable launch files
- Use `IncludeLaunchDescription` to compose complex systems

### 2. Parameterization
- Use launch arguments to make launch files configurable
- Separate environment-specific configurations from general ones

### 3. Error Handling
- Set proper output options for debugging
- Use appropriate restart policies for nodes

### 4. Resource Management
- Set resource limits where necessary
- Configure node lifecycle appropriately

## Using Launch Files

To use a launch file from the command line:

```bash
# Basic launch
ros2 launch humanoid_bringup basic_humanoid.launch.py

# Launch with arguments
ros2 launch humanoid_bringup basic_humanoid.launch.py use_sim_time:=true

# Launch with simulation time
ros2 launch humanoid_bringup simulation.launch.py
```

## Integration with Modules

### Module 1 (ROS 2 Fundamentals) - Launch Basics
- Understanding launch file structure
- Basic multi-node launch files
- Parameter passing to nodes

### Module 2 (Digital Twin Simulation) - Simulation Launch
- Launching robot in simulation with Gazebo
- Spawning robots in simulation environments
- Simulation-specific parameters

### Module 3 (AI-Robot Brain) - AI Integration Launch
- Launching perception nodes
- Starting navigation systems
- AI model integration

### Module 4 (Vision-Language-Action) - Cognitive System Launch
- Launching voice processing nodes
- Starting cognitive robotics systems
- Multi-modal integration

These foundational launch files provide a structured way to start complex humanoid robot systems, making it easier to manage dependencies and ensure all required nodes are running together.