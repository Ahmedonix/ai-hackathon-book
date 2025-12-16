# Creating Custom Simulation Environment/World for Humanoid Robotics

## Overview

In this section, we'll create a comprehensive custom simulation environment specifically designed for humanoid robot testing and development. This environment will incorporate the concepts we've learned about physics, sensors, and world-building to create a realistic testing ground for our humanoid robot.

## Designing the Humanoid Testing Environment

### 1. Environment Requirements

Our custom humanoid testing environment will include:
- Flat areas for basic locomotion testing
- Obstacles of varying sizes for navigation challenges
- Stairs for testing climbing capabilities
- Interactive objects for manipulation tasks
- Multiple rooms for indoor navigation
- Lighting that simulates real-world conditions

### 2. World File Structure

Create the main world file `humanoid_test_world.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version='1.7'>
  <world name="humanoid_test_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
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
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.7 0.7 0.7</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Custom models and obstacles -->
    
    <!-- 1. Main testing area -->
    <model name="test_area_boundary">
      <static>true</static>
      <link name="wall_north">
        <pose>0 7.5 1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_south">
        <pose>0 -7.5 1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_east">
        <pose>7.5 0 1 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_west">
        <pose>-7.5 0 1 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 2. Navigation obstacles -->
    <model name="obstacle_1">
      <pose>-3 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <model name="obstacle_2">
      <pose>3 2 0.3 0 0 0.5</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- 3. Stairs for climbing test -->
    <model name="stairs">
      <static>true</static>
      <pose>5 -4 0 0 0 0</pose>
      
      <!-- Step 1 -->
      <link name="step_1">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 2 -->
      <link name="step_2">
        <pose>0 0 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 3 -->
      <link name="step_3">
        <pose>0 0 0.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 4. Indoor room section -->
    <model name="room_partition">
      <static>true</static>
      <pose>-5 0 1 0 0 0</pose>
      
      <!-- Wall -->
      <link name="wall">
        <collision name="collision">
          <geometry>
            <box><size>0.2 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Door opening -->
      <link name="door_frame">
        <pose>0 1 1 0 0 0</pose>
        <visual name="visual">
          <geometry>
            <box><size>0.2 1 1.5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 5. Manipulation area with objects -->
    <model name="table">
      <pose>-5 -3 0.7 0 0 0</pose>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box><size>1.2 0.8 0.7</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.2 0.8 0.7</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.1 1</ambient>
            <diffuse>0.4 0.2 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Objects on table -->
    <model name="cup">
      <pose>-4.8 -3 1.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.2 1</ambient>
            <diffuse>0.8 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <model name="box_on_table">
      <pose>-5.2 -2.8 1.05 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 0.2 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 0.2 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- 6. Sloped terrain -->
    <model name="slope">
      <static>true</static>
      <pose>-2 4 0 0 0 0</pose>
      <link name="slope_link">
        <pose>0 0 0.25 0 0.3 0</pose>  <!-- 0.3 radians ~ 17 degrees slope -->
        <collision name="collision">
          <geometry>
            <box><size>3 2 0.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>3 2 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.7 0.5 1</ambient>
            <diffuse>0.5 0.7 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 7. Humanoid robot -->
    <model name="simple_humanoid">
      <pose>0 0 1 0 0 0</pose>
      <!-- This will be loaded from your URDF model -->
    </model>
    
    <!-- 8. Additional lighting for indoor section -->
    <model name="light_1">
      <pose>-5 0 2 0 0 0</pose>
      <link name="link">
        <light type="point">
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.8 0.8 0.8 1</specular>
          <attenuation>
            <range>10</range>
            <constant>0.5</constant>
            <linear>0.1</linear>
            <quadratic>0.01</quadratic>
          </attenuation>
          <cast_shadows>true</cast_shadows>
          <pose>0 0 0 0 0 0</pose>
        </light>
      </link>
    </model>
  </world>
</sdf>
```

## Creating a Launch File for the Environment

### 1. Environment Launch File

Create `launch/humanoid_test_world.launch.py`:

```python
# launch/humanoid_test_world.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='humanoid_test_world.sdf')
    
    # Get the package share directory
    pkg_share = get_package_share_directory('humanoid_description')  # Adjust package name as needed
    
    # Gazebo launch with custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', world_name]),
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Joint State Publisher (for non-controlled joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='both',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # RViz2 for visualization
    rviz_config = PathJoinSubstitution([
        FindPackageShare('humanoid_description'),
        'rviz',
        'humanoid_test.rviz'
    ])
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # Robot spawning node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0',
            '-y', '0', 
            '-z', '1.0'
        ],
        output='screen'
    )
    
    # Create worlds directory if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    if not os.path.exists(worlds_dir):
        os.makedirs(worlds_dir)
    
    # Copy world file to package directory
    import shutil
    world_src = os.path.join(pkg_share, '..', 'docs', 'module2-simulation', 'humanoid_test_world.sdf')
    world_dst = os.path.join(worlds_dir, 'humanoid_test_world.sdf')
    try:
        shutil.copy2(world_src, world_dst)
    except FileNotFoundError:
        # If the source doesn't exist, create it from the content in this documentation
        with open(world_dst, 'w') as f:
            f.write("""<?xml version="1.0" ?>
<sdf version='1.7'>
  <world name="humanoid_test_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
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
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.7 0.7 0.7</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Custom models and obstacles -->
    
    <!-- 1. Main testing area -->
    <model name="test_area_boundary">
      <static>true</static>
      <link name="wall_north">
        <pose>0 7.5 1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_south">
        <pose>0 -7.5 1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_east">
        <pose>7.5 0 1 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="wall_west">
        <pose>-7.5 0 1 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>15 0.2 2</size></box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.3 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 2. Navigation obstacles -->
    <model name="obstacle_1">
      <pose>-3 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1 1 1</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <model name="obstacle_2">
      <pose>3 2 0.3 0 0 0.5</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>0.6</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- 3. Stairs for climbing test -->
    <model name="stairs">
      <static>true</static>
      <pose>5 -4 0 0 0 0</pose>
      
      <!-- Step 1 -->
      <link name="step_1">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 2 -->
      <link name="step_2">
        <pose>0 0 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 3 -->
      <link name="step_3">
        <pose>0 0 0.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2 1.5 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 4. Indoor room section -->
    <model name="room_partition">
      <static>true</static>
      <pose>-5 0 1 0 0 0</pose>
      
      <!-- Wall -->
      <link name="wall">
        <collision name="collision">
          <geometry>
            <box><size>0.2 6 2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 6 2</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Door opening -->
      <link name="door_frame">
        <pose>0 1 1 0 0 0</pose>
        <visual name="visual">
          <geometry>
            <box><size>0.2 1 1.5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 5. Manipulation area with objects -->
    <model name="table">
      <pose>-5 -3 0.7 0 0 0</pose>
      <link name="base">
        <collision name="collision">
          <geometry>
            <box><size>1.2 0.8 0.7</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.2 0.8 0.7</size></box>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.1 1</ambient>
            <diffuse>0.4 0.2 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Objects on table -->
    <model name="cup">
      <pose>-4.8 -3 1.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.2 1</ambient>
            <diffuse>0.8 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <model name="box_on_table">
      <pose>-5.2 -2.8 1.05 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>0.2 0.2 0.2</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.2 0.2 0.2</size></box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- 6. Sloped terrain -->
    <model name="slope">
      <static>true</static>
      <pose>-2 4 0 0 0 0</pose>
      <link name="slope_link">
        <pose>0 0 0.25 0 0.3 0</pose>  <!-- 0.3 radians ~ 17 degrees slope -->
        <collision name="collision">
          <geometry>
            <box><size>3 2 0.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>3 2 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.5 0.7 0.5 1</ambient>
            <diffuse>0.5 0.7 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 7. Humanoid robot placeholder -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
    
    <!-- 8. Additional lighting for indoor section -->
    <model name="light_1">
      <pose>-5 0 2 0 0 0</pose>
      <link name="link">
        <light type="point">
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.8 0.8 0.8 1</specular>
          <attenuation>
            <range>10</range>
            <constant>0.5</constant>
            <linear>0.1</linear>
            <quadratic>0.01</quadratic>
          </attenuation>
          <cast_shadows>true</cast_shadows>
          <pose>0 0 0 0 0 0</pose>
        </light>
      </link>
    </model>
  </world>
</sdf>""")
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        # Add delay before spawning robot to ensure Gazebo is ready
        TimerAction(
            period=5.0,
            actions=[spawn_robot]
        ),
        rviz
    ])
```

## Creating Environment-Specific Nodes

### 1. Test Scenario Node

Create a node to implement specific test scenarios in the environment:

```python
# scripts/test_scenarios.py
#!/usr/bin/env python3

"""
Test scenarios for the humanoid robot in the custom environment.
This node implements various test scenarios to validate robot capabilities.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, Imu
from std_msgs.msg import String, Float64MultiArray
from builtin_interfaces.msg import Duration
import time
import math


class HumanoidTestScenarios(Node):
    def __init__(self):
        super().__init__('humanoid_test_scenarios')
        
        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        
        # Subscribers for sensor data
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        
        # State variables
        self.current_scenario = 0
        self.robot_pose = Pose()
        self.laser_data = None
        self.imu_data = None
        self.test_start_time = self.get_clock().now()
        
        # Timer for scenario execution
        self.scenario_timer = self.create_timer(0.1, self.execute_scenario)
        
        # Scenario execution flag
        self.execute_scenarios = True
        
        self.get_logger().info('Humanoid Test Scenarios Node Started')
    
    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.robot_pose = msg.pose.pose
    
    def scan_callback(self, msg):
        """Store laser scan data"""
        self.laser_data = msg
    
    def imu_callback(self, msg):
        """Store IMU data"""
        self.imu_data = msg
    
    def camera_callback(self, msg):
        """Process camera data if needed"""
        # For now, just acknowledge the data
        pass
    
    def execute_scenario(self):
        """Execute the current test scenario"""
        if not self.execute_scenarios:
            return
        
        # Switch scenarios based on time or conditions
        if self.current_scenario == 0:
            self.basic_locomotion_test()
        elif self.current_scenario == 1:
            self.obstacle_avoidance_test()
        elif self.current_scenario == 2:
            self.stair_navigation_test()
        elif self.current_scenario == 3:
            self.manipulation_test()
        else:
            self.current_scenario = 0  # Reset to first scenario
    
    def basic_locomotion_test(self):
        """Test basic walking and turning"""
        self.get_logger().info('Executing Basic Locomotion Test')
        
        # Publish velocity commands for forward movement
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.0  # No turning
        
        self.cmd_vel_pub.publish(cmd)
        
        # After 5 seconds, move to next scenario
        if (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9 > 5.0:
            self.current_scenario = 1
            self.test_start_time = self.get_clock().now()
    
    def obstacle_avoidance_test(self):
        """Test obstacle avoidance with laser data"""
        self.get_logger().info('Executing Obstacle Avoidance Test')
        
        cmd = Twist()
        
        if self.laser_data:
            # Check if there's an obstacle in front
            min_distance = min(self.laser_data.ranges[:10] + self.laser_data.ranges[-10:])
            
            if min_distance < 1.0:  # Obstacle within 1 meter
                cmd.linear.x = 0.0  # Stop
                cmd.angular.z = 0.5  # Turn right
            else:
                cmd.linear.x = 0.3  # Move forward
                cmd.angular.z = 0.0  # No turn
        
        self.cmd_vel_pub.publish(cmd)
        
        # After 10 seconds, move to next scenario
        if (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9 > 10.0:
            self.current_scenario = 2
            self.test_start_time = self.get_clock().now()
    
    def stair_navigation_test(self):
        """Test stair climbing (conceptual - real implementation would be more complex)"""
        self.get_logger().info('Executing Stair Navigation Test')
        
        # For this demo, we'll just move toward the stairs area
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = -0.1  # Slight turn toward stairs at (5, -4)
        
        self.cmd_vel_pub.publish(cmd)
        
        # After 15 seconds, move to next scenario
        if (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9 > 15.0:
            self.current_scenario = 3
            self.test_start_time = self.get_clock().now()
    
    def manipulation_test(self):
        """Test manipulation in the room section"""
        self.get_logger().info('Executing Manipulation Test')
        
        # Move near the table in the room section
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.2  # Turn toward room (-5, -3)
        
        self.cmd_vel_pub.publish(cmd)
        
        # After 10 seconds, complete the cycle
        if (self.get_clock().now() - self.test_start_time).nanoseconds / 1e9 > 10.0:
            self.current_scenario = 0
            self.test_start_time = self.get_clock().now()
    
    def stop_robot(self):
        """Stop all robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    
    test_node = HumanoidTestScenarios()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info('Test interrupted by user')
    finally:
        test_node.stop_robot()
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Testing the Custom Environment

### 1. Running the Environment

```bash
# Make sure your script is executable
chmod +x install/humanoid_description/lib/humanoid_description/test_scenarios.py

# Launch the custom environment
ros2 launch humanoid_description humanoid_test_world.launch.py
```

### 2. Verifying the Environment

Check that all elements are loaded correctly:

```bash
# Check if all models are loaded
gz model -l

# Check physics properties
gz topic -e -t /world/humanoid_test_world/physics

# Monitor robot state
ros2 topic echo /joint_states

# Monitor sensor data
ros2 topic echo /scan
ros2 topic echo /imu
ros2 topic echo /camera/image_raw
```

## Advanced Testing Scenarios

### 1. Performance Testing Script

Create a script to validate the environment performance:

```python
# scripts/environment_validation.py
#!/usr/bin/env python3

"""
Environment validation script for the humanoid test world.
Checks various aspects of the simulation environment.
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetWorldProperties, GetEntityState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
import time


class EnvironmentValidator(Node):
    def __init__(self):
        super().__init__('environment_validator')
        
        # Create service clients for Gazebo
        self.get_world_props_cli = self.create_client(GetWorldProperties, '/get_world_properties')
        self.get_entity_state_cli = self.create_client(GetEntityState, '/get_entity_state')
        
        # Wait for services to be available
        while not self.get_world_props_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_world_properties service...')
        
        while not self.get_entity_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_entity_state service...')
        
        self.get_logger().info('Environment Validator Started')
        
        # Run validation tests
        self.run_validations()
    
    def run_validations(self):
        """Run all validation tests"""
        self.get_logger().info('Starting Environment Validations...')
        
        # 1. Check world properties
        self.validate_world_properties()
        
        # 2. Check key entities
        self.validate_entities()
        
        # 3. Performance check
        self.performance_check()
        
        self.get_logger().info('Environment Validations Complete')
    
    def validate_world_properties(self):
        """Validate world physics properties"""
        self.get_logger().info('Validating World Properties...')
        
        request = GetWorldProperties.Request()
        future = self.get_world_props_cli.call_async(request)
        
        # Wait for response
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        
        if response.success:
            self.get_logger().info(f'World Name: {response.world_name}')
            self.get_logger().info(f'Paused: {response.paused}')
            self.get_logger().info(f'Sim Time: {response.sim_time.sec}.{response.sim_time.nanosec}')
            self.get_logger().info(f'Gravity: ({response.gravity.x}, {response.gravity.y}, {response.gravity.z})')
        else:
            self.get_logger().error('Failed to get world properties')
    
    def validate_entities(self):
        """Validate key entities in the environment"""
        self.get_logger().info('Validating Key Entities...')
        
        entities_to_check = [
            'simple_humanoid',
            'obstacle_1',
            'obstacle_2',
            'table',
            'cup',
            'box_on_table'
        ]
        
        for entity in entities_to_check:
            request = GetEntityState.Request()
            request.name = entity
            request.relative_entity_name = 'world'
            
            future = self.get_entity_state_cli.call_async(request)
            
            # Wait for response
            rclpy.spin_until_future_complete(self, future)
            
            response = future.result()
            
            if response.success:
                self.get_logger().info(f'{entity}: Position ({response.state.pose.position.x:.2f}, {response.state.pose.position.y:.2f}, {response.state.pose.position.z:.2f})')
            else:
                self.get_logger().warn(f'{entity}: Not found or error occurred')
    
    def performance_check(self):
        """Basic performance check by timing physics updates"""
        self.get_logger().info('Performance Check: Timing physics updates...')
        
        # This is a basic check - real performance validation would be more complex
        start_time = time.time()
        
        # Perform some operations that would affect performance
        # (In a real implementation, this would be more sophisticated)
        
        end_time = time.time()
        self.get_logger().info(f'Sample operation took {end_time - start_time:.4f} seconds')


def main(args=None):
    rclpy.init(args=args)
    
    validator = EnvironmentValidator()
    
    # Spin once to execute validations
    rclpy.spin_once(validator, timeout_sec=1.0)
    
    validator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Environment Configuration Files

### 1. RViz Configuration

Create an RViz configuration file to visualize the environment properly:

```yaml
# rviz/humanoid_test.rviz
Panels:
  - Class: rviz_common/Displays
    Help Height: 0
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /LaserScan1
        - /Imu1
        - /TF1
      Splitter Ratio: 0.5
    Tree Height: 1000
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 20
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: RobotModel
      Description Topic:
        Depth: 5
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Name Topic:
        Depth: 5
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /tf
      Visual Enabled: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Min Color: 0; 0; 0
      Name: LaserScan
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/Imu
      Color: 204; 51; 51
      Enabled: true
      Name: Imu
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /imu
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: world
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Topic:
        Depth: 5
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1000
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd0000000400000000000001f0000003a2fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000003a2000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000003a2fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000003a2000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000004c7000003a200000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1800
  X: 0
  Y: 0
```

## Running Complete Environment Test

### 1. Full System Test Command

```bash
# First, make sure to build your package
colcon build --packages-select humanoid_description
source install/setup.bash

# Then run the complete environment test
ros2 launch humanoid_description humanoid_test_world.launch.py
```

### 2. Additional Test Commands

```bash
# Run environment validation
ros2 run humanoid_description environment_validation

# Run test scenarios
ros2 run humanoid_description test_scenarios
```

## Troubleshooting Common Issues

### 1. Environment Loading Problems

If the environment doesn't load properly:
- Check the SDF file syntax with `gz sdf -k humanoid_test_world.sdf`
- Verify that all referenced models exist
- Check Gazebo console output for errors

### 2. Performance Issues

If simulation runs slowly:
- Reduce the number of objects in the environment
- Simplify collision geometries
- Lower the physics update rate slightly
- Disable unnecessary visualizations

### 3. Robot-Environment Interaction Issues

If the robot doesn't interact properly:
- Verify that the robot's collision geometries are properly defined
- Check that the robot has appropriate mass and inertia values
- Ensure that friction parameters are realistic for humanoid locomotion

## Next Steps

With your custom simulation environment successfully created, you're now ready to learn about Unity integration for enhanced visualization. Unity provides advanced rendering capabilities and can be used alongside Gazebo for high-fidelity visualization of your humanoid robot.

The comprehensive environment you've built provides a solid foundation for testing all aspects of humanoid robot capabilities.