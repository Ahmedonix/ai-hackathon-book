---
sidebar_position: 12
---

# Exercise 3: Robot Description with URDF and XACRO

## Objective

In this exercise, you will learn how to create and work with robot description files using URDF (Unified Robot Description Format) and XACRO (XML Macros). You'll build a simple humanoid robot model and visualize it in RViz.

## Prerequisites

Before starting this exercise, ensure you have:
- ROS 2 Iron installed
- RViz2 installed
- Basic understanding of 3D coordinate systems
- Completed previous exercises on nodes and communication

## Step 1: Understanding URDF Structure

URDF describes robot models using XML. The basic structure includes:
- `<link>`: Represents a rigid part of the robot
- `<joint>`: Connects two links (defines their relationship)
- `<material>`: Defines visual properties like color
- `<geometry>`: Defines shape and size

## Step 2: Creating Your First URDF

Create a package for this exercise:

```bash
# Create a new package
ros2 pkg create --build-type ament_cmake robot_description_exercise --dependencies urdf xacro joint_state_publisher robot_state_publisher rviz2
```

Now create a URDF file `simple_robot.urdf` in the `urdf` directory:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link of the robot -->
  <link name="base_link">
    <visual>
      <!-- Visual representation of the link -->
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.4 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <!-- Collision geometry for physics simulation -->
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 0.4 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <!-- Inertial properties for simulation -->
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- A simple arm link connected to the base -->
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.2 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="arm_link">
    <visual>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- End effector for the arm -->
  <joint name="gripper_joint" type="fixed">
    <parent link="arm_link"/>
    <child link="gripper_link"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
  </joint>

  <link name="gripper_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>
</robot>
```

## Step 3: Converting to XACRO

Create a more complex humanoid robot using XACRO in `humanoid_robot.xacro`:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties for reusable values -->
  <xacro:property name="M_PI" value="3.14159"/>

  <!-- Define a macro for creating limbs -->
  <xacro:macro name="limb" params="name parent xyz joint_limit_lower joint_limit_upper">
    <!-- Joint connecting to parent -->
    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${joint_limit_lower}" upper="${joint_limit_upper}" effort="100" velocity="1"/>
    </joint>

    <!-- Link for the limb -->
    <link name="${name}_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="light_red">
          <color rgba="0.8 0.4 0.4 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.5"/>
      </geometry>
      <material name="light_blue">
        <color rgba="0.4 0.4 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-M_PI/4}" upper="${M_PI/4}" effort="100" velocity="1"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Arms -->
  <xacro:limb name="left_arm" parent="base_link" xyz="0.1 0.15 0.2" 
              joint_limit_lower="${-M_PI/2}" joint_limit_upper="${M_PI/2}"/>

  <xacro:limb name="right_arm" parent="base_link" xyz="0.1 -0.15 0.2" 
              joint_limit_lower="${-M_PI/2}" joint_limit_upper="${M_PI/2}"/>

  <!-- Legs -->
  <xacro:limb name="left_leg" parent="base_link" xyz="-0.1 0.05 0" 
              joint_limit_lower="${-M_PI/2}" joint_limit_upper="${M_PI/2}"/>

  <xacro:limb name="right_leg" parent="base_link" xyz="-0.1 -0.05 0" 
              joint_limit_lower="${-M_PI/2}" joint_limit_upper="${M_PI/2}"/>

  <!-- Joint Limits Plugin -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>/</namespace>
      </ros>
      <update_rate>30</update_rate>
      <joint_name>neck_joint</joint_name>
      <joint_name>left_arm_joint</joint_name>
      <joint_name>right_arm_joint</joint_name>
      <joint_name>left_leg_joint</joint_name>
      <joint_name>right_leg_joint</joint_name>
    </plugin>
  </gazebo>

  <!-- Joint Position Controllers -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

## Step 4: Creating Launch Files

Create a launch directory and launch file `display_robot.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('robot_description_exercise')
    
    # Launch configuration variables
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Declare launch arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # Path to URDF/XACRO file
    robot_description_path = os.path.join(
        pkg_share,
        'urdf',
        'humanoid_robot.xacro'
    )
    
    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'robot_description': Command(['xacro ', robot_description_path])
            }
        ]
    )
    
    # Joint State Publisher node (GUI for controlling joints)
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
    
    # RViz node
    rviz_node = Node(
        condition=LaunchConfigurationEquals('use_rviz', 'true'),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'urdf.rviz')]
    )
    
    # Create launch description and add actions
    ld = LaunchDescription()
    
    ld.add_action(declare_use_rviz)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(rviz_node)
    
    return ld
```

Actually, let me create the Python launch file with the correct syntax:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('robot_description_exercise')
    
    # Launch configuration variables
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Declare launch arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # Path to URDF/XACRO file
    robot_description_path = os.path.join(
        pkg_share,
        'urdf',
        'humanoid_robot.xacro'
    )
    
    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'robot_description': Command(['xacro ', robot_description_path])
            }
        ]
    )
    
    # Joint State Publisher node (GUI for controlling joints)
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
    
    # RViz node
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'urdf.rviz')]
    )
    
    # Create launch description and add actions
    ld = LaunchDescription()
    
    ld.add_action(declare_use_rviz)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(rviz_node)
    
    return ld
```

Create the RViz configuration file `urdf.rviz` in the `rviz` directory:

```
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
      Splitter Ratio: 0.5
    Tree Height: 602
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
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        head_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        left_arm_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        left_leg_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        right_arm_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        right_leg_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
    - Class: rviz_default_plugins/SetInitialPose
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
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
      Distance: 2.0
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
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d0065010000000000000450000000000000000000000390000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 72
  Y: 60
```

## Step 5: Create the package.xml file

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>robot_description_exercise</name>
  <version>0.1.0</version>
  <description>Robot description exercise package</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>urdf</depend>
  <depend>xacro</depend>
  <depend>joint_state_publisher</depend>
  <depend>joint_state_publisher_gui</depend>
  <depend>robot_state_publisher</depend>
  <depend>rviz2</depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Step 6: Build and Run the Robot Model

Build your package:

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select robot_description_exercise

# Source the setup files
source install/setup.bash
```

Run the visualization:

```bash
# Launch the robot model visualization
ros2 launch robot_description_exercise display_robot.launch.py
```

## Step 7: Validating Your URDF

You can validate your URDF files using the check_urdf tool:

```bash
# Check the URDF for errors
check_urdf <(ros2 run xacro xacro `ros2 pkg prefix robot_description_exercise`/share/robot_description_exercise/urdf/humanoid_robot.xacro)
```

## Exercise Challenges

### Challenge 1: Add a Sensor
- Extend the humanoid model by adding a camera or LiDAR sensor to the head
- Define the sensor's mounting and properties in URDF

### Challenge 2: Create a Wheeled Robot
- Design a simple wheeled robot with differential drive
- Include proper wheel joints and define the transmission

### Challenge 3: Use XACRO Macros
- Create a macro for a wheel and instantiate multiple wheels using it
- Create a macro for a complete robot limb

## Solution for Challenge 1: Adding a Sensor

Here's an example of how to add a camera sensor to the head:

```xml
<!-- In the head link section, add a camera joint and link -->
<joint name="camera_joint" type="fixed">
  <parent link="head_link"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
    <material name="black">
      <color rgba="0 0 0 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
  </inertial>
</link>

<!-- Add Gazebo plugin for the camera -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera_sensor">
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/camera</namespace>
        <remapping>~/image_raw:=/camera/image_raw</remapping>
        <remapping>~/image_info:=/camera/image_info</remapping>
      </ros>
      <camera_name>head_camera</camera_name>
      <frame_name>camera_link</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

## Key Learning Points

This exercise taught you:

1. **URDF Structure**: Understanding how links and joints form robot models
2. **XACRO Macros**: Using macros to create reusable components
3. **Visual vs Collision**: Defining both visual and collision properties
4. **Inertial Properties**: Setting mass and inertia for physics simulation
5. **RViz Visualization**: Seeing your robot model in the visualization tool
6. **Launch Files**: Using launch files to start robot visualization
7. **Robot State Publisher**: Understanding how joint states relate to transforms

## Common Issues and Solutions

### Issue 1: URDF Parsing Errors
- **Problem**: XML syntax errors or missing attributes
- **Solution**: Carefully check tag pairs and required attributes like `name`, `type`, etc.

### Issue 2: Joint Configuration Problems
- **Problem**: Robot not displaying correctly in RViz
- **Solution**: Verify joint origins, parent-child relationships, and joint limits

### Issue 3: Missing Dependencies
- **Problem**: Package build errors
- **Solution**: Install missing packages like `joint_state_publisher_gui`

## Summary

In this exercise, you've learned how to:
- Create robot models using URDF and XACRO
- Define links, joints, and their properties
- Visualize robot models in RViz
- Use XACRO macros to create reusable components
- Set up launch files for robot visualization

Robot description is fundamental to robotics applications as it provides the spatial relationships and physical properties that govern how a robot interacts with the world. Well-designed URDF files are essential for simulation, visualization, and proper functioning of navigation and manipulation algorithms.

The next exercise will focus on multi-node systems, bringing together all the concepts learned so far.