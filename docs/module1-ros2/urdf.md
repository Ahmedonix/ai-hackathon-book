---
sidebar_position: 5
---

# Robot Description: URDF and XACRO

## Overview

URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS 2. It defines the physical and visual properties of a robot, including its links, joints, inertial properties, and visual elements. XACRO (XML Macros) is an XML macro language that adds macros and expressions to URDF, making it more readable and maintainable. Understanding these formats is essential for representing your humanoid robot in simulation and real-world applications.

## URDF Basics

URDF defines a robot as a collection of rigid bodies (links) connected by joints. Each link can have visual, collision, and inertial properties.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## URDF Components

### Links

Links represent rigid bodies in the robot. Each link can have:

- **Visual**: How the link appears in simulation/visualization
- **Collision**: How the link interacts with physics simulation
- **Inertial**: Physical properties for dynamics simulation

#### Visual Properties
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Geometry can be one of: box, cylinder, sphere, or mesh -->
    <box size="1 1 1"/>
    <!-- OR -->
    <cylinder radius="0.5" length="1.0"/>
    <!-- OR -->
    <sphere radius="0.5"/>
    <!-- OR -->
    <mesh filename="package://my_robot/meshes/link.stl"/>
  </geometry>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</visual>
```

#### Collision Properties
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="1 1 1"/>
  </geometry>
</collision>
```

#### Inertial Properties
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
</inertial>
```

### Joints

Joints connect links and define their relative motion. Joint types include:

- **fixed**: No movement between links
- **revolute**: Rotational joint with limits
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint with limits
- **floating**: 6 DOF with no limits
- **planar**: Movement on a plane

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="1 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## XACRO: Making URDF More Maintainable

XACRO adds macro capabilities to URDF, allowing for:

- **Property definitions**: Define constants
- **Macros**: Reusable components
- **Mathematical expressions**: Calculate values
- **Conditional blocks**: Include/exclude parts

### Basic XACRO Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot_xacro" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Properties -->
  <xacro:property name="base_radius" value="0.2"/>
  <xacro:property name="base_height" value="0.6"/>
  <xacro:property name="wheel_radius" value="0.05"/>
  <xacro:property name="wheel_width" value="0.1"/>

  <!-- Define a macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>

    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Define the base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="${base_height}" radius="${base_radius}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="${base_height}" radius="${base_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="2.0"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.3 0.3 -0.3"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.3 -0.3 -0.3"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.3 0.3 -0.3"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.3 -0.3 -0.3"/>
</robot>
```

## Humanoid Robot URDF Example

Here's a simplified URDF for a basic humanoid robot with legs:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Hip joint and link -->
  <joint name="hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="hip_link"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <link name="hip_link">
    <visual>
      <geometry>
        <box size="0.25 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.25 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Left leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="hip_link"/>
    <child link="left_thigh"/>
    <origin xyz="-0.05 0.15 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.15 0.2 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.2 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Right leg (similar structure) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="hip_link"/>
    <child link="right_thigh"/>
    <origin xyz="-0.05 -0.15 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.15 0.2 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.2 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>
</robot>
```

## Working with URDF in ROS 2

### Loading URDF into ROS 2

To use your URDF model in ROS 2, you typically load it using the `robot_state_publisher` node:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from math import sin, cos
import numpy as np

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # 10Hz
        
    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)  # Initialize all to 0
        
        # Simulate simple movement
        t = self.get_clock().now().nanoseconds / 1e9
        msg.position[0] = 0.3 * sin(t)    # Left hip
        msg.position[1] = -0.5 * sin(t)   # Left knee
        msg.position[3] = 0.3 * sin(t)    # Right hip
        msg.position[4] = -0.5 * sin(t)   # Right knee
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File for Robot State Publisher

You'll typically use a launch file to load the URDF and start the robot state publisher:

```python
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('my_humanoid_robot')
    urdf_file = os.path.join(pkg_share, 'urdf', 'simple_humanoid.urdf.xacro')
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': Command(['xacro ', urdf_file])}
        ]
    )
    
    joint_state_publisher_node = Node(
        package='my_humanoid_robot',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node
    ])
```

## Best Practices for URDF/XACRO

1. **Start Simple**: Begin with a basic model and add complexity gradually
2. **Use Units**: Always specify units (meters for distances, kg for mass)
3. **Realistic Inertias**: Calculate or estimate inertial properties accurately
4. **Use XACRO**: Leverage XACRO for reusability and cleaner definitions
5. **Validate**: Check your URDF with tools like `check_urdf` before simulation
6. **Documentation**: Comment your URDF to explain complex parts
7. **Mesh Organization**: Keep mesh files in `meshes/` directory within your package

## Common Issues and Solutions

- **Invalid Inertias**: Make sure diagonal elements of inertia matrix are positive
- **Joint Limits**: Set appropriate limits for realistic movement ranges
- **Collision vs Visual**: Use simpler geometry for collision than visual when possible
- **Mass Distribution**: Ensure mass is distributed realistically across links

## Summary

URDF and XACRO form the foundation of robot representation in ROS 2. They allow you to define:
- Physical structure (links and joints)
- Visual appearance
- Collision properties
- Inertial properties for simulation

With the combination of URDF for robot description and the joint state publisher for animation, you can represent your humanoid robot in both simulation and real-world applications. The next section will cover parameter management and launch files to coordinate multiple nodes in your ROS 2 system.