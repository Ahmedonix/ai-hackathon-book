# Practical Hands-On Exercise: Sensor Simulation for Humanoid Robotics

## Exercise Overview

In this hands-on exercise, you'll create and configure various sensors for your humanoid robot in simulation. You'll learn to implement LiDAR, camera, and IMU sensors in Gazebo, validate their functionality, and process the sensor data in ROS 2.

## Learning Objectives

By the end of this exercise, you will be able to:
1. Configure LiDAR, camera, and IMU sensors in Gazebo
2. Implement proper sensor mounting on your humanoid robot
3. Validate that sensors publish correct data to ROS 2 topics
4. Process sensor data in ROS 2 nodes for perception tasks
5. Troubleshoot common sensor simulation issues

## Prerequisites

Before starting this exercise, you should have:
- Completed the Gazebo setup exercise or have a functioning simulation environment
- Basic knowledge of ROS 2 concepts (topics, messages, nodes)
- The simple humanoid robot model from the previous exercise
- Access to a ROS 2 Iron workspace

## Part 1: Review Your Robot Model

### Step 1.1: Verify Existing Robot Model

First, make sure your robot model from the previous exercise is properly set up:

```bash
# Navigate to your workspace
cd ~/humanoid_ws

# Source your workspace
source install/setup.bash

# Check if your robot model URDF exists
ls src/humanoid_simple_robot/urdf/simple_humanoid.urdf
```

### Step 1.2: Launch Robot Model to Verify

```bash
# Run the previous exercise's launch file to verify everything works
ros2 launch humanoid_simple_robot spawn_humanoid.launch.py
```

Keep Gazebo running for the next steps.

## Part 2: Adding LiDAR Sensors

### Step 2.1: Understanding LiDAR Configuration

In your previous exercise, you already added a basic LiDAR sensor. Let's enhance it and add a second one for comparison.

First, let's create a more comprehensive URDF with multiple sensor types. Create `urdf/advanced_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="advanced_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include basic humanoid from previous exercise -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1.0 1.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
  </joint>

  <!-- Left Leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="-0.1 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <link name="left_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0015" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip"/>
    <child link="left_knee"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="100.0" velocity="2.0"/>
    <dynamics damping="2.0" friction="0.1"/>
  </joint>

  <link name="left_ankle">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_knee"/>
    <child link="left_ankle"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.5"/>
    <dynamics damping="0.5" friction="0.05"/>
  </joint>

  <!-- Right Leg (mirror of left leg) -->
  <link name="right_hip">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="0.1 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <link name="right_knee">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0015" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip"/>
    <child link="right_knee"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="1.57" effort="100.0" velocity="2.0"/>
    <dynamics damping="2.0" friction="0.1"/>
  </joint>

  <link name="right_ankle">
    <visual>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_knee"/>
    <child link="right_ankle"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.5"/>
    <dynamics damping="0.5" friction="0.05"/>
  </joint>

  <!-- LiDAR Sensor on top of head -->
  <link name="lidar_top">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="lidar_top_joint" type="fixed">
    <parent link="head"/>
    <child link="lidar_top"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Front-facing LiDAR on chest -->
  <link name="lidar_front">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.03"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.2 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.08"/>
      <inertia ixx="0.00005" ixy="0.0" ixz="0.0" iyy="0.00005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <joint name="lidar_front_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_front"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Camera in head -->
  <link name="camera">
    <visual>
      <geometry>
        <box size="0.02 0.08 0.04"/>
      </geometry>
      <material name="black">
        <color rgba="0.0 0.0 0.0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.08 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- IMU in torso -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <!-- Joint state publisher -->
  <gazebo>
    <plugin filename="libgazebo_ros_joint_state_publisher.so" name="joint_state_publisher">
      <ros>
        <namespace>/advanced_humanoid</namespace>
        <remapping>~/out:=joint_states</remapping>
      </ros>
      <update_rate>30</update_rate>
    </plugin>
  </gazebo>

  <!-- LiDAR on top of head -->
  <gazebo reference="lidar_top">
    <sensor name="lidar_top_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14</min_angle>
            <max_angle>3.14</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_top_controller_plugin" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/advanced_humanoid</namespace>
          <remapping>~/out:=top_scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_top</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Front-facing LiDAR -->
  <gazebo reference="lidar_front">
    <sensor name="lidar_front_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>15</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>180</samples>
            <resolution>1</resolution>
            <min_angle>-1.57</min_angle>
            <max_angle>1.57</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>8.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_front_controller_plugin" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/advanced_humanoid</namespace>
          <remapping>~/out:=front_scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_front</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera">
    <sensor name="camera_sensor" type="camera">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.0472</horizontal_fov> <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/advanced_humanoid</namespace>
          <remapping>image_raw:=image_raw</remapping>
          <remapping>camera_info:=camera_info</remapping>
        </ros>
        <frame_name>camera</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001745329</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001745329</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001745329</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_controller" filename="libgazebo_ros_imu_sensor.so">
        <ros>
          <namespace>/advanced_humanoid</namespace>
          <remapping>~/out:=imu</remapping>
        </ros>
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Step 2.2: Create Updated Launch File

Create a launch file for the advanced robot: `launch/sensor_demo.launch.py`

```python
# launch/sensor_demo.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_simple_robot')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_share, 'worlds', 'simple_humanoid_world.sdf']),
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_simple_robot'),
                'urdf',
                'advanced_humanoid.urdf'
            ])
        }],
        output='screen'
    )
    
    # Spawn Entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'advanced_humanoid',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )
    
    # Sensor validation node (created in next section)
    sensor_validator = Node(
        package='humanoid_simple_robot',
        executable='sensor_validator',
        name='sensor_validator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Create worlds directory if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    if not os.path.exists(worlds_dir):
        os.makedirs(worlds_dir)
    
    # Create a simple world file if it doesn't exist
    world_file = os.path.join(worlds_dir, 'simple_humanoid_world.sdf')
    if not os.path.exists(world_file):
        with open(world_file, 'w') as f:
            f.write("""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_humanoid_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>""")
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo first
        gazebo,
        
        # Launch robot state publisher after a delay
        TimerAction(
            period=2.0,
            actions=[robot_state_publisher]
        ),
        
        # Launch spawn entity after more delay
        TimerAction(
            period=4.0,
            actions=[spawn_entity]
        ),
        
        # Launch sensor validator after robot is in simulation
        TimerAction(
            period=6.0,
            actions=[sensor_validator]
        ),
    ])
```

## Part 3: Creating Sensor Processing Nodes

### Step 3.1: Create the Sensor Validator Node

First, create a directory for scripts in your package:

```bash
# In your terminal, navigate to the package directory
cd ~/humanoid_ws/src/humanoid_simple_robot

# Create the script directory if it doesn't exist
mkdir -p scripts
```

Create the sensor validator script: `scripts/sensor_validator.py`

```python
#!/usr/bin/env python3

"""
Sensor Validator Node for Humanoid Robot Simulation.
Monitors and validates that all sensors are publishing data correctly.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, JointState
from std_msgs.msg import String
import time


class SensorValidatorNode(Node):
    def __init__(self):
        super().__init__('sensor_validator_node')
        
        # Track sensor data
        self.top_scan_received = False
        self.front_scan_received = False
        self.camera_received = False
        self.imu_received = False
        self.joint_states_received = False
        
        self.top_scan_count = 0
        self.front_scan_count = 0
        self.camera_count = 0
        self.imu_count = 0
        self.joint_states_count = 0
        
        # Setup subscribers
        self.top_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/top_scan',
            self.top_scan_callback,
            10
        )
        
        self.front_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/front_scan',
            self.front_scan_callback,
            10
        )
        
        self.camera_sub = self.create_subscription(
            Image,
            '/advanced_humanoid/image_raw',
            self.camera_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/advanced_humanoid/imu',
            self.imu_callback,
            10
        )
        
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/advanced_humanoid/joint_states',
            self.joint_states_callback,
            10
        )
        
        # Publisher for validation status
        self.status_pub = self.create_publisher(
            String,
            '/sensor_validation_status',
            10
        )
        
        # Setup timer for validation checks
        self.validation_timer = self.create_timer(2.0, self.validate_sensors)
        
        self.get_logger().info('Sensor Validator Node Started')

    def top_scan_callback(self, msg):
        """Handle top LiDAR scan messages"""
        self.top_scan_received = True
        self.top_scan_count += 1

    def front_scan_callback(self, msg):
        """Handle front LiDAR scan messages"""
        self.front_scan_received = True
        self.front_scan_count += 1

    def camera_callback(self, msg):
        """Handle camera image messages"""
        self.camera_received = True
        self.camera_count += 1

    def imu_callback(self, msg):
        """Handle IMU messages"""
        self.imu_received = True
        self.imu_count += 1

    def joint_states_callback(self, msg):
        """Handle joint state messages"""
        self.joint_states_received = True
        self.joint_states_count += 1

    def validate_sensors(self):
        """Validate that all sensors are working"""
        validation_status = []
        
        if self.top_scan_received:
            validation_status.append(f"Top LiDAR OK ({self.top_scan_count} msgs)")
        else:
            validation_status.append("Top LiDAR: NO DATA")
        
        if self.front_scan_received:
            validation_status.append(f"Front LiDAR OK ({self.front_scan_count} msgs)")
        else:
            validation_status.append("Front LiDAR: NO DATA")
        
        if self.camera_received:
            validation_status.append(f"Camera OK ({self.camera_count} imgs)")
        else:
            validation_status.append("Camera: NO DATA")
        
        if self.imu_received:
            validation_status.append(f"IMU OK ({self.imu_count} msgs)")
        else:
            validation_status.append("IMU: NO DATA")
        
        if self.joint_states_received:
            validation_status.append(f"Joint States OK ({self.joint_states_count} msgs)")
        else:
            validation_status.append("Joint States: NO DATA")
        
        # Determine overall status
        all_working = all([
            self.top_scan_received,
            self.front_scan_received, 
            self.camera_received,
            self.imu_received,
            self.joint_states_received
        ])
        
        overall_status = "ALL_SENSORS_WORKING" if all_working else "ISSUES_DETECTED"
        
        status_msg = String()
        status_msg.data = f"{overall_status} - {', '.join(validation_status)}"
        self.status_pub.publish(status_msg)
        
        # Log the status
        if all_working:
            self.get_logger().info(f"✓ All sensors working: {status_msg.data}")
        else:
            self.get_logger().warn(f"✗ Sensor issues detected: {status_msg.data}")

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Sensor Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator_node = SensorValidatorNode()
    
    try:
        rclpy.spin(validator_node)
    except KeyboardInterrupt:
        validator_node.get_logger().info('Node interrupted by user')
    finally:
        validator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3.2: Create a Sensor Processing Node

Create `scripts/laser_processor.py` for processing LiDAR data:

```python
#!/usr/bin/env python3

"""
LiDAR Data Processor Node for Humanoid Robot Simulation.
Processes and analyzes LiDAR data from multiple sensors.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String
import numpy as np


class LaserProcessorNode(Node):
    def __init__(self):
        super().__init__('laser_processor_node')
        
        # Subscriptions for both LiDAR sensors
        self.top_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/top_scan',
            self.top_scan_callback,
            10
        )
        
        self.front_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/front_scan',
            self.front_scan_callback,
            10
        )
        
        # Publishers for processed data
        self.top_processed_pub = self.create_publisher(
            Float64MultiArray,
            '/processed/top_scan',
            10
        )
        
        self.front_processed_pub = self.create_publisher(
            Float64MultiArray,
            '/processed/front_scan',
            10
        )
        
        self.obstacle_detection_pub = self.create_publisher(
            String,
            '/environment_status',
            10
        )
        
        # Processing parameters
        self.min_obstacle_distance = 0.5  # meters
        self.front_view_angle = 0.52  # radians (~30 degrees for front detection)
        
        self.get_logger().info('Laser Processor Node Started')

    def top_scan_callback(self, msg):
        """Process top LiDAR scan data"""
        try:
            # Calculate processed metrics
            metrics = self.process_laser_scan(msg)
            
            # Publish processed data
            processed_msg = Float64MultiArray()
            processed_msg.data = metrics
            self.top_processed_pub.publish(processed_msg)
            
            # Log if we detect nearby obstacles
            min_distance = min(metrics[:3])  # First few values contain min distances
            if min_distance < self.min_obstacle_distance:
                status_msg = String()
                status_msg.data = f"OBSTACLE_DETECTED_TOP: {min_distance:.2f}m"
                self.obstacle_detection_pub.publish(status_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing top scan: {str(e)}')

    def front_scan_callback(self, msg):
        """Process front LiDAR scan data"""
        try:
            # Calculate processed metrics
            metrics = self.process_laser_scan(msg)
            
            # Publish processed data
            processed_msg = Float64MultiArray()
            processed_msg.data = metrics
            self.front_processed_pub.publish(processed_msg)
            
            # Check for obstacles in front view
            front_obstacle = self.check_front_obstacles(msg)
            if front_obstacle:
                status_msg = String()
                status_msg.data = f"OBSTACLE_DETECTED_FRONT: {front_obstacle:.2f}m"
                self.obstacle_detection_pub.publish(status_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing front scan: {str(e)}')

    def process_laser_scan(self, scan_msg):
        """Process laser scan to extract meaningful metrics"""
        # Convert to numpy array for efficient processing
        ranges = np.array(scan_msg.ranges)
        
        # Filter out invalid ranges (infinity, NaN)
        valid_ranges = ranges[np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)]
        
        if len(valid_ranges) > 0:
            min_distance = float(np.min(valid_ranges))
            max_distance = float(np.max(valid_ranges))
            avg_distance = float(np.mean(valid_ranges))
            valid_points = len(valid_ranges)
        else:
            min_distance = float('inf')
            max_distance = 0.0
            avg_distance = 0.0
            valid_points = 0
        
        return [
            min_distance,      # Minimum distance to object
            max_distance,      # Maximum distance to object
            avg_distance,      # Average distance to objects
            float(valid_points)  # Number of valid points
        ]

    def check_front_obstacles(self, scan_msg):
        """Check for obstacles in the front view of the robot"""
        # Calculate indices for the front portion of the scan
        total_beams = len(scan_msg.ranges)
        front_start_idx = int(total_beams / 2 - (self.front_view_angle / scan_msg.angle_increment) / 2)
        front_end_idx = int(total_beams / 2 + (self.front_view_angle / scan_msg.angle_increment) / 2)
        
        # Ensure indices are within bounds
        front_start_idx = max(0, front_start_idx)
        front_end_idx = min(total_beams, front_end_idx)
        
        # Get front range readings
        front_ranges = scan_msg.ranges[front_start_idx:front_end_idx]
        
        # Filter valid ranges
        valid_front_ranges = [r for r in front_ranges if np.isfinite(float(r))]
        
        if valid_front_ranges:
            closest_front = min(valid_front_ranges)
            if closest_front < self.min_obstacle_distance:
                return closest_front
        
        return None

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Laser Processor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    processor_node = LaserProcessorNode()
    
    try:
        rclpy.spin(processor_node)
    except KeyboardInterrupt:
        processor_node.get_logger().info('Node interrupted by user')
    finally:
        processor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3.3: Make Scripts Executable and Update Package

```bash
# Make the scripts executable
chmod +x ~/humanoid_ws/src/humanoid_simple_robot/scripts/sensor_validator.py
chmod +x ~/humanoid_ws/src/humanoid_simple_robot/scripts/laser_processor.py

# Update package.xml to include dependencies
# We'll edit the package.xml file to add required dependencies
```

Edit the `package.xml` file to include the required dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_simple_robot</name>
  <version>0.0.0</version>
  <description>Simple humanoid robot for Gazebo simulation</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>cv_bridge</depend>
  <depend>message_filters</depend>

  <exec_depend>gazebo_ros_pkgs</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update the `setup.py` file to include the scripts:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'humanoid_simple_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include URDF and launch files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Simple humanoid robot for Gazebo simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_validator = humanoid_simple_robot.scripts.sensor_validator:main',
            'laser_processor = humanoid_simple_robot.scripts.laser_processor:main',
        ],
    },
)
```

Now create a directory for Python scripts:

```bash
# Create the scripts subdirectory in the package
mkdir -p ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts

# Move the scripts to the correct location
mv ~/humanoid_ws/src/humanoid_simple_robot/scripts/sensor_validator.py \
   ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts/sensor_validator.py

mv ~/humanoid_ws/src/humanoid_simple_robot/scripts/laser_processor.py \
   ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts/laser_processor.py
```

## Part 4: Building and Testing the Sensor Setup

### Step 4.1: Build Your Package

```bash
# Navigate to the workspace root
cd ~/humanoid_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build the package with our new scripts
colcon build --packages-select humanoid_simple_robot

# Source the workspace again
source ~/humanoid_ws/install/setup.bash
```

### Step 4.2: Test the Sensor Setup

```bash
# Launch the sensor demo
ros2 launch humanoid_simple_robot sensor_demo.launch.py
```

In a separate terminal, monitor the sensor data:

```bash
# Source the workspace
source ~/humanoid_ws/install/setup.bash

# Check available topics
ros2 topic list | grep advanced_humanoid

# Monitor the sensor validation status
ros2 topic echo /sensor_validation_status

# Monitor laser scan data
ros2 topic echo /advanced_humanoid/top_scan --field ranges

# Monitor camera data (this will be more verbose)
ros2 topic echo /advanced_humanoid/image_raw --field header

# Monitor processed data
ros2 topic echo /processed/top_scan
```

## Part 5: Processing Sensor Data

### Step 5.1: Create a Perception Node

Create `scripts/perception_node.py` to demonstrate processing of multiple sensor types:

```python
#!/usr/bin/env python3

"""
Perception Node for Humanoid Robot Simulation.
Combines data from multiple sensors to form a unified understanding of the environment.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Twist
import numpy as np
import math


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        
        # Subscriptions for all sensors
        self.top_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/top_scan',
            self.top_scan_callback,
            10
        )
        
        self.front_scan_sub = self.create_subscription(
            LaserScan,
            '/advanced_humanoid/front_scan',
            self.front_scan_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/advanced_humanoid/imu',
            self.imu_callback,
            10
        )
        
        # Publishers for processed information
        self.environment_map_pub = self.create_publisher(
            Float64MultiArray,
            '/environment_map',
            10
        )
        
        self.navigation_commands_pub = self.create_publisher(
            Twist,
            '/navigation_commands',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/perception_status',
            10
        )
        
        # Internal state
        self.last_top_scan = None
        self.last_front_scan = None
        self.last_imu = None
        
        # Processing parameters
        self.safe_distance = 0.8  # meters
        self.turn_threshold = 0.5  # meters for obstacle avoidance
        self.need_turn = False
        
        # Timer for perception processing
        self.perception_timer = self.create_timer(0.5, self.process_environment)
        
        self.get_logger().info('Perception Node Started')

    def top_scan_callback(self, msg):
        """Store top LiDAR data"""
        self.last_top_scan = msg

    def front_scan_callback(self, msg):
        """Store front LiDAR data"""
        self.last_front_scan = msg

    def imu_callback(self, msg):
        """Store IMU data"""
        self.last_imu = msg

    def process_environment(self):
        """Process environment data from all sensors"""
        if self.last_front_scan is None or self.last_top_scan is None:
            return
        
        # Process front scan for obstacle detection
        front_clear = self.is_direction_clear(self.last_front_scan, 0, 0.26)  # ~15 degrees left/right
        left_clear = self.is_direction_clear(self.last_front_scan, -0.52, 0.26)  # ~30 degrees left
        right_clear = self.is_direction_clear(self.last_front_scan, 0.52, 0.26)  # ~30 degrees right
        ahead_clear = front_clear  # Same as front since we're looking forward
        
        # Create environment map
        env_map = Float64MultiArray()
        env_map.data = [
            float(front_clear),
            float(left_clear), 
            float(right_clear),
            float(ahead_clear),
            self.get_closest_object_distance(self.last_front_scan)
        ]
        
        self.environment_map_pub.publish(env_map)
        
        # Make navigation decision based on environment
        cmd = Twist()
        
        if not front_clear:
            # Obstacle ahead - need to turn
            if left_clear and not right_clear:
                # Turn left
                cmd.angular.z = 0.5
                self.need_turn = True
            elif right_clear and not left_clear:
                # Turn right
                cmd.angular.z = -0.5
                self.need_turn = True
            elif left_clear and right_clear:
                # Both directions clear, choose randomly
                cmd.angular.z = 0.5 if np.random.rand() > 0.5 else -0.5
                self.need_turn = True
            else:
                # Nowhere to go, stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        else:
            # Path clear, move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.need_turn = False
        
        # Publish navigation command
        self.navigation_commands_pub.publish(cmd)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Front:{'CLEAR' if front_clear else 'BLOCKED'}, " \
                         f"Left:{'CLEAR' if left_clear else 'BLOCKED'}, " \
                         f"Right:{'CLEAR' if right_clear else 'BLOCKED'}, " \
                         f"Closest:{self.get_closest_object_distance(self.last_front_scan):.2f}m"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f'Environment: {status_msg.data}')

    def is_direction_clear(self, scan_msg, angle_offset, angle_range):
        """Check if a particular direction is clear of obstacles"""
        # Calculate beam indices for the specified angle range
        beam_idx_offset = int(angle_offset / scan_msg.angle_increment)
        beam_idx_range = int(angle_range / scan_msg.angle_increment)
        
        center_beam = int(len(scan_msg.ranges) / 2)
        start_idx = max(0, center_beam + beam_idx_offset - beam_idx_range)
        end_idx = min(len(scan_msg.ranges), center_beam + beam_idx_offset + beam_idx_range)
        
        # Check if any distances in the range are below safe distance
        for i in range(start_idx, end_idx):
            range_val = scan_msg.ranges[i]
            if np.isfinite(range_val) and range_val < self.safe_distance:
                return False
        
        return True

    def get_closest_object_distance(self, scan_msg):
        """Get the distance to the closest object in the scan"""
        valid_ranges = [r for r in scan_msg.ranges if np.isfinite(r)]
        return min(valid_ranges) if valid_ranges else float('inf')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Perception Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    perception_node = PerceptionNode()
    
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Node interrupted by user')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Add the new script to the setup.py and update the entry points:

```bash
# Create the file
mv ~/humanoid_ws/src/humanoid_simple_robot/scripts/perception_node.py \
   ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts/perception_node.py
```

Update the entry points in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'sensor_validator = humanoid_simple_robot.scripts.sensor_validator:main',
        'laser_processor = humanoid_simple_robot.scripts.laser_processor:main',
        'perception_node = humanoid_simple_robot.scripts.perception_node:main',
    ],
},
```

## Part 6: Testing and Validation

### Step 6.1: Build After Adding New Script

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS
source /opt/ros/iron/setup.bash

# Rebuild the package
colcon build --packages-select humanoid_simple_robot

# Source the workspace
source install/setup.bash
```

### Step 6.2: Test All Sensors Together

```bash
# Launch the sensor demo with all nodes
ros2 launch humanoid_simple_robot sensor_demo.launch.py
```

In separate terminals:

```bash
# Terminal 1: Monitor sensor validation
source ~/humanoid_ws/install/setup.bash
ros2 topic echo /sensor_validation_status

# Terminal 2: Monitor perception output
source ~/humanoid_ws/install/setup.bash
ros2 topic echo /perception_status

# Terminal 3: Monitor processed sensor data
source ~/humanoid_ws/install/setup.bash
ros2 topic echo /processed/front_scan

# Terminal 4: Monitor navigation commands
source ~/humanoid_ws/install/setup.bash
ros2 topic echo /navigation_commands
```

### Step 6.3: Visualize Sensor Data in RViz

```bash
# In a new terminal
source ~/humanoid_ws/install/setup.bash
ros2 run rviz2 rviz2
```

In RViz:
1. Set Fixed Frame to "base_link" or "map"
2. Add a LaserScan display and set topic to `/advanced_humanoid/front_scan`
3. Add another LaserScan display for the top sensor `/advanced_humanoid/top_scan`
4. Add a RobotModel display to see the robot model
5. Add TF to visualize coordinate frames

## Part 7: Troubleshooting Common Issues

### Issue 1: No Camera Data

**Symptoms:** Camera topic is available but no images are received.

**Solutions:**
1. Check that `gz sim` is running and the rendering engine works
2. Verify that the camera plugin is properly configured in the URDF
3. Check for OpenGL-related error messages in the console
4. Try installing additional graphics libraries:

```bash
sudo apt update
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

### Issue 2: High CPU Usage from Sensor Processing

**Symptoms:** System becomes sluggish, high CPU usage.

**Solutions:**
1. Reduce sensor update rates in the URDF:
   ```xml
   <update_rate>10</update_rate>  <!-- Instead of 30+ -->
   ```
2. Increase the processing timer intervals in your nodes
3. Implement message throttling using `message_filters`

### Issue 3: Sensor Data Arrives Too Fast

**Symptoms:** Buffer overflow errors, dropped messages.

**Solutions:**
1. Use message filters to sample data at specific intervals
2. Increase subscriber queue sizes
3. Throttle processing with timers

### Issue 4: Sensor Frames Not Aligned

**Symptoms:** Data comes from wrong frame or TF errors.

**Solutions:**
1. Use `ros2 run tf2_tools view_frames` to visualize the TF tree
2. Verify that sensor frames are properly attached to the robot model
3. Check that frame IDs match between Gazebo plugins and ROS nodes

## Part 8: Validation and Assessment

### Step 8.1: Validate Sensor Integration

Complete these validation checks:

1. **LiDAR Sensors**: Verify both top and front LiDAR sensors publish data
2. **Camera Sensor**: Confirm camera publishes image data
3. **IMU Sensor**: Validate IMU publishes orientation and acceleration data
4. **Data Processing**: Verify that your processing nodes handle the data correctly
5. **Integration**: Ensure all sensors work together without interference

### Step 8.2: Self-Assessment Questions

After completing the exercise, answer these questions:

1. What is the difference in configuration between a 2D LiDAR and a 3D LiDAR?
2. How would you calibrate the camera sensor for accurate depth perception?
3. What is the purpose of the noise parameters in the IMU configuration?
4. How can you reduce the computational load of processing high-frequency sensor data?
5. What steps would you take to add a new sensor type (e.g., sonar) to the robot model?

## Bonus Challenge

Create a sensor fusion algorithm that combines LiDAR and IMU data to improve navigation accuracy. Implement a simple Kalman filter that uses IMU data to predict robot position between LiDAR measurements and corrects the prediction when new LiDAR data arrives.

## Exercise Completion

Congratulations! You have successfully:
- Configured multiple sensor types on your humanoid robot
- Implemented ROS 2 nodes to process sensor data
- Validated that sensors publish correct data to ROS topics
- Created a perception system that combines multiple sensor inputs
- Learned to troubleshoot common sensor simulation issues

These skills are fundamental to building a capable humanoid robot that can perceive and navigate in its environment.