# Practical Hands-On Exercise: Gazebo Setup for Humanoid Robotics

## Exercise Overview

In this hands-on exercise, you'll set up a complete Gazebo simulation environment specifically for humanoid robotics. You'll learn to configure Gazebo with proper physics, integrate it with ROS 2, and validate the setup with a simple humanoid model.

## Learning Objectives

By the end of this exercise, you will be able to:
1. Install and configure Gazebo Garden for humanoid robotics applications
2. Set up the necessary ROS 2 Gazebo integration packages
3. Create and test a simple humanoid robot model in Gazebo
4. Validate that sensors are properly configured and publishing data
5. Debug common setup issues

## Prerequisites

Before starting this exercise, ensure you have:
- Ubuntu 22.04 LTS installed
- ROS 2 Iron properly installed and sourced
- Basic familiarity with terminal commands
- Internet access to download required packages

## Part 1: Installing Gazebo Garden

### Step 1.1: Add the Gazebo Package Repository

Open a terminal and run the following commands to add the Gazebo repository to your system:

```bash
# Add the Gazebo repository
sudo sh -c 'echo "deb [arch=amd64] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

# Add the Gazebo signing key
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

# Update package list
sudo apt update
```

### Step 1.2: Install Gazebo Garden

```bash
# Install Gazebo Garden
sudo apt install gazebo
```

### Step 1.3: Verify Installation

Test that Gazebo is properly installed:

```bash
# Open Gazebo GUI
gz sim

# You should see the Gazebo interface with a default empty world.
# Close the GUI by clicking the X button.
```

## Part 2: Installing ROS 2 Integration Packages

### Step 2.1: Install Required ROS 2 Gazebo Packages

```bash
# Source ROS 2 Iron
source /opt/ros/iron/setup.bash

# Install Gazebo ROS packages
sudo apt install ros-iron-gazebo-ros ros-iron-gazebo-plugins ros-iron-gazebo-dev

# Install additional useful packages
sudo apt install ros-iron-joint-state-publisher ros-iron-robot-state-publisher ros-iron-xacro
```

### Step 2.2: Create a Workspace for Your Robot Package

```bash
# Create workspace directory
mkdir -p ~/humanoid_ws/src

# Navigate to workspace
cd ~/humanoid_ws

# Source ROS 2 before building
source /opt/ros/iron/setup.bash

# Build the workspace initially (will have nothing to build yet)
colcon build --symlink-install
```

### Step 2.3: Source the Workspace

```bash
# Add to your bashrc to source automatically (optional)
echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc

# Source the workspace for this session
source ~/humanoid_ws/install/setup.bash
```

## Part 3: Creating a Simple Humanoid Robot Model

### Step 3.1: Create a Robot Package

```bash
# Navigate to the src directory of your workspace
cd ~/humanoid_ws/src

# Create a new package for your humanoid robot
ros2 pkg create --build-type ament_python humanoid_simple_robot --dependencies rclpy std_msgs sensor_msgs geometry_msgs

# Navigate to the package directory
cd humanoid_simple_robot
```

### Step 3.2: Create a URDF Model

Create a simple humanoid robot model by creating a URDF file. First, create the necessary directories:

```bash
# Create directories for URDF files
mkdir -p urdf
mkdir -p launch
```

Now create the URDF file `urdf/simple_humanoid.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
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

  <!-- Simple LiDAR sensor -->
  <link name="lidar">
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

  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for ROS integration -->
  <gazebo>
    <plugin filename="libgazebo_ros_joint_state_publisher.so" name="joint_state_publisher">
      <ros>
        <namespace>/simple_humanoid</namespace>
        <argument>~/out:=joint_states</argument>
      </ros>
      <update_rate>30</update_rate>
    </plugin>
  </gazebo>

  <!-- Gazebo plugins for the LiDAR -->
  <gazebo reference="lidar">
    <sensor name="lidar_sensor" type="ray">
      <always_on>true</always_on>
      <visualize>false</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/simple_humanoid</namespace>
          <argument>~/out:=scan</argument>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Step 3.3: Create a Launch File

Create a launch file to spawn your robot in Gazebo. Create the file `launch/spawn_humanoid.launch.py`:

```python
# launch/spawn_humanoid.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Get the package share directory
    pkg_share = get_package_share_directory('humanoid_simple_robot')
    
    # Launch Gazebo
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
                'simple_humanoid.urdf'
            ])
        }],
        output='screen'
    )
    
    # Joint State Publisher (for non-controlled joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )
    
    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0', '-y', '0', '-z', '1.0'  # Spawn 1m above ground to ensure it lands properly
        ],
        output='screen'
    )
    
    # Create the world file if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    if not os.path.exists(worlds_dir):
        os.makedirs(worlds_dir)
    
    # Create a simple world file
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
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity
    ])
```

## Part 4: Building and Testing the Setup

### Step 4.1: Build Your Package

```bash
# Navigate back to the workspace root
cd ~/humanoid_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build the package
colcon build --packages-select humanoid_simple_robot

# Source the workspace again to update with your changes
source ~/humanoid_ws/install/setup.bash
```

### Step 4.2: Test the Robot in Gazebo

```bash
# Launch the simulation
ros2 launch humanoid_simple_robot spawn_humanoid.launch.py
```

You should see:
1. Gazebo window opens with your humanoid robot model
2. The robot should fall onto the ground plane due to gravity
3. The robot should remain stable after landing

### Step 4.3: Verify Sensor Data

Open a new terminal (while keeping the simulation running) and check if sensor data is being published:

```bash
# Source the workspace
source ~/humanoid_ws/install/setup.bash

# Check if the scan topic is being published
ros2 topic echo /simple_humanoid/scan --field ranges

# In another terminal, check joint states
ros2 topic echo /joint_states

# Verify all expected topics are present
ros2 topic list | grep simple_humanoid
```

Expected output should include:
- `/simple_humanoid/scan` - LiDAR scan data
- `/joint_states` - Joint position/velocity/effort data

## Part 5: Troubleshooting Common Issues

### Issue 1: Robot Doesn't Appear in Gazebo

**Symptoms:** Gazebo launches but the robot doesn't appear.

**Solutions:**
1. Check the spawn_entity output in the terminal for error messages
2. Verify the URDF file has correct syntax: `check_urdf ~/humanoid_ws/src/humanoid_simple_robot/urdf/simple_humanoid.urdf`
3. Ensure the robot_description parameter is correctly set

### Issue 2: Robot Falls Through Ground

**Symptoms:** Robot falls infinitely through the ground.

**Solutions:**
1. Verify that the robot model has proper collision geometry
2. Check that the physics parameters are properly configured
3. Ensure the robot has mass and inertia values

### Issue 3: No Sensor Data Published

**Symptoms:** Robot appears but sensor topics show no data.

**Solutions:**
1. Check if plugins are properly defined in the URDF
2. Verify that the Gazebo ROS packages are installed
3. Look for error messages in the Gazebo console

## Part 6: Validation and Assessment

### Step 6.1: Validate Setup Completion

Complete the following validation checks:

1. **Robot Visualization**: Verify the humanoid robot model appears correctly in Gazebo
2. **Physics Simulation**: Confirm the robot responds to gravity and remains stable
3. **Joint States**: Verify that joint positions are being published via `/joint_states`
4. **Sensor Data**: Validate that LiDAR data is being published via `/simple_humanoid/scan`
5. **RViz Visualization** (Bonus): Launch RViz to visualize the robot:
   ```bash
   ros2 run rviz2 rviz2
   # In RViz, set the Fixed Frame to 'base_link' and add a RobotModel display
   # Subscribe to 'robot_description' topic and set the TF topic to '/tf'
   ```

### Step 6.2: Self-Assessment Questions

After completing the exercise, answer these questions:

1. What command would you use to check which topics are currently active in your ROS system?
2. How would you verify that your robot's physics properties (mass, inertia) are correctly configured?
3. What would happen if you removed the `<gazebo>` plugin section from your URDF?
4. How can you adjust the physics update rate in Gazebo to improve simulation stability?

## Bonus Challenge

Modify the URDF to add a simple camera sensor to the robot's head and verify that it publishes image data. Hint: You'll need to add a camera sensor plugin similar to the LiDAR sensor.

## Exercise Completion

Congratulations! You have successfully:
- Set up Gazebo Garden with ROS 2 integration
- Created a simple humanoid robot model with sensors
- Validated that the robot appears correctly in simulation
- Confirmed that sensor data is being published properly
- Learned to troubleshoot common setup issues

This foundation will allow you to build more complex humanoid robot simulations with confidence that your basic setup is working correctly.