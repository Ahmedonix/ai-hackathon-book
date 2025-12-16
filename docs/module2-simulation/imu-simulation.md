# Implementing IMU Sensor Simulation in Gazebo

## Overview

Inertial Measurement Units (IMUs) are critical sensors for humanoid robots, providing essential data about the robot's orientation, angular velocity, and linear acceleration. Accurate IMU simulation is vital for tasks such as balance control, navigation, and motion planning. This section covers implementing various IMU configurations in Gazebo.

## Understanding IMU Requirements for Humanoid Robots

### 1. IMU Placement Strategy

For humanoid robots, IMU placement is critical for balance and motion control:

- **Torso/Middle**: Provides overall body orientation and balance information
- **Head**: For head tracking and visual orientation
- **Limbs**: For detailed motion tracking of arms/legs
- **Feet**: For ground contact detection and foot placement

### 2. IMU Data Types

IMUs typically provide:
- **Orientation (Quaternion)**: 3D orientation of the sensor
- **Angular Velocity**: Rate of rotation around each axis
- **Linear Acceleration**: Acceleration along each axis (including gravity)

## Basic IMU Implementation

### 1. IMU Mounting Link

Create a mounting link for the IMU in your URDF:

```xml
<!-- IMU mounting link -->
<link name="imu_mount">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<!-- Fixed joint to mount IMU in torso -->
<joint name="imu_mount_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_mount"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>  <!-- Near center of mass -->
</joint>
```

### 2. Basic IMU Sensor Implementation

```xml
<!-- IMU link -->
<link name="imu_link">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<!-- Joint connecting IMU to mount -->
<joint name="imu_joint" type="fixed">
  <parent link="imu_mount"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>

<!-- Gazebo IMU sensor definition -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>  <!-- 100Hz for balance control -->
    <visualize>false</visualize>
    <imu>
      <!-- Angular velocity parameters -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001745329</stddev> <!-- ~0.1 deg/s in rad/s -->
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0010</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001745329</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0010</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001745329</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0010</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <!-- Linear acceleration parameters -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev> <!-- 1.7% of 1G -->
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0098</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0098</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0098</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    
    <!-- Plugin for ROS 2 integration -->
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>imu</topicName>
      <serviceName>imu_service</serviceName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>
```

## Advanced IMU Configurations

### 1. High-Performance IMU for Balance Control

For critical balance applications, use higher accuracy settings:

```xml
<gazebo reference="high_performance_imu_link">
  <sensor name="high_performance_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>  <!-- Higher rate for precise control -->
    <visualize>false</visualize>
    <imu>
      <!-- More accurate angular velocity (like ADIS16448) -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.000174533</stddev> <!-- ~0.01 deg/s in rad/s -->
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.000174533</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.000174533</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <!-- More accurate linear acceleration -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev> <!-- 0.17% of 1G -->
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.00098</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.00098</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
            <bias_mean>0.0000</bias_mean>
            <bias_stddev>0.00098</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    
    <plugin name="high_performance_imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>200.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>imu_high_performance</topicName>
      <serviceName>imu_hp_service</serviceName>
      <gaussianNoise>0.001</gaussianNoise>
      <frameName>high_performance_imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>
```

### 2. Multiple IMU Configuration for Humanoid Robot

For comprehensive motion sensing, implement multiple IMUs:

```xml
<!-- Head IMU for head orientation -->
<link name="head_imu_link">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="head_imu_joint" type="fixed">
  <parent link="head"/>
  <child link="head_imu_link"/>
  <origin xyz="0 0 0.05" rpy="0 0 0"/>
</joint>

<gazebo reference="head_imu_link">
  <sensor name="head_imu" type="imu">
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
    
    <plugin name="head_imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>head</bodyName>
      <topicName>imu/head</topicName>
      <frameName>head_imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>

<!-- Foot IMUs for ground contact detection -->
<link name="left_foot_imu_link">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="left_foot_imu_joint" type="fixed">
  <parent link="left_ankle"/>
  <child link="left_foot_imu_link"/>
  <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
</joint>

<gazebo reference="left_foot_imu_link">
  <sensor name="left_foot_imu" type="imu">
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
    
    <plugin name="left_foot_imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>left_ankle</bodyName>
      <topicName>imu/left_foot</topicName>
      <frameName>left_foot_imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>

<!-- Right foot IMU (similar to left) -->
<link name="right_foot_imu_link">
  <inertial>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="right_foot_imu_joint" type="fixed">
  <parent link="right_ankle"/>
  <child link="right_foot_imu_link"/>
  <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
</joint>

<gazebo reference="right_foot_imu_link">
  <sensor name="right_foot_imu" type="imu">
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
    
    <plugin name="right_foot_imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>100.0</updateRate>
      <bodyName>right_ankle</bodyName>
      <topicName>imu/right_foot</topicName>
      <frameName>right_foot_imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>
```

## IMU with Magnetometer for Absolute Orientation

### 1. Compass Simulation

For applications requiring absolute heading information:

```xml
<gazebo reference="imu_mag_link">
  <sensor name="imu_mag_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>50</update_rate>
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
    
    <!-- Plugin for IMU with magnetometer -->
    <plugin name="imu_mag_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>50.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>imu_mag</topicName>
      <serviceName>imu_mag_service</serviceName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>imu_mag_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>
```

## Performance Optimization

### 1. Multi-Priority IMU Setup

Create different IMUs for different purposes:

```xml
<!-- High-rate IMU for balance control -->
<gazebo reference="balance_imu_link">
  <sensor name="balance_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>200</update_rate>  <!-- High rate for balance -->
    <visualize>false</visualize>
    <!-- [IMU configuration details] -->
  </sensor>
</gazebo>

<!-- Low-rate IMU for navigation -->
<gazebo reference="nav_imu_link">
  <sensor name="nav_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>20</update_rate>   <!-- Lower rate for navigation -->
    <visualize>false</visualize>
    <!-- [IMU configuration details] -->
  </sensor>
</gazebo>
```

### 2. Adaptive IMU Configuration

For systems with limited computational resources:

```xml
<!-- Less accurate but more efficient IMU -->
<gazebo reference="efficient_imu_link">
  <sensor name="efficient_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>50</update_rate>   <!-- Lower update rate -->
    <visualize>false</visualize>
    <imu>
      <!-- Reduced noise parameters for better performance -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.005</stddev>  <!-- Higher noise for better performance -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.005</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.005</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.05</stddev>  <!-- Higher noise for better performance -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.05</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.05</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="efficient_imu_controller" filename="libgazebo_ros_imu.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>50.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>imu/efficient</topicName>
      <frameName>efficient_imu_link</frameName>
      <initialOrientationAsReference>false</initialOrientationAsReference>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Calibration Considerations

### 1. Simulated Calibration Parameters

In real applications, IMUs require calibration. You can simulate this:

```xml
<!-- IMU with calibration offsets -->
<gazebo reference="calibrated_imu_link">
  <sensor name="calibrated_imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <!-- Angular velocity with bias simulation -->
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.001</mean>  <!-- Simulated bias -->
            <stddev>0.001745329</stddev>
            <bias_mean>0.001</bias_mean>  <!-- Additional calibration offset -->
            <bias_stddev>0.0005</bias_stddev>
          </noise>
        </x>
        <!-- Similar for Y and Z axes -->
      </angular_velocity>
      
      <!-- Linear acceleration with bias simulation -->
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.01</mean>  <!-- Simulated bias -->
            <stddev>0.017</stddev>
            <bias_mean>0.01</bias_mean>  <!-- Additional calibration offset -->
            <bias_stddev>0.008</bias_stddev>
          </noise>
        </x>
        <!-- Similar for Y and Z axes -->
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

## Testing IMU Implementation

### 1. Launch Configuration

Create a launch file to test IMU implementation:

```python
# launch/imu_test.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both'
    )
    
    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='both'
    )
    
    # IMU processor node
    imu_processor = Node(
        package='imu_tools',
        executable='r2b_imu_processor',
        name='imu_processor',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'rviz',
            'imu_test.rviz'
        ])],
        output='screen'
    )
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        imu_processor,
        rviz
    ])
```

### 2. Verification Commands

Test your IMU implementation:

```bash
# Check if IMU topic is publishing
ros2 topic echo /imu

# Check IMU message info
ros2 topic info /imu

# Monitor IMU data statistics
ros2 run plotjuggler plotjuggler -d /imu

# Check multiple IMU topics (for multi-IMU setup)
ros2 topic list | grep imu

# Visualize in RViz2 (add an Imu display and set the topic to /imu)
```

## Troubleshooting Common Issues

### 1. No IMU Data

If the IMU is not publishing data:
- Verify the IMU plugin is loaded correctly
- Check that the IMU link is properly connected in URDF
- Ensure the Gazebo simulation is running

### 2. Constant Values

If IMU values remain constant:
- Check that the IMU is attached to a moving part of the robot
- Verify that the robot is actually moving in simulation
- Ensure physics properties are properly configured

### 3. Unexpected Drift

If IMU shows drift over time:
- This is expected behavior in real IMUs; verify noise parameters are realistic
- Check that the update rate is appropriate for your application
- Verify that integration algorithms account for drift

### 4. High-Frequency Noise

If IMU values are too noisy:
- Check that the noise parameters are appropriate for your simulation
- Consider whether some applications need filtering
- Verify that the update rate matches your processing capabilities

## Integration with Humanoid Control Systems

### 1. Balance Control Integration

IMUs are critical for humanoid balance:

```python
# Example ROS 2 node for balance control using IMU data
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np
import math


class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')
        
        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Publisher for joint position commands
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )
        
        # Control loop
        self.control_timer = self.create_timer(0.01, self.balance_control_loop)
        
        # State variables
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angular_velocity = Vector3()
        
    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation (in a real system, you'd integrate angular velocity 
        # or use sensor fusion to get orientation)
        _, self.pitch, self.roll = self.quaternion_to_euler(
            msg.orientation.x,
            msg.orientation.y, 
            msg.orientation.z,
            msg.orientation.w
        )
        
        # Store angular velocity
        self.angular_velocity = msg.angular_velocity
        
    def balance_control_loop(self):
        """Main balance control loop"""
        # Simple PD controller concept (real implementation would be more complex)
        if abs(self.roll) > 0.1 or abs(self.pitch) > 0.1:
            # Generate joint commands to counteract imbalance
            cmd = Float64MultiArray()
            # This would contain actual control algorithm to adjust joint positions
            # for balance maintenance
            self.joint_cmd_pub.publish(cmd)
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
```

## Next Steps

With IMU sensors properly implemented, you'll next explore environment design and world-building in Gazebo. The environment in which your humanoid robot operates is crucial for testing its capabilities under various conditions.

The IMU implementation provides essential data for balance control, navigation, and motion planning in your humanoid robot system.