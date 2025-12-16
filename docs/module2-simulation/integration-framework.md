# Integrating ROS 2 with Both Gazebo and Unity Simulators

## Overview

In humanoid robotics development, combining the physics accuracy of Gazebo with the visualization capabilities of Unity provides a comprehensive simulation environment. This section covers how to properly integrate ROS 2 with both simulators to create a powerful development and testing platform for humanoid robots.

## Architecture Overview

### 1. System Architecture

The integrated system architecture involves:

```
ROS 2 Nodes
     |
[rosbridge_suite / ROS TCP Connector]
     |
--------------------------------------
     |              |
   Gazebo        Unity
   (Physics)    (Visualization)
     |              |
   Sensors &    High-fidelity
   Actuators    Rendering
```

The architecture enables:
- Physics-accurate simulation in Gazebo
- High-fidelity visualization in Unity
- Synchronized state between both simulators
- Realistic sensor data generation
- Natural human-robot interaction

### 2. Data Flow Patterns

The system supports multiple data flow patterns:

**Pattern A: Gazebo-Driven Simulation**
- Gazebo handles physics and robot dynamics
- Unity visualizes the robot state from Gazebo
- Sensors modeled in Gazebo, visualized in Unity
- Commands originate from ROS 2 nodes, executed in Gazebo

**Pattern B: Unity-Enhanced Visualization**
- Gazebo maintains physics simulation
- Unity enhances visual representation
- Unity provides user interface elements
- Sensor data processed in ROS 2, visualized in Unity

## Setting Up the Integration Framework

### 1. ROS Bridge Components

The integration relies on two main ROS bridge components:

**For Gazebo:**
- `ros_gazebo_pkgs` - Provides communication between ROS 2 and Gazebo
- `gazebo_ros` - Gazebo plugins for ROS 2 integration
- Standard ROS 2 message publishers/subscribers for sensor and control data

**For Unity:**
- `ROS TCP Connector` - Enables TCP-based communication with Unity
- Custom message serializers for Unity compatibility
- Network configuration for reliable communication

### 2. Network Configuration

Configure network settings for reliable communication between all components:

```bash
# In your terminal, set up ROS environment for multi-machine communication
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0  # Enable communication with external systems like Unity
```

For Unity-ROS communication, ensure the TCP connector endpoint is running:

```bash
# Start the ROS TCP endpoint server
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000
```

## Implementation Patterns

### 1. State Synchronization Framework

Create a framework to synchronize state between Gazebo and Unity:

```python
# scripts/state_synchronizer.py
#!/usr/bin/env python3

"""
State synchronization node for Gazebo-Unity integration.
Synchronizes robot state between physics simulation (Gazebo) and visualization (Unity).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import math


class StateSynchronizer(Node):
    def __init__(self):
        super().__init__('state_synchronizer')
        
        # Subscribers for Gazebo data
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers for Unity visualization
        self.unity_joint_pub = self.create_publisher(
            JointState,
            '/unity/joint_states',
            10
        )
        
        self.unity_pose_pub = self.create_publisher(
            Pose,
            '/unity/robot_pose',
            10
        )
        
        # Timer for state synchronization
        self.sync_timer = self.create_timer(0.033, self.sync_callback)  # ~30 Hz
        
        # Internal state storage
        self.current_joint_state = JointState()
        self.current_odom = Odometry()
        self.last_sync_time = self.get_clock().now()
        
        self.get_logger().info('State Synchronizer Node Initialized')

    def joint_state_callback(self, msg):
        """Store joint state from Gazebo"""
        self.current_joint_state = msg

    def odom_callback(self, msg):
        """Store odometry from Gazebo"""
        self.current_odom = msg

    def sync_callback(self):
        """Synchronize state to Unity"""
        # Publish joint states to Unity
        if self.current_joint_state.name:  # Check if we have valid data
            unity_joint_msg = JointState()
            unity_joint_msg.header = Header()
            unity_joint_msg.header.stamp = self.get_clock().now().to_msg()
            unity_joint_msg.header.frame_id = 'unity_robot'
            
            # Copy joint names and positions
            unity_joint_msg.name = self.current_joint_state.name
            unity_joint_msg.position = self.current_joint_msg.position
            unity_joint_msg.velocity = self.current_joint_state.velocity
            unity_joint_msg.effort = self.current_joint_state.effort
            
            self.unity_joint_pub.publish(unity_joint_msg)
        
        # Publish pose to Unity
        unity_pose_msg = Pose()
        unity_pose_msg.position = self.current_odom.pose.pose.position
        unity_pose_msg.orientation = self.current_odom.pose.pose.orientation
        self.unity_pose_pub.publish(unity_pose_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('State Synchronizer Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    synchronizer = StateSynchronizer()
    
    try:
        rclpy.spin(synchronizer)
    except KeyboardInterrupt:
        synchronizer.get_logger().info('Node interrupted by user')
    finally:
        synchronizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Sensor Data Integration

Create a sensor integrator that collects data from Gazebo and prepares it for Unity visualization:

```python
# scripts/sensor_integrator.py
#!/usr/bin/env python3

"""
Sensor data integrator for Gazebo-Unity integration.
Processes sensor data from Gazebo and formats it for Unity visualization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, CameraInfo, Imu
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np


class SensorIntegrator(Node):
    def __init__(self):
        super().__init__('sensor_integrator')
        
        # Subscribers for Gazebo sensor data
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Publishers for Unity visualization
        self.unity_laser_pub = self.create_publisher(
            MarkerArray,
            '/unity/laser_scan',
            10
        )
        
        self.unity_camera_pub = self.create_publisher(
            Image,
            '/unity/camera/display',
            10
        )
        
        self.unity_imu_pub = self.create_publisher(
            Float32MultiArray,
            '/unity/imu_visualization',
            10
        )
        
        # Internal storage
        self.last_laser_scan = None
        self.last_camera_image = None
        self.last_imu_data = None
        
        # Visualization parameters
        self.laser_point_size = 0.05
        self.visualization_frame = 'unity_visualization'
        
        self.get_logger().info('Sensor Integrator Node Initialized')

    def laser_callback(self, msg):
        """Process laser scan data from Gazebo"""
        self.last_laser_scan = msg
        self.process_laser_scan(msg)

    def camera_callback(self, msg):
        """Process camera data from Gazebo"""
        self.last_camera_image = msg
        # Forward camera data to Unity with possible format conversion
        self.unity_camera_pub.publish(msg)

    def imu_callback(self, msg):
        """Process IMU data from Gazebo"""
        self.last_imu_data = msg
        self.process_imu_data(msg)

    def process_laser_scan(self, scan_msg):
        """Convert laser scan to Unity visualization markers"""
        marker_array = MarkerArray()
        
        # Create markers for each laser point
        for i, range_val in enumerate(scan_msg.ranges):
            if not (np.isnan(range_val) or np.isinf(range_val)) and range_val <= scan_msg.range_max:
                # Calculate angle for this laser beam
                angle = scan_msg.angle_min + (i * scan_msg.angle_increment)
                
                # Calculate Cartesian coordinates
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                
                # Create marker for this point
                marker = Marker()
                marker.header = scan_msg.header
                marker.header.frame_id = self.visualization_frame
                marker.ns = "laser_scan"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = self.laser_point_size
                marker.scale.y = self.laser_point_size
                marker.scale.z = self.laser_point_size
                
                marker.color.a = 1.0  # Don't forget to set the alpha!
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                
                marker_array.markers.append(marker)
        
        self.unity_laser_pub.publish(marker_array)

    def process_imu_data(self, imu_msg):
        """Process IMU data for Unity visualization"""
        # Create a simple representation of IMU orientation
        imu_viz = Float32MultiArray()
        imu_viz.data = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ]
        
        self.unity_imu_pub.publish(imu_viz)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Sensor Integrator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    integrator = SensorIntegrator()
    
    try:
        rclpy.spin(integrator)
    except KeyboardInterrupt:
        integrator.get_logger().info('Node interrupted by user')
    finally:
        integrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Control Command Routing

Implement a command router that directs commands to the appropriate simulator:

```python
# scripts/control_router.py
#!/usr/bin/env python3

"""
Control command router for Gazebo-Unity integration.
Routes control commands to the appropriate simulator based on configuration.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
import json


class ControlRouter(Node):
    def __init__(self):
        super().__init__('control_router')
        
        # Configuration for routing
        self.use_gazebo_for_physics = True  # Set to True for physics simulation in Gazebo
        self.enable_unity_feedback = True   # Enable Unity to receive state feedback
        
        # Subscribers for commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            '/joint_commands',
            self.joint_cmd_callback,
            10
        )
        
        self.nav_goal_sub = self.create_subscription(
            Pose,
            '/goal_pose',
            self.nav_goal_callback,
            10
        )
        
        # Publishers for Gazebo control
        self.gazebo_cmd_vel_pub = self.create_publisher(
            Twist,
            '/gazebo/cmd_vel',
            10
        )
        
        self.gazebo_joint_pub = self.create_publisher(
            JointState,
            '/gazebo/joint_commands',
            10
        )
        
        self.gazebo_nav_goal_pub = self.create_publisher(
            Pose,
            '/gazebo/goal_pose',
            10
        )
        
        # Publishers for Unity visualization
        self.unity_cmd_vel_pub = self.create_publisher(
            Twist,
            '/unity/cmd_vel',
            10
        )
        
        self.unity_joint_pub = self.create_publisher(
            JointState,
            '/unity/joint_commands_visualization',
            10
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/integration_status',
            10
        )
        
        # Timer for status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info('Control Router Node Initialized')

    def cmd_vel_callback(self, msg):
        """Route velocity commands"""
        if self.use_gazebo_for_physics:
            # Send to Gazebo for physics simulation
            self.gazebo_cmd_vel_pub.publish(msg)
        
        # Always send to Unity for visualization feedback
        if self.enable_unity_feedback:
            self.unity_cmd_vel_pub.publish(msg)

    def joint_cmd_callback(self, msg):
        """Route joint commands"""
        if self.use_gazebo_for_physics:
            # Send to Gazebo for physics simulation
            self.gazebo_joint_pub.publish(msg)
        
        # Always send to Unity for visualization feedback
        if self.enable_unity_feedback:
            self.unity_joint_pub.publish(msg)

    def nav_goal_callback(self, msg):
        """Route navigation goals"""
        if self.use_gazebo_for_physics:
            # Send to Gazebo for navigation simulation
            self.gazebo_nav_goal_pub.publish(msg)

    def publish_status(self):
        """Publish integration status"""
        status = {
            'use_gazebo_for_physics': self.use_gazebo_for_physics,
            'enable_unity_feedback': self.enable_unity_feedback,
            'timestamp': self.get_clock().now().seconds_nanoseconds()
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Control Router Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    router = ControlRouter()
    
    try:
        rclpy.spin(router)
    except KeyboardInterrupt:
        router.get_logger().info('Node interrupted by user')
    finally:
        router.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Unity Integration Components

### 1. Unity Message Handling

Create Unity scripts to handle messages from ROS:

```csharp
// Scripts/UnityROSIntegrator.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class UnityROSIntegrator : MonoBehaviour
{
    [Header("Integration Settings")]
    public string rosIpAddress = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Visualization Settings")]
    public GameObject robotModel;
    public bool visualizeLaserScan = true;
    public bool visualizeCamera = true;
    
    private ROSConnection rosConnection;
    
    void Start()
    {
        // Initialize ROS connection
        rosConnection = ROSConnection.instance;
        if (rosConnection == null)
        {
            rosConnection = gameObject.AddComponent<ROSConnection>();
        }
        rosConnection.Initialize(rosIpAddress, rosPort);
        
        // Subscribe to relevant topics
        rosConnection.Subscribe<JointStateMsg>("/unity/joint_states", OnJointStateReceived);
        rosConnection.Subscribe<PoseMsg>("/unity/robot_pose", OnPoseReceived);
        rosConnection.Subscribe<TwistMsg>("/unity/cmd_vel", OnCmdVelReceived);
        
        Debug.Log("Unity-ROS Integration Initialized");
    }
    
    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update robot model based on joint states
        if (robotModel != null)
        {
            for (int i = 0; i < jointState.name.Length; i++)
            {
                string jointName = jointState.name[i];
                double jointPosition = jointState.position[i];
                
                // Find and update the corresponding joint in the Unity model
                Transform jointTransform = robotModel.transform.FindRecursive(jointName);
                if (jointTransform != null)
                {
                    // Convert ROS joint position to Unity rotation
                    float angle = (float)(jointPosition * Mathf.Rad2Deg);
                    jointTransform.Rotate(Vector3.right, angle);
                }
            }
        }
    }
    
    void OnPoseReceived(PoseMsg pose)
    {
        // Update robot position and orientation in Unity
        if (robotModel != null)
        {
            robotModel.transform.position = new Vector3(
                (float)pose.position.x,
                (float)pose.position.y,
                (float)pose.position.z
            );
            
            robotModel.transform.rotation = new Quaternion(
                (float)pose.orientation.x,
                (float)pose.orientation.y,
                (float)pose.orientation.z,
                (float)pose.orientation.w
            );
        }
    }
    
    void OnCmdVelReceived(TwistMsg cmd)
    {
        // Process velocity commands for visualization
        Debug.Log($"Received cmd_vel: linear=({cmd.linear.x}, {cmd.linear.y}, {cmd.linear.z}), " +
                  $"angular=({cmd.angular.x}, {cmd.angular.y}, {cmd.angular.z})");
    }
    
    void Update()
    {
        // Additional update logic for Unity visualization
    }
}

// Extension method to help find transforms recursively
public static class TransformExtension
{
    public static Transform FindRecursive(this Transform parent, string name)
    {
        if (parent.name == name)
            return parent;
            
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform result = parent.GetChild(i).FindRecursive(name);
            if (result != null)
                return result;
        }
        
        return null;
    }
}
```

### 2. Visualization Control

Add visualization control components:

```csharp
// Scripts/VisualizationController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Visualization;

public class VisualizationController : MonoBehaviour
{
    [Header("Visualization Components")]
    public GameObject laserScanVisualization;
    public GameObject cameraFeedDisplay;
    public GameObject pathVisualization;
    
    [Header("Performance Settings")]
    public float maxVisualizationRate = 30.0f; // Hz
    private float lastUpdateTime;
    
    void Start()
    {
        lastUpdateTime = Time.time;
        
        // Subscribe to visualization topics
        ROSConnection.instance.Subscribe<MarkerArrayMsg>("/unity/laser_scan", OnLaserScanReceived);
        ROSConnection.instance.Subscribe<MarkerArrayMsg>("/unity/path_plan", OnPathReceived);
        
        Debug.Log("Visualization Controller Initialized");
    }
    
    void Update()
    {
        // Limit update rate to maintain performance
        if (Time.time - lastUpdateTime < 1.0f / maxVisualizationRate)
        {
            return;
        }
        
        lastUpdateTime = Time.time;
        
        // Additional visualization update logic
        UpdateVisualization();
    }
    
    void UpdateVisualization()
    {
        // Update various visualization components based on settings
        if (laserScanVisualization != null)
        {
            laserScanVisualization.SetActive(ROSConnection.instance != null);
        }
        
        if (cameraFeedDisplay != null)
        {
            cameraFeedDisplay.SetActive(ROSConnection.instance != null);
        }
    }
    
    void OnLaserScanReceived(MarkerArrayMsg markerArray)
    {
        // Update laser scan visualization
        if (laserScanVisualization != null)
        {
            foreach (var marker in markerArray.markers)
            {
                UpdateLaserPoint(marker);
            }
        }
    }
    
    void OnPathReceived(MarkerArrayMsg markerArray)
    {
        // Update path visualization
        if (pathVisualization != null)
        {
            // Process path markers
            foreach (var marker in markerArray.markers)
            {
                UpdatePathPoint(marker);
            }
        }
    }
    
    void UpdateLaserPoint(RosMessageTypes.Visualization.MarkerMsg marker)
    {
        // Create or update laser point visualization
        // Implementation depends on your specific visualization needs
    }
    
    void UpdatePathPoint(RosMessageTypes.Visualization.MarkerMsg marker)
    {
        // Create or update path point visualization
        // Implementation depends on your specific visualization needs
    }
}
```

## Launch Configuration

### 1. Integration Launch File

Create a launch file to start the entire integration:

```python
# launch/gazebo_unity_integration.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    gazebo_world = LaunchConfiguration('gazebo_world', default='empty.sdf')
    
    # Get package directory
    pkg_share = get_package_share_directory('humanoid_description')  # Adjust package name as needed
    
    # Launch Gazebo with specified world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([get_package_share_directory('gazebo_ros'), 'worlds', gazebo_world]),
            'gui': 'true',
            'verbose': 'false'
        }.items()
    )
    
    # Launch ROS TCP endpoint for Unity
    ros_tcp_endpoint = ExecuteProcess(
        cmd=['ros2', 'run', 'ros_tcp_endpoint', 'default_server_endpoint', '--ros-args', 
             '-p', 'ROS_IP:=127.0.0.1', '-p', 'ROS_TCP_PORT:=10000'],
        output='screen'
    )
    
    # State synchronization node
    state_synchronizer = Node(
        package='humanoid_description',  # Adjust package name
        executable='state_synchronizer',
        name='state_synchronizer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Sensor integrator node
    sensor_integrator = Node(
        package='humanoid_description',  # Adjust package name
        executable='sensor_integrator',
        name='sensor_integrator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Control router node
    control_router = Node(
        package='humanoid_description',  # Adjust package name
        executable='control_router',
        name='control_router',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # RViz for additional visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'rviz',
            'gazebo_unity_integration.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Launch sequence with delays to ensure proper initialization
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo first
        gazebo,
        
        # Launch ROS TCP endpoint
        ros_tcp_endpoint,
        
        # Launch state synchronization after a delay
        TimerAction(
            period=3.0,
            actions=[state_synchronizer]
        ),
        
        # Launch sensor integrator
        TimerAction(
            period=4.0,
            actions=[sensor_integrator]
        ),
        
        # Launch control router
        TimerAction(
            period=4.0,
            actions=[control_router]
        ),
        
        # Launch publishers
        TimerAction(
            period=1.0,
            actions=[robot_state_publisher]
        ),
        
        TimerAction(
            period=2.0,
            actions=[joint_state_publisher]
        ),
        
        # Launch RViz
        TimerAction(
            period=5.0,
            actions=[rviz]
        ),
    ])
```

## Performance Optimization

### 1. Network Optimization

Optimize network communication between ROS, Gazebo, and Unity:

```python
# performance_monitor.py
#!/usr/bin/env python3

"""
Performance monitoring for Gazebo-Unity integration.
Monitors network and processing performance of the integrated system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from builtin_interfaces.msg import Time
import time


class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Publishers for performance metrics
        self.ros_cpu_pub = self.create_publisher(Float32, '/performance/ros_cpu', 10)
        self.network_latency_pub = self.create_publisher(Float32, '/performance/network_latency', 10)
        self.unity_framerate_pub = self.create_publisher(Float32, '/performance/unity_framerate', 10)
        
        # Timers for performance checks
        self.network_timer = self.create_timer(0.1, self.check_network_performance)
        self.system_timer = self.create_timer(1.0, self.check_system_performance)
        
        # Internal tracking
        self.ping_times = []
        self.last_ping_time = None
        
        self.get_logger().info('Performance Monitor Initialized')

    def check_network_performance(self):
        """Check network communication performance"""
        # In a real implementation, this would send ping messages
        # and measure round-trip time to Unity
        latency_msg = Float32()
        latency_msg.data = 0.0  # Placeholder - actual implementation would measure real latency
        self.network_latency_pub.publish(latency_msg)

    def check_system_performance(self):
        """Check overall system performance"""
        # Publish system performance metrics
        cpu_msg = Float32()
        cpu_msg.data = 0.0  # Placeholder - actual implementation would measure CPU usage
        self.ros_cpu_pub.publish(cpu_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Performance Monitor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    monitor = PerformanceMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Node interrupted by user')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Message Optimization

Optimize message structures for efficient communication:

```python
# optimized_messages.py
#!/usr/bin/env python3

"""
Optimized message structures for Gazebo-Unity integration.
Provides efficient data representations for common robot state information.
"""

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion
from builtin_interfaces.msg import Time


class OptimizedRobotState:
    """
    Optimized representation of robot state for efficient transmission.
    Combines position, orientation, and joint states in a single message.
    """
    def __init__(self):
        self.header = Header()
        self.position = Vector3()
        self.orientation = Quaternion()
        self.joint_names = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_efforts = []

    def to_msg(self):
        """Convert to ROS message format"""
        from humanoid_msgs.msg import RobotState  # Assuming custom message
        
        msg = RobotState()
        msg.header = self.header
        msg.position = self.position
        msg.orientation = self.orientation
        msg.joint_names = self.joint_names
        msg.joint_positions = self.joint_positions
        msg.joint_velocities = self.joint_velocities
        msg.joint_efforts = self.joint_efforts
        
        return msg

    @classmethod
    def from_msg(cls, msg):
        """Create from ROS message"""
        state = cls()
        state.header = msg.header
        state.position = msg.position
        state.orientation = msg.orientation
        state.joint_names = msg.joint_names
        state.joint_positions = msg.joint_positions
        state.joint_velocities = msg.joint_velocities
        state.joint_efforts = msg.joint_efforts
        return state

    def compress(self):
        """Compress the state data for transmission"""
        # Implementation would depend on specific compression needs
        # This could apply quantization, delta encoding, etc.
        pass


class OptimizedSensorData:
    """
    Optimized representation of sensor data for efficient transmission.
    Combines multiple sensor readings in a single message.
    """
    def __init__(self):
        self.header = Header()
        self.laser_scan = None  # Compressed laser scan data
        self.camera_data = None  # Compressed camera data
        self.imu_data = None  # IMU data
        self.other_sensors = {}  # Dictionary for other sensor types

    def to_msg(self):
        """Convert to ROS message format"""
        from humanoid_msgs.msg import SensorData  # Assuming custom message
        
        msg = SensorData()
        msg.header = self.header
        # Convert and assign other fields
        # Implementation depends on exact message specification
        
        return msg

    @classmethod
    def from_msg(cls, msg):
        """Create from ROS message"""
        # Implementation depends on exact message specification
        pass
```

## Troubleshooting Integration Issues

### 1. Common Integration Problems

**Network Connectivity Issues:**
- Verify ROS IP configuration between all components
- Check firewall settings that may block communication
- Ensure ROS TCP endpoint is running with correct parameters
- Test connectivity with simple ping messages

**Synchronization Problems:**
- Check timestamp consistency between Gazebo and Unity
- Verify that update rates are appropriate for both systems
- Implement buffer mechanisms for handling latency differences

**Performance Degradation:**
- Monitor update rates and message frequencies
- Implement message throttling where appropriate
- Use visualization LODs to maintain performance

### 2. Debugging Tools

Create debugging nodes to help troubleshoot integration issues:

```python
# debug_integration.py
#!/usr/bin/env python3

"""
Debugging tools for Gazebo-Unity integration.
Provides diagnostic information about the integration status.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
import time


class IntegrationDebugger(Node):
    def __init__(self):
        super().__init__('integration_debugger')
        
        # Publishers for debug information
        self.gazebo_status_pub = self.create_publisher(String, '/debug/gazebo_status', 10)
        self.unity_status_pub = self.create_publisher(String, '/debug/unity_status', 10)
        self.sync_status_pub = self.create_publisher(String, '/debug/sync_status', 10)
        self.connection_status_pub = self.create_publisher(Bool, '/debug/connection_status', 10)
        
        # Subscribers to monitor system state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Timer for status updates
        self.debug_timer = self.create_timer(2.0, self.publish_debug_info)
        
        # Internal tracking
        self.last_joint_update = None
        self.joint_update_count = 0
        
        self.get_logger().info('Integration Debugger Initialized')

    def joint_state_callback(self, msg):
        """Track joint state updates"""
        self.last_joint_update = self.get_clock().now()
        self.joint_update_count += 1

    def publish_debug_info(self):
        """Publish debug information"""
        # Check Gazebo status
        gazebo_status = String()
        if self.last_joint_update is not None:
            time_since_update = (self.get_clock().now() - self.last_joint_update).nanoseconds / 1e9
            if time_since_update < 1.0:  # If we received data in the last second
                gazebo_status.data = f"Active - {self.joint_update_count} updates, {time_since_update:.2f}s since last"
            else:
                gazebo_status.data = f"Inactive - {time_since_update:.2f}s since last update"
        else:
            gazebo_status.data = "No data received"
        
        self.gazebo_status_pub.publish(gazebo_status)
        
        # Check connection status
        connection_status = Bool()
        connection_status.data = self.last_joint_update is not None
        self.connection_status_pub.publish(connection_status)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Integration Debugger Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    debugger = IntegrationDebugger()
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info('Node interrupted by user')
    finally:
        debugger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Best Practices for Integration

### 1. Architecture Guidelines

- **Separation of Concerns**: Keep physics simulation in Gazebo, visualization in Unity
- **Consistent Time**: Use ROS time across all components for proper synchronization
- **Error Handling**: Implement robust error handling for network disconnections
- **Performance Monitoring**: Continuously monitor performance metrics

### 2. Development Workflow

- **Iterative Development**: Test components individually before integration
- **Version Control**: Track both ROS package and Unity project changes
- **Documentation**: Maintain clear documentation of the integration architecture
- **Testing**: Create automated tests for the integrated system

## Next Steps

With the integration framework established, you'll next connect Gazebo sensor data to ROS 2 nodes, enabling the full sensor pipeline in your integrated simulation environment.

The integration architecture you've implemented provides a solid foundation for combining the physics capabilities of Gazebo with the visualization power of Unity for your humanoid robot.