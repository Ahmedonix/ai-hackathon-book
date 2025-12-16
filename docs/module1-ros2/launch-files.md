---
sidebar_position: 6
---

# Parameter Management and Launch Files

## Overview

In ROS 2, parameter management and launch files are crucial for configuring and starting complex systems with multiple nodes. Parameters allow you to configure node behavior without recompiling, while launch files provide a way to start multiple nodes with appropriate configurations simultaneously. This is especially important for humanoid robots, which typically involve many coordinated nodes working together.

## Parameters in ROS 2

Parameters are named values that control how nodes behave. They can be set at runtime and changed dynamically, making them ideal for configuration without code changes.

### Setting Parameters from Code

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class ParameterExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_example_node')
        
        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('wheel_diameter', 0.1)
        
        # Access parameter values
        robot_name = self.get_parameter('robot_name').value
        max_speed = self.get_parameter('max_speed').value
        wheel_diameter = self.get_parameter('wheel_diameter').value
        
        self.get_logger().info(f'Robot: {robot_name}, Max Speed: {max_speed}, Wheel Diameter: {wheel_diameter}')
        
        # Set a parameter callback to handle dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterExampleNode()
    
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

### Setting Parameters from Command Line

Parameters can be set when launching a node:

```bash
ros2 run my_package parameter_example_node --ros-args -p robot_name:=my_robot -p max_speed:=2.0
```

### YAML Parameter Files

Create parameter files in YAML format to manage configurations:

```yaml
# config/humanoid_params.yaml
parameter_example_node:
  ros__parameters:
    robot_name: "humanoid_robot"
    max_speed: 1.0
    wheel_diameter: 0.1
    joints:
      hip:
        min_angle: -1.0
        max_angle: 1.0
      knee:
        min_angle: -2.0
        max_angle: 0.5
      ankle:
        min_angle: -0.5
        max_angle: 0.5
```

Load parameters from YAML file:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

class ParameterFileExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_file_example_node')
        
        # Declare parameters with descriptions
        self.declare_parameter(
            'robot_name', 
            'default_robot',
            ParameterDescriptor(description='Name of the robot')
        )
        
        # Access nested parameters
        hip_params = self.get_parameter('joints.hip').value
        if hip_params:
            min_angle = hip_params.get('min_angle', -1.0)
            max_angle = hip_params.get('max_angle', 1.0)
            self.get_logger().info(f'Hip joint limits: {min_angle} to {max_angle}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterFileExampleNode()
    
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

## Launch Files

Launch files allow you to start multiple nodes with specific configurations simultaneously. They are written in Python using the `launch` library.

### Basic Launch File Structure

```python
# launch/humanoid_basic.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Declare launch argument
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    # Create nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([FindPackageShare('my_humanoid_package'), 'config', 'robot_params.yaml'])
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    joint_state_publisher = Node(
        package='my_humanoid_package',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        robot_state_publisher,
        joint_state_publisher
    ])
```

### Launch File with Parameter Files

```python
# launch/humanoid_with_params.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_humanoid_package'),
            'config',
            'humanoid_params.yaml'
        ]),
        description='Path to parameter file'
    )
    
    # Robot state publisher with parameters
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            use_sim_time,
            config_file
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{config_file}]
    )
    
    # Joint state publisher GUI (optional)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_gui'))
    )
    
    # Declare GUI argument
    declare_use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='false',
        description='Enable joint_state_publisher_gui'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_config_file,
        declare_use_gui,
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui
    ])
```

### Launch File with Multiple Robot Configurations

```python
# launch/humanoid_multi_config.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    config_file = LaunchConfiguration('config_file')
    enable_gazebo = LaunchConfiguration('enable_gazebo')
    
    # Launch argument declarations
    launch_args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('robot_name', default_value='humanoid'),
        DeclareLaunchArgument('config_file', default_value='humanoid_config.yaml'),
        DeclareLaunchArgument('enable_gazebo', default_value='false'),
    ]
    
    # Robot-specific nodes
    robot_group = GroupAction(
        actions=[
            PushRosNamespace(robot_name),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[
                    {'robot_description': open(FindPackageShare('my_humanoid_package') + '/urdf/humanoid.urdf.xacro').read()}
                ]
            ),
            Node(
                package='my_humanoid_package',
                executable='humanoid_controller',
                name='humanoid_controller',
                parameters=[{'use_sim_time': use_sim_time}]
            )
        ]
    )
    
    # Gazebo simulation nodes (only if enabled)
    gazebo_nodes = [
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', robot_name,
                '-file', FindPackageShare('my_humanoid_package') + '/urdf/humanoid.urdf.xacro',
                '-robot_namespace', robot_name
            ],
            output='screen',
            condition=IfCondition(enable_gazebo)
        )
    ]
    
    return LaunchDescription(
        launch_args + 
        [robot_group] + 
        gazebo_nodes
    )
```

## Advanced Launch Features

### Conditional Launch

```python
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    enable_logging = LaunchConfiguration('enable_logging')
    use_gpu = LaunchConfiguration('use_gpu')
    
    declare_enable_logging = DeclareLaunchArgument('enable_logging', default_value='true')
    declare_use_gpu = DeclareLaunchArgument('use_gpu', default_value='false')
    
    # Node that only runs if logging is enabled
    logging_node = Node(
        package='my_humanoid_package',
        executable='data_logger',
        name='data_logger',
        condition=IfCondition(enable_logging)
    )
    
    # Different nodes based on hardware
    perception_node = Node(
        package='my_humanoid_package',
        executable='perception_node_cpu' if LaunchConfiguration('use_gpu') == 'false' else 'perception_node_gpu',
        name='perception_node',
        parameters=[{'use_gpu': use_gpu}]
    )
    
    return LaunchDescription([
        declare_enable_logging,
        declare_use_gpu,
        logging_node,
        perception_node
    ])
```

### Launch File with Remappings

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Example of remapping topics in launch files
    robot_controller = Node(
        package='my_humanoid_package',
        executable='robot_controller',
        name='robot_controller',
        remappings=[
            ('/joint_commands', '/humanoid/joint_commands'),
            ('/sensor_data', '/humanoid/sensor_data'),
            ('/robot_state', '/humanoid/robot_state')
        ],
        parameters=[
            {'control_frequency': 50.0},
            {'max_joint_speed': 2.0}
        ]
    )
    
    return LaunchDescription([
        robot_controller
    ])
```

## Best Practices for Parameters and Launch Files

### Parameter Best Practices
1. **Use Descriptive Names**: Name parameters clearly to indicate their purpose
2. **Set Default Values**: Always provide sensible defaults
3. **Group Related Parameters**: Use namespaces for related settings
4. **Document Parameters**: Add descriptions to parameters using ParameterDescriptor
5. **Validate Values**: Validate parameter values at runtime
6. **Use Proper Types**: Declare parameters with correct types (int, double, string, bool)

### Launch File Best Practices
1. **Modular Design**: Break complex systems into smaller launch files
2. **Use Launch Arguments**: Make launch files flexible with arguments
3. **Include Descriptions**: Document launch arguments with descriptions
4. **Use Path Substitutions**: Use PathJoinSubstitution for file paths
5. **Handle Conditions**: Use conditions for optional components
6. **Parameter Files**: Externalize parameters in YAML files
7. **Namespace Management**: Use namespaces for multi-robot systems

## Real-World Example: Humanoid Robot Launch

Here's a complete example of launching a humanoid robot system:

```python
# launch/humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    config_dir = LaunchConfiguration('config_dir')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_humanoid',
        description='Robot name for namespacing'
    )
    
    declare_config_dir = DeclareLaunchArgument(
        'config_dir',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_humanoid_package'),
            'config'
        ]),
        description='Path to configuration files'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([config_dir, 'robot_params.yaml'])
        ]
    )
    
    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rate': 50.0},
            PathJoinSubstitution([config_dir, 'joints_params.yaml'])
        ]
    )
    
    # Robot controller
    robot_controller = Node(
        package='my_humanoid_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([config_dir, 'controller_params.yaml'])
        ],
        remappings=[
            ('/joint_states', 'joint_states'),
            ('/cmd_vel', 'cmd_vel')
        ]
    )
    
    # Diagnostics aggregator
    diagnostics_aggregator = Node(
        package='diagnostic_aggregator',
        executable='aggregator_node',
        name='diagnostics_aggregator',
        parameters=[
            PathJoinSubstitution([config_dir, 'diagnostics.yaml'])
        ]
    )
    
    # Launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_config_dir)
    
    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_controller)
    ld.add_action(diagnostics_aggregator)
    
    return ld
```

## Summary

In this section, we've covered two critical aspects of ROS 2 systems:

1. **Parameter Management**: Using parameters to configure nodes at runtime, with support for:
   - Default values
   - Command-line overrides
   - YAML configuration files
   - Dynamic parameter changes

2. **Launch Files**: Using Python launch files to:
   - Start multiple nodes simultaneously
   - Configure nodes with parameters
   - Handle conditional launching
   - Remap topics
   - Create modular launch systems

These tools are essential for creating robust, configurable humanoid robot systems that can adapt to different hardware configurations, operating environments, and operational requirements without code changes. The next section will cover integrating AI agents with ROS 2 nodes.