# launch/humanoid_robot_system.launch.py
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
            FindPackageShare('humanoid_examples'),
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
        package='humanoid_examples',
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