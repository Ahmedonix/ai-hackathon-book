# Agent Context for ROS 2 Fundamentals Module

## Technology Stack
- **ROS 2 Version**: Iron
- **Programming Language**: Python 3.10+
- **Primary Client Library**: rclpy
- **Robot Description**: URDF/XACRO
- **Transform Library**: TF2
- **Build System**: Colcon
- **Simulation Environment**: Gazebo (for later modules)
- **Target Platform**: Ubuntu 22.04 LTS

## Key Concepts to Understand
1. **Nodes**: Independent computational units that communicate via messages
2. **Topics**: Asynchronous communication for data streams (publish/subscribe)
3. **Services**: Synchronous request/response communication
4. **Actions**: Asynchronous goal-oriented communication
5. **Messages**: Data structures for communication between nodes
6. **Launch Files**: Configuration files to start multiple nodes together
7. **URDF/XACRO**: XML-based robot description format with macro support
8. **TF2**: Transform library for coordinate frame management

## ROS 2 Architecture Patterns
- Publisher-Subscriber pattern for data streams
- Client-Service pattern for request-response communication
- Action pattern for complex, goal-oriented tasks
- Parameter server for configuration management
- Launch system for multi-node orchestration

## Humanoid Robotics Concepts
- Joint state management (positions, velocities, efforts)
- Coordinate frame transformations (TF2)
- Simple bipedal model with 6-12 DOF legs and basic torso
- Sensor integration (IMU, joint encoders)
- Rule-based AI agent for high-level control

## Educational Approach
- Beginner-to-intermediate level concepts
- Practical hands-on examples
- Step-by-step workflows
- Code examples tested in ROS 2 Iron environment
- Integration with AI agent demonstration