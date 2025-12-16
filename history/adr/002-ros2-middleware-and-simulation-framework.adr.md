# 002: ROS 2 Middleware and Simulation Framework

Date: 2025-01-15

## Status

Accepted

## Context

We need to establish the foundational communication system for the humanoid robotics platform. This system must support real-time communication between robot components, be widely adopted in robotics research and industry, support both low-level control and high-level planning, and integrate well with simulation environments and AI frameworks. The chosen middleware must be suitable for educational purposes and support the progression from basic to advanced robotics concepts.

## Decision

We will use ROS 2 Iron as the primary middleware framework with the following components:
- ROS Distribution: ROS 2 Iron (Irrwaddy) for latest features and long-term support until 2027
- Simulation: Gazebo for physics simulation and Unity via Robotics Hub for enhanced visualization
- ROS Client Library: rclpy for Python-based nodes and controllers
- Robot Description: URDF/XACRO for robot modeling
- Navigation: Nav2 for path planning and navigation capabilities

## Consequences

Positive:
- ROS 2 Iron provides the latest features and active community support
- Gazebo offers accurate physics simulation essential for realistic robot behavior
- Unity adds high-fidelity visualization and interaction capabilities
- Large community and extensive documentation available
- Industry standard for robotics development
- Good integration with AI and perception libraries
- Comprehensive tooling for debugging and visualization

Negative:
- Steep learning curve for beginners
- Complex setup and configuration requirements
- Performance overhead compared to custom solutions
- Platform dependencies may cause compatibility issues
- Hardware requirements for simulation (especially Isaac Sim) are demanding

## Alternatives

1. ROS 2 Humble: Longer support period (until 2027) but older features and fewer improvements
2. ROS 2 Rolling: Cutting-edge features but lack of stability and long-term support
3. Custom middleware: Complete control over architecture but significant development overhead and maintenance
4. Other robotics frameworks like Webots or CoppeliaSim: Different ecosystems that might not align with industry standards
5. Only Gazebo without Unity: Adequate physics simulation but less visual appeal and interaction capabilities

## References

- plan.md: Technical Context section
- research.md: ROS 2 Iron decision and Simulation Strategy
- data-model.md: ROS2Node and SimulationEnvironment entities