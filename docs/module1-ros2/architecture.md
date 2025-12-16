---
sidebar_position: 2
---

# ROS 2 Architecture and Communication Patterns

## Overview

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robotic platforms. Understanding ROS 2 architecture is fundamental to building humanoid robots that can effectively communicate between different components.

## Core Architecture Concepts

### Nodes
A node is a process that performs computation. Nodes are the fundamental building blocks of ROS 2 applications. In the context of a humanoid robot, you might have nodes for:
- Joint controllers
- Sensor data processing
- Movement planning
- High-level decision making
- Communication with external systems

Nodes allow for the distribution of computation across multiple processes and machines, which is essential for complex humanoid robots with many sensors and actuators.

### Topics
Topics provide a way for nodes to communicate with each other using a publish-subscribe model. A topic has a name and a data type, and nodes can publish messages to the topic or subscribe to it to receive messages.

In a humanoid robot:
- Sensor data (IMU, camera, LIDAR) is typically published to topics
- Actuator commands (joint positions, velocities) are sent via topics
- Internal state information is shared using topics

### Services
Services allow nodes to send a request to another node and receive a response. This follows a client-server model where one node provides a service and other nodes use the service.

In humanoid robots, services might be used for:
- Requesting specific robot poses
- Getting state information on demand
- Triggering specific behaviors
- Configuration changes

### Actions
Actions are similar to services but are designed for long-running tasks. They allow clients to send goals to action servers, receive feedback during execution, and get a result when the goal is completed (or fails).

For humanoid robots, actions are particularly useful for:
- Navigation tasks
- Complex manipulation sequences
- Multi-step behaviors
- Tasks with continuous feedback

## Communication Patterns

### Publisher-Subscriber Pattern
The publisher-subscriber pattern is the most common communication pattern in ROS 2. Publishers send messages to topics without knowledge of which subscribers (if any) exist. Subscribers receive messages from topics without knowledge of which publishers (if any) exist.

This pattern enables:
- Loose coupling between nodes
- Multiple subscribers to the same topic
- Multiple publishers to the same topic (with appropriate coordination)
- Efficient data distribution

### Client-Server Pattern
The client-server pattern is used for request-response communication. A client node sends a request to a server node, which processes the request and returns a response.

This pattern is appropriate for:
- On-demand services
- Configuration requests
- State queries
- Synchronous operations

### Action Pattern
The action pattern is used for long-running tasks that provide continuous feedback. A client sends a goal to an action server, which executes the goal and returns feedback during execution. When the goal is completed (or fails), the server returns a result.

This pattern is ideal for:
- Navigation tasks
- Complex manipulation tasks
- Any task that takes time and benefits from feedback
- Cancellable long-running operations

## ROS 2 Middleware and DDS

ROS 2 uses DDS (Data Distribution Service) as its underlying middleware. This provides:
- Language independence
- Platform independence
- Scalability
- Performance
- Quality of service settings for different communication needs

DDS handles the low-level details of message passing, allowing developers to focus on robot behavior rather than communication infrastructure.

## Quality of Service (QoS)

ROS 2 provides Quality of Service profiles that allow fine-tuning of communication behavior:
- Reliability: Best effort or reliable delivery
- Durability: Volatile or transient local history
- History: Keep all messages or only the last N messages
- Rate: Publish frequency constraints

These settings are crucial for humanoid robots that need to balance real-time performance with communication reliability.

## Practical Considerations for Humanoid Robots

When designing ROS 2 architecture for humanoid robots, consider:
- Processing power limitations on humanoid platforms
- Real-time constraints for stable locomotion
- Safety requirements for human interaction
- Communication latency between robot components
- Redundancy for critical functions
- Scalability as robot capabilities expand

## Next Steps

Now that you understand the ROS 2 architecture and communication patterns, the next section will guide you through implementing these concepts with practical examples in Python using the `rclpy` library.