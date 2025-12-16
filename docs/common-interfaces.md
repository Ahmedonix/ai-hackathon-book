# Common Interfaces and Message Types for Humanoid Robotics

## Overview

This document defines the common interfaces and message types used across all modules of the Physical AI & Humanoid Robotics Book. These standardized interfaces ensure consistency and interoperability between different components and modules.

## Standardized Message Types

### 1. Robot State Messages

#### JointState Extended
- **Topic**: `/joint_states_extended`
- **Type**: `sensor_msgs/JointState` with additional fields
- **Purpose**: Extended joint state information with additional humanoid-specific data
- **Additional Fields**:
  - `temperatures`: array of floats (motor temperatures)
  - `hw_states`: array of strings (hardware states: normal, warning, error)

#### HumanoidPose
- **Topic**: `/humanoid_pose`
- **Type**: Custom message `humanoid_msgs/HumanoidPose`
- **Purpose**: Represent humanoid robot pose with full body configuration
- **Fields**:
  - `header`: std_msgs/Header
  - `base_pose`: geometry_msgs/Pose (position of robot base)
  - `left_arm_joints`: sensor_msgs/JointState (left arm configuration)
  - `right_arm_joints`: sensor_msgs/JointState (right arm configuration)
  - `left_leg_joints`: sensor_msgs/JointState (left leg configuration)
  - `right_leg_joints`: sensor_msgs/JointState (right leg configuration)
  - `head_joints`: sensor_msgs/JointState (head/neck configuration)
  - `pose_type`: string (standing, walking, sitting, etc.)

### 2. Control Messages

#### HumanoidCommand
- **Topic**: `/humanoid_command`
- **Type**: Custom message `humanoid_msgs/HumanoidCommand`
- **Purpose**: High-level humanoid robot commands
- **Fields**:
  - `header`: std_msgs/Header
  - `command_type`: string (move, grasp, speak, gesture, etc.)
  - `target_pose`: geometry_msgs/Pose (for navigation commands)
  - `target_joint_state`: sensor_msgs/JointState (for pose commands)
  - `command_params`: dictionary of parameters (command-specific)

#### GaitCommand
- **Topic**: `/gait_command`
- **Type**: Custom message `humanoid_msgs/GaitCommand`
- **Purpose**: Command for bipedal locomotion patterns
- **Fields**:
  - `header`: std_msgs/Header
  - `gait_type`: string (walk, run, stair_climb, etc.)
  - `speed`: float (normalized speed 0.0-1.0)
  - `step_height`: float (step height in meters)
  - `step_length`: float (step length in meters)
  - `step_width`: float (step width in meters)

### 3. Perception Messages

#### SceneDescription
- **Topic**: `/scene_description`
- **Type**: Custom message `humanoid_msgs/SceneDescription`
- **Purpose**: Structured representation of the robot's perception of its environment
- **Fields**:
  - `header`: std_msgs/Header
  - `objects`: array of `humanoid_msgs/Object` (detected objects)
  - `surfaces`: array of `humanoid_msgs/Surface` (navigable surfaces)
  - `obstacles`: array of `humanoid_msgs/Obstacle` (navigation obstacles)
  - `human_poses`: array of `geometry_msgs/Pose` (detected human poses)

#### Object
- **Message Type**: `humanoid_msgs/Object`
- **Purpose**: Representation of detected objects
- **Fields**:
  - `name`: string (object identifier)
  - `type`: string (object category: cup, chair, table, etc.)
  - `pose`: geometry_msgs/Pose (object location and orientation)
  - `dimensions`: geometry_msgs/Vector3 (object size)
  - `confidence`: float (detection confidence)

### 4. Interaction Messages

#### HumanCommand
- **Topic**: `/human_command`
- **Type**: Custom message `humanoid_msgs/HumanCommand`
- **Purpose**: Commands from humans to the robot (from voice, gesture, etc.)
- **Fields**:
  - `header`: std_msgs/Header
  - `command_text`: string (original command text)
  - `command_type`: string (navigation, manipulation, information, etc.)
  - `target_object`: string (object name if applicable)
  - `target_location`: geometry_msgs/Pose (location if applicable)
  - `confidence`: float (command interpretation confidence)

#### RobotResponse
- **Topic**: `/robot_response`
- **Type**: Custom message `humanoid_msgs/RobotResponse`
- **Purpose**: Robot's response to human commands
- **Fields**:
  - `header`: std_msgs/Header
  - `response_text`: string (text to speak)
  - `response_type`: string (acknowledgment, question, status, error)
  - `emotional_state`: string (robot's emotional state to convey)

## Standardized Service Interfaces

### 1. Navigation Services

#### NavigateToPose
- **Service**: `/navigate_to_pose`
- **Type**: Custom service `humanoid_srvs/NavigateToPose`
- **Purpose**: Navigate humanoid robot to a specific pose
- **Request**:
  - `target_pose`: geometry_msgs/PoseStamped
  - `planning_options`: dictionary of navigation parameters
- **Response**:
  - `success`: bool (navigation completed successfully)
  - `message`: string (status or error message)
  - `actual_path`: nav_msgs/Path (executed path)

#### ComputeTrajectory
- **Service**: `/compute_trajectory`
- **Type**: Custom service `humanoid_srvs/ComputeTrajectory`
- **Purpose**: Compute whole-body trajectory for humanoid robot
- **Request**:
  - `start_pose`: humanoid_msgs/HumanoidPose
  - `goal_pose`: humanoid_msgs/HumanoidPose
  - `constraints`: dictionary of kinematic constraints
- **Response**:
  - `success`: bool (trajectory computed successfully)
  - `trajectory`: humanoid_msgs/HumanoidTrajectory
  - `message`: string (status or error message)

### 2. Manipulation Services

#### GraspObject
- **Service**: `/grasp_object`
- **Type**: Custom service `humanoid_srvs/GraspObject`
- **Purpose**: Grasp an object with humanoid robot's hand
- **Request**:
  - `object_name`: string (name of object to grasp)
  - `grasp_pose`: geometry_msgs/Pose (pose to grasp the object)
  - `hand`: string ("left" or "right")
- **Response**:
  - `success`: bool (grasping successful)
  - `message`: string (status or error message)

## Standardized Action Interfaces

### 1. Complex Task Actions

#### ExecuteBehavior
- **Action**: `/execute_behavior`
- **Type**: Custom action `humanoid_actions/ExecuteBehavior`
- **Purpose**: Execute complex humanoid behaviors (walking, talking, gesturing simultaneously)
- **Goal**:
  - `behavior_type`: string (complex behavior to execute)
  - `behavior_params`: dictionary of behavior-specific parameters
  - `timeout`: duration (maximum time to complete behavior)
- **Feedback**:
  - `current_action`: string (current step in behavior)
  - `progress`: float (0.0 to 1.0)
  - `status`: string (current status)
- **Result**:
  - `success`: bool (behavior completed successfully)
  - `message`: string (status or error message)

## Module Integration Points

### Module 1 (ROS 2 Fundamentals) - Interface Introduction
- Basic publisher/subscriber patterns using standard ROS 2 messages
- Introduction to custom message types defined in this document
- Understanding of message fields and data types

### Module 2 (Digital Twin Simulation) - Simulation Integration
- Simulation of custom messages in Gazebo and Unity
- Mapping of real robot messages to simulated equivalents
- Validation of message consistency between real and simulated robots

### Module 3 (AI-Robot Brain) - Intelligence Integration
- AI perception systems publishing SceneDescription messages
- Navigation systems using NavigateToPose service
- Behavior execution using ExecuteBehavior action

### Module 4 (Vision-Language-Action) - Cognitive Integration
- Voice command processing producing HumanCommand messages
- LLM-based action planning using HumanoidCommand messages
- Multi-modal integration combining perception and commands

## Quality Requirements

### Message Format Consistency
- All custom messages must follow ROS 2 message definition conventions
- Standard headers must be included in all messages where temporal/spatial context is relevant
- All fields must have appropriate validation to prevent invalid data

### Interface Compatibility
- All interfaces must be compatible with ROS 2 Iron
- Message types should reuse standard ROS 2 message definitions where possible
- Backward compatibility must be maintained when updating message definitions

### Performance
- Message publishing rates must meet real-time requirements for humanoid control
- Service calls must complete within specified time bounds
- Action feedback must be updated at appropriate frequency (typically 10-100 Hz)

## Implementation Guidelines

### Creating New Message Types
1. Use the custom message types as templates for creating new types
2. Follow ROS 2 message definition conventions
3. Document any new message types in this document
4. Ensure new types integrate well with the existing message taxonomy

### Using Standard Interfaces
1. Always prefer standard interfaces when available
2. Create new interfaces only when existing ones don't meet requirements
3. Follow the same field naming and semantic conventions as existing interfaces
4. Test new implementations with both real and simulated robots

These standardized interfaces provide a consistent foundation for building humanoid robotics applications across all modules of the book, ensuring modularity, maintainability, and interoperability.