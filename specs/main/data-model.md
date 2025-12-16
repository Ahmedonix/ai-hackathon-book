# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document outlines the key entities, relationships, and data structures for the Physical AI & Humanoid Robotics educational book project. It covers the main concepts and components across all four modules.

## Core Entities

### 1. Module
- **Description**: A major section of the book that covers specific aspects of humanoid robotics
- **Fields**:
  - id: string (e.g., "001-ros2-module", "002-digital-twin-sim", etc.)
  - title: string
  - description: string
  - duration: integer (estimated hours)
  - prerequisites: array of Module IDs
  - learning_outcomes: array of strings
  - components: array of strings
  - tech_introduced: array of strings
  - deliverables: array of strings

### 2. Chapter
- **Description**: A subsection within a module that focuses on specific topics
- **Fields**:
  - id: string (e.g., "001-ros2-architecture", "001-ros2-nodes", etc.)
  - title: string
  - module_id: string
  - content_type: enum ["conceptual", "practical", "exercise", "assessment"]
  - objectives: array of strings
  - concepts: array of strings
  - code_examples: array of strings (file references)
  - exercises: array of exercise objects

### 3. Exercise
- **Description**: Practical tasks for students to complete
- **Fields**:
  - id: string
  - title: string
  - chapter_id: string
  - description: string
  - difficulty: enum ["beginner", "intermediate", "advanced"]
  - estimated_time: integer (minutes)
  - steps: array of strings
  - acceptance_criteria: array of strings
  - hints: array of strings (optional)

### 4. CodeExample
- **Description**: Runnable code samples included in the book
- **Fields**:
  - id: string
  - title: string
  - chapter_id: string
  - language: enum ["python", "bash", "yaml", "xml", "json", "other"]
  - description: string
  - code: string (actual code content)
  - requirements: array of strings (dependencies, environment setup)
  - test_validation: string (how to verify it works)
  - file_path: string (where to place the file in the repo)

### 5. Concept
- **Description**: Key ideas or technical concepts explained in the book
- **Fields**:
  - id: string
  - name: string
  - definition: string
  - related_concepts: array of Concept IDs
  - modules: array of Module IDs where it's covered
  - examples: array of strings
  - visual_representations: array of image file paths

### 6. Resource
- **Description**: External resources referenced in the book
- **Fields**:
  - id: string
  - title: string
  - url: string
  - category: enum ["documentation", "tutorial", "research_paper", "tool", "video"]
  - description: string
  - module_id: string (optional)
  - chapter_id: string (optional)

## Module-Specific Entities

### 7. ROS2Node
- **Description**: ROS 2 nodes specifically for Module 1
- **Fields**:
  - id: string
  - name: string
  - module_id: string (should be "001-ros2-module")
  - description: string
  - node_type: enum ["publisher", "subscriber", "service", "action", "other"]
  - topics_published: array of strings
  - topics_subscribed: array of strings
  - services: array of strings
  - actions: array of strings
  - parameters: object (parameter_name: type)

### 8. SimulationEnvironment
- **Description**: Gazebo/Unity simulation environments for Module 2
- **Fields**:
  - id: string
  - name: string
  - module_id: string (should be "002-digital-twin-sim")
  - description: string
  - world_file: string (path to .world file)
  - robot_models: array of strings (URDF file paths)
  - sensors: array of Sensor objects
  - physics_properties: object (gravity, friction, etc.)
  - lighting: object (for Unity)

### 9. Sensor
- **Description**: Sensors simulated in Gazebo/Unity
- **Fields**:
  - id: string
  - name: string
  - type: enum ["lidar", "camera", "imu", "depth_camera", "other"]
  - simulation_platform: enum ["gazebo", "unity", "both"]
  - parameters: object (specific to sensor type)
  - ros_topic: string (topic where sensor data is published)
  - data_format: string

### 10. PerceptionPipeline
- **Description**: AI perception systems for Module 3
- **Fields**:
  - id: string
  - name: string
  - module_id: string (should be "003-ai-robot-brain")
  - description: string
  - input_types: array of enum ["image", "depth", "lidar", "imu", "other"]
  - output_types: array of enum ["classification", "detection", "segmentation", "tracking", "other"]
  - algorithm: string (e.g., "VSLAM", "YOLO", etc.)
  - ros_nodes: array of strings (node names)
  - dependencies: array of strings (required packages)
  - compute_requirements: string (GPU specs, etc.)

### 11. VAInterface
- **Description**: Vision-language-action interface for Module 4
- **Fields**:
  - id: string
  - name: string
  - module_id: string (should be "004-vla-cognitive-robotics")
  - description: string
  - input_type: enum ["voice", "text", "vision"]
  - processing_method: string (e.g., "Whisper", "GPT", etc.)
  - output_action: string (how it translates to robot action)
  - ros_integration: string (how it connects to ROS)
  - planning_approach: string (how high-level commands are decomposed)

## Relationships

- Module 1→2→3→4: Sequential learning path with increasing complexity
- Module contains multiple Chapters
- Chapter contains multiple Exercises and CodeExamples
- CodeExample and Exercise reference Concepts
- Module uses Resources for additional learning
- Module 2's SimulationEnvironment contains Sensors and RobotModels
- Module 3's PerceptionPipeline builds on Module 1's ROS2Node communication
- Module 4's VAInterface integrates with all previous modules

## Validation Rules

1. Each module must have clearly defined learning outcomes
2. Each code example must be executable in the specified environment
3. All referenced ROS 2 nodes must follow proper ROS 2 conventions
4. All simulation environments must be reproducible with provided instructions
5. All perception pipelines must be testable with simulated data
6. All VLA interfaces must have safety guards for action execution
7. All exercises must have verifiable acceptance criteria