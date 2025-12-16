# Module Configuration and Navigation Structure

## Overview

This document outlines the configuration and navigation structure for all 4 modules of the Physical AI & Humanoid Robotics Book. This structure ensures consistent navigation and organization throughout the educational content.

## Navigation Structure

The documentation site is organized into 4 main modules, each building upon the previous one:

1. Module 1: ROS 2 Fundamentals (The Robotic Nervous System)
2. Module 2: Digital Twin Simulation (Creating Virtual Worlds)
3. Module 3: AI-Robot Brain (Perception and Navigation)
4. Module 4: Vision-Language-Action (Cognitive Robotics)

## Module 1: ROS 2 Fundamentals

### Navigation Path
- `/docs/module1-ros2/index.md` - Introduction and Learning Objectives
- `/docs/module1-ros2/architecture.md` - ROS 2 Architecture Concepts
- `/docs/module1-ros2/nodes.md` - Creating and Managing Nodes
- `/docs/module1-ros2/topics-services-actions.md` - Communication Patterns
- `/docs/module1-ros2/urdf.md` - Robot Description with URDF
- `/docs/module1-ros2/launch-files.md` - Launching Multi-Node Systems
- `/docs/module1-ros2/rclpy-usage.md` - Python Client Library Usage
- `/docs/module1-ros2/ai-integration.md` - Integrating AI Agents with ROS
- `/docs/module1-ros2/humanoid-control.md` - Humanoid Robot Control Concepts
- `/docs/module1-ros2/exercises/` - Practical Exercises
  - `/docs/module1-ros2/exercises/exercise-1.md` - Basic Node Communication
  - `/docs/module1-ros2/exercises/exercise-2.md` - Publisher-Subscriber Pattern
  - `/docs/module1-ros2/exercises/exercise-3.md` - Multi-Node Systems
  - `/docs/module1-ros2/exercises/exercise-4.md` - Robot Control Interface
  - `/docs/module1-ros2/exercises/assessment.md` - Module Assessment

## Module 2: Digital Twin Simulation

### Navigation Path
- `/docs/module2-simulation/index.md` - Introduction and Learning Objectives
- `/docs/module2-simulation/gazebo-setup.md` - Gazebo Simulation Environment
- `/docs/module2-simulation/urdf-import.md` - Importing Robot Models to Simulation
- `/docs/module2-simulation/physics.md` - Physics Simulation Concepts
- `/docs/module2-simulation/sensors.md` - Sensor Simulation in Gazebo
- `/docs/module2-simulation/worlds.md` - Creating Simulation Environments
- `/docs/module2-simulation/unity.md` - Unity Visualization Layer
- `/docs/module2-simulation/robotics-hub.md` - Unity Robotics Hub Integration
- `/docs/module2-simulation/exercises/` - Practical Exercises
  - `/docs/module2-simulation/exercises/exercise-1.md` - Basic Simulation Setup
  - `/docs/module2-simulation/exercises/exercise-2.md` - Environment Creation

## Module 3: AI-Robot Brain

### Navigation Path
- `/docs/module3-ai/index.md` - Introduction and Learning Objectives
- `/docs/module3-ai/isaac-sim.md` - NVIDIA Isaac Sim Setup
- `/docs/module3-ai/perception.md` - AI-Based Perception Systems
- `/docs/module3-ai/vslam.md` - Visual SLAM for Robot Navigation
- `/docs/module3-ai/navigation.md` - Navigation with Nav2
- `/docs/module3-ai/sim-to-real.md` - Sim-to-Real Transfer Techniques
- `/docs/module3-ai/jetson-deployment.md` - Deploying AI Models to Jetson
- `/docs/module3-ai/motion-planning.md` - Humanoid Motion Planning
- `/docs/module3-ai/exercises/` - Practical Exercises
  - `/docs/module3-ai/exercises/exercise-1.md` - Perception Pipeline Implementation
  - `/docs/module3-ai/exercises/exercise-2.md` - Navigation System Setup

## Module 4: Vision-Language-Action

### Navigation Path
- `/docs/module4-vla/index.md` - Introduction and Learning Objectives
- `/docs/module4-vla/whisper.md` - Whisper Integration for Speech Processing
- `/docs/module4-vla/llm-planning.md` - LLM-Based Action Planning
- `/docs/module4-vla/task-decomposition.md` - Natural Language Task Decomposition
- `/docs/module4-vla/multimodal.md` - Multimodal Perception Integration
- `/docs/module4-vla/vla-architecture.md` - VLA System Architecture
- `/docs/module4-vla/multi-modal-interaction.md` - Multi-Modal Interaction
- `/docs/module4-vla/gpt-integration.md` - GPT Integration with ROS
- `/docs/module4-vla/exercises/` - Practical Exercises
  - `/docs/module4-vla/exercises/exercise-1.md` - Voice Command Processing
  - `/docs/module4-vla/exercises/exercise-2.md` - Complete VLA System

## Configuration Files

### Docusaurus Configuration
The navigation is configured in `docusaurus.config.js` with the following structure:

```javascript
module.exports = {
  // ... other config options
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // ... other options
        },
        // ... other preset options
      },
    ],
  ],
  // ... rest of config
};
```

### Sidebar Configuration
The sidebar navigation is configured in `sidebars.js` to reflect the hierarchical structure of the modules, with each module containing its sub-topics and exercises organized in a logical learning sequence.

## Cross-Module Navigation Aids

### Prerequisites and Dependencies
- Each module clearly indicates prerequisites from previous modules
- Links to foundational concepts in earlier modules when referenced
- Common terminology glossary accessible from all modules

### Progress Tracking
- Learning objectives clearly stated at the beginning of each module
- Self-assessment quizzes at the end of each module
- Capstone projects that integrate concepts from multiple modules

## Consistency Guidelines

### Content Structure
Each module section follows a consistent structure:
1. Learning objectives
2. Theoretical concepts
3. Practical examples
4. Hands-on exercises
5. Summary and key takeaways

### Code Example Standards
- All code examples use Python for ROS 2 nodes (rclpy)
- Consistent naming conventions
- Proper error handling and logging
- Modular, reusable components

### Assessment Standards
- Exercises increase in complexity throughout each module
- Clear acceptance criteria for each exercise
- Solutions and hints available separately
- Cross-module integration projects in later modules

This configuration provides a clear, consistent navigation structure that helps students progress systematically through the content while maintaining clear connections between related concepts across modules.