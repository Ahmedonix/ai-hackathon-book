# VLA Interface Contract

## Overview

This document defines the interface contract for the Vision-Language-Action (VLA) system that integrates perception, language understanding, and robotic action in the humanoid robot system. This contract establishes the API specifications, data formats, and communication protocols between the different components of the VLA system.

## System Architecture

The VLA system consists of three main interconnected components:

```
Vision Component ←→ Language Component ←→ Action Component
       ↓                    ↓                   ↓
   Perception      Natural Language      Robot Control
   Processing      Understanding         Execution
```

## Interface Specifications

### 1. Vision Interface

#### Input Requirements
- Camera feeds (RGB, depth, thermal)
- Sensor data (LiDAR, IMU, etc.)
- Timestamp synchronization

#### Output Format
```json
{
  "timestamp": "2023-10-01T12:00:00.000Z",
  "frame_id": "camera_rgb_optical_frame",
  "detected_objects": [
    {
      "id": "object_001",
      "class": "cup",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 50,
        "height": 60
      },
      "pose": {
        "position": {
          "x": 1.2,
          "y": 0.5,
          "z": 0.8
        },
        "orientation": {
          "x": 0.0,
          "y": 0.0,
          "z": 0.0,
          "w": 1.0
        }
      }
    }
  ],
  "scene_description": "A kitchen scene with a wooden table, a blue cup on the table, and a refrigerator in the background",
  "features": {
    "optical_flow": [],
    "keypoints": [],
    "segmentation": {}
  }
}
```

#### ROS Message Definition
```
# VisionOutput.msg
Header header
string scene_description
ObjectDetection[] detected_objects
VisionFeatures features
geometry_msgs/TransformStamped[] transforms
sensor_msgs/Image[] raw_images

# ObjectDetection.msg
string id
string class
float32 confidence
BoundingBox bbox
geometry_msgs/Pose pose

# BoundingBox.msg
uint32 x
uint32 y
uint32 width
uint32 height

# VisionFeatures.msg
float32[] optical_flow
Keypoint[] keypoints
sensor_msgs/Image segmentation
```

### 2. Language Interface

#### Input Requirements
- Natural language commands
- Context information
- Conversation history

#### Output Format
```json
{
  "timestamp": "2023-10-01T12:00:00.000Z",
  "request_id": "req_001",
  "intent": "grasp_object",
  "intent_confidence": 0.92,
  "entities": {
    "object": "cup",
    "location": "kitchen table",
    "color": "blue"
  },
  "action_sequence": [
    {
      "action_type": "navigate",
      "target_location": {
        "x": 1.2,
        "y": 0.5,
        "z": 0.0
      },
      "description": "Move to the location of the blue cup"
    },
    {
      "action_type": "identify_object",
      "object_class": "cup",
      "description": "Identify the blue cup"
    },
    {
      "action_type": "grasp",
      "object_id": "object_001",
      "description": "Grasp the blue cup"
    }
  ],
  "context": {
    "conversation_history": [
      "User: Please get me the blue cup from the kitchen table",
      "Robot: On my way to get the blue cup"
    ],
    "relevant_objects": ["object_001"],
    "environment": "kitchen"
  },
  "response_text": "I will get the blue cup from the kitchen table for you"
}
```

#### ROS Message Definition
```
# LanguageOutput.msg
Header header
string request_id
string intent
float32 intent_confidence
EntityMap entities
ActionSequence[] action_sequence
Context context
string response_text

# EntityMap.msg
string[] object_classes
string[] locations
string[] attributes
string[] actions

# Action.msg
string action_type
geometry_msgs/PoseStamped target_pose
string description
string[] parameters

# Context.msg
string[] conversation_history
string[] relevant_objects
string environment
```

### 3. Action Interface

#### Input Requirements
- Planned action sequence
- Robot state information
- Environmental constraints

#### Output Format
```json
{
  "timestamp": "2023-10-01T12:00:00.000Z",
  "action_id": "act_001",
  "status": "completed",
  "execution_time": 15.2,
  "feedback": {
    "object_grasped": true,
    "navigation_success": true,
    "execution_errors": [],
    "metrics": {
      "path_efficiency": 0.89,
      "grasp_success": 1.0,
      "safety_compliance": true
    }
  },
  "next_action": null
}
```

#### ROS Message Definition
```
# ActionOutput.msg
Header header
string action_id
string status  # "pending", "executing", "completed", "failed", "cancelled"
float32 execution_time
ActionFeedback feedback
Action next_action

# ActionFeedback.msg
bool object_grasped
bool navigation_success
string[] execution_errors
Metrics metrics

# Metrics.msg
float32 path_efficiency
float32 grasp_success
bool safety_compliance
```

## VLA Integration Interface

### Main VLA Controller Service

```
# VLAControl.srv
string command
string context
---
bool success
string message
ActionSequence planned_actions
string response_text
```

### VLA State Monitor Topic

```
# VLAState.msg
Header header
string system_status  # "idle", "processing", "executing", "error"
string current_action
float32 system_confidence
string[] active_interfaces  # vision, language, action
```

## Communication Protocols

### 1. Synchronous Communication
For time-critical operations, use ROS services:
- `/vla/process_command` - Service to process a VLA command synchronously
- `/vla/get_system_status` - Service to get current system status

### 2. Asynchronous Communication
For continuous monitoring, use ROS topics:
- `/vla/vision_input` - Input for vision processing
- `/vla/vision_output` - Output from vision processing
- `/vla/language_input` - Input for language processing
- `/vla/language_output` - Output from language processing
- `/vla/action_input` - Input for action execution
- `/vla/action_output` - Output from action execution
- `/vla/control_input` - High-level control commands
- `/vla/state` - System state monitoring

### 3. Action Server Communication
For complex multi-step actions:
- `/vla/action_server` - Action server for long-running tasks
- Provides goal, feedback, and result interfaces

## Error Handling and Recovery

### Vision Component Errors
- `VISION_SENSOR_ERROR`: Vision sensor unavailable
- `VISION_PERFORMANCE_ERROR`: Processing too slow
- `VISION_DETECTION_ERROR`: Object detection failure

### Language Component Errors
- `LANGUAGE_PARSE_ERROR`: Natural language parsing failed
- `LANGUAGE_UNDERSTANDING_ERROR`: Intent understanding failed
- `LANGUAGE_CONTEXT_ERROR`: Context management error

### Action Component Errors
- `ACTION_EXECUTION_ERROR`: Action execution failed
- `ACTION_SAFETY_ERROR`: Safety constraint violation
- `ACTION_PLANNING_ERROR`: Action planning failed

### VLA System Errors
- `VLA_INTEGRATION_ERROR`: Component integration failure
- `VLA_TIMEOUT_ERROR`: Operation timed out
- `VLA_RESOURCE_ERROR`: Insufficient computational resources

## Performance Requirements

### Latency Requirements
- Vision processing: < 100ms per frame
- Language processing: < 500ms per command
- Action planning: < 200ms per action
- End-to-end VLA processing: < 2s per command

### Accuracy Requirements
- Vision detection accuracy: > 90%
- Language understanding accuracy: > 85%
- Action execution success rate: > 80%

### Reliability Requirements
- System uptime: > 95%
- Error recovery time: < 30s
- Graceful degradation: Maintain basic functionality during partial failures

## Security Considerations

### Authentication
- All VLA interfaces require authentication
- API keys for external services (GPT, etc.)
- Certificate-based communication for sensitive operations

### Authorization
- Role-based access control for different system components
- Command validation and sanitization
- Rate limiting for API calls

## Testing Protocols

### Unit Testing
- Individual component interfaces
- Message validation
- Error handling paths

### Integration Testing
- End-to-end VLA flow
- Component communication
- State synchronization

### Stress Testing
- High load scenarios
- Network failure simulations
- Component failure recovery

## Versioning

The VLA interface contract follows semantic versioning:
- Major version changes indicate breaking interface changes
- Minor version changes add new optional fields or methods
- Patch version changes fix bugs or improve documentation

Current version: 1.0.0