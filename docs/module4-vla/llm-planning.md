---
title: Using LLMs for Robot Action Planning
description: How to use large language models for generating robot action plans from natural language commands
sidebar_position: 4
---

# Using LLMs for Robot Action Planning

## Overview

Large Language Models (LLMs) like GPT have revolutionized the way we approach natural language understanding and task planning. In humanoid robotics, LLMs can serve as a high-level reasoning system that translates natural language commands into executable robot behaviors. This chapter explores how to leverage LLMs for robot action planning, creating systems that can understand complex, multi-step instructions and generate appropriate action sequences.

## Learning Objectives

- Understand the role of LLMs in robot action planning
- Learn to structure prompts for effective robot task planning
- Implement LLM-based action planners for humanoid robots
- Handle multi-step task decomposition using LLMs
- Integrate LLM outputs with robot execution systems

## LLM Architecture and Capabilities for Robotics

### Why Use LLMs for Robot Action Planning?

Traditional robot programming requires precise, structured commands. LLMs can bridge the gap between natural language and structured robot commands, enabling:

1. **Intuitive Interaction**: Users can communicate with robots in natural language
2. **Adaptive Planning**: LLMs can generate plans for new situations not explicitly programmed
3. **Task Decomposition**: Complex instructions can be broken into simpler, executable steps
4. **Context Understanding**: LLMs can use context to disambiguate commands

### Challenges in Robotics Applications

However, using LLMs for robotics comes with specific challenges:

1. **Precision Requirements**: Robots need precise commands, while LLMs generate natural language
2. **Safety Constraints**: Robot actions must be safe and predictable
3. **Real-time Processing**: Robot systems often need quick responses
4. **Environmental Context**: LLMs may not have real-time environmental information

## LLM Integration Architecture

### Basic Architecture Components

```python
# llm_robot_planner.py
import openai
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: str  # e.g., "move", "grasp", "speak"
    parameters: Dict[str, Any]
    description: str

@dataclass
class PlanResult:
    """Represents the result of a planning operation"""
    success: bool
    actions: List[RobotAction]
    reasoning: str
    execution_time: float

class LLMRobotPlanner:
    """
    LLM-based robot action planner
    """
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """
        Initialize the LLM robot planner
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        openai.api_key = api_key
        self.model = model
        self.action_schema = self._define_action_schema()
        self.system_prompt = self._define_system_prompt()
    
    def _define_action_schema(self) -> Dict[str, Any]:
        """
        Define the schema for robot actions
        """
        return {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action_type": {
                                "type": "string",
                                "enum": ["move", "grasp", "place", "speak", "detect", "navigate", "wait", "turn"]
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters specific to the action type"
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable description of the action"
                            }
                        },
                        "required": ["action_type", "parameters", "description"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the planning process"
                }
            },
            "required": ["actions", "reasoning"]
        }
    
    def _define_system_prompt(self) -> str:
        """
        Define the system prompt that guides the LLM's behavior
        """
        return """
You are an advanced robot action planning system. Your task is to take natural language commands and convert them into a sequence of actions that a humanoid robot can execute. 

The robot has the following capabilities:
- move: Move the robot to a specific location (parameters: x, y, z coordinates)
- grasp: Grasp an object (parameters: object_type, object_position)
- place: Place an object at a location (parameters: position)
- speak: Speak a message (parameters: message)
- detect: Detect objects in the environment (parameters: object_type)
- navigate: Navigate to a named location (parameters: location_name)
- wait: Wait for a specified time (parameters: duration_seconds)
- turn: Turn to face a specific direction (parameters: direction, degrees)

Your response must be in JSON format following the provided schema. Each action should be concrete and executable by the robot. Be precise with locations and object specifications. Always include reasoning for your plan.
"""
    
    def plan_action(self, command: str, robot_state: Optional[Dict] = None, 
                   environment_context: Optional[Dict] = None) -> PlanResult:
        """
        Plan robot actions based on a natural language command
        
        Args:
            command: Natural language command
            robot_state: Current state of the robot
            environment_context: Current environmental information
        
        Returns:
            PlanResult containing planned actions
        """
        start_time = time.time()
        
        # Construct the full prompt
        user_prompt = self._construct_user_prompt(command, robot_state, environment_context)
        
        try:
            # Call the LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,  # Lower temperature for more consistent outputs
                max_tokens=1000
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            # Convert to RobotAction objects
            actions = [RobotAction(
                action_type=action['action_type'],
                parameters=action['parameters'],
                description=action['description']
            ) for action in result['actions']]
            
            execution_time = time.time() - start_time
            
            return PlanResult(
                success=True,
                actions=actions,
                reasoning=result['reasoning'],
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Planning error: {e}")
            return PlanResult(
                success=False,
                actions=[],
                reasoning=f"Error occurred during planning: {str(e)}",
                execution_time=execution_time
            )
    
    def _construct_user_prompt(self, command: str, robot_state: Optional[Dict], 
                              environment_context: Optional[Dict]) -> str:
        """
        Construct the user prompt with context
        """
        prompt = f"Command: {command}\n\n"
        
        if robot_state:
            prompt += f"Current robot state: {json.dumps(robot_state)}\n\n"
        
        if environment_context:
            prompt += f"Environment context: {json.dumps(environment_context)}\n\n"
        
        prompt += "Generate a sequence of actions for the robot to execute this command."
        
        return prompt
    
    def validate_plan(self, plan_result: PlanResult) -> bool:
        """
        Validate that the plan is safe and executable
        """
        # Check each action against known safe actions
        valid_action_types = {"move", "grasp", "place", "speak", "detect", "navigate", "wait", "turn"}
        
        for action in plan_result.actions:
            if action.action_type not in valid_action_types:
                print(f"Invalid action type: {action.action_type}")
                return False
            
            # Additional validation could go here
            # For example, checking parameter ranges
        
        return True

# Example usage
def example_usage():
    """
    Example of using the LLM robot planner
    """
    # Initialize planner (you would use your own API key)
    # planner = LLMRobotPlanner(api_key="your-api-key")
    
    # Example command
    command = "Go to the kitchen, pick up the red cup, and bring it to the table"
    
    # Example robot state
    robot_state = {
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "battery_level": 0.85,
        "gripper_status": "open",
        "current_task": "idle"
    }
    
    # Example environment context
    environment_context = {
        "locations": {
            "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
            "table": {"x": 5.0, "y": 1.0, "z": 0.0}
        },
        "objects": [
            {"type": "cup", "color": "red", "position": {"x": 3.2, "y": 2.1, "z": 0.8}}
        ]
    }
    
    # This would actually call the LLM
    # result = planner.plan_action(command, robot_state, environment_context)
    # print(f"Plan: {result.actions}")
    # print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    example_usage()
```

## Advanced Prompt Engineering

### Context-Aware Prompting

For more sophisticated planning, we need to provide the LLM with rich context:

```python
class ContextAwareLLMPlanner(LLMRobotPlanner):
    """
    LLM planner with enhanced context awareness
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        super().__init__(api_key, model)
        self.task_history = []
        self.robot_capabilities = self._define_capabilities()
    
    def _define_capabilities(self) -> Dict[str, Any]:
        """
        Define detailed robot capabilities
        """
        return {
            "locomotion": {
                "max_speed": 0.8,  # m/s
                "turn_rate": 0.5,  # rad/s
                "step_size": 0.3,  # m per step
                "navigation_modes": ["path_planning", "obstacle_avoidance"]
            },
            "manipulation": {
                "max_reach": 1.2,  # m
                "gripper_types": ["parallel", "suction"],
                "payload": 2.0,  # kg
                "precision": 0.01  # m
            },
            "sensors": {
                "camera": {"resolution": [640, 480], "fov": 60},
                "lidar": {"range": 10.0, "resolution": 0.05},
                "imu": True,
                "force_torque": True
            },
            "communication": {
                "speaking": True,
                "led_indicators": True
            }
        }
    
    def _create_detailed_system_prompt(self) -> str:
        """
        Create a more detailed system prompt with capabilities
        """
        return f"""
You are an advanced robot action planning system. Your task is to take natural language commands and convert them into a sequence of actions that a humanoid robot can execute.

Robot Capabilities:
- Locomotion: Max speed {self.robot_capabilities['locomotion']['max_speed']} m/s, turn rate {self.robot_capabilities['locomotion']['turn_rate']} rad/s
- Manipulation: Max reach {self.robot_capabilities['manipulation']['max_reach']} m, payload {self.robot_capabilities['manipulation']['payload']} kg
- Sensors: Camera ({self.robot_capabilities['sensors']['camera']['resolution']} resolution), LIDAR (max range {self.robot_capabilities['sensors']['lidar']['range']} m)

The robot has the following actions available:
- move: Move to specific coordinates (parameters: x, y, z in meters)
- grasp: Grasp an object (parameters: object_type, object_position, grasp_type)
- place: Place an object (parameters: position, placement_strategy)
- speak: Speak a message aloud (parameters: message, volume)
- detect: Detect objects in the environment (parameters: object_type, detection_method)
- navigate: Navigate to a named location (parameters: location_name, navigation_mode)
- wait: Wait for a specified time (parameters: duration_seconds)
- turn: Turn to face a direction (parameters: direction_vector or angle)

When planning:
1. Always consider robot capabilities and limitations
2. Include safety checks in your plan
3. Consider the most efficient sequence of actions
4. Account for environmental constraints
5. Plan for error recovery where appropriate

Your response must be in JSON format with the following structure:
{{
  "actions": [
    {{
      "action_type": "string",
      "parameters": {{"key": "value"}},
      "description": "Human-readable description",
      "estimated_duration": "float_seconds"
    }}
  ],
  "reasoning": "Explain your planning process",
  "safety_considerations": ["list", "of", "safety", "factors"],
  "estimated_total_time": "float_seconds"
}}

Be concise but comprehensive in your planning.
"""
    
    def plan_with_history(self, command: str, num_recent_tasks: int = 3) -> PlanResult:
        """
        Plan action considering recent task history
        """
        # Include recent task history in the prompt
        recent_tasks = self.task_history[-num_recent_tasks:] if self.task_history else []
        
        # Update the system prompt with history
        system_prompt_with_history = self._create_detailed_system_prompt()
        if recent_tasks:
            system_prompt_with_history += f"\n\nRecent Task History: {recent_tasks}"
        
        # For this implementation, we'll use the parent class method but with the enhanced prompt
        # In a real implementation, you'd modify the call to the LLM to use the new system prompt
        return super().plan_action(command)
    
    def update_task_history(self, command: str, result: PlanResult):
        """
        Update the task history with the completed task
        """
        self.task_history.append({
            "command": command,
            "plan_success": result.success,
            "action_count": len(result.actions),
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
        
        # Keep only recent history to prevent memory issues
        if len(self.task_history) > 100:  # Keep last 100 tasks
            self.task_history = self.task_history[-100:]
```

## LLM-to-Robot Action Mapping

### Converting LLM Outputs to Robot Commands

The conversion from LLM output to actual robot commands requires careful validation:

```python
class ActionMapper:
    """
    Maps LLM-generated actions to actual robot commands
    """
    
    def __init__(self):
        self.action_converters = {
            "move": self._convert_move_action,
            "grasp": self._convert_grasp_action,
            "place": self._convert_place_action,
            "speak": self._convert_speak_action,
            "detect": self._convert_detect_action,
            "navigate": self._convert_navigate_action,
            "wait": self._convert_wait_action,
            "turn": self._convert_turn_action
        }
    
    def _convert_move_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert move action to robot command
        """
        try:
            x = float(parameters.get('x', 0))
            y = float(parameters.get('y', 0))
            z = float(parameters.get('z', 0))
            
            # Validate coordinates are within reasonable bounds
            if not (-10 <= x <= 10 and -10 <= y <= 10 and -1 <= z <= 3):
                print(f"Warning: Move coordinates {x}, {y}, {z} may be outside operational bounds")
            
            return {
                "command_type": "navigation",
                "target_position": [x, y, z],
                "motion_type": "move"
            }
        except (TypeError, ValueError):
            print(f"Invalid move parameters: {parameters}")
            return None
    
    def _convert_grasp_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert grasp action to robot command
        """
        try:
            obj_type = parameters.get('object_type', 'object')
            position = parameters.get('object_position', {})
            
            if not position or 'x' not in position or 'y' not in position or 'z' not in position:
                print(f"Invalid position for grasp: {position}")
                return None
            
            # Validate grasp type
            grasp_type = parameters.get('grasp_type', 'pinch')
            valid_grasps = ['pinch', 'power', 'suction']
            if grasp_type not in valid_grasps:
                grasp_type = 'pinch'  # Default to pinch grasp
            
            return {
                "command_type": "manipulation",
                "action": "grasp",
                "object_type": obj_type,
                "position": [position['x'], position['y'], position['z']],
                "grasp_type": grasp_type
            }
        except (TypeError, ValueError):
            print(f"Invalid grasp parameters: {parameters}")
            return None
    
    def _convert_place_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert place action to robot command
        """
        try:
            position = parameters.get('position', {})
            
            if not position or 'x' not in position or 'y' not in position or 'z' not in position:
                print(f"Invalid position for place: {position}")
                return None
            
            return {
                "command_type": "manipulation",
                "action": "place",
                "position": [position['x'], position['y'], position['z']]
            }
        except (TypeError, ValueError):
            print(f"Invalid place parameters: {parameters}")
            return None
    
    def _convert_speak_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert speak action to robot command
        """
        message = parameters.get('message', '')
        
        if not message:
            print("No message specified for speak action")
            return None
        
        return {
            "command_type": "communication",
            "action": "speak",
            "message": str(message),
            "volume": parameters.get('volume', 0.7)  # Default to 70% volume
        }
    
    def _convert_detect_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert detect action to robot command
        """
        obj_type = parameters.get('object_type', 'any')
        
        return {
            "command_type": "perception",
            "action": "detect",
            "target_object": str(obj_type)
        }
    
    def _convert_navigate_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert navigate action to robot command
        """
        location_name = parameters.get('location_name', '')
        
        if not location_name:
            print("No location specified for navigate action")
            return None
        
        return {
            "command_type": "navigation",
            "action": "navigate",
            "target_location": str(location_name),
            "navigation_mode": parameters.get('navigation_mode', 'path_planning')
        }
    
    def _convert_wait_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert wait action to robot command
        """
        try:
            duration = float(parameters.get('duration_seconds', 1.0))
            
            # Validate duration is reasonable
            if duration <= 0 or duration > 300:  # Max 5 minutes
                print(f"Invalid wait duration: {duration}")
                return None
            
            return {
                "command_type": "control",
                "action": "wait",
                "duration": duration
            }
        except (TypeError, ValueError):
            print(f"Invalid wait parameters: {parameters}")
            return None
    
    def _convert_turn_action(self, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert turn action to robot command
        """
        try:
            # Handle both angle and direction vector representations
            if 'angle' in parameters:
                angle = float(parameters['angle'])
                return {
                    "command_type": "navigation",
                    "action": "turn",
                    "angle": angle
                }
            elif 'direction' in parameters:
                direction = parameters['direction']
                if isinstance(direction, list) and len(direction) == 3:
                    # Direction vector [x, y, z]
                    return {
                        "command_type": "navigation",
                        "action": "turn",
                        "direction_vector": direction
                    }
                else:
                    print(f"Invalid direction format: {direction}")
                    return None
            else:
                print("Neither angle nor direction specified for turn action")
                return None
        except (TypeError, ValueError):
            print(f"Invalid turn parameters: {parameters}")
            return None
    
    def map_actions(self, llm_actions: List[RobotAction]) -> List[Dict[str, Any]]:
        """
        Map LLM-generated actions to robot commands
        """
        robot_commands = []
        
        for action in llm_actions:
            converter = self.action_converters.get(action.action_type)
            if converter:
                command = converter(action.parameters)
                if command:
                    command['original_description'] = action.description
                    robot_commands.append(command)
                else:
                    print(f"Failed to convert action: {action.action_type} with parameters {action.parameters}")
            else:
                print(f"Unknown action type: {action.action_type}")
        
        return robot_commands
```

## Safety and Validation

### Implementing Safety Checks

LLMs can generate unsafe or infeasible actions, so validation is crucial:

```python
class SafetyValidator:
    """
    Validates robot action plans for safety
    """
    
    def __init__(self):
        # Define safety constraints
        self.safety_constraints = {
            "max_move_distance": 10.0,  # Maximum distance robot can move at once (m)
            "max_payload": 2.0,  # Maximum weight robot can carry (kg)
            "max_velocity": 1.0,  # Maximum speed (m/s)
            "forbidden_locations": set(),  # Locations robot should not go
            "safety_zones": {},  # Define safety zones around obstacles
            "max_operation_time": 1800  # Maximum operation time before pause (seconds)
        }
    
    def validate_action_sequence(self, commands: List[Dict[str, Any]], 
                                robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an entire sequence of commands for safety
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check each command
        for i, command in enumerate(commands):
            command_issues, command_warnings = self._validate_single_command(
                command, robot_state
            )
            issues.extend([f"Command {i}: {issue}" for issue in command_issues])
            warnings.extend([f"Command {i}: {warning}" for warning in command_warnings])
        
        # Check overall sequence constraints
        sequence_issues, sequence_warnings = self._validate_sequence_properties(commands)
        issues.extend(sequence_issues)
        warnings.extend(sequence_warnings)
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "can_execute": len(issues) == 0  # For now, we'll say issues prevent execution
        }
    
    def _validate_single_command(self, command: Dict[str, Any], 
                                robot_state: Dict[str, Any]) -> tuple:
        """
        Validate a single command
        """
        issues = []
        warnings = []
        
        cmd_type = command.get("command_type", "")
        
        if cmd_type == "navigation":
            # Validate navigation commands
            target_pos = command.get("target_position")
            if target_pos and len(target_pos) >= 3:
                x, y, z = target_pos[0], target_pos[1], target_pos[2]
                
                # Check if destination is in forbidden locations
                if self._is_forbidden_location(x, y):
                    issues.append(f"Destination ({x}, {y}) is in forbidden location")
                
                # Check distance constraints
                current_pos = robot_state.get("location", {"x": 0, "y": 0, "z": 0})
                dist = ((x - current_pos["x"])**2 + (y - current_pos["y"])**2 + (z - current_pos["z"])**2)**0.5
                if dist > self.safety_constraints["max_move_distance"]:
                    issues.append(f"Move distance {dist:.2f}m exceeds maximum {self.safety_constraints['max_move_distance']}m")
        
        elif cmd_type == "manipulation" and command.get("action") == "grasp":
            # Validate grasp commands
            obj_weight = command.get("object_weight", 0.0)
            if obj_weight > self.safety_constraints["max_payload"]:
                issues.append(f"Object weight {obj_weight}kg exceeds maximum payload {self.safety_constraints['max_payload']}kg")
        
        elif cmd_type == "communication" and command.get("action") == "speak":
            # Validate speak commands (e.g., check for inappropriate content)
            message = command.get("message", "")
            if self._contains_inappropriate_content(message):
                issues.append("Message contains inappropriate content")
        
        return issues, warnings
    
    def _validate_sequence_properties(self, commands: List[Dict[str, Any]]) -> tuple:
        """
        Validate properties of the entire command sequence
        """
        issues = []
        warnings = []
        
        # Check for operation time limits
        total_time = sum(
            cmd.get("estimated_duration", 1.0) 
            for cmd in commands 
            if "estimated_duration" in cmd
        )
        
        if total_time > self.safety_constraints["max_operation_time"]:
            issues.append(f"Total operation time {total_time}s exceeds maximum {self.safety_constraints['max_operation_time']}s")
        
        # Check for potential infinite loops or excessive repetition
        action_types = [cmd.get("command_type") for cmd in commands]
        if len(action_types) > 10:
            # Check for repeated patterns
            for i in range(len(action_types) - 5):
                pattern = action_types[i:i+3]
                if action_types[i+3:i+6] == pattern:
                    warnings.append(f"Potential repetitive pattern detected: {pattern}")
        
        return issues, warnings
    
    def _is_forbidden_location(self, x: float, y: float) -> bool:
        """
        Check if a location is forbidden
        """
        # In a real implementation, this would check against a map of forbidden locations
        return False
    
    def _contains_inappropriate_content(self, message: str) -> bool:
        """
        Check if a message contains inappropriate content
        """
        # Simple, basic check - in a real implementation, use more sophisticated content filtering
        inappropriate_keywords = [
            "harm", "dangerous", "unsafe", "attack", "destroy", "break"
        ]
        
        msg_lower = message.lower()
        return any(keyword in msg_lower for keyword in inappropriate_keywords)
```

## Integration with ROS 2

### ROS 2 Interface for LLM Planner

```python
# llm_planner_ros.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import json
from typing import Dict, Any

from llm_robot_planner import LLMRobotPlanner, PlanResult
from action_mapper import ActionMapper
from safety_validator import SafetyValidator

class LLMPlannerROSNode(Node):
    """
    ROS 2 node that integrates LLM-based planning with robot execution
    """
    
    def __init__(self):
        super().__init__('llm_planner_node')
        
        # Initialize LLM planner (you need to provide API key)
        # self.llm_planner = LLMRobotPlanner(api_key="your-api-key")
        self.action_mapper = ActionMapper()
        self.safety_validator = SafetyValidator()
        
        # Subscribe to command topics
        self.command_subscriber = self.create_subscription(
            String,
            'robot_command',
            self.command_callback,
            10
        )
        
        # Subscribe to robot state
        self.state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )
        
        # Publish planned actions
        self.action_publisher = self.create_publisher(
            String,
            'planned_actions',
            10
        )
        
        # Status publisher
        self.status_publisher = self.create_publisher(
            String,
            'planner_status',
            10
        )
        
        # Store current robot state
        self.current_state = {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "battery_level": 1.0,
            "gripper_status": "open"
        }
        
        # Parameters
        self.declare_parameter('api_key', '')
        self.declare_parameter('model', 'gpt-4-turbo')
        self.declare_parameter('max_retry_attempts', 3)
        
        # Initialize planner with parameters
        api_key = self.get_parameter('api_key').value
        if api_key:
            model = self.get_parameter('model').value
            # self.llm_planner = LLMRobotPlanner(api_key=api_key, model=model)
            self.get_logger().info('LLM Planner initialized with parameters')
        else:
            self.get_logger().warn('API key not provided, planning will not work')
        
        self.get_logger().info('LLM Planner ROS Node initialized')
    
    def command_callback(self, msg):
        """
        Handle incoming robot command
        """
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')
        
        # Plan actions using LLM
        if hasattr(self, 'llm_planner'):
            plan_result = self.plan_command(command_text)
            
            if plan_result.success:
                # Validate plan for safety
                robot_commands = self.action_mapper.map_actions(plan_result.actions)
                validation_result = self.safety_validator.validate_action_sequence(
                    robot_commands, self.current_state
                )
                
                if validation_result["is_safe"]:
                    # Publish planned actions
                    actions_msg = String()
                    actions_msg.data = json.dumps(robot_commands)
                    self.action_publisher.publish(actions_msg)
                    
                    self.get_logger().info(f'Published {len(robot_commands)} actions')
                else:
                    self.get_logger().error(f'Plan failed safety validation: {validation_result["issues"]}')
                    self._publish_status("PLANNING_FAILED_SAFETY", validation_result["issues"])
            else:
                self.get_logger().error(f'Planning failed: {plan_result.reasoning}')
                self._publish_status("PLANNING_FAILED", plan_result.reasoning)
        else:
            self.get_logger().error('LLM Planner not initialized (missing API key)')
            self._publish_status("PLANNING_ERROR", "LLM Planner not initialized")
    
    def plan_command(self, command: str) -> PlanResult:
        """
        Plan robot actions for a given command using the LLM
        """
        # Get environment context (in a real system, this would come from perception nodes)
        environment_context = self._get_environment_context()
        
        # Plan the action
        plan_result = self.llm_planner.plan_action(
            command=command,
            robot_state=self.current_state,
            environment_context=environment_context
        )
        
        return plan_result
    
    def _get_environment_context(self) -> Dict[str, Any]:
        """
        Get current environment context for planning
        """
        # In a real implementation, this would gather data from perception nodes
        # For now, return a simple static context
        return {
            "locations": {
                "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
                "table": {"x": 5.0, "y": 1.0, "z": 0.0}
            },
            "objects": [
                {"type": "cup", "color": "red", "position": {"x": 3.2, "y": 2.1, "z": 0.8}}
            ]
        }
    
    def state_callback(self, msg: JointState):
        """
        Update robot state from joint states
        """
        # Update internal robot state based on joint values
        # This is a simplified example
        self.current_state["joint_positions"] = dict(zip(msg.name, msg.position))
        self.current_state["timestamp"] = self.get_clock().now().nanoseconds / 1e9
    
    def _publish_status(self, status: str, details: Any = None):
        """
        Publish planning status
        """
        status_msg = String()
        status_data = {"status": status, "details": details}
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerROSNode()
    
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

## Practical Implementation Guide

### Setting Up the LLM Planner

To implement the LLM planner in a real robot system, you need:

1. An OpenAI API key
2. Appropriate ROS 2 interfaces
3. Safety validation mechanisms
4. Error handling for LLM failures

Here's a complete example setup:

```python
# setup_llm_planner.py
import os
from llm_robot_planner import LLMRobotPlanner, PlanResult
from action_mapper import ActionMapper
from safety_validator import SafetyValidator

def setup_llm_planner(api_key_env_var="OPENAI_API_KEY"):
    """
    Set up the LLM planner with proper configuration
    """
    # Get API key from environment
    api_key = os.getenv(api_key_env_var)
    
    if not api_key:
        raise ValueError(f"API key not found in environment variable {api_key_env_var}")
    
    # Initialize planner
    planner = LLMRobotPlanner(api_key=api_key)
    action_mapper = ActionMapper()
    safety_validator = SafetyValidator()
    
    return {
        'planner': planner,
        'action_mapper': action_mapper,
        'safety_validator': safety_validator
    }

def run_example_plan():
    """
    Example of running a complete planning cycle
    """
    # Setup
    components = setup_llm_planner()
    
    # Example command
    command = "Pick up the red cup from the kitchen counter and place it on the dining table"
    
    # Example robot state
    robot_state = {
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "battery_level": 0.85,
        "gripper_status": "open"
    }
    
    # Example environment context
    environment_context = {
        "locations": {
            "kitchen_counter": {"x": 2.5, "y": 1.5, "z": 0.9},
            "dining_table": {"x": 4.0, "y": 2.0, "z": 0.0}
        },
        "objects": [
            {"type": "cup", "color": "red", "position": {"x": 2.5, "y": 1.5, "z": 0.9}}
        ]
    }
    
    # Plan the action
    plan_result = components['planner'].plan_action(
        command, robot_state, environment_context
    )
    
    if plan_result.success:
        print(f"Plan generated successfully in {plan_result.execution_time:.2f}s")
        print(f"Reasoning: {plan_result.reasoning}")
        print(f"Actions:")
        
        for i, action in enumerate(plan_result.actions):
            print(f"  {i+1}. {action.action_type}: {action.description}")
        
        # Map to robot commands
        robot_commands = components['action_mapper'].map_actions(plan_result.actions)
        print(f"\nMapped to {len(robot_commands)} robot commands")
        
        # Validate for safety
        validation_result = components['safety_validator'].validate_action_sequence(
            robot_commands, robot_state
        )
        
        if validation_result["is_safe"]:
            print("✓ Plan passed safety validation")
        else:
            print("✗ Plan failed safety validation:")
            for issue in validation_result["issues"]:
                print(f"  - {issue}")
    else:
        print(f"Planning failed: {plan_result.reasoning}")

if __name__ == "__main__":
    run_example_plan()
```

## Summary

In this chapter, we've explored how to use Large Language Models for robot action planning:

1. **Architecture**: We developed a comprehensive architecture for integrating LLMs with robot systems
2. **Prompt Engineering**: We created detailed system prompts that guide the LLM toward generating appropriate robot actions
3. **Action Mapping**: We implemented systems to convert LLM outputs into executable robot commands
4. **Safety Validation**: We created validation mechanisms to ensure planned actions are safe and feasible
5. **ROS 2 Integration**: We showed how to integrate the planning system with ROS 2 for real robot applications

The LLM-based action planning system enables robots to understand and execute complex, natural language commands by leveraging the language understanding capabilities of LLMs while ensuring safety through careful validation and mapping to executable robot commands. This approach makes robots more accessible to non-programmers and enables more flexible, adaptive behaviors.

In the next chapter, we'll implement an actual LLM-based action planner following the patterns we've established.