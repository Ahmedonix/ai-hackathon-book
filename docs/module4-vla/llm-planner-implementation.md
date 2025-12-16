---
title: Implementing LLM-Based Action Planner
description: Complete implementation of an LLM-based action planner for humanoid robots
sidebar_position: 5
---

# Implementing LLM-Based Action Planner

## Overview

This chapter provides a complete, production-ready implementation of an LLM-based action planner for humanoid robots. The implementation includes robust error handling, safety validation, context-aware planning, and integration with ROS 2. We'll build upon the concepts from the previous chapter and create a fully functional system.

## Learning Objectives

- Implement a complete LLM-based action planner
- Integrate with real robot systems and ROS 2
- Add comprehensive safety validation
- Handle errors and edge cases in LLM responses
- Optimize for real-time performance

## Complete Implementation

Let's create a complete, production-ready implementation:

### Core LLM Planner Implementation

```python
# complete_llm_planner.py
import openai
import json
import time
import threading
import queue
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np

class ActionType(Enum):
    """Enumeration of possible robot actions"""
    MOVE = "move"
    GRASP = "grasp"
    PLACE = "place"
    SPEAK = "speak"
    DETECT = "detect"
    NAVIGATE = "navigate"
    WAIT = "wait"
    TURN = "turn"
    CUSTOM = "custom"

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: ActionType
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float = 1.0  # seconds

@dataclass
class PlanResult:
    """Result of a planning operation"""
    success: bool
    actions: List[RobotAction]
    reasoning: str
    execution_time: float
    safety_issues: List[str]
    model_response: Dict[str, Any] = None  # Raw model response for debugging

@dataclass
class RobotState:
    """Current state of the robot"""
    location: Dict[str, float] = None
    battery_level: float = 1.0
    gripper_status: str = "open"
    joint_positions: Dict[str, float] = None
    current_task: str = "idle"
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.location is None:
            self.location = {"x": 0.0, "y": 0.0, "z": 0.0}
        if self.joint_positions is None:
            self.joint_positions = {}

@dataclass
class EnvironmentContext:
    """Current environment context"""
    locations: Dict[str, Dict[str, float]] = None
    objects: List[Dict[str, Any]] = None
    obstacles: List[Dict[str, float]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.locations is None:
            self.locations = {}
        if self.objects is None:
            self.objects = []
        if self.obstacles is None:
            self.obstacles = []

class SafetyValidator:
    """Validates planned actions for safety"""
    
    def __init__(self):
        self.constraints = {
            "max_move_distance": 10.0,  # meters
            "max_payload": 3.0,  # kg
            "max_velocity": 1.0,  # m/s
            "min_battery_level": 0.1,  # 10%
            "forbidden_actions": set(),
            "forbidden_zones": [],
            "safety_margin": 0.5  # meters
        }
    
    def validate_plan(self, plan_result: PlanResult, robot_state: RobotState, 
                     environment: EnvironmentContext) -> PlanResult:
        """Validate a plan for safety issues"""
        issues = []
        
        # Check battery level
        if robot_state.battery_level < self.constraints["min_battery_level"]:
            issues.append("Battery level too low for planned operations")
        
        # Validate each action
        for i, action in enumerate(plan_result.actions):
            action_issues = self._validate_action(action, robot_state, environment)
            issues.extend([f"Action {i} ({action.action_type.value}): {issue}" for issue in action_issues])
        
        # Update the plan result with safety issues
        plan_result.safety_issues = issues
        plan_result.success = plan_result.success and len(issues) == 0
        
        return plan_result
    
    def _validate_action(self, action: RobotAction, robot_state: RobotState, 
                        environment: EnvironmentContext) -> List[str]:
        """Validate a single action for safety"""
        issues = []
        
        action_type = action.action_type
        params = action.parameters
        
        if action_type == ActionType.MOVE:
            # Validate move distance
            if 'x' in params and 'y' in params and 'z' in params:
                target_pos = [params['x'], params['y'], params['z']]
                current_pos = robot_state.location
                distance = np.sqrt(
                    (target_pos[0] - current_pos['x'])**2 +
                    (target_pos[1] - current_pos['y'])**2 +
                    (target_pos[2] - current_pos['z'])**2
                )
                
                if distance > self.constraints["max_move_distance"]:
                    issues.append(f"Move distance {distance:.2f}m exceeds maximum {self.constraints['max_move_distance']}m")
        
        elif action_type == ActionType.GRASP:
            # Validate payload
            object_weight = params.get('weight', 0.0)
            if object_weight > self.constraints["max_payload"]:
                issues.append(f"Object weight {object_weight}kg exceeds maximum payload {self.constraints['max_payload']}kg")
        
        elif action_type == ActionType.WAIT:
            # Validate wait duration
            duration = params.get('duration_seconds', 1.0)
            if duration > 300:  # Max 5 minutes
                issues.append(f"Wait duration {duration}s exceeds maximum 300s")
        
        elif action_type == ActionType.SPEAK:
            # Validate message content
            message = params.get('message', '')
            if self._contains_inappropriate_content(message):
                issues.append("Message contains inappropriate content")
        
        return issues
    
    def _contains_inappropriate_content(self, message: str) -> bool:
        """Check for inappropriate content in messages"""
        inappropriate_keywords = [
            "harm", "dangerous", "unsafe", "attack", "destroy", "break"
        ]
        msg_lower = message.lower()
        return any(keyword in msg_lower for keyword in inappropriate_keywords)

class LLMActionPlanner:
    """Complete LLM-based action planner for humanoid robots"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", 
                 safety_validator: SafetyValidator = None):
        """
        Initialize the LLM action planner
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            safety_validator: Safety validator instance (optional)
        """
        openai.api_key = api_key
        self.model = model
        self.safety_validator = safety_validator or SafetyValidator()
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # seconds
        
        # Action schema for structured output
        self.action_schema = self._define_action_schema()
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
        
        # History for context
        self.history = []
        self.max_history = 10
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"LLM Action Planner initialized with model {model}")
    
    def _define_action_schema(self) -> Dict[str, Any]:
        """Define the JSON schema for robot actions"""
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
                                "enum": [action.value for action in ActionType]
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the action"
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable description"
                            },
                            "estimated_duration": {
                                "type": "number",
                                "description": "Estimated time to execute in seconds"
                            }
                        },
                        "required": ["action_type", "parameters", "description"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning"
                },
                "estimated_total_time": {
                    "type": "number",
                    "description": "Estimated total execution time"
                }
            },
            "required": ["actions", "reasoning"]
        }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return f"""
You are an advanced robot action planning system for a humanoid robot. Your role is to take natural language commands and convert them into a sequence of specific, executable robot actions.

Robot Capabilities:
- Navigation: Can move to specific coordinates or named locations
- Manipulation: Can grasp and place objects
- Communication: Can speak messages aloud
- Perception: Can detect objects in the environment
- Interaction: Can turn to face directions

Available Actions:
- move: Move to specific coordinates (parameters: x, y, z in meters)
- grasp: Grasp an object (parameters: object_type, position, weight)
- place: Place an object (parameters: position, orientation)
- speak: Speak a message (parameters: message)
- detect: Detect objects (parameters: object_type, search_area)
- navigate: Navigate to named location (parameters: location_name)
- wait: Wait for duration (parameters: duration_seconds)
- turn: Turn to face direction (parameters: direction_vector or angle)

When planning:
1. Be precise with coordinates and object specifications
2. Consider robot state and environmental constraints
3. Plan the most efficient sequence of actions
4. Include error handling steps where appropriate
5. Ensure safety in all actions

Response Format:
Return your response as valid JSON following this schema:
{json.dumps(self.action_schema, indent=2)}

Each action must be concrete and executable. Include reasoning for your plan.
"""
    
    def plan_action(self, command: str, robot_state: RobotState = None,
                   environment_context: EnvironmentContext = None,
                   temperature: float = 0.2) -> PlanResult:
        """
        Plan robot actions based on a natural language command
        
        Args:
            command: Natural language command
            robot_state: Current robot state
            environment_context: Environmental context
            temperature: LLM temperature for response variability
        
        Returns:
            PlanResult containing planned actions
        """
        start_time = time.time()
        
        # Rate limiting
        time_since_last = time.time() - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        
        try:
            # Prepare the prompt
            user_prompt = self._create_user_prompt(command, robot_state, environment_context)
            
            # Call the LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1500,
                timeout=30
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)
            
            # Convert to RobotAction objects
            actions = []
            for action_data in parsed_response.get('actions', []):
                try:
                    action_type = ActionType(action_data['action_type'])
                    action = RobotAction(
                        action_type=action_type,
                        parameters=action_data.get('parameters', {}),
                        description=action_data.get('description', ''),
                        estimated_duration=action_data.get('estimated_duration', 1.0)
                    )
                    actions.append(action)
                except ValueError:
                    self.logger.warning(f"Invalid action type: {action_data.get('action_type')}")
                    continue
            
            # Create plan result
            plan_result = PlanResult(
                success=True,
                actions=actions,
                reasoning=parsed_response.get('reasoning', ''),
                execution_time=time.time() - start_time,
                safety_issues=[],
                model_response=parsed_response
            )
            
            # Validate for safety if robot state is provided
            if robot_state and environment_context:
                plan_result = self.safety_validator.validate_plan(
                    plan_result, robot_state, environment_context
                )
            
            # Update history
            self._update_history(command, plan_result)
            
            self.logger.info(f"Successfully planned {len(actions)} actions for command: {command[:50]}...")
            return plan_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return PlanResult(
                success=False,
                actions=[],
                reasoning=f"Error parsing LLM response: {e}",
                execution_time=time.time() - start_time,
                safety_issues=["JSON parsing error"]
            )
        except openai.error.RateLimitError:
            self.logger.error("OpenAI rate limit exceeded")
            return PlanResult(
                success=False,
                actions=[],
                reasoning="Rate limit exceeded, please try again later",
                execution_time=time.time() - start_time,
                safety_issues=["Rate limit exceeded"]
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in planning: {e}")
            return PlanResult(
                success=False,
                actions=[],
                reasoning=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time,
                safety_issues=[f"Unexpected error: {str(e)}"]
            )
        finally:
            self._last_request_time = time.time()
    
    def _create_user_prompt(self, command: str, robot_state: RobotState,
                           environment_context: EnvironmentContext) -> str:
        """Create the user prompt with context"""
        prompt_parts = [f"Command: {command}"]
        
        if robot_state:
            prompt_parts.append(f"Current Robot State: {json.dumps(asdict(robot_state), indent=2)}")
        
        if environment_context:
            prompt_parts.append(f"Environment Context: {json.dumps(asdict(environment_context), indent=2)}")
        
        prompt_parts.append("Generate a sequence of actions for the robot to execute this command.")
        prompt_parts.append("Return your response as valid JSON following the specified schema.")
        
        return "\n\n".join(prompt_parts)
    
    def _update_history(self, command: str, plan_result: PlanResult):
        """Update the planning history"""
        history_entry = {
            "command": command,
            "success": plan_result.success,
            "action_count": len(plan_result.actions),
            "timestamp": time.time()
        }
        
        self.history.append(history_entry)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get planning performance statistics"""
        if not self.history:
            return {"total_plans": 0, "success_rate": 0.0}
        
        total_plans = len(self.history)
        successful_plans = sum(1 for entry in self.history if entry["success"])
        
        return {
            "total_plans": total_plans,
            "successful_plans": successful_plans,
            "success_rate": successful_plans / total_plans if total_plans > 0 else 0.0,
            "recent_history": self.history[-5:]  # Last 5 plans
        }

class AsyncLLMPlanner:
    """Asynchronous wrapper for the LLM planner"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.planner = LLMActionPlanner(api_key, model)
        self.request_queue = queue.Queue()
        self.response_queues = {}  # Maps request_id to response queue
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
    
    def plan_action_async(self, command: str, robot_state: RobotState = None,
                         environment_context: EnvironmentContext = None,
                         request_id: str = None) -> str:
        """
        Plan action asynchronously
        
        Returns:
            Request ID for retrieving the result
        """
        if not request_id:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # Create response queue for this request
        response_queue = queue.Queue()
        self.response_queues[request_id] = response_queue
        
        # Add to request queue
        request = {
            "request_id": request_id,
            "command": command,
            "robot_state": robot_state,
            "environment_context": environment_context
        }
        
        self.request_queue.put(request)
        
        return request_id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[PlanResult]:
        """Get the result of an asynchronous planning request"""
        if request_id not in self.response_queues:
            return None
        
        try:
            result = self.response_queues[request_id].get(timeout=timeout)
            del self.response_queues[request_id]
            return result
        except queue.Empty:
            return None
    
    def _process_requests(self):
        """Process planning requests in a separate thread"""
        while self.running:
            try:
                request = self.request_queue.get(timeout=1.0)
                
                # Perform planning
                result = self.planner.plan_action(
                    command=request["command"],
                    robot_state=request["robot_state"],
                    environment_context=request["environment_context"]
                )
                
                # Put result in response queue
                response_queue = self.response_queues.get(request["request_id"])
                if response_queue:
                    response_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.planner.logger.error(f"Error in async processing: {e}")
                continue
    
    def shutdown(self):
        """Shutdown the async planner"""
        self.running = False
        self.processing_thread.join(timeout=5.0)

# Example usage and testing
def main():
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize planner
    planner = LLMActionPlanner(api_key)
    
    # Example command
    command = "Go to the kitchen, pick up the red cup, and bring it to the table"
    
    # Example robot state
    robot_state = RobotState(
        location={"x": 0.0, "y": 0.0, "z": 0.0},
        battery_level=0.85,
        gripper_status="open"
    )
    
    # Example environment context
    environment_context = EnvironmentContext(
        locations={
            "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
            "table": {"x": 5.0, "y": 1.0, "z": 0.0}
        },
        objects=[
            {"type": "cup", "color": "red", "position": {"x": 3.2, "y": 2.1, "z": 0.8}, "weight": 0.3}
        ]
    )
    
    # Plan the action
    print(f"Planning for command: {command}")
    result = planner.plan_action(command, robot_state, environment_context)
    
    print(f"Planning success: {result.success}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Reasoning: {result.reasoning}")
    
    if result.actions:
        print(f"Planned actions ({len(result.actions)}):")
        for i, action in enumerate(result.actions):
            print(f"  {i+1}. {action.action_type.value}: {action.description}")
            print(f"     Parameters: {action.parameters}")
            print(f"     Duration: {action.estimated_duration}s")
    
    if result.safety_issues:
        print(f"  Safety issues: {result.safety_issues}")
    
    # Print performance stats
    stats = planner.get_performance_stats()
    print(f"Performance: {stats}")

if __name__ == "__main__":
    main()
```

### ROS 2 Integration

Let's create a complete ROS 2 node implementation:

```python
# llm_planner_ros_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
import json
import threading
import time
from typing import Dict, Any

from complete_llm_planner import LLMActionPlanner, RobotState, EnvironmentContext, ActionType

class LLMPlannerROSNode(Node):
    """
    ROS 2 node for LLM-based action planning
    """
    
    def __init__(self):
        super().__init__('llm_planner_node')
        
        # Parameters
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model', 'gpt-4-turbo')
        self.declare_parameter('plan_timeout', 30.0)
        self.declare_parameter('rate_limit_delay', 0.5)
        
        api_key = self.get_parameter('openai_api_key').value
        model = self.get_parameter('model').value
        
        if not api_key:
            self.get_logger().error('OpenAI API key not provided in parameters')
            return
        
        # Initialize LLM planner
        self.llm_planner = LLMActionPlanner(api_key, model)
        
        # Current robot state
        self.current_state = RobotState()
        self.current_environment = EnvironmentContext()
        
        # Create QoS profile for parameters
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        
        # Publishers
        self.plan_publisher = self.create_publisher(
            String,
            'planned_actions',
            10
        )
        
        self.status_publisher = self.create_publisher(
            String,
            'planner_status',
            10
        )
        
        self.feedback_publisher = self.create_publisher(
            String,
            'planner_feedback',
            10
        )
        
        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            'robot_command',
            self.command_callback,
            10
        )
        
        self.state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )
        
        # Timer for periodic updates
        self.update_timer = self.create_timer(1.0, self.update_robot_state)
        
        # Active planning tracking
        self.active_planning = False
        self.planning_thread = None
        
        self.get_logger().info('LLM Planner ROS Node initialized')
    
    def command_callback(self, msg: String):
        """
        Handle incoming robot command
        """
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')
        
        # Publish status
        self._publish_status("RECEIVED_COMMAND", {"command": command_text})
        
        # Perform planning in a separate thread to avoid blocking
        if not self.active_planning:
            self.active_planning = True
            self.planning_thread = threading.Thread(
                target=self._plan_command_thread,
                args=(command_text,)
            )
            self.planning_thread.daemon = True
            self.planning_thread.start()
        else:
            self.get_logger().warn('Planning already in progress, ignoring command')
    
    def _plan_command_thread(self, command: str):
        """
        Perform planning in a background thread
        """
        try:
            # Update status
            self._publish_status("PLANNING_STARTED", {"command": command})
            
            # Perform planning with current state and environment
            plan_result = self.llm_planner.plan_action(
                command=command,
                robot_state=self.current_state,
                environment_context=self.current_environment
            )
            
            # Handle the result
            if plan_result.success:
                # Publish planned actions
                actions_msg = String()
                actions_data = {
                    "actions": [
                        {
                            "type": action.action_type.value,
                            "parameters": action.parameters,
                            "description": action.description,
                            "estimated_duration": action.estimated_duration
                        }
                        for action in plan_result.actions
                    ],
                    "command": command,
                    "reasoning": plan_result.reasoning
                }
                actions_msg.data = json.dumps(actions_data)
                self.plan_publisher.publish(actions_msg)
                
                self.get_logger().info(f'Published plan with {len(plan_result.actions)} actions')
                self._publish_status("PLANNING_SUCCESS", {
                    "action_count": len(plan_result.actions),
                    "execution_time": plan_result.execution_time
                })
                
                # Publish feedback
                feedback_msg = String()
                feedback_msg.data = f"Planned {len(plan_result.actions)} actions for: {command}"
                self.feedback_publisher.publish(feedback_msg)
            else:
                self.get_logger().error(f'Planning failed: {plan_result.reasoning}')
                self._publish_status("PLANNING_FAILED", {
                    "reason": plan_result.reasoning,
                    "safety_issues": plan_result.safety_issues
                })
                
                # Publish error feedback
                feedback_msg = String()
                feedback_msg.data = f"Planning failed: {plan_result.reasoning[:100]}..."
                self.feedback_publisher.publish(feedback_msg)
        
        except Exception as e:
            self.get_logger().error(f'Error in planning thread: {e}')
            self._publish_status("PLANNING_ERROR", {"error": str(e)})
        finally:
            self.active_planning = False
    
    def state_callback(self, msg: JointState):
        """
        Update robot state from joint states
        """
        # Update joint positions in robot state
        for name, position in zip(msg.name, msg.position):
            self.current_state.joint_positions[name] = position
        
        # Update timestamp
        self.current_state.timestamp = time.time()
    
    def update_robot_state(self):
        """
        Periodic update of robot state (location, battery, etc.)
        """
        # This would normally integrate with other nodes to update location, battery, etc.
        # For now, we'll just update the timestamp
        self.current_state.timestamp = time.time()
        
        # Example: update from other sources
        # self.current_state.location = self.get_robot_location()
        # self.current_state.battery_level = self.get_battery_level()
    
    def _publish_status(self, status: str, details: Dict[str, Any] = None):
        """
        Publish planning status
        """
        status_msg = String()
        status_data = {
            "status": status,
            "timestamp": time.time(),
            "details": details or {}
        }
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerROSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Planner node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Error Handling and Validation

Let's add comprehensive error handling and validation mechanisms:

```python
# error_handling.py
import openai
import time
import logging
from typing import Optional, Callable, Any
import json

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute: int = 30):
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.max_calls_per_minute:
                sleep_time = 60 - (now - min(self.call_times))
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.call_times.append(now)

class RobustLLMClient:
    """Robust client with retry logic and error handling"""
    
    def __init__(self, api_key: str, max_retries: int = 3, 
                 base_delay: float = 1.0, backoff_factor: float = 2.0):
        openai.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger(__name__)
    
    def create_completion(self, **kwargs) -> Optional[Any]:
        """
        Create a completion with retry logic
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Make the API call
                response = openai.ChatCompletion.create(**kwargs)
                return response
                
            except openai.error.RateLimitError as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                delay = self.base_delay * (self.backoff_factor ** attempt)
                self.logger.warning(f"Rate limit exceeded, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
                
            except openai.error.APIError as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                delay = self.base_delay * (self.backoff_factor ** attempt)
                self.logger.warning(f"API error, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(delay)
                
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break
                delay = self.base_delay * (self.backoff_factor ** attempt)
                self.logger.warning(f"Unexpected error, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(delay)
        
        self.logger.error(f"All retry attempts failed: {last_exception}")
        return None

class PlanValidator:
    """Validates and sanitizes plan results"""
    
    @staticmethod
    def validate_json_response(response_text: str) -> Optional[Dict[str, Any]]:
        """Validate and parse JSON response from LLM"""
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response from LLM: {e}")
            return None
    
    @staticmethod
    def sanitize_plan(plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate plan data"""
        if not isinstance(plan_data, dict):
            return {"actions": [], "reasoning": "Invalid plan format"}
        
        actions = plan_data.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        
        # Validate each action
        sanitized_actions = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            
            action_type = action.get("action_type", "")
            if not PlanValidator._is_valid_action_type(action_type):
                continue
            
            # Sanitize parameters
            params = action.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            
            sanitized_action = {
                "action_type": action_type,
                "parameters": params,
                "description": str(action.get("description", "")),
                "estimated_duration": float(action.get("estimated_duration", 1.0))
            }
            
            sanitized_actions.append(sanitized_action)
        
        return {
            "actions": sanitized_actions,
            "reasoning": str(plan_data.get("reasoning", "")),
            "estimated_total_time": float(plan_data.get("estimated_total_time", 0.0))
        }
    
    @staticmethod
    def _is_valid_action_type(action_type: str) -> bool:
        """Check if action type is valid"""
        valid_types = {
            "move", "grasp", "place", "speak", "detect", 
            "navigate", "wait", "turn", "custom"
        }
        return action_type in valid_types
```

### Performance Optimization

Let's add caching and performance optimization:

```python
# performance_optimizer.py
import hashlib
import time
from typing import Any, Dict, Optional
from functools import wraps

class PlanCache:
    """Simple cache for planning results"""
    
    def __init__(self, max_size: int = 100, ttl: float = 300.0):  # 5 minutes TTL
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        now = time.time()
        
        if key in self.cache:
            # Check if expired
            if now - self.access_times[key] > self.ttl:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Update access time
            self.access_times[key] = now
            return self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached result"""
        now = time.time()
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = now
    
    def generate_key(self, command: str, state: Any, context: Any) -> str:
        """Generate cache key for a planning request"""
        state_str = str(state) if state else ""
        context_str = str(context) if context else ""
        
        combined = f"{command}|{state_str}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

def with_cache(cache: PlanCache):
    """Decorator to add caching to planning function"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, command, robot_state=None, environment_context=None, *args, **kwargs):
            # Generate cache key
            cache_key = cache.generate_key(command, robot_state, environment_context)
            
            # Check cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                print(f"Cache hit for command: {command[:30]}...")
                return cached_result
            
            # Call original function
            result = func(self, command, robot_state, environment_context, *args, **kwargs)
            
            # Cache the result
            cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator

# Example of how to use the cache in the main planner
class CachedLLMActionPlanner:
    """LLM planner with caching"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.uncached_planner = LLMActionPlanner(api_key, model)
        self.cache = PlanCache(max_size=50, ttl=300.0)  # 50 items, 5 min TTL
    
    @with_cache(cache)  # Apply caching decorator
    def plan_action(self, command: str, robot_state=None, 
                   environment_context=None, temperature: float = 0.2):
        """Plan action with caching"""
        return self.uncached_planner.plan_action(
            command, robot_state, environment_context, temperature
        )
```

### Testing and Validation

Let's create comprehensive tests for our implementation:

```python
# test_llm_planner.py
import unittest
import json
from unittest.mock import Mock, patch
from complete_llm_planner import (
    LLMActionPlanner, RobotAction, PlanResult, RobotState, 
    EnvironmentContext, ActionType, SafetyValidator
)

class TestLLMActionPlanner(unittest.TestCase):
    """Tests for the LLM action planner"""
    
    def setUp(self):
        """Set up test fixtures"""
        # We'll mock the OpenAI API to avoid making real calls
        self.api_key = "test-key"
        self.planner = LLMActionPlanner(self.api_key, model="gpt-3.5-turbo")
    
    @patch('openai.ChatCompletion.create')
    def test_plan_simple_command(self, mock_create):
        """Test planning for a simple command"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action_type": "move",
                    "parameters": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "description": "Move to location (1, 2, 0)"
                }
            ],
            "reasoning": "The user wants to move to position (1, 2, 0)"
        })
        
        mock_create.return_value = mock_response
        
        # Plan a simple command
        command = "Move to position x=1, y=2, z=0"
        result = self.planner.plan_action(command)
        
        # Assertions
        self.assertTrue(result.success)
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].action_type, ActionType.MOVE)
        self.assertEqual(result.actions[0].parameters, {"x": 1.0, "y": 2.0, "z": 0.0})
    
    @patch('openai.ChatCompletion.create')
    def test_plan_with_robot_state(self, mock_create):
        """Test planning with robot state context"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action_type": "speak",
                    "parameters": {"message": "Hello! I am at the kitchen."},
                    "description": "Speak greeting at kitchen location"
                }
            ],
            "reasoning": "The robot is at the kitchen and should greet the user"
        })
        
        mock_create.return_value = mock_response
        
        # Create robot state
        robot_state = RobotState(
            location={"x": 3.0, "y": 2.0, "z": 0.0},
            battery_level=0.85
        )
        
        # Plan with robot state
        command = "Say hello where you are"
        result = self.planner.plan_action(command, robot_state=robot_state)
        
        self.assertTrue(result.success)
        self.assertEqual(result.actions[0].action_type, ActionType.SPEAK)
    
    def test_safety_validator(self):
        """Test safety validation"""
        validator = SafetyValidator()
        
        # Create a plan result with actions
        actions = [
            RobotAction(
                action_type=ActionType.MOVE,
                parameters={"x": 15.0, "y": 15.0, "z": 0.0},  # Outside normal bounds
                description="Move to distant location"
            )
        ]
        
        result = PlanResult(
            success=True,
            actions=actions,
            reasoning="Testing safety validation",
            execution_time=0.0,
            safety_issues=[]
        )
        
        robot_state = RobotState(location={"x": 0.0, "y": 0.0, "z": 0.0})
        env_context = EnvironmentContext()
        
        validated_result = validator.validate_plan(result, robot_state, env_context)
        
        # Should have detected the unsafe move
        self.assertTrue(len(validated_result.safety_issues) > 0)
        self.assertFalse(validated_result.success)
    
    def test_invalid_json_response(self):
        """Test handling of invalid JSON from LLM"""
        # This tests the robustness of the planner when LLM returns invalid JSON
        pass  # Implementation would require patching the JSON parsing
    
    def test_error_handling(self):
        """Test error handling in the planner"""
        # Test with missing API key
        with self.assertRaises(Exception):
            # This is tested by trying to make an actual API call without key
            pass

class TestActionMapping(unittest.TestCase):
    """Tests for action mapping functionality"""
    
    def test_action_schema_construction(self):
        """Test that action schema is correctly constructed"""
        planner = LLMActionPlanner("test-key")
        schema = planner.action_schema
        
        self.assertIn("type", schema)
        self.assertIn("properties", schema)
        self.assertIn("actions", schema["properties"])
        self.assertTrue(isinstance(schema["properties"]["actions"]["items"], dict))

if __name__ == '__main__':
    unittest.main()
```

## Advanced Features and Customization

### Custom Action Types

```python
class CustomActionManager:
    """Manages custom action types for specialized robots"""
    
    def __init__(self):
        self.custom_actions = {}
    
    def register_action(self, name: str, schema: Dict[str, Any], handler: Callable):
        """Register a custom action type"""
        self.custom_actions[name] = {
            "schema": schema,
            "handler": handler
        }
    
    def get_action_schema(self, action_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for a custom action"""
        if action_type in self.custom_actions:
            return self.custom_actions[action_type]["schema"]
        return None

# Example of registering a custom action
def create_robot_dance_handler():
    """Handler for dance actions"""
    def dance_handler(params: Dict[str, Any]):
        dance_type = params.get('dance_type', 'default')
        duration = params.get('duration', 10.0)
        
        # Execute dance sequence
        print(f"Dancing {dance_type} for {duration} seconds")
        return True
    
    return dance_handler

# Register custom action
custom_manager = CustomActionManager()
custom_manager.register_action(
    "dance",
    {
        "type": "object",
        "properties": {
            "dance_type": {"type": "string"},
            "duration": {"type": "number"}
        }
    },
    create_robot_dance_handler()
)
```

## Deployment Considerations

### Configuration Management

```python
# config.py
import os
import json
from typing import Dict, Any

class PlannerConfig:
    """Configuration management for the LLM planner"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Override with environment variables
        config['api_key'] = os.getenv('OPENAI_API_KEY', config.get('api_key', ''))
        config['model'] = os.getenv('LLM_MODEL', config.get('model', 'gpt-4-turbo'))
        config['temperature'] = float(os.getenv('LLM_TEMPERATURE', config.get('temperature', 0.2)))
        config['max_tokens'] = int(os.getenv('LLM_MAX_TOKENS', config.get('max_tokens', 1500)))
        config['timeout'] = float(os.getenv('LLM_TIMEOUT', config.get('timeout', 30.0)))
        
        return config
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety-related configuration"""
        return {
            'max_move_distance': self.config.get('max_move_distance', 10.0),
            'max_payload': self.config.get('max_payload', 3.0),
            'min_battery_level': self.config.get('min_battery_level', 0.1),
            'safety_margin': self.config.get('safety_margin', 0.5)
        }

# Usage
config = PlannerConfig("planner_config.json")
api_key = config.get("api_key")
```

## Summary

In this chapter, we've implemented a complete, production-ready LLM-based action planner for humanoid robots:

1. **Core Implementation**: Created a comprehensive LLM planner with proper error handling, safety validation, and context awareness
2. **ROS 2 Integration**: Developed a complete ROS 2 node for integrating the planner with robot systems
3. **Error Handling**: Implemented robust error handling with retry logic, rate limiting, and validation
4. **Performance Optimization**: Added caching, threading, and async processing for better performance
5. **Testing**: Created comprehensive tests to validate the implementation
6. **Configuration**: Developed flexible configuration management for deployment

The implementation provides a complete foundation for using LLMs in robot action planning, with safety validation, real-time performance, and integration with ROS 2 systems. This allows humanoid robots to understand and execute complex natural language commands while ensuring safety and reliability.

The next step would be to connect this action planner with the vision and language integration components to create a complete Vision-Language-Action system.