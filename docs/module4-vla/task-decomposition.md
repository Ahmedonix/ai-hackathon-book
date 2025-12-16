---
title: Natural-Language Task Decomposition
description: Breaking down complex natural language commands into executable robot tasks
sidebar_position: 6
---

# Natural-Language Task Decomposition

## Overview

Natural-language task decomposition is a critical capability in vision-language-action systems, enabling robots to understand and execute complex, multi-step commands expressed in natural language. This chapter explores advanced techniques for parsing, understanding, and decomposing high-level commands into sequences of low-level, executable robot actions.

## Learning Objectives

- Understand the challenges of natural language command interpretation
- Learn techniques for decomposing complex commands into subtasks
- Implement effective task decomposition systems
- Validate and verify decomposed tasks for safety and correctness
- Integrate task decomposition with action planning and execution

## Understanding Task Decomposition Challenges

### The Complexity of Natural Language

Natural language commands can be remarkably complex, containing implicit information, multiple steps, and contextual references. Consider the command: "Go to the kitchen, pick up the red cup from the counter, and bring it to the living room table." This single sentence contains:
- Multiple sequential actions (navigate → detect/grasp → navigate → place)
- Spatial references (kitchen, counter, living room table)
- Object specifications (red cup)
- Temporal sequence

### Key Challenges

1. **Ambiguity**: Natural language often lacks precision
2. **Context Dependency**: Understanding requires environmental context
3. **Multi-Step Planning**: Complex commands need decomposition into sequences
4. **Error Recovery**: Plans need to account for potential failures
5. **Temporal Constraints**: Some tasks have timing requirements

## Task Decomposition Architecture

### Hierarchical Task Structure

```python
# task_decomposition.py
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

class TaskType(Enum):
    """Types of robot tasks"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    CONTROL = "control"
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"

class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Base class for robot tasks"""
    id: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    prerequisite_tasks: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: float = 1.0  # in seconds
    parent_task_id: Optional[str] = None
    subtasks: List['Task'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Tasks that must complete before this
    success_conditions: List[str] = field(default_factory=list)  # Conditions for success
    failure_conditions: List[str] = field(default_factory=list)  # Conditions for failure
    
    def add_subtask(self, subtask: 'Task'):
        """Add a subtask to this task"""
        subtask.parent_task_id = self.id
        self.subtasks.append(subtask)
    
    def get_all_subtasks(self) -> List['Task']:
        """Recursively get all subtasks"""
        all_subtasks = []
        for subtask in self.subtasks:
            all_subtasks.append(subtask)
            all_subtasks.extend(subtask.get_all_subtasks())
        return all_subtasks

@dataclass
class TaskDecompositionResult:
    """Result of a task decomposition operation"""
    success: bool
    root_task: Task
    all_tasks: List[Task]
    reasoning: str
    validation_issues: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class TaskValidator:
    """Validates tasks for safety and executability"""
    
    def __init__(self):
        self.constraints = {
            "max_navigation_distance": 15.0,  # meters
            "max_manipulation_weight": 3.0,   # kg
            "min_battery_for_task": 0.15,     # 15%
            "max_task_sequence_length": 50,   # number of tasks
            "max_execution_time": 600.0       # seconds (10 minutes)
        }
    
    def validate_task(self, task: Task, context: Dict[str, Any] = None) -> List[str]:
        """Validate a single task for safety and executability"""
        issues = []
        
        if task.task_type == TaskType.NAVIGATION:
            # Check navigation distance
            target_pos = task.parameters.get('target_position', [0, 0, 0])
            if len(target_pos) >= 3:
                dist = (target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)**0.5
                if dist > self.constraints["max_navigation_distance"]:
                    issues.append(f"Navigation distance {dist:.2f}m exceeds limit of {self.constraints['max_navigation_distance']}m")
        
        elif task.task_type == TaskType.MANIPULATION:
            # Check manipulation weight
            obj_weight = task.parameters.get('object_weight', 0.0)
            if obj_weight > self.constraints["max_manipulation_weight"]:
                issues.append(f"Object weight {obj_weight}kg exceeds limit of {self.constraints['max_manipulation_weight']}kg")
        
        elif task.task_type == TaskType.COMMUNICATION:
            # Check for inappropriate content
            message = task.parameters.get('message', '')
            if self._contains_inappropriate_content(message):
                issues.append("Communication task contains inappropriate content")
        
        return issues
    
    def validate_task_sequence(self, tasks: List[Task], context: Dict[str, Any] = None) -> List[str]:
        """Validate a sequence of tasks"""
        issues = []
        
        # Check sequence length
        if len(tasks) > self.constraints["max_task_sequence_length"]:
            issues.append(f"Task sequence length {len(tasks)} exceeds limit of {self.constraints['max_task_sequence_length']}")
        
        # Validate each task
        for task in tasks:
            task_issues = self.validate_task(task, context)
            issues.extend([f"Task {task.id}: {issue}" for issue in task_issues])
        
        # Check for dependency cycles and validity
        dependency_issues = self._check_dependencies(tasks)
        issues.extend(dependency_issues)
        
        return issues
    
    def _contains_inappropriate_content(self, message: str) -> bool:
        """Check for inappropriate content in messages"""
        inappropriate_keywords = [
            "harm", "dangerous", "unsafe", "attack", "destroy", "break",
            "inappropriate", "offensive"
        ]
        msg_lower = message.lower()
        return any(keyword in msg_lower for keyword in inappropriate_keywords)
    
    def _check_dependencies(self, tasks: List[Task]) -> List[str]:
        """Check for dependency issues"""
        issues = []
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    issues.append(f"Task {task.id} has dependency on non-existent task {dep_id}")
        
        return issues
```

### Task Decomposition Strategy

```python
import re
from typing import Pattern, Match
from collections import defaultdict

class NaturalLanguageParser:
    """Parses natural language commands and identifies task components"""
    
    def __init__(self):
        # Define patterns for different types of commands
        self.patterns = {
            'navigation': [
                r'go to (?:the )?(?P<location>\w+(?: \w+)*)',
                r'move to (?:the )?(?P<location>\w+(?: \w+)*)',
                r'navigate to (?:the )?(?P<location>\w+(?: \w+)*)',
                r'walk to (?:the )?(?P<location>\w+(?: \w+)*)',
            ],
            'perception': [
                r'detect (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'find (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'locate (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'look for (?:the )?(?P<object_type>\w+(?: \w+)*)',
            ],
            'manipulation': [
                r'pick up (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'grasp (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'take (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'get (?:the )?(?P<object_type>\w+(?: \w+)*)',
                r'place (?:the )?(?P<object_type>\w+(?: \w+)*) (?:on|at) (?:the )?(?P<destination>\w+(?: \w+)*)',
            ],
            'communication': [
                r'say "(?P<message>.*?)"',
                r'speak "(?P<message>.*?)"',
                r'tell "(?P<message>.*?)"',
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for task_type, pattern_list in self.patterns.items():
            self.compiled_patterns[task_type] = [re.compile(p, re.IGNORECASE) for p in pattern_list]
    
    def parse_command(self, command: str) -> List[Dict[str, Any]]:
        """Parse a command and return identified task components"""
        command = command.strip()
        identified_tasks = []
        
        # Check each pattern against the command
        for task_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(command)
                for match in matches:
                    task_info = {
                        'task_type': task_type,
                        'parameters': match.groupdict(),
                        'matched_text': match.group(0),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    }
                    identified_tasks.append(task_info)
        
        # Sort tasks by position in the command
        identified_tasks.sort(key=lambda x: x['start_pos'])
        
        # Determine task sequence based on command structure
        sequential_tasks = self._determine_sequence(command, identified_tasks)
        
        return sequential_tasks
    
    def _determine_sequence(self, command: str, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine the correct sequence for tasks based on command structure"""
        # For now, return tasks in the order they appear
        # In a more sophisticated system, we might use LLMs to determine
        # the correct sequence based on context
        return tasks

class TaskDecomposer:
    """Decomposes natural language commands into task hierarchies"""
    
    def __init__(self, parser: NaturalLanguageParser = None, validator: TaskValidator = None):
        self.parser = parser or NaturalLanguageParser()
        self.validator = validator or TaskValidator()
        self.task_counter = 0
    
    def decompose(self, command: str, context: Dict[str, Any] = None) -> TaskDecompositionResult:
        """Decompose a natural language command into executable tasks"""
        start_time = __import__('time').time()
        
        try:
            # Parse the command
            parsed_tasks = self.parser.parse_command(command)
            
            # Convert parsed tasks to Task objects
            root_task = self._create_root_task(command)
            task_objects = []
            
            for i, parsed_task in enumerate(parsed_tasks):
                task_obj = self._create_task_from_parsed(parsed_task, i, context)
                task_objects.append(task_obj)
                root_task.add_subtask(task_obj)
            
            # Validate the task sequence
            validation_issues = self.validator.validate_task_sequence(task_objects, context)
            
            result = TaskDecompositionResult(
                success=len(validation_issues) == 0,
                root_task=root_task,
                all_tasks=[root_task] + task_objects + [subtask for task in task_objects for subtask in task.get_all_subtasks()],
                reasoning=f"Decomposed command '{command}' into {len(task_objects)} tasks",
                validation_issues=validation_issues,
                execution_time=__import__('time').time() - start_time
            )
            
            return result
            
        except Exception as e:
            return TaskDecompositionResult(
                success=False,
                root_task=self._create_root_task(command),
                all_tasks=[],
                reasoning=f"Error during decomposition: {str(e)}",
                validation_issues=[str(e)],
                execution_time=__import__('time').time() - start_time
            )
    
    def _create_root_task(self, command: str) -> Task:
        """Create the root task for a command"""
        self.task_counter += 1
        return Task(
            id=f"root_{self.task_counter}",
            task_type=TaskType.SEQUENCE,
            description=f"Root task for command: {command}",
            parameters={"original_command": command}
        )
    
    def _create_task_from_parsed(self, parsed_task: Dict[str, Any], index: int, context: Dict[str, Any] = None) -> Task:
        """Create a Task object from parsed task information"""
        self.task_counter += 1
        
        task_type = TaskType[parsed_task['task_type'].upper()]
        
        # Enhance parameters with context information
        enhanced_params = self._enhance_parameters(parsed_task['parameters'], task_type, context)
        
        # Create description
        description = self._create_task_description(task_type, enhanced_params)
        
        return Task(
            id=f"task_{self.task_counter}",
            task_type=task_type,
            description=description,
            parameters=enhanced_params,
            estimated_duration=self._estimate_duration(task_type)
        )
    
    def _enhance_parameters(self, params: Dict[str, Any], task_type: TaskType, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance parameters with context information"""
        enhanced = params.copy()
        
        if context and task_type == TaskType.NAVIGATION:
            # Map location names to coordinates if available in context
            location = params.get('location')
            if location and 'locations' in context:
                location_coords = context['locations'].get(location)
                if location_coords:
                    enhanced['target_position'] = location_coords
        
        elif context and task_type == TaskType.PERCEPTION:
            # Add object information from context
            obj_type = params.get('object_type')
            if obj_type and 'objects' in context:
                # Find objects of the specified type
                matching_objects = [obj for obj in context['objects'] if obj.get('type') == obj_type]
                if matching_objects:
                    # Use the first matching object
                    enhanced['object_info'] = matching_objects[0]
        
        return enhanced
    
    def _create_task_description(self, task_type: TaskType, params: Dict[str, Any]) -> str:
        """Create a human-readable description of the task"""
        if task_type == TaskType.NAVIGATION:
            location = params.get('location', 'unknown location')
            return f"Navigate to {location}"
        elif task_type == TaskType.MANIPULATION:
            if 'destination' in params:
                return f"Place {params.get('object_type', 'object')} at {params.get('destination', 'location')}"
            else:
                return f"Manipulate {params.get('object_type', 'object')}"
        elif task_type == TaskType.PERCEPTION:
            return f"Perceive {params.get('object_type', 'object')}"
        elif task_type == TaskType.COMMUNICATION:
            return f"Communicate: {params.get('message', '...')}"
        else:
            return f"{task_type.value} task"
    
    def _estimate_duration(self, task_type: TaskType) -> float:
        """Estimate the duration of a task"""
        base_durations = {
            TaskType.NAVIGATION: 10.0,
            TaskType.MANIPULATION: 5.0,
            TaskType.PERCEPTION: 3.0,
            TaskType.COMMUNICATION: 2.0,
            TaskType.CONTROL: 1.0
        }
        return base_durations.get(task_type, 5.0)
```

## Advanced Task Decomposition with LLMs

### LLM-Enhanced Task Decomposition

```python
import openai
import json
from typing import Dict, Any, List

class LLMEnhancedTaskDecomposer:
    """Task decomposer enhanced with LLM capabilities"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", 
                 base_decomposer: TaskDecomposer = None):
        openai.api_key = api_key
        self.model = model
        self.base_decomposer = base_decomposer or TaskDecomposer()
        self.task_schema = self._define_task_schema()
        self.system_prompt = self._create_system_prompt()
    
    def _define_task_schema(self) -> Dict[str, Any]:
        """Define the JSON schema for tasks"""
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "task_type": {
                                "type": "string",
                                "enum": ["navigation", "manipulation", "perception", "communication", "control", "sequence", "conditional"]
                            },
                            "description": {"type": "string"},
                            "parameters": {"type": "object"},
                            "estimated_duration": {"type": "number"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["id", "task_type", "description", "parameters"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning for the decomposition"
                }
            },
            "required": ["tasks", "reasoning"]
        }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the LLM"""
        return f"""
You are an advanced task decomposition expert for humanoid robots. Your role is to take complex natural language commands and decompose them into a sequence of specific, executable robot tasks.

Task Types:
- navigation: Moving to specific locations or coordinates
- manipulation: Grasping, placing, or manipulating objects
- perception: Detecting, recognizing, or identifying objects/locations
- communication: Speaking messages or providing feedback
- control: Low-level control actions
- sequence: A sequence of other tasks
- conditional: Tasks that depend on conditions

Each task should have:
- An ID (string)
- A description (what the robot should do)
- Parameters (specific details like coordinates, object properties)
- Dependencies (other tasks that must complete first)

Response Format:
Return your response as valid JSON following this schema:
{json.dumps(self.task_schema, indent=2)}

For complex commands, break them into multiple simple tasks with proper dependencies.
"""
    
    def decompose_with_llm(self, command: str, context: Dict[str, Any] = None,
                          temperature: float = 0.2) -> TaskDecompositionResult:
        """Decompose a command using LLM for enhanced understanding"""
        start_time = __import__('time').time()
        
        try:
            # Create the user prompt
            user_prompt = self._create_user_prompt(command, context)
            
            # Call the LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1500
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)
            
            # Convert to Task objects
            tasks = []
            task_mapping = {}  # Maps task IDs to Task objects
            
            # Create all tasks first
            for task_data in parsed_response.get('tasks', []):
                task_obj = Task(
                    id=task_data['id'],
                    task_type=TaskType(task_data['task_type']),
                    description=task_data['description'],
                    parameters=task_data.get('parameters', {}),
                    dependencies=task_data.get('dependencies', []),
                    estimated_duration=task_data.get('estimated_duration', 5.0)
                )
                tasks.append(task_obj)
                task_mapping[task_data['id']] = task_obj
            
            # Create root task that contains all subtasks
            root_task = Task(
                id="root_llm",
                task_type=TaskType.SEQUENCE,
                description=f"Root task for LLM decomposition of: {command}",
                parameters={"original_command": command}
            )
            
            # Add all tasks as subtasks of the root
            for task in tasks:
                root_task.add_subtask(task)
            
            # Validate the decomposition
            validation_issues = self.base_decomposer.validator.validate_task_sequence(tasks, context)
            
            result = TaskDecompositionResult(
                success=len(validation_issues) == 0,
                root_task=root_task,
                all_tasks=[root_task] + tasks + [subtask for task in tasks for subtask in task.get_all_subtasks()],
                reasoning=parsed_response.get('reasoning', ''),
                validation_issues=validation_issues,
                execution_time=__import__('time').time() - start_time
            )
            
            return result
            
        except json.JSONDecodeError as e:
            return TaskDecompositionResult(
                success=False,
                root_task=Task(id="root_error", task_type=TaskType.SEQUENCE, description="Error in decomposition"),
                all_tasks=[],
                reasoning=f"LLM returned invalid JSON: {e}",
                validation_issues=[f"JSON parsing error: {e}"],
                execution_time=__import__('time').time() - start_time
            )
        except Exception as e:
            return TaskDecompositionResult(
                success=False,
                root_task=Task(id="root_error", task_type=TaskType.SEQUENCE, description="Error in decomposition"),
                all_tasks=[],
                reasoning=f"Error during LLM decomposition: {e}",
                validation_issues=[str(e)],
                execution_time=__import__('time').time() - start_time
            )
    
    def _create_user_prompt(self, command: str, context: Dict[str, Any] = None) -> str:
        """Create the user prompt for the LLM"""
        prompt_parts = [f"Command: {command}"]
        
        if context:
            prompt_parts.append(f"Context: {json.dumps(context, indent=2)}")
        
        prompt_parts.append("Decompose this command into specific, executable robot tasks.")
        prompt_parts.append("Return your response as valid JSON following the specified schema.")
        
        return "\n\n".join(prompt_parts)
    
    def decompose_hybrid(self, command: str, context: Dict[str, Any] = None,
                        use_llm_threshold: int = 4) -> TaskDecompositionResult:
        """
        Hybrid decomposition that uses LLM for complex commands and rule-based for simple ones
        
        Args:
            command: The natural language command
            context: Environmental and robot state context
            use_llm_threshold: Use LLM if command has more than this many words
        """
        word_count = len(command.split())
        
        if word_count > use_llm_threshold:
            # Use LLM for complex commands
            return self.decompose_with_llm(command, context)
        else:
            # Use rule-based approach for simple commands
            return self.base_decomposer.decompose(command, context)
```

## Context-Aware Task Decomposition

### Managing Context for Task Decomposition

```python
from datetime import datetime
from typing import Any

class ContextManager:
    """Manages context for task decomposition"""
    
    def __init__(self):
        self.robot_state = {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "battery_level": 1.0,
            "gripper_status": "open",
            "current_task": None
        }
        
        self.environment = {
            "locations": {},
            "objects": [],
            "obstacles": [],
            "maps": {}
        }
        
        self.temporal_context = {
            "last_command_time": datetime.now(),
            "task_history": [],
            "recent_interactions": []
        }
        
        self.user_context = {
            "preferences": {},
            "familiarity_level": "beginner",
            "language": "en"
        }
    
    def update_robot_state(self, new_state: Dict[str, Any]):
        """Update robot state"""
        self.robot_state.update(new_state)
    
    def update_environment(self, new_env: Dict[str, Any]):
        """Update environmental information"""
        self.environment.update(new_env)
    
    def get_context_for_decomposition(self) -> Dict[str, Any]:
        """Get context relevant for task decomposition"""
        return {
            "robot_state": self.robot_state,
            "environment": self.environment,
            "temporal_context": self.temporal_context,
            "user_context": self.user_context
        }
    
    def add_task_to_history(self, task_id: str, success: bool, duration: float):
        """Add a task to the history"""
        self.temporal_context["task_history"].append({
            "task_id": task_id,
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.temporal_context["task_history"]) > 50:
            self.temporal_context["task_history"] = self.temporal_context["task_history"][-50:]

class ContextAwareDecomposer:
    """Task decomposer that uses rich contextual information"""
    
    def __init__(self, llm_decomposer: LLMEnhancedTaskDecomposer = None):
        self.llm_decomposer = llm_decomposer
        self.context_manager = ContextManager()
        self.task_validator = TaskValidator()
    
    def decompose_with_context(self, command: str) -> TaskDecompositionResult:
        """Decompose a command using full contextual information"""
        context = self.context_manager.get_context_for_decomposition()
        
        if self.llm_decomposer:
            result = self.llm_decomposer.decompose_with_llm(command, context)
        else:
            base_decomposer = TaskDecomposer()
            result = base_decomposer.decompose(command, context)
        
        # Update context based on the decomposition
        self.context_manager.temporal_context["last_command_time"] = datetime.now()
        
        return result
    
    def adapt_decomposition(self, original_result: TaskDecompositionResult,
                           new_context: Dict[str, Any]) -> TaskDecompositionResult:
        """
        Adapt an existing decomposition based on new context information
        """
        # This is a simplified example - in a real system, this would involve
        # re-evaluating the task sequence based on new information
        
        adapted_tasks = []
        
        for task in original_result.all_tasks:
            adapted_task = self._adapt_task_to_context(task, new_context)
            adapted_tasks.append(adapted_task)
        
        # Update root task to reference adapted subtasks
        root_task = original_result.root_task
        root_task.subtasks = [t for t in adapted_tasks if t.parent_task_id == root_task.id]
        
        return TaskDecompositionResult(
            success=original_result.success,
            root_task=root_task,
            all_tasks=adapted_tasks,
            reasoning=original_result.reasoning + " (adapted to new context)",
            validation_issues=original_result.validation_issues,
            execution_time=original_result.execution_time
        )
    
    def _adapt_task_to_context(self, task: Task, context: Dict[str, Any]) -> Task:
        """Adapt a single task to new context information"""
        adapted_task = task  # In a real implementation, this would modify the task
        
        # Example: Update navigation task if robot location changed
        if task.task_type == TaskType.NAVIGATION and 'robot_state' in context:
            robot_pos = context['robot_state'].get('location', {})
            if robot_pos:
                # Adjust target position based on new robot position
                target_pos = task.parameters.get('target_position', [0, 0, 0])
                # Update the parameters to reflect the new starting position
                adapted_task.parameters = task.parameters.copy()
                adapted_task.parameters['adjusted_target'] = target_pos
                adapted_task.parameters['from_position'] = robot_pos
        
        return adapted_task
```

## Task Execution Planning

### Coordinating Decomposed Tasks

```python
from enum import Enum
import asyncio
from typing import Callable, Awaitable

class ExecutionStatus(Enum):
    """Status of task execution"""
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    EXECUTING = "executing"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskExecutionState:
    """State of task during execution"""
    task_id: str
    status: ExecutionStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    result_data: Optional[Dict[str, Any]] = None

class TaskExecutor:
    """Executes decomposed tasks in the proper order"""
    
    def __init__(self, context_manager: ContextManager = None):
        self.context_manager = context_manager or ContextManager()
        self.execution_states: Dict[str, TaskExecutionState] = {}
        self.active_execution = False
    
    async def execute_task_sequence(self, root_task: Task,
                                   progress_callback: Callable[[TaskExecutionState], None] = None) -> bool:
        """Execute a sequence of tasks respecting dependencies"""
        if self.active_execution:
            raise RuntimeError("Task execution already in progress")
        
        self.active_execution = True
        
        try:
            # Get all tasks to execute
            all_tasks = [root_task] + root_task.get_all_subtasks()
            
            # Initialize execution states
            for task in all_tasks:
                self.execution_states[task.id] = TaskExecutionState(
                    task_id=task.id,
                    status=ExecutionStatus.NOT_STARTED
                )
            
            # Execute tasks respecting dependencies
            completed_tasks = set()
            total_tasks = len(all_tasks)
            completed_count = 0
            
            while len(completed_tasks) < total_tasks:
                # Find tasks ready to execute (dependencies satisfied)
                ready_tasks = []
                
                for task in all_tasks:
                    if task.id in completed_tasks:
                        continue
                    
                    # Check if all dependencies are satisfied
                    dependencies_satisfied = True
                    for dep_id in task.dependencies:
                        dep_state = self.execution_states.get(dep_id)
                        if not dep_state or dep_state.status != ExecutionStatus.COMPLETED:
                            dependencies_satisfied = False
                            break
                    
                    if dependencies_satisfied:
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # No tasks can be executed (possible dependency cycle)
                    break
                
                # Execute ready tasks (in parallel would be here)
                for task in ready_tasks:
                    success = await self._execute_single_task(task, progress_callback)
                    if success:
                        completed_tasks.add(task.id)
                        completed_count += 1
                    else:
                        # Task failed - mark it and dependent tasks as failed
                        await self._mark_failed_and_dependents(task.id)
                        return False
            
            return len(completed_tasks) == total_tasks
            
        finally:
            self.active_execution = False
    
    async def _execute_single_task(self, task: Task,
                                  progress_callback: Callable[[TaskExecutionState], None] = None) -> bool:
        """Execute a single task"""
        state = self.execution_states[task.id]
        state.status = ExecutionStatus.QUEUED
        state.start_time = __import__('time').time()
        
        if progress_callback:
            progress_callback(state)
        
        try:
            state.status = ExecutionStatus.EXECUTING
            
            # Simulate task execution based on type
            success = await self._execute_task_logic(task)
            
            state.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
            state.end_time = __import__('time').time()
            
            if progress_callback:
                progress_callback(state)
            
            return success
            
        except Exception as e:
            state.status = ExecutionStatus.FAILED
            state.error_message = str(e)
            state.end_time = __import__('time').time()
            
            if progress_callback:
                progress_callback(state)
            
            return False
    
    async def _execute_task_logic(self, task: Task) -> bool:
        """Execute the actual logic for a task"""
        # In a real implementation, this would interface with the actual robot
        # For simulation, we'll just return success after a delay
        
        # Simulate different execution times based on task type
        if task.task_type == TaskType.NAVIGATION:
            await asyncio.sleep(3.0)  # 3 seconds for navigation
        elif task.task_type == TaskType.MANIPULATION:
            await asyncio.sleep(2.0)  # 2 seconds for manipulation
        else:
            await asyncio.sleep(1.0)  # 1 second for other tasks
        
        # Simulate success/failure (95% success rate)
        import random
        return random.random() > 0.05
    
    async def _mark_failed_and_dependents(self, failed_task_id: str):
        """Mark a failed task and all tasks that depend on it as failed"""
        # This is a simplified implementation
        # In a real system, this would need to handle complex dependency graphs
        for task_id, state in self.execution_states.items():
            if state.status == ExecutionStatus.NOT_STARTED:
                state.status = ExecutionStatus.CANCELLED
                state.error_message = f"Cancelled due to dependency failure of task {failed_task_id}"

class TaskOrchestrator:
    """Orchestrates the complete task decomposition and execution pipeline"""
    
    def __init__(self, decomposer: ContextAwareDecomposer, executor: TaskExecutor):
        self.decomposer = decomposer
        self.executor = executor
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a command from decomposition to execution"""
        # Decompose the command
        decomposition_result = self.decomposer.decompose_with_context(command)
        
        if not decomposition_result.success:
            return {
                "success": False,
                "error": f"Task decomposition failed: {', '.join(decomposition_result.validation_issues)}",
                "decomposition_result": decomposition_result
            }
        
        # Execute the decomposed tasks
        execution_success = await self.executor.execute_task_sequence(
            decomposition_result.root_task
        )
        
        return {
            "success": execution_success,
            "decomposition_result": decomposition_result,
            "execution_summary": {
                "total_tasks": len(decomposition_result.all_tasks),
                "execution_time": __import__('time').time() - decomposition_result.execution_time
            }
        }
```

## Integration with ROS 2

### ROS 2 Integration for Task Decomposition

```python
# llm_task_decomposer_ros.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import json
import asyncio
from threading import Thread

from task_decomposition import TaskDecompositionResult, TaskValidator
from context_aware_decomposer import ContextAwareDecomposer, ContextManager
from llm_enhanced_decomposer import LLMEnhancedTaskDecomposer

class LLMTasksDecomposerROSNode(Node):
    """
    ROS 2 node for LLM-enhanced task decomposition
    """
    
    def __init__(self):
        super().__init__('llm_tasks_decomposer_node')
        
        # Parameters
        self.declare_parameter('openai_api_key', '')
        self.declare_parameter('model', 'gpt-4-turbo')
        self.declare_parameter('enable_llm', True)
        
        api_key = self.get_parameter('openai_api_key').value
        model = self.get_parameter('model').value
        enable_llm = self.get_parameter('enable_llm').value
        
        if not api_key:
            self.get_logger().error('OpenAI API key not provided in parameters')
            return
        
        # Initialize components
        if enable_llm:
            llm_decomposer = LLMEnhancedTaskDecomposer(api_key, model)
            self.decomposer = ContextAwareDecomposer(llm_decomposer)
        else:
            self.decomposer = ContextAwareDecomposer()
        
        # Publishers
        self.decomposition_publisher = self.create_publisher(
            String,
            'decomposed_tasks',
            10
        )
        
        self.status_publisher = self.create_publisher(
            String,
            'tasks_decomposer_status',
            10
        )
        
        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            'natural_language_command',
            self.command_callback,
            10
        )
        
        self.state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.state_callback,
            10
        )
        
        # Initialize context
        self.context_manager = self.decomposer.context_manager
        
        self.get_logger().info('LLM Tasks Decomposer ROS Node initialized')
    
    def command_callback(self, msg: String):
        """
        Handle incoming natural language command
        """
        command_text = msg.data
        self.get_logger().info(f'Received command: {command_text}')
        
        # Publish status
        self._publish_status("RECEIVED_COMMAND", {"command": command_text})
        
        # Perform decomposition in a separate thread to avoid blocking
        thread = Thread(target=self._decompose_command_thread, args=(command_text,))
        thread.daemon = True
        thread.start()
    
    def _decompose_command_thread(self, command: str):
        """
        Perform task decomposition in a background thread
        """
        try:
            self._publish_status("DECOMPOSITION_STARTED", {"command": command})
            
            # Decompose the command
            result = self.decomposer.decompose_with_context(command)
            
            if result.success:
                # Publish decomposed tasks
                tasks_msg = String()
                tasks_data = {
                    "command": command,
                    "root_task": self._task_to_dict(result.root_task),
                    "all_tasks": [self._task_to_dict(task) for task in result.all_tasks],
                    "reasoning": result.reasoning,
                    "validation_issues": result.validation_issues
                }
                tasks_msg.data = json.dumps(tasks_data)
                self.decomposition_publisher.publish(tasks_msg)
                
                self.get_logger().info(f'Decomposed into {len(result.all_tasks)} tasks')
                self._publish_status("DECOMPOSITION_SUCCESS", {
                    "task_count": len(result.all_tasks),
                    "execution_time": result.execution_time
                })
            else:
                self.get_logger().error(f'Decomposition failed: {result.reasoning}')
                self._publish_status("DECOMPOSITION_FAILED", {
                    "reason": result.reasoning,
                    "issues": result.validation_issues
                })
        
        except Exception as e:
            self.get_logger().error(f'Error in decomposition thread: {e}')
            self._publish_status("DECOMPOSITION_ERROR", {"error": str(e)})
    
    def _task_to_dict(self, task) -> Dict[str, Any]:
        """Convert a task to a dictionary for JSON serialization"""
        return {
            "id": task.id,
            "task_type": task.task_type.value,
            "description": task.description,
            "parameters": task.parameters,
            "prerequisite_tasks": task.prerequisite_tasks,
            "dependencies": task.dependencies,
            "estimated_duration": task.estimated_duration,
            "subtasks": [self._task_to_dict(subtask) for subtask in task.subtasks]
        }
    
    def state_callback(self, msg: JointState):
        """
        Update robot state from joint states
        """
        # Update joint positions in context
        joint_positions = dict(zip(msg.name, msg.position))
        self.context_manager.update_robot_state({
            "joint_positions": joint_positions,
            "timestamp": self.get_clock().now().nanoseconds / 1e9
        })
    
    def _publish_status(self, status: str, details: Dict[str, Any] = None):
        """Publish decomposition status"""
        status_msg = String()
        status_data = {
            "status": status,
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "details": details or {}
        }
        status_msg.data = json.dumps(status_data)
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMTasksDecomposerROSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Tasks Decomposer node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Implementation Example

### Complete System Integration

```python
# complete_integration_example.py
import os
import asyncio
from typing import Dict, Any

def run_complete_task_decomposition_example():
    """
    Complete example of task decomposition system
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize the complete system
    llm_decomposer = LLMEnhancedTaskDecomposer(api_key)
    context_decomposer = ContextAwareDecomposer(llm_decomposer)
    
    # Example command
    command = "Go to the kitchen, pick up the red cup from the counter, and bring it to the living room table"
    
    # Example context
    context = {
        "robot_state": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "battery_level": 0.85,
            "gripper_status": "open"
        },
        "environment": {
            "locations": {
                "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
                "living_room": {"x": 5.0, "y": 1.0, "z": 0.0},
                "living_room_table": {"x": 5.2, "y": 1.1, "z": 0.8}
            },
            "objects": [
                {"type": "cup", "color": "red", "position": {"x": 3.2, "y": 2.1, "z": 0.9}}
            ]
        }
    }
    
    # Update context manager
    context_decomposer.context_manager.environment.update(context["environment"])
    context_decomposer.context_manager.robot_state.update(context["robot_state"])
    
    print(f"Decomposing command: {command}")
    result = context_decomposer.decompose_with_context(command)
    
    print(f"Decomposition success: {result.success}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if result.validation_issues:
        print(f"Validation issues: {result.validation_issues}")
    else:
        print("No validation issues found")
    
    # Show the task hierarchy
    print(f"\nTask Hierarchy:")
    print(f"Root task: {result.root_task.description}")
    
    for i, task in enumerate(result.all_tasks):
        if task.id != result.root_task.id:  # Don't show root task details again
            indent = "  " * (task.parent_task_id.count('_') if task.parent_task_id else 1)
            print(f"{indent}{i+1}. {task.task_type.value}: {task.description}")
            print(f"{indent}   Parameters: {task.parameters}")
            if task.dependencies:
                print(f"{indent}   Dependencies: {task.dependencies}")
    
    # Example of adapting a decomposition to new context
    print(f"\nAdapting to new context...")
    new_context = {
        "robot_state": {
            "location": {"x": 2.0, "y": 1.5, "z": 0.0},  # Robot moved closer to kitchen
            "battery_level": 0.75
        }
    }
    
    adapted_result = context_decomposer.adapt_decomposition(result, new_context)
    print(f"Adaptation completed. New execution time estimate: {adapted_result.execution_time:.2f}s")

def run_ros_integration_example():
    """
    Example of how the system would integrate with ROS 2
    """
    # This would be the entry point for the ROS 2 node
    # The actual implementation is in llm_task_decomposer_ros.py
    pass

if __name__ == "__main__":
    run_complete_task_decomposition_example()
```

## Summary

In this chapter, we've covered natural-language task decomposition for humanoid robots:

1. **Task Structure**: Created a hierarchical task representation system with proper types, dependencies, and validation
2. **Parsing and Decomposition**: Implemented both rule-based and LLM-enhanced decomposition approaches
3. **Context Management**: Developed systems for maintaining and utilizing contextual information
4. **Execution Planning**: Created task execution systems with dependency management
5. **ROS Integration**: Designed ROS 2 interfaces for the task decomposition system
6. **Validation and Safety**: Implemented comprehensive validation systems to ensure safe execution

The task decomposition system enables humanoid robots to understand and execute complex natural language commands by breaking them down into sequences of simpler, executable actions. The system incorporates safety validation, context awareness, and the ability to adapt to changing conditions.

This capability is essential for the Vision-Language-Action architecture, as it provides the bridge between high-level language understanding and low-level robot control.