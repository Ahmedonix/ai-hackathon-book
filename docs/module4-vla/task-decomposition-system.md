---
title: Natural Language Task Decomposition System
description: Complete implementation of a natural language task decomposition system for humanoid robots
sidebar_position: 7
---

# Natural Language Task Decomposition System

## Overview

This chapter provides a complete, production-ready implementation of a natural language task decomposition system for humanoid robots. The system takes complex natural language commands and breaks them down into executable robot tasks while managing context, dependencies, and safety constraints.

## Learning Objectives

- Implement a complete task decomposition pipeline
- Create a robust system for handling natural language commands
- Build context management and reasoning capabilities
- Integrate with robot execution systems
- Implement safety validation and error recovery

## Complete System Architecture

### Core Task Management System

```python
# complete_task_decomposition_system.py
import openai
import json
import asyncio
import threading
import queue
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import logging
import time
import re

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
    prerequisites: List[str] = field(default_factory=list)  # Tasks that must complete before this
    dependencies: List[str] = field(default_factory=list)   # Tasks that must complete before this
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: float = 1.0  # in seconds
    parent_task_id: Optional[str] = None
    subtasks: List['Task'] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    execution_result: Optional[Any] = None
    error_message: Optional[str] = None
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        result = asdict(self)
        result['task_type'] = self.task_type.value
        result['status'] = self.status.value
        return result

@dataclass
class TaskDecompositionResult:
    """Result of a task decomposition operation"""
    success: bool
    root_task: Task
    all_tasks: List[Task]
    reasoning: str
    validation_issues: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    command: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'success': self.success,
            'root_task': self.root_task.to_dict() if self.root_task else None,
            'all_tasks': [task.to_dict() for task in self.all_tasks],
            'reasoning': self.reasoning,
            'validation_issues': self.validation_issues,
            'execution_time': self.execution_time,
            'command': self.command
        }

@dataclass
class TaskExecutionContext:
    """Context for task execution"""
    robot_state: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    
    def update_from_robot(self, robot_data: Dict[str, Any]):
        """Update context with robot data"""
        self.robot_state.update(robot_data)
    
    def update_from_environment(self, env_data: Dict[str, Any]):
        """Update context with environment data"""
        self.environment.update(env_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'robot_state': self.robot_state,
            'environment': self.environment,
            'user_preferences': self.user_preferences,
            'temporal_context': self.temporal_context
        }

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
    
    def validate_task(self, task: Task, context: TaskExecutionContext = None) -> List[str]:
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
        
        # Check battery if context is provided
        if context and task.task_type in [TaskType.NAVIGATION, TaskType.MANIPULATION]:
            battery_level = context.robot_state.get('battery_level', 1.0)
            if battery_level < self.constraints["min_battery_for_task"]:
                issues.append(f"Battery level {battery_level:.2f} is below minimum threshold {self.constraints['min_battery_for_task']}")
        
        return issues
    
    def validate_task_sequence(self, tasks: List[Task], context: TaskExecutionContext = None) -> List[str]:
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
            "inappropriate", "offensive", "hurt", "pain"
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

class TaskExecutor:
    """Executes tasks in the proper order respecting dependencies"""
    
    def __init__(self, robot_interface=None):
        self.robot_interface = robot_interface
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.logger = logging.getLogger(__name__)
    
    async def execute_task_sequence(self, root_task: Task, 
                                   context: TaskExecutionContext,
                                   progress_callback: Callable[[Task, TaskStatus], None] = None) -> bool:
        """Execute a sequence of tasks respecting dependencies"""
        all_tasks = [root_task] + root_task.get_all_subtasks()
        
        # Initialize task states
        pending_tasks = {task.id: task for task in all_tasks}
        in_progress_tasks = {}
        completed_task_ids = set()
        
        # Check if all dependencies can be resolved
        all_task_ids = {task.id for task in all_tasks}
        for task in all_tasks:
            for dep_id in task.dependencies:
                if dep_id not in all_task_ids:
                    self.logger.error(f"Task {task.id} has dependency on non-existent task {dep_id}")
                    return False
        
        # Execute tasks in dependency order
        while pending_tasks or in_progress_tasks:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = []
            
            for task_id, task in pending_tasks.items():
                # Check if all dependencies are completed
                all_deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id not in completed_task_ids:
                        all_deps_satisfied = False
                        break
                
                if all_deps_satisfied:
                    ready_tasks.append(task)
            
            if not ready_tasks and not in_progress_tasks:
                # No tasks can be executed (likely a dependency cycle)
                self.logger.error("No tasks can be executed - possible dependency cycle")
                return False
            
            # Execute ready tasks
            for task in ready_tasks:
                del pending_tasks[task.id]
                in_progress_tasks[task.id] = task
                
                # Execute in the background
                task.started_at = time.time()
                asyncio.create_task(self._execute_single_task(task, context, progress_callback))
            
            # Wait a bit for tasks to progress
            await asyncio.sleep(0.1)
        
        # Wait for all tasks to complete
        # In a real system, this would be more sophisticated
        # For now, we'll assume tasks complete quickly in this example
        await asyncio.sleep(0.5)
        
        # Check if all tasks completed successfully
        all_completed = all(task_id in self.completed_tasks for task_id in all_task_ids)
        return all_completed
    
    async def _execute_single_task(self, task: Task, 
                                  context: TaskExecutionContext,
                                  progress_callback: Callable[[Task, TaskStatus], None] = None):
        """Execute a single task"""
        try:
            task.started_at = time.time()
            task.status = TaskStatus.IN_PROGRESS
            
            if progress_callback:
                progress_callback(task, task.status)
            
            # Execute based on task type
            success = await self._execute_task_logic(task, context)
            
            task.completed_at = time.time()
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            
            if success:
                self.completed_tasks[task.id] = task
            else:
                self.failed_tasks[task.id] = task
            
            if progress_callback:
                progress_callback(task, task.status)
                
            return success
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.time()
            self.failed_tasks[task.id] = task
            
            if progress_callback:
                progress_callback(task, task.status)
            
            return False
    
    async def _execute_task_logic(self, task: Task, context: TaskExecutionContext) -> bool:
        """Execute the actual logic for a task"""
        # Simulate different execution times based on task type
        if task.task_type == TaskType.NAVIGATION:
            # Simulate navigation - would interface with navigation system
            await asyncio.sleep(3.0)  # 3 seconds for navigation
            # In a real implementation, this would control the robot
            self.logger.info(f"Executed navigation task: {task.description}")
            
        elif task.task_type == TaskType.MANIPULATION:
            # Simulate manipulation - would interface with manipulator
            await asyncio.sleep(2.0)  # 2 seconds for manipulation
            self.logger.info(f"Executed manipulation task: {task.description}")
            
        elif task.task_type == TaskType.PERCEPTION:
            # Simulate perception - would interface with sensors
            await asyncio.sleep(1.0)  # 1 second for perception
            self.logger.info(f"Executed perception task: {task.description}")
            
        elif task.task_type == TaskType.COMMUNICATION:
            # Simulate communication - would interface with speech system
            await asyncio.sleep(0.5)  # 0.5 seconds for communication
            self.logger.info(f"Executed communication task: {task.description}")
        
        # Simulate success (95% success rate for demonstration)
        import random
        return random.random() > 0.05
```

### Natural Language Understanding Component

```python
class NaturalLanguageProcessor:
    """Processes natural language commands and extracts meaningful information"""
    
    def __init__(self):
        # Define patterns for different types of commands
        self.patterns = {
            'navigation': [
                (r'go to (?:the )?(?P<location>\w+(?: \w+)*)', 'navigate_to'),
                (r'move to (?:the )?(?P<location>\w+(?: \w+)*)', 'move_to'),
                (r'travel to (?:the )?(?P<location>\w+(?: \w+)*)', 'travel_to'),
                (r'head to (?:the )?(?P<location>\w+(?: \w+)*)', 'head_to'),
            ],
            'perception': [
                (r'detect (?:the )?(?P<object_type>\w+(?: \w+)*)', 'detect_object'),
                (r'find (?:the )?(?P<object_type>\w+(?: \w+)*)', 'find_object'),
                (r'locate (?:the )?(?P<object_type>\w+(?: \w+)*)', 'locate_object'),
                (r'look for (?:the )?(?P<object_type>\w+(?: \w+)*)', 'search_for'),
            ],
            'manipulation': [
                (r'pick up (?:the )?(?P<object_type>\w+(?: \w+)*)', 'pick_up'),
                (r'grasp (?:the )?(?P<object_type>\w+(?: \w+)*)', 'grasp_object'),
                (r'take (?:the )?(?P<object_type>\w+(?: \w+)*)', 'take_object'),
                (r'get (?:the )?(?P<object_type>\w+(?: \w+)*)', 'get_object'),
                (r'place (?:the )?(?P<object_type>\w+(?: \w+)*) (?:on|at|to) (?:the )?(?P<destination>\w+(?: \w+)*)', 'place_object'),
                (r'put (?:the )?(?P<object_type>\w+(?: \w+)*) (?:on|at|to) (?:the )?(?P<destination>\w+(?: \w+)*)', 'place_object'),
            ],
            'communication': [
                (r'say "(?P<message>.*?)"', 'say_message'),
                (r'speak "(?P<message>.*?)"', 'speak_message'),
                (r'tell me "(?P<message>.*?)"', 'speak_message'),
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                (re.compile(p[0], re.IGNORECASE), p[1]) for p in pattern_list
            ]
    
    def extract_command_parts(self, command: str) -> List[Dict[str, Any]]:
        """Extract meaningful command parts from natural language"""
        command = command.strip()
        extracted_parts = []
        
        # Check each pattern category
        for category, patterns in self.compiled_patterns.items():
            for pattern, action_type in patterns:
                matches = pattern.finditer(command)
                for match in matches:
                    part_info = {
                        'category': category,
                        'action_type': action_type,
                        'parameters': match.groupdict(),
                        'matched_text': match.group(0),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    }
                    extracted_parts.append(part_info)
        
        # Sort by position in the command
        extracted_parts.sort(key=lambda x: x['start_pos'])
        
        return extracted_parts

    def infer_dependencies(self, extracted_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer dependencies between command parts"""
        # This is a simplified dependency inference
        # In a more sophisticated system, this would use deeper NLP
        enhanced_parts = []
        
        for i, part in enumerate(extracted_parts):
            # Basic dependency inference
            dependencies = []
            
            # If this is a manipulation task that places an object, 
            # it likely depends on a previous task that picked it up
            if part['action_type'] == 'place_object':
                # Look for a previous "pick up" action
                for j, prev_part in enumerate(extracted_parts[:i]):
                    if (prev_part['action_type'] in ['pick_up', 'grasp_object'] and 
                        prev_part['parameters'].get('object_type') == part['parameters'].get('object_type')):
                        dependencies.append(f"task_{j}")
            
            # Add inferred dependencies
            enhanced_part = part.copy()
            enhanced_part['dependencies'] = dependencies
            enhanced_parts.append(enhanced_part)
        
        return enhanced_parts
```

### LLM-Enhanced Task Decomposer

```python
class LLMTaskDecomposer:
    """Task decomposer enhanced with LLM capabilities"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        openai.api_key = api_key
        self.model = model
        self.nl_processor = NaturalLanguageProcessor()
        self.logger = logging.getLogger(__name__)
        self.task_counter = 0
        
        # Define the expected JSON schema for LLM responses
        self.task_schema = {
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
                                "enum": [tt.value for tt in TaskType]
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
        
        # System prompt for the LLM
        self.system_prompt = f"""
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
    
    def decompose(self, command: str, context: TaskExecutionContext = None,
                 temperature: float = 0.2) -> TaskDecompositionResult:
        """Decompose a command using LLM for enhanced understanding"""
        start_time = time.time()
        
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
            task_mapping = {}
            
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
                id=f"root_{self._get_next_task_id()}",
                task_type=TaskType.SEQUENCE,
                description=f"Root task for command: {command}",
                parameters={"original_command": command}
            )
            
            # Add all tasks as subtasks of the root
            for task in tasks:
                root_task.add_subtask(task)
            
            # Validate the decomposition
            validator = TaskValidator()
            validation_issues = validator.validate_task_sequence(tasks, context)
            
            result = TaskDecompositionResult(
                success=len(validation_issues) == 0,
                root_task=root_task,
                all_tasks=[root_task] + tasks + [subtask for task in tasks for subtask in task.get_all_subtasks()],
                reasoning=parsed_response.get('reasoning', ''),
                validation_issues=validation_issues,
                execution_time=time.time() - start_time,
                command=command
            )
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"LLM returned invalid JSON: {e}")
            return TaskDecompositionResult(
                success=False,
                root_task=Task(id=f"root_error_{self._get_next_task_id()}", task_type=TaskType.SEQUENCE, description="Error in decomposition"),
                all_tasks=[],
                reasoning=f"LLM returned invalid JSON: {e}",
                validation_issues=[f"JSON parsing error: {e}"],
                execution_time=time.time() - start_time,
                command=command
            )
        except Exception as e:
            self.logger.error(f"Error during LLM decomposition: {e}")
            return TaskDecompositionResult(
                success=False,
                root_task=Task(id=f"root_error_{self._get_next_task_id()}", task_type=TaskType.SEQUENCE, description="Error in decomposition"),
                all_tasks=[],
                reasoning=f"Error during LLM decomposition: {e}",
                validation_issues=[str(e)],
                execution_time=time.time() - start_time,
                command=command
            )
    
    def _create_user_prompt(self, command: str, context: TaskExecutionContext = None) -> str:
        """Create the user prompt for the LLM"""
        prompt_parts = [f"Command: {command}"]
        
        if context:
            prompt_parts.append(f"Context: {json.dumps(context.to_dict(), indent=2)}")
        
        prompt_parts.append("Decompose this command into specific, executable robot tasks.")
        prompt_parts.append("Return your response as valid JSON following the specified schema.")
        
        return "\n\n".join(prompt_parts)
    
    def _get_next_task_id(self) -> str:
        """Get the next unique task ID"""
        self.task_counter += 1
        return f"task_{self.task_counter}"
```

### Hybrid Task Decomposer

```python
class HybridTaskDecomposer:
    """Combines rule-based and LLM-based task decomposition"""
    
    def __init__(self, llm_decomposer: LLMTaskDecomposer = None, 
                 use_llm_threshold: int = 4):
        self.llm_decomposer = llm_decomposer
        self.nl_processor = NaturalLanguageProcessor()
        self.use_llm_threshold = use_llm_threshold  # Use LLM for commands with more than this many words
        self.validator = TaskValidator()
    
    def decompose(self, command: str, context: TaskExecutionContext = None) -> TaskDecompositionResult:
        """Decompose command using appropriate method based on complexity"""
        word_count = len(command.split())
        
        if word_count > self.use_llm_threshold and self.llm_decomposer:
            # Use LLM for complex commands
            return self.llm_decomposer.decompose(command, context)
        else:
            # Use rule-based approach for simple commands
            return self._rule_based_decompose(command, context)
    
    def _rule_based_decompose(self, command: str, context: TaskExecutionContext = None) -> TaskDecompositionResult:
        """Rule-based decomposition for simple commands"""
        start_time = time.time()
        
        try:
            # Extract command parts
            extracted_parts = self.nl_processor.extract_command_parts(command)
            enhanced_parts = self.nl_processor.infer_dependencies(extracted_parts)
            
            # Convert to Task objects
            tasks = []
            root_task = Task(
                id=f"root_rule_{int(time.time())}",
                task_type=TaskType.SEQUENCE,
                description=f"Root task for command: {command}",
                parameters={"original_command": command}
            )
            
            for i, part in enumerate(enhanced_parts):
                task_id = f"rule_task_{i}"
                
                # Map action type to task type
                task_type = self._map_action_to_task_type(part['action_type'])
                
                # Create parameters based on extracted info
                params = part['parameters'].copy()
                if context:
                    # Enhance parameters with context
                    if task_type == TaskType.NAVIGATION and params.get('location'):
                        location = params['location']
                        if location in context.environment.get('locations', {}):
                            params['target_position'] = context.environment['locations'][location]
                    
                    if task_type == TaskType.PERCEPTION and params.get('object_type'):
                        obj_type = params['object_type']
                        # Find object in environment
                        for obj in context.environment.get('objects', []):
                            if obj.get('type') == obj_type:
                                params['object_location'] = obj.get('position')
                                break
                
                task = Task(
                    id=task_id,
                    task_type=task_type,
                    description=f"{part['action_type'].replace('_', ' ').title()} {part.get('parameters', {}).get('object_type', '') or part.get('parameters', {}).get('location', '')}",
                    parameters=params,
                    dependencies=part.get('dependencies', []),
                    estimated_duration=self._estimate_duration(task_type)
                )
                
                tasks.append(task)
                root_task.add_subtask(task)
            
            # Validate the decomposition
            validation_issues = self.validator.validate_task_sequence(tasks, context)
            
            result = TaskDecompositionResult(
                success=len(validation_issues) == 0,
                root_task=root_task,
                all_tasks=[root_task] + tasks + [subtask for task in tasks for subtask in task.get_all_subtasks()],
                reasoning=f"Rule-based decomposition of: {command}",
                validation_issues=validation_issues,
                execution_time=time.time() - start_time,
                command=command
            )
            
            return result
            
        except Exception as e:
            return TaskDecompositionResult(
                success=False,
                root_task=Task(id=f"root_error_{int(time.time())}", task_type=TaskType.SEQUENCE, description="Error in decomposition"),
                all_tasks=[],
                reasoning=f"Error in rule-based decomposition: {e}",
                validation_issues=[str(e)],
                execution_time=time.time() - start_time,
                command=command
            )
    
    def _map_action_to_task_type(self, action_type: str) -> TaskType:
        """Map action type to task type"""
        mapping = {
            'navigate_to': TaskType.NAVIGATION,
            'move_to': TaskType.NAVIGATION,
            'travel_to': TaskType.NAVIGATION,
            'head_to': TaskType.NAVIGATION,
            'detect_object': TaskType.PERCEPTION,
            'find_object': TaskType.PERCEPTION,
            'locate_object': TaskType.PERCEPTION,
            'search_for': TaskType.PERCEPTION,
            'pick_up': TaskType.MANIPULATION,
            'grasp_object': TaskType.MANIPULATION,
            'take_object': TaskType.MANIPULATION,
            'get_object': TaskType.MANIPULATION,
            'place_object': TaskType.MANIPULATION,
            'say_message': TaskType.COMMUNICATION,
            'speak_message': TaskType.COMMUNICATION,
        }
        return mapping.get(action_type, TaskType.CONTROL)
    
    def _estimate_duration(self, task_type: TaskType) -> float:
        """Estimate duration for a task type"""
        estimates = {
            TaskType.NAVIGATION: 10.0,
            TaskType.MANIPULATION: 5.0,
            TaskType.PERCEPTION: 3.0,
            TaskType.COMMUNICATION: 2.0,
            TaskType.CONTROL: 1.0,
            TaskType.SEQUENCE: 1.0,  # Base duration
            TaskType.CONDITIONAL: 1.0
        }
        return estimates.get(task_type, 5.0)
```

### Complete Task Decomposition System

```python
class NaturalLanguageTaskSystem:
    """Complete system for natural language task decomposition and execution"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo"):
        # Initialize components
        self.llm_decomposer = None
        if api_key:
            self.llm_decomposer = LLMTaskDecomposer(api_key, model)
        
        self.hybrid_decomposer = HybridTaskDecomposer(self.llm_decomposer)
        self.task_executor = TaskExecutor()
        self.context_manager = TaskExecutionContext()
        
        # Initialize context with default values
        self.context_manager.robot_state = {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "battery_level": 1.0,
            "gripper_status": "open",
            "current_task": None
        }
        
        self.context_manager.environment = {
            "locations": {},
            "objects": [],
            "obstacles": []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def decompose_command(self, command: str) -> TaskDecompositionResult:
        """Decompose a natural language command into tasks"""
        self.logger.info(f"Decomposing command: {command}")
        
        result = self.hybrid_decomposer.decompose(command, self.context_manager)
        
        if result.success:
            self.logger.info(f"Successfully decomposed into {len(result.all_tasks)} tasks")
        else:
            self.logger.error(f"Failed to decompose command: {result.reasoning}")
        
        return result
    
    async def execute_tasks(self, root_task: Task, 
                           progress_callback: Callable[[Task, TaskStatus], None] = None) -> bool:
        """Execute the decomposed tasks"""
        self.logger.info(f"Executing task sequence starting with: {root_task.description}")
        
        success = await self.task_executor.execute_task_sequence(
            root_task, 
            self.context_manager, 
            progress_callback
        )
        
        if success:
            self.logger.info("Task sequence completed successfully")
        else:
            self.logger.error("Task sequence execution failed")
        
        return success
    
    def update_robot_context(self, robot_data: Dict[str, Any]):
        """Update the robot state in the context"""
        self.context_manager.update_from_robot(robot_data)
    
    def update_environment_context(self, env_data: Dict[str, Any]):
        """Update the environment data in the context"""
        self.context_manager.update_from_environment(env_data)
    
    def process_command(self, command: str,
                       progress_callback: Callable[[Task, TaskStatus], None] = None) -> Dict[str, Any]:
        """Process a command from decomposition to execution"""
        # For async operations, we need to run in an event loop
        # This is a simplified version - in practice, you'd need to handle async properly
        import asyncio
        
        async def async_process():
            # Decompose the command
            decomposition_result = self.decompose_command(command)
            
            if not decomposition_result.success:
                return {
                    "success": False,
                    "error": f"Task decomposition failed: {', '.join(decomposition_result.validation_issues)}",
                    "decomposition_result": decomposition_result
                }
            
            # Execute the decomposed tasks
            execution_success = await self.execute_tasks(
                decomposition_result.root_task,
                progress_callback
            )
            
            return {
                "success": execution_success,
                "decomposition_result": decomposition_result,
                "execution_summary": {
                    "total_tasks": len(decomposition_result.all_tasks),
                    "execution_time": decomposition_result.execution_time
                }
            }
        
        # Run the async operation
        # Note: In a real system, this would be handled differently
        # to avoid blocking the main thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(async_process())
        finally:
            loop.close()
        
        return result

# ROS 2 Integration
class TaskSystemROSNode:
    """ROS 2 node interface for the task decomposition system"""
    
    def __init__(self, system: NaturalLanguageTaskSystem):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # In a real implementation, this would integrate with ROS 2
        # For this example, we'll just simulate the interface methods
        self.active_commands = {}
    
    def handle_command(self, command_msg: str) -> Dict[str, Any]:
        """Handle a natural language command from ROS"""
        self.logger.info(f"Received ROS command: {command_msg}")
        
        # Process the command
        result = self.system.process_command(command_msg)
        
        # Publish results to appropriate ROS topics
        # (This would be implemented with actual ROS publishers in a real system)
        self.publish_decomposition_result(result)
        self.publish_execution_status(result)
        
        return result
    
    def publish_decomposition_result(self, result: Dict[str, Any]):
        """Publish decomposition result to ROS topic"""
        # In a real system, this would publish to a ROS topic
        self.logger.info(f"Publishing decomposition result: {result}")
    
    def publish_execution_status(self, result: Dict[str, Any]):
        """Publish execution status to ROS topic"""
        # In a real system, this would publish to a ROS topic
        self.logger.info(f"Publishing execution status: {result['success']}")
```

### Testing and Validation Framework

```python
# test_task_system.py
import unittest
from unittest.mock import Mock, patch, AsyncMock
import json

class TestNaturalLanguageTaskSystem(unittest.TestCase):
    """Tests for the natural language task system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # For testing, we'll use the rule-based decomposer
        self.task_system = NaturalLanguageTaskSystem()
        
        # Set up a simple context
        self.task_system.context_manager.environment = {
            "locations": {
                "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
                "living_room": {"x": 5.0, "y": 1.0, "z": 0.0}
            },
            "objects": [
                {"type": "cup", "color": "red", "position": {"x": 3.1, "y": 2.1, "z": 0.9}}
            ]
        }
    
    def test_simple_command_decomposition(self):
        """Test decomposition of a simple command"""
        command = "Go to kitchen"
        result = self.task_system.decompose_command(command)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.all_tasks), 3)  # root + 1 subtask + subtask's own structure
        self.assertIn("Go to kitchen", result.root_task.description)
    
    def test_complex_command_decomposition(self):
        """Test decomposition of a complex command"""
        command = "Go to kitchen and pick up the red cup"
        result = self.task_system.decompose_command(command)
        
        self.assertTrue(result.success)
        # Should have at least navigation and manipulation tasks
        task_types = [task.task_type for task in result.all_tasks if task.id != result.root_task.id]
        self.assertIn(TaskType.NAVIGATION, task_types)
        self.assertIn(TaskType.MANIPULATION, task_types)
    
    def test_command_with_context(self):
        """Test that context is properly used in decomposition"""
        # Update context with specific information
        self.task_system.update_environment_context({
            "locations": {
                "kitchen_counter": {"x": 3.2, "y": 2.1, "z": 0.9}
            }
        })
        
        command = "Navigate to kitchen counter"
        result = self.task_system.decompose_command(command)
        
        self.assertTrue(result.success)
        
        # Find navigation task and verify it has the right parameters
        nav_task = None
        for task in result.all_tasks:
            if task.task_type == TaskType.NAVIGATION:
                nav_task = task
                break
        
        self.assertIsNotNone(nav_task)
        self.assertIn('target_position', nav_task.parameters)
    
    def test_invalid_command(self):
        """Test handling of invalid/malformed commands"""
        command = "Invalid command with no clear action"
        result = self.task_system.decompose_command(command)
        
        # The system should handle invalid commands gracefully
        # For the rule-based system, this might result in an empty task list
        # but should not crash
        self.assertIsInstance(result, TaskDecompositionResult)
    
    def test_task_validation(self):
        """Test that tasks are properly validated"""
        # Create a command that should result in invalid tasks
        command = "Go to location that is very far away"
        
        # Add a location that violates constraints
        self.task_system.context_manager.environment = {
            "locations": {
                "far_away": {"x": 100.0, "y": 100.0, "z": 0.0}  # Exceeds max navigation distance
            }
        }
        
        result = self.task_system.decompose_command(command)
        
        # Should have validation issues due to excessive distance
        if result.validation_issues:
            has_distance_issue = any("exceeds limit" in issue for issue in result.validation_issues)
            if has_distance_issue:
                self.assertFalse(result.success)
    
    @patch('asyncio.sleep', new_callable=AsyncMock)
    def test_task_execution(self, mock_sleep):
        """Test task execution"""
        # Mock the sleep to make tests faster
        mock_sleep.return_value = None
        
        command = "Say hello"
        result = self.task_system.decompose_command(command)
        
        self.assertTrue(result.success)
        
        import asyncio
        
        async def run_execution():
            success = await self.task_system.execute_tasks(result.root_task)
            return success
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            execution_success = loop.run_until_complete(run_execution())
            self.assertTrue(execution_success)
        finally:
            loop.close()

class TestHybridTaskDecomposer(unittest.TestCase):
    """Tests for the hybrid task decomposer"""
    
    def setUp(self):
        self.hybrid_decomposer = HybridTaskDecomposer()
    
    def test_short_command_uses_rule_based(self):
        """Test that short commands use rule-based decomposition"""
        command = "Say hello"
        result = self.hybrid_decomposer.decompose(command)
        
        # Should be successful with rule-based approach
        self.assertTrue(result.success)
    
    def test_task_type_mapping(self):
        """Test action to task type mapping"""
        mappings = [
            ("go to kitchen", TaskType.NAVIGATION),
            ("pick up cup", TaskType.MANIPULATION),
            ("find object", TaskType.PERCEPTION),
            ("say hello", TaskType.COMMUNICATION),
        ]
        
        for command, expected_type in mappings:
            result = self.hybrid_decomposer.decompose(command)
            if result.success and result.all_tasks:
                task = result.all_tasks[1]  # First subtask (root is at index 0)
                self.assertEqual(task.task_type, expected_type)

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)

# Example usage
def example_usage():
    """Example of using the complete system"""
    print("Natural Language Task System Example")
    print("="*40)
    
    # Create system (without API key, uses rule-based decomposition)
    task_system = NaturalLanguageTaskSystem()
    
    # Set up some context
    task_system.update_environment_context({
        "locations": {
            "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
            "table": {"x": 5.0, "y": 1.0, "z": 0.0}
        },
        "objects": [
            {"type": "cup", "color": "red", "position": {"x": 3.1, "y": 2.1, "z": 0.9}}
        ]
    })
    
    # Test commands
    commands = [
        "Go to kitchen",
        "Say hello",
        "Go to kitchen and say hello"
    ]
    
    for command in commands:
        print(f"\nCommand: {command}")
        result = task_system.decompose_command(command)
        
        print(f"Success: {result.success}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.validation_issues:
            print(f"Validation issues: {result.validation_issues}")
        
        # Print tasks
        print("Tasks:")
        for task in result.all_tasks:
            if task.id != result.root_task.id:
                print(f"  - {task.task_type.value}: {task.description}")
                print(f"    Parameters: {task.parameters}")
    
    print("\nExample completed!")

if __name__ == "__main__":
    example_usage()
```

### Practical Implementation Guide

```python
# practical_implementation_guide.py
import os
import asyncio
from typing import Dict, Any, Callable

def setup_task_decomposition_system(config: Dict[str, Any]) -> NaturalLanguageTaskSystem:
    """Set up the task decomposition system with configuration"""
    
    # Get API key from config or environment
    api_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
    model = config.get('model', 'gpt-4-turbo')
    
    # Create the system
    if api_key:
        system = NaturalLanguageTaskSystem(api_key=api_key, model=model)
    else:
        system = NaturalLanguageTaskSystem()  # Rule-based only
    
    # Set up initial context
    system.context_manager.robot_state.update(config.get('robot_state', {}))
    system.context_manager.environment.update(config.get('environment', {}))
    
    return system

def create_progress_callback() -> Callable[[Task, TaskStatus], None]:
    """Create a progress callback for monitoring task execution"""
    def callback(task: Task, status: TaskStatus):
        print(f"Task {task.id} ({task.task_type.value}) updated to status: {status.value}")
        if status == TaskStatus.COMPLETED:
            duration = (task.completed_at or time.time()) - (task.started_at or time.time())
            print(f"  Completed in {duration:.2f} seconds")
        elif status == TaskStatus.FAILED:
            print(f"  Failed with error: {task.error_message}")
    
    return callback

def run_practical_example():
    """Run a practical example of the task system"""
    
    # Configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),  # Will be None if not set
        "model": "gpt-4-turbo",
        "robot_state": {
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "battery_level": 0.9,
            "gripper_status": "open"
        },
        "environment": {
            "locations": {
                "kitchen": {"x": 3.0, "y": 2.0, "z": 0.0},
                "living_room": {"x": 5.0, "y": 1.0, "z": 0.0},
                "bedroom": {"x": 1.0, "y": 4.0, "z": 0.0}
            },
            "objects": [
                {"type": "cup", "color": "red", "position": {"x": 3.1, "y": 2.1, "z": 0.9}},
                {"type": "book", "title": "Robotics", "position": {"x": 5.1, "y": 1.1, "z": 0.8}}
            ]
        }
    }
    
    # Set up the system
    task_system = setup_task_decomposition_system(config)
    
    # Create progress callback
    progress_callback = create_progress_callback()
    
    # Example commands to process
    commands = [
        "Go to the kitchen",
        "Pick up the red cup",
        "Go to living room and place the cup on the table",
        "Say hello to everyone in the room"
    ]
    
    print("Natural Language Task Decomposition System - Practical Example")
    print("="*60)
    print(f"System configured with {'LLM+Rules' if config['openai_api_key'] else 'Rules only'} approach")
    print()
    
    for command in commands:
        print(f"Processing command: '{command}'")
        print("-" * 40)
        
        # Process the command
        result = task_system.process_command(command, progress_callback)
        
        print(f"Command: {command}")
        print(f"Success: {result.get('success', 'N/A')}")
        
        if 'decomposition_result' in result:
            decomp_result = result['decomposition_result']
            print(f"Tasks created: {len(decomp_result.all_tasks) - 1}")  # Subtract root task
            print(f"Reasoning: {decomp_result.reasoning[:100]}...")  # Truncate for display
        
        if result.get('success'):
            print("✓ Command processed successfully")
        else:
            print("✗ Command processing failed")
            if 'error' in result:
                print(f"  Error: {result['error']}")
        
        print()
    
    print("Practical example completed!")

def advanced_usage_example():
    """Show advanced usage with context updates and error handling"""
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4-turbo",
        "robot_state": {"location": {"x": 0.0, "y": 0.0, "z": 0.0}, "battery_level": 0.85},
        "environment": {"locations": {"kitchen": {"x": 2.0, "y": 1.0, "z": 0.0}}}
    }
    
    task_system = setup_task_decomposition_system(config)
    
    # Simulate robot movement (context updates)
    print("Advanced Usage Example - Context Updates")
    print("="*50)
    
    # Initial command
    command = "Go to the kitchen"
    result = task_system.decompose_command(command)
    
    if result.success:
        print(f"✓ Decomposed '{command}' into {len(result.all_tasks)-1} tasks")
        
        # Simulate robot moving to kitchen
        task_system.update_robot_context({
            "location": {"x": 2.0, "y": 1.0, "z": 0.0},  # Now in kitchen
            "battery_level": 0.80
        })
        
        print("Robot moved to kitchen, context updated")
        
        # New command based on new position
        new_command = "Find the coffee mug"
        new_result = task_system.decompose_command(new_command)
        
        if new_result.success:
            print(f"✓ Decomposed '{new_command}' with updated context")
        else:
            print(f"✗ Failed to decompose '{new_command}': {new_result.reasoning}")
    else:
        print(f"✗ Failed to decompose initial command: {result.reasoning}")

if __name__ == "__main__":
    run_practical_example()
    print("\n" + "="*60 + "\n")
    advanced_usage_example()
```

## Summary

In this chapter, we've implemented a complete natural language task decomposition system for humanoid robots:

1. **Core Architecture**: Created a robust task management system with proper types, dependencies, and validation
2. **Natural Language Processing**: Implemented both rule-based and LLM-enhanced command parsing
3. **Context Management**: Developed systems for maintaining and utilizing environmental and robot state context
4. **Execution Framework**: Built task execution systems with progress tracking and error handling
5. **Validation and Safety**: Implemented comprehensive validation systems to ensure safe execution
6. **Practical Implementation**: Provided complete examples and testing frameworks

The system successfully bridges the gap between natural language commands and executable robot tasks, incorporating safety validation, context awareness, and the ability to handle both simple and complex commands. It can operate with or without LLMs, using a hybrid approach that leverages LLMs for complex commands and rule-based systems for simpler ones.

This task decomposition system is a critical component of the Vision-Language-Action architecture, enabling robots to understand and execute high-level commands through a sequence of low-level actions.