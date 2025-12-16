---
title: Complete VLA Pipeline Architecture
description: Building a complete Vision-Language-Action pipeline for humanoid robotics
sidebar_position: 11
---

# Complete VLA Pipeline Architecture

## Overview

This chapter provides a comprehensive implementation of the complete Vision-Language-Action (VLA) pipeline architecture for humanoid robotics. We'll build a fully integrated system that processes visual and linguistic inputs to generate robotic actions, with emphasis on real-time performance, safety, and scalability.

## Learning Objectives

- Build a complete VLA pipeline with real-time processing capabilities
- Implement safety mechanisms and error recovery
- Design scalable architecture for multi-robot systems
- Create monitoring and debugging tools for VLA systems
- Validate the pipeline with real-world scenarios

## Complete VLA Pipeline Architecture

### Core Pipeline Components

```python
# complete_vla_pipeline.py
import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

@dataclass
class VLAPipelineInput:
    """Input to the VLA pipeline"""
    image: Optional[np.ndarray] = None
    text: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VLAPipelineOutput:
    """Output from the VLA pipeline"""
    action_plan: List[Dict[str, Any]]
    visual_features: Optional[torch.Tensor] = None
    semantic_features: Optional[torch.Tensor] = None
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    execution_status: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class VLAPipelineComponent(nn.Module):
    """Base class for VLA pipeline components"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__()
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"VLA.{name}")
        self.performance_metrics = {
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'errors': 0
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs with performance monitoring"""
        start_time = time.time()
        self.performance_metrics['call_count'] += 1
        
        try:
            output = self.process(inputs)
            processing_time = time.time() - start_time
            self.performance_metrics['total_time'] += processing_time
            self.performance_metrics['avg_time'] = (
                self.performance_metrics['total_time'] / 
                self.performance_metrics['call_count']
            )
            return output
        except Exception as e:
            self.performance_metrics['errors'] += 1
            self.logger.error(f"Error in {self.name}: {e}")
            self.logger.debug(traceback.format_exc())
            return {'error': str(e)}

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs - to be implemented in subclasses"""
        raise NotImplementedError
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this component"""
        return self.performance_metrics.copy()

class VLAPerceptionComponent(VLAPipelineComponent):
    """Perception component of the VLA pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("perception", config)
        
        # Vision processing model
        from transformers import CLIPProcessor, CLIPModel
        clip_model_name = config.get('clip_model', 'openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Object detection model
        self.object_detector = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=True
        )
        
        # Scene understanding module
        self.scene_graph_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Freezing pre-trained models if specified
        if config.get('freeze_vision_backbone', True):
            for param in self.clip_model.parameters():
                param.requires_grad = False
            for param in self.object_detector.parameters():
                param.requires_grad = False
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual input"""
        image = inputs.get('image')
        if image is None:
            raise ValueError("Image input is required for perception component")
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            # Convert from HWC to CHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        else:
            image_tensor = image
        
        device = next(self.clip_model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Extract visual features
        with torch.no_grad():
            visual_features = self.clip_model.get_image_features(image_tensor)
        
        # Run object detection
        detection_results = self.object_detector(image)
        detected_objects = self._parse_detections(detection_results)
        
        # Scene understanding
        scene_features = self.scene_graph_generator(visual_features)
        
        return {
            'visual_features': visual_features,
            'detected_objects': detected_objects,
            'scene_features': scene_features,
            'image_tensor': image_tensor
        }
    
    def _parse_detections(self, detection_results) -> List[Dict[str, Any]]:
        """Parse detection results from YOLOv5"""
        detections = detection_results.xyxy[0]  # [x1, y1, x2, y2, conf, class]
        
        objects = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            objects.append({
                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                'confidence': conf.item(),
                'class_id': int(cls.item()),
                'class_name': self.object_detector.names[int(cls.item())]
            })
        
        return objects

class VLALanguageComponent(VLAPipelineComponent):
    """Language component of the VLA pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("language", config)
        
        from transformers import AutoTokenizer, AutoModel
        model_name = config.get('language_model', 'bert-base-uncased')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Intent classification head
        self.intent_classifier = nn.Linear(768, 100)
        
        # Entity extraction
        self.entity_extractor = nn.Linear(768, 50)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input"""
        text = inputs.get('text', "")
        if not text:
            raise ValueError("Text input is required for language component")
        
        # Tokenize and encode text
        encoded_inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(next(self.model.parameters()).device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            semantic_features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Classify intent
        intent_logits = self.intent_classifier(semantic_features)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_score, intent_id = torch.max(intent_probs, dim=-1)
        
        # Extract entities
        entities = self._extract_entities(text, intent_id[0].item())
        
        return {
            'semantic_features': semantic_features,
            'text': text,
            'intent': self._id_to_intent(intent_id[0].item()),
            'intent_confidence': intent_score[0].item(),
            'entities': entities,
            'tokenized_input': encoded_inputs
        }
    
    def _extract_entities(self, text: str, intent_id: int) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        # This is a simplified entity extractor
        # In a real system, use NER models like spaCy or BERT-NER
        
        entities = []
        text_lower = text.lower()
        
        # Define entity patterns for different intents
        entity_patterns = {
            0: ['kitchen', 'bedroom', 'living room'],  # navigation intents
            1: ['cup', 'bottle', 'book', 'phone'],     # manipulation intents
            2: ['hello', 'goodbye', 'please', 'thanks'] # communication intents
        }
        
        intent_patterns = entity_patterns.get(intent_id % len(entity_patterns), [])
        
        for pattern in intent_patterns:
            if pattern in text_lower:
                entities.append({
                    'text': pattern,
                    'label': 'LOCATION' if pattern in entity_patterns[0] else 
                             'OBJECT' if pattern in entity_patterns[1] else 
                             'COMMUNICATION',
                    'confidence': 0.8
                })
        
        return entities
    
    def _id_to_intent(self, intent_id: int) -> str:
        """Convert intent ID to label"""
        intents = [
            'navigation', 'manipulation', 'communication', 'perception',
            'greeting', 'farewell', 'question', 'command'
        ]
        return intents[intent_id % len(intents)]

class VLAReasoningComponent(VLAPipelineComponent):
    """Reasoning component of the VLA pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("reasoning", config)
        
        # Action planner
        self.action_planner = nn.Sequential(
            nn.Linear(512 + 768, 512),  # Combined vision + language features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 100 possible actions
        )
        
        # Action templates for different intent types
        self.action_templates = {
            'navigation': [
                {
                    'action': 'move_to_location',
                    'required_params': ['location'],
                    'preconditions': ['robot_operational', 'path_clear']
                }
            ],
            'manipulation': [
                {
                    'action': 'grasp_object',
                    'required_params': ['object_id', 'object_position'],
                    'preconditions': ['object_visible', 'gripper_free', 'reachable']
                },
                {
                    'action': 'place_object',
                    'required_params': ['location'],
                    'preconditions': ['object_held', 'location_free']
                }
            ],
            'communication': [
                {
                    'action': 'speak',
                    'required_params': ['message'],
                    'preconditions': ['speakers_operational']
                }
            ]
        }
        
        # Confidence threshold for action execution
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process perception and language inputs to generate action plan"""
        visual_features = inputs.get('visual_features')
        semantic_features = inputs.get('semantic_features')
        intent = inputs.get('intent', 'unknown')
        entities = inputs.get('entities', [])
        
        if visual_features is None or semantic_features is None:
            raise ValueError("Both visual and semantic features are required for reasoning")
        
        # Combine features
        combined_features = torch.cat([visual_features, semantic_features], dim=-1)
        
        # Generate action predictions
        with torch.no_grad():
            action_logits = self.action_planner(combined_features)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        # Generate action plan based on intent
        action_plan = self._generate_action_plan(intent, entities, action_probs)
        
        # Compute confidence as the average of top action probabilities
        top_probs, _ = torch.topk(action_probs, k=min(3, action_probs.size(1)), dim=-1)
        confidence = torch.mean(top_probs).item()
        
        reasoning_trace = [
            f"Intent: {intent}",
            f"Entities: {[e['text'] for e in entities]}",
            f"Action plan generated with {len(action_plan)} steps",
            f"Overall confidence: {confidence:.3f}"
        ]
        
        return {
            'action_plan': action_plan,
            'confidence': confidence,
            'reasoning_trace': reasoning_trace,
            'combined_features': combined_features
        }
    
    def _generate_action_plan(self, intent: str, entities: List[Dict[str, Any]], 
                            action_probs: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate action plan based on intent and entities"""
        action_plan = []
        
        # Get action templates for this intent
        templates = self.action_templates.get(intent, [])
        
        for template in templates:
            # Check if we have required parameters
            action = {
                'action_type': template['action'],
                'preconditions': template['preconditions'],
                'parameters': {},
                'metadata': {}
            }
            
            # Add entity-based parameters
            for entity in entities:
                if entity['label'].lower() in ['location']:
                    action['parameters']['location'] = entity['text']
                elif entity['label'].lower() in ['object']:
                    action['parameters']['object'] = entity['text']
                elif entity['label'].lower() in ['communication']:
                    action['parameters']['message'] = entity['text']
            
            action_plan.append(action)
        
        return action_plan

class VLAControlComponent(VLAPipelineComponent):
    """Control component of the VLA pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("control", config)
        
        # Robot interface (simulated)
        self.robot_interface = self._initialize_robot_interface()
        
        # Action execution parameters
        self.timeout = config.get('execution_timeout', 30.0)
        self.max_retries = config.get('max_retries', 3)
        
        # Safety constraints
        self.safety_constraints = config.get('safety_constraints', {
            'min_battery': 0.1,
            'max_velocity': 1.0,
            'safe_zone_boundary': 10.0
        })
    
    def _initialize_robot_interface(self) -> Dict[str, Any]:
        """Initialize robot interface (simulated for this example)"""
        return {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion
            'gripper_status': 'open',
            'battery_level': 1.0,
            'operational': True
        }
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action plan on robot"""
        action_plan = inputs.get('action_plan', [])
        if not action_plan:
            return {
                'execution_status': {'success': True, 'actions_executed': 0},
                'robot_state': self.robot_interface
            }
        
        execution_results = []
        all_successful = True
        
        for i, action in enumerate(action_plan):
            success = self._execute_action(action, i)
            execution_results.append({
                'action_index': i,
                'action': action['action_type'],
                'success': success,
                'timestamp': time.time()
            })
            
            if not success:
                all_successful = False
                break  # Stop execution on failure
        
        return {
            'execution_status': {
                'success': all_successful,
                'actions_executed': len(execution_results),
                'action_results': execution_results
            },
            'robot_state': self.robot_interface.copy()
        }
    
    def _execute_action(self, action: Dict[str, Any], action_index: int) -> bool:
        """Execute a single action on the robot"""
        action_type = action['action_type']
        parameters = action.get('parameters', {})
        
        self.logger.info(f"Executing action {action_index}: {action_type}")
        
        # Check safety constraints
        if not self._check_safety_constraints():
            self.logger.error("Safety constraints violated")
            return False
        
        # Execute action based on type
        success = False
        
        if action_type == 'move_to_location':
            success = self._execute_navigation(parameters)
        elif action_type == 'grasp_object':
            success = self._execute_manipulation(parameters)
        elif action_type == 'speak':
            success = self._execute_communication(parameters)
        else:
            self.logger.warning(f"Unknown action type: {action_type}")
            success = False
        
        # Update robot state based on execution
        if success:
            self._update_robot_state(action, parameters)
        
        return success
    
    def _check_safety_constraints(self) -> bool:
        """Check if safety constraints are satisfied"""
        battery_ok = self.robot_interface['battery_level'] > self.safety_constraints['min_battery']
        operational = self.robot_interface['operational']
        
        return battery_ok and operational
    
    def _execute_navigation(self, params: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        location = params.get('location', 'unknown')
        
        self.logger.info(f"Navigating to {location}")
        
        # Update robot position (simulated)
        if location == 'kitchen':
            self.robot_interface['position'] = [3.0, 2.0, 0.0]
        elif location == 'bedroom':
            self.robot_interface['position'] = [1.0, 4.0, 0.0]
        elif location == 'living room':
            self.robot_interface['position'] = [5.0, 1.0, 0.0]
        else:
            # Default movement
            self.robot_interface['position'][0] += 1.0
        
        # Simulate battery drain
        self.robot_interface['battery_level'] = max(0.0, self.robot_interface['battery_level'] - 0.05)
        
        return True
    
    def _execute_manipulation(self, params: Dict[str, Any]) -> bool:
        """Execute manipulation action"""
        obj = params.get('object', 'unknown')
        
        self.logger.info(f"Manipulating object: {obj}")
        
        # Update gripper status
        self.robot_interface['gripper_status'] = 'closed' if self.robot_interface['gripper_status'] == 'open' else 'open'
        
        # Simulate battery drain
        self.robot_interface['battery_level'] = max(0.0, self.robot_interface['battery_level'] - 0.02)
        
        return True
    
    def _execute_communication(self, params: Dict[str, Any]) -> bool:
        """Execute communication action"""
        message = params.get('message', 'Hello')
        
        self.logger.info(f"Speaking: {message}")
        
        # No battery drain for communication
        return True
    
    def _update_robot_state(self, action: Dict[str, Any], params: Dict[str, Any]):
        """Update robot state after action execution"""
        # Update based on action type
        pass

class VLAPipeline(nn.Module):
    """Complete VLA Pipeline with all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        
        # Initialize components
        self.perception_module = VLAPerceptionComponent(
            self.config.get('perception', {})
        )
        self.language_module = VLALanguageComponent(
            self.config.get('language', {})
        )
        self.reasoning_module = VLAReasoningComponent(
            self.config.get('reasoning', {})
        )
        self.control_module = VLAControlComponent(
            self.config.get('control', {})
        )
        
        # Performance tracking
        self.overall_performance = {
            'pipeline_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'success_rate': 0.0,
            'failures': 0
        }
        
        # Safety monitoring
        self.safety_monitor = VLASafetyMonitor()
        
        # Logging
        self.logger = logging.getLogger("VLA.Pipeline")
    
    def forward(self, inputs: Dict[str, Any]) -> VLAPipelineOutput:
        """Process inputs through the complete VLA pipeline"""
        start_time = time.time()
        self.overall_performance['pipeline_calls'] += 1
        
        try:
            # Step 1: Perception
            perception_inputs = {
                'image': inputs.get('image')
            }
            perception_output = self.perception_module(perception_inputs)
            
            if 'error' in perception_output:
                raise Exception(f"Perception error: {perception_output['error']}")
            
            # Step 2: Language processing
            language_inputs = {
                'text': inputs.get('text', '')
            }
            language_output = self.language_module(language_inputs)
            
            if 'error' in language_output:
                raise Exception(f"Language error: {language_output['error']}")
            
            # Step 3: Reasoning
            reasoning_inputs = {
                'visual_features': perception_output['visual_features'],
                'semantic_features': language_output['semantic_features'],
                'intent': language_output['intent'],
                'entities': language_output['entities']
            }
            reasoning_output = self.reasoning_module(reasoning_inputs)
            
            if 'error' in reasoning_output:
                raise Exception(f"Reasoning error: {reasoning_output['error']}")
            
            # Step 4: Control
            control_inputs = {
                'action_plan': reasoning_output['action_plan']
            }
            control_output = self.control_module(control_inputs)
            
            if 'error' in control_output:
                raise Exception(f"Control error: {control_output['error']}")
            
            # Calculate overall confidence
            confidence = min(
                perception_output.get('visual_features', torch.tensor([1.0])).mean().item(),
                language_output.get('intent_confidence', 0.0),
                reasoning_output.get('confidence', 0.0)
            )
            
            # Generate reasoning trace
            reasoning_trace = reasoning_output.get('reasoning_trace', [])
            reasoning_trace.append(f"Control execution: {control_output['execution_status']}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.overall_performance['total_time'] += processing_time
            self.overall_performance['avg_time'] = (
                self.overall_performance['total_time'] / 
                self.overall_performance['pipeline_calls']
            )
            
            # Update success rate
            success = control_output['execution_status']['success']
            if success:
                self.overall_performance['success_rate'] = (
                    self.overall_performance['success_rate'] * 
                    (self.overall_performance['pipeline_calls'] - 1) + 1.0
                ) / self.overall_performance['pipeline_calls']
            else:
                self.overall_performance['failures'] += 1
                self.overall_performance['success_rate'] = (
                    self.overall_performance['success_rate'] * 
                    (self.overall_performance['pipeline_calls'] - 1)
                ) / self.overall_performance['pipeline_calls']
            
            return VLAPipelineOutput(
                action_plan=reasoning_output['action_plan'],
                visual_features=perception_output['visual_features'],
                semantic_features=language_output['semantic_features'],
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                execution_status=control_output['execution_status'],
                timestamp=time.time()
            )
            
        except Exception as e:
            self.overall_performance['failures'] += 1
            self.logger.error(f"Pipeline execution failed: {e}")
            self.logger.debug(traceback.format_exc())
            
            return VLAPipelineOutput(
                action_plan=[],
                confidence=0.0,
                reasoning_trace=[f"Pipeline failed: {str(e)}"],
                execution_status={'success': False, 'error': str(e)},
                timestamp=time.time()
            )
    
    def process_input(self, image: np.ndarray, text: str) -> VLAPipelineOutput:
        """Process image and text input through the pipeline"""
        inputs = {
            'image': image,
            'text': text
        }
        
        return self(inputs)
    
    def get_component_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all components"""
        return {
            'perception': self.perception_module.get_performance_stats(),
            'language': self.language_module.get_performance_stats(),
            'reasoning': self.reasoning_module.get_performance_stats(),
            'control': self.control_module.get_performance_stats(),
            'overall_pipeline': self.overall_performance
        }
```

### Real-Time Processing Pipeline

```python
# real_time_pipeline.py
import asyncio
import threading
from collections import deque
import multiprocessing as mp

class RealTimeVLAPipeline:
    """Real-time VLA pipeline with optimized processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize the VLA pipeline
        self.pipeline = VLAPipeline(config)
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=self.config.get('input_queue_size', 10))
        self.output_queue = queue.Queue(maxsize=self.config.get('output_queue_size', 10))
        self.error_queue = queue.Queue(maxsize=self.config.get('error_queue_size', 5))
        
        # Processing state
        self.running = False
        self.processing_thread = None
        self.main_loop = None
        self.loop_thread = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=1000)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Callbacks
        self.result_callbacks = []
        self.error_callbacks = []
        
        # Logging
        self.logger = logging.getLogger("VLA.RealTimePipeline")
    
    def add_result_callback(self, callback: Callable[[VLAPipelineOutput], None]):
        """Add callback for processing results"""
        self.result_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for error notification"""
        self.error_callbacks.append(callback)
    
    def start_pipeline(self):
        """Start the real-time pipeline"""
        self.running = True
        
        # Start the processing thread
        self.processing_thread = threading.Thread(
            target=self._process_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        # Start the main event loop in a separate thread
        self.loop_thread = threading.Thread(
            target=self._run_async_loop, 
            daemon=True
        )
        self.loop_thread.start()
        
        self.logger.info("Real-time VLA pipeline started")
    
    def stop_pipeline(self):
        """Stop the real-time pipeline"""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.loop_thread:
            # Stop the event loop
            if self.main_loop:
                self.main_loop.call_soon_threadsafe(self.main_loop.stop)
            self.loop_thread.join(timeout=2.0)
        
        self.logger.info("Real-time VLA pipeline stopped")
    
    def submit_input(self, image: np.ndarray, text: str) -> bool:
        """Submit input to the pipeline"""
        try:
            pipeline_input = VLAPipelineInput(image=image, text=text)
            self.input_queue.put(pipeline_input, block=False)
            return True
        except queue.Full:
            self.logger.warning("Input queue is full, dropping frame")
            return False
    
    def get_result(self) -> Optional[VLAPipelineOutput]:
        """Get the next result from the pipeline"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_component_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.pipeline.get_component_performance()
    
    def _run_async_loop(self):
        """Run the async event loop"""
        self.main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.main_loop)
        self.main_loop.run_forever()
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get input
                try:
                    pipeline_input = self.input_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Process input
                start_time = time.time()
                
                output = self.pipeline({
                    'image': pipeline_input.image,
                    'text': pipeline_input.text
                })
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Update FPS counter
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.logger.info(f"Processing FPS: {fps:.2f}")
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Handle results
                try:
                    self.output_queue.put(output, block=False)
                    
                    # Call result callbacks
                    for callback in self.result_callbacks:
                        try:
                            callback(output)
                        except Exception as e:
                            self.logger.error(f"Error in result callback: {e}")
                            
                except queue.Full:
                    self.logger.warning("Output queue is full, dropping result")
                
                # Check for errors
                if not output.execution_status.get('success', True):
                    error_msg = output.execution_status.get('error', 'Unknown error')
                    error_ex = Exception(error_msg)
                    
                    # Add to error queue
                    try:
                        self.error_queue.put(error_ex, block=False)
                    except queue.Full:
                        pass  # Ignore if error queue is full
                    
                    # Call error callbacks
                    for callback in self.error_callbacks:
                        try:
                            callback(error_ex)
                        except Exception as e:
                            self.logger.error(f"Error in error callback: {e}")
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.01)  # Brief pause to prevent excessive errors
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        if not self.processing_times:
            avg_time = 0.0
            min_time = 0.0
            max_time = 0.0
            fps = 0.0
        else:
            times_list = list(self.processing_times)
            avg_time = sum(times_list) / len(times_list)
            min_time = min(times_list)
            max_time = max(times_list)
            
            # Calculate FPS based on processing time
            if avg_time > 0:
                fps = 1.0 / avg_time
            else:
                fps = float('inf')
        
        return {
            'avg_processing_time': avg_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'processing_fps': fps,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'error_queue_size': self.error_queue.qsize()
        }

class VLAWorkerPool:
    """Worker pool for parallel VLA processing"""
    
    def __init__(self, num_workers: int = 4, config: Dict[str, Any] = None):
        self.num_workers = num_workers
        self.config = config or {}
        
        # Shared queues
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.error_queue = mp.Queue()
        
        # Workers
        self.workers = []
        self.running = False
        
        # Performance tracking
        self.start_time = time.time()
        self.total_processed = 0
    
    def start_workers(self):
        """Start processing workers"""
        self.running = True
        
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_process,
                args=(i, self.config)
            )
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        """Stop all workers"""
        self.running = False
        
        # Terminate workers
        for worker in self.workers:
            worker.terminate()
            worker.join(timeout=2.0)
        
        self.workers.clear()
    
    def submit_input(self, input_data: Dict[str, Any]) -> bool:
        """Submit input to worker pool"""
        try:
            self.input_queue.put(input_data, timeout=0.1)
            return True
        except:
            return False
    
    def get_output(self) -> Optional[Dict[str, Any]]:
        """Get output from worker pool"""
        try:
            return self.output_queue.get_nowait()
        except:
            return None
    
    def _worker_process(self, worker_id: int, config: Dict[str, Any]):
        """Worker process function"""
        # Create a pipeline for this worker
        pipeline = VLAPipeline(config)
        
        while self.running:
            try:
                # Get input
                input_data = self.input_queue.get(timeout=0.1)
                
                # Process input
                output = pipeline(input_data)
                
                # Put result
                result = {
                    'worker_id': worker_id,
                    'input_data': input_data,
                    'output': output,
                    'timestamp': time.time()
                }
                
                self.output_queue.put(result)
                
                # Update statistics
                if hasattr(self, '_lock'):
                    self.total_processed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                # Put error in error queue
                error_result = {
                    'worker_id': worker_id,
                    'error': str(e),
                    'timestamp': time.time()
                }
                self.error_queue.put(error_result)
```

### Safety and Monitoring Systems

```python
# safety_monitoring.py
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Callable

@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float
    component: str
    details: Dict[str, Any] = None

class VLASafetyMonitor:
    """Safety monitoring system for VLA pipeline"""
    
    def __init__(self):
        self.violations = []
        self.violation_callbacks = []
        self.running = False
        self.monitor_thread = None
        
        # Safety constraints
        self.constraints = {
            'max_velocity': 1.0,  # m/s
            'min_battery': 0.1,   # 10%
            'safe_zone': {  # Safe zone boundaries
                'x_min': -10.0,
                'x_max': 10.0,
                'y_min': -10.0,
                'y_max': 10.0
            },
            'max_acceleration': 2.0,  # m/s^2
            'max_force': 50.0,        # Newtons
            'temperature_limits': {
                'cpu_max': 85.0,      # Celsius
                'gpu_max': 80.0,
                'motors_max': 70.0
            }
        }
        
        # Current robot state
        self.current_state = {
            'position': [0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0, 0.0],
            'battery_level': 1.0,
            'temperatures': {'cpu': 45.0, 'gpu': 40.0, 'motors': 30.0},
            'collision': False,
            'emergency_stop': False
        }
        
        # Logging
        self.logger = logging.getLogger("VLA.SafetyMonitor")
    
    def start_monitoring(self):
        """Start safety monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop, 
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Safety monitoring stopped")
    
    def add_violation_callback(self, callback: Callable[[SafetyViolation], None]):
        """Add callback for safety violations"""
        self.violation_callbacks.append(callback)
    
    def update_robot_state(self, state: Dict[str, Any]):
        """Update robot state for monitoring"""
        self.current_state.update(state)
    
    def check_safety_constraints(self) -> List[SafetyViolation]:
        """Check all safety constraints and report violations"""
        violations = []
        
        # Check battery level
        if self.current_state['battery_level'] < self.constraints['min_battery']:
            violations.append(SafetyViolation(
                type='battery_low',
                description=f'Battery level {self.current_state["battery_level"]:.2f} below minimum {self.constraints["min_battery"]}',
                severity='high',
                timestamp=time.time(),
                component='power_system'
            ))
        
        # Check safe zone boundaries
        x, y, z = self.current_state['position']
        if (x < self.constraints['safe_zone']['x_min'] or 
            x > self.constraints['safe_zone']['x_max'] or
            y < self.constraints['safe_zone']['y_min'] or 
            y > self.constraints['safe_zone']['y_max']):
            violations.append(SafetyViolation(
                type='boundary_violation',
                description=f'Position [{x:.2f}, {y:.2f}, {z:.2f}] outside safe zone',
                severity='medium',
                timestamp=time.time(),
                component='navigation_system'
            ))
        
        # Check velocity limits
        v_x, v_y, v_z = self.current_state['velocity']
        velocity_magnitude = (v_x**2 + v_y**2 + v_z**2)**0.5
        if velocity_magnitude > self.constraints['max_velocity']:
            violations.append(SafetyViolation(
                type='velocity_exceeded',
                description=f'Velocity {velocity_magnitude:.2f} m/s exceeds maximum {self.constraints["max_velocity"]} m/s',
                severity='high',
                timestamp=time.time(),
                component='motion_control'
            ))
        
        # Check temperature limits
        temps = self.current_state['temperatures']
        if temps['cpu'] > self.constraints['temperature_limits']['cpu_max']:
            violations.append(SafetyViolation(
                type='temperature_critical',
                description=f'CPU temperature {temps["cpu"]:.1f}°C exceeds maximum {self.constraints["temperature_limits"]["cpu_max"]}°C',
                severity='critical',
                timestamp=time.time(),
                component='thermal_system'
            ))
        
        if temps['gpu'] > self.constraints['temperature_limits']['gpu_max']:
            violations.append(SafetyViolation(
                type='temperature_critical',
                description=f'GPU temperature {temps["gpu"]:.1f}°C exceeds maximum {self.constraints["temperature_limits"]["gpu_max"]}°C',
                severity='critical',
                timestamp=time.time(),
                component='thermal_system'
            ))
        
        # Check for collision
        if self.current_state.get('collision', False):
            violations.append(SafetyViolation(
                type='collision_detected',
                description='Physical collision detected',
                severity='critical',
                timestamp=time.time(),
                component='collision_detection'
            ))
        
        # Check for emergency stop
        if self.current_state.get('emergency_stop', False):
            violations.append(SafetyViolation(
                type='emergency_stop',
                description='Emergency stop triggered',
                severity='critical',
                timestamp=time.time(),
                component='safety_system'
            ))
        
        # Store violations
        for violation in violations:
            self.violations.append(violation)
            
            # Call violation callbacks
            for callback in self.violation_callbacks:
                try:
                    callback(violation)
                except Exception as e:
                    self.logger.error(f"Error in violation callback: {e}")
        
        return violations
    
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            try:
                violations = self.check_safety_constraints()
                
                if violations:
                    self.logger.warning(f"Safety violations detected: {len(violations)}")
                    for violation in violations:
                        self.logger.warning(f"  - {violation.type}: {violation.description}")
                
                # Sleep for monitoring interval
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in safety monitoring loop: {e}")
                time.sleep(0.1)

class VLAPipelineMonitor:
    """Comprehensive monitoring for VLA pipeline"""
    
    def __init__(self, pipeline: VLAPipeline):
        self.pipeline = pipeline
        self.metrics_history = deque(maxlen=1000)
        self.alert_callbacks = []
        self.running = False
        self.monitor_thread = None
        self.logger = logging.getLogger("VLA.PipelineMonitor")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start pipeline monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """Stop pipeline monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Pipeline monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        return self.pipeline.get_component_performance()
    
    def _monitoring_loop(self):
        """Monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.get_current_metrics()
                metrics['timestamp'] = time.time()
                self.metrics_history.append(metrics)
                
                # Check for anomalies
                self._check_anomalies(metrics)
                
                # Sleep for monitoring interval
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _check_anomalies(self, metrics: Dict[str, Any]):
        """Check for performance anomalies"""
        # Check if any component is taking too long
        components = ['perception', 'language', 'reasoning', 'control']
        for comp in components:
            if comp in metrics and 'avg_time' in metrics[comp]:
                avg_time = metrics[comp]['avg_time']
                if avg_time > 1.0:  # More than 1 second is concerning
                    alert_msg = f"Slow performance in {comp} component: {avg_time:.3f}s avg"
                    self._trigger_alert('performance_anomaly', {
                        'component': comp,
                        'average_time': avg_time,
                        'alert_message': alert_msg
                    })
        
        # Check success rates
        if 'overall_pipeline' in metrics:
            success_rate = metrics['overall_pipeline']['success_rate']
            if success_rate < 0.8:  # Less than 80% success rate
                self._trigger_alert('success_rate_low', {
                    'success_rate': success_rate,
                    'alert_message': f'Low pipeline success rate: {success_rate:.3f}'
                })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger an alert"""
        self.logger.warning(f"Alert: {alert_type} - {details}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, details)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
```

### Multi-Robot Coordination System

```python
# multi_robot_system.py
import asyncio
import json
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class RobotState:
    """Current state of a robot"""
    id: str
    position: List[float]
    orientation: List[float]
    battery_level: float
    status: str
    last_update: float
    capabilities: List[str]

class MultiRobotVLAController:
    """Controller for coordinating multiple VLA-enabled robots"""
    
    def __init__(self):
        self.robots = {}  # robot_id -> RobotState
        self.queues = {}  # robot_id -> asyncio.Queue
        self.assignments = {}  # task_id -> robot_id
        self.running = False
        self.controller_task = None
        self.logger = logging.getLogger("VLA.MultiRobotController")
    
    def register_robot(self, robot_id: str, capabilities: List[str] = None):
        """Register a new robot with the controller"""
        self.robots[robot_id] = RobotState(
            id=robot_id,
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            battery_level=1.0,
            status='idle',
            last_update=time.time(),
            capabilities=capabilities or ['navigation', 'manipulation', 'communication']
        )
        self.queues[robot_id] = asyncio.Queue()
        self.logger.info(f"Robot {robot_id} registered with capabilities: {capabilities}")
    
    def start_coordinator(self):
        """Start the multi-robot coordinator"""
        self.running = True
        self.controller_task = asyncio.create_task(self._coordinator_loop())
        self.logger.info("Multi-robot coordinator started")
    
    async def stop_coordinator(self):
        """Stop the multi-robot coordinator"""
        self.running = False
        if self.controller_task:
            self.controller_task.cancel()
            try:
                await self.controller_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Multi-robot coordinator stopped")
    
    async def assign_task(self, task: Dict[str, Any]) -> str:
        """Assign a task to an appropriate robot"""
        task_type = task.get('type', 'unknown')
        required_capabilities = task.get('capabilities', [])
        
        # Find available robot that can handle this task
        best_robot = None
        best_score = -1
        
        for robot_id, robot_state in self.robots.items():
            if robot_state.status != 'busy' and robot_state.battery_level > 0.2:
                # Calculate score based on capability match and proximity
                capability_score = sum(
                    1 for cap in required_capabilities 
                    if cap in robot_state.capabilities
                ) / len(required_capabilities) if required_capabilities else 1.0
                
                # Bonus for battery level
                battery_score = robot_state.battery_level
                
                total_score = capability_score * 0.7 + battery_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_robot = robot_id
        
        if best_robot:
            # Assign task to robot
            assignment = {
                'task_id': f"task_{int(time.time() * 1000000)}",
                'robot_id': best_robot,
                'task': task,
                'assigned_time': time.time()
            }
            
            self.assignments[assignment['task_id']] = best_robot
            
            # Send task to robot
            await self.queues[best_robot].put(assignment)
            
            # Update robot status
            self.robots[best_robot].status = 'busy'
            
            self.logger.info(f"Task assigned to robot {best_robot}: {task_type}")
            return assignment['task_id']
        else:
            self.logger.warning(f"No available robot found for task: {task_type}")
            return None
    
    async def _coordinator_loop(self):
        """Main coordinator loop"""
        while self.running:
            try:
                # Update robot states
                await self._update_robot_states()
                
                # Process task assignments
                await self._process_assignments()
                
                # Check for completed tasks
                await self._check_completed_tasks()
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.logger.error(f"Error in coordinator loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_robot_states(self):
        """Update robot states (in a real system, this would come from robots)"""
        current_time = time.time()
        
        for robot_id in self.robots:
            # Simulate state updates
            self.robots[robot_id].last_update = current_time
            # In real system, get actual state from robot
            # self.robots[robot_id] = await self._get_robot_state(robot_id)
    
    async def _process_assignments(self):
        """Process pending assignments"""
        # This would handle more complex task assignment logic
        pass
    
    async def _check_completed_tasks(self):
        """Check for completed tasks and update robot status"""
        # In a real system, this would listen for task completion messages from robots
        pass

class VLAPipelineFactory:
    """Factory for creating VLA pipelines with different configurations"""
    
    def __init__(self):
        self.configurations = {
            'standard': {
                'perception': {'model': 'openai/clip-vit-base-patch32'},
                'language': {'model': 'bert-base-uncased'},
                'reasoning': {'confidence_threshold': 0.7},
                'control': {'execution_timeout': 30.0}
            },
            'high_accuracy': {
                'perception': {'model': 'openai/clip-vit-large-patch14'},
                'language': {'model': 'bert-large-uncased'},
                'reasoning': {'confidence_threshold': 0.85},
                'control': {'execution_timeout': 60.0}
            },
            'real_time': {
                'perception': {'model': 'facebook/detr-resnet-50'},
                'language': {'model': 'distilbert-base-uncased'},
                'reasoning': {'confidence_threshold': 0.6},
                'control': {'execution_timeout': 10.0}
            }
        }
    
    def create_pipeline(self, config_name: str = 'standard') -> VLAPipeline:
        """Create a VLA pipeline with the specified configuration"""
        if config_name not in self.configurations:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config = self.configurations[config_name]
        return VLAPipeline(config)
    
    def create_real_time_pipeline(self) -> RealTimeVLAPipeline:
        """Create a real-time VLA pipeline"""
        config = self.configurations['real_time']
        return RealTimeVLAPipeline(config)

# Example usage and integration
def run_complete_vla_pipeline_demo():
    """Run a complete VLA pipeline demo"""
    print("Complete VLA Pipeline Architecture Demo")
    print("="*50)
    
    # Create pipeline factory
    factory = VLAPipelineFactory()
    
    # Create a standard pipeline
    print("\n1. Creating VLA pipeline...")
    pipeline = factory.create_pipeline('standard')
    print("   ✓ Pipeline created successfully")
    
    # Create real-time pipeline
    print("\n2. Creating real-time pipeline...")
    rt_pipeline = factory.create_real_time_pipeline()
    rt_pipeline.start_pipeline()
    print("   ✓ Real-time pipeline created and started")
    
    # Set up safety monitoring
    print("\n3. Setting up safety monitoring...")
    safety_monitor = VLASafetyMonitor()
    safety_monitor.start_monitoring()
    print("   ✓ Safety monitoring started")
    
    # Set up pipeline monitoring
    print("\n4. Setting up pipeline monitoring...")
    pipeline_monitor = VLAPipelineMonitor(pipeline)
    pipeline_monitor.start_monitoring()
    print("   ✓ Pipeline monitoring started")
    
    # Create a dummy image and text
    print("\n5. Processing sample inputs...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    sample_text = "Go to the kitchen and pick up the red cup"
    
    # Process through pipeline
    start_time = time.time()
    output = pipeline.process_input(dummy_image, sample_text)
    processing_time = time.time() - start_time
    
    print(f"   ✓ Input processed in {processing_time:.3f}s")
    print(f"   Action plan: {len(output.action_plan)} steps")
    print(f"   Confidence: {output.confidence:.3f}")
    
    # Process through real-time pipeline
    print("\n6. Submitting to real-time pipeline...")
    submitted = rt_pipeline.submit_input(dummy_image, sample_text)
    if submitted:
        print("   ✓ Input submitted to real-time pipeline")
        
        # Wait a bit and get result
        time.sleep(1.0)
        result = rt_pipeline.get_result()
        if result:
            print(f"   ✓ Result received: {len(result.action_plan)} actions")
            print(f"   Confidence: {result.confidence:.3f}")
        else:
            print("   ⚠ No result received (might be processing)")
    
    # Show performance metrics
    print("\n7. Performance metrics:")
    metrics = pipeline.get_component_performance()
    for component, stats in metrics.items():
        if 'avg_time' in stats:
            print(f"   {component}: avg_time={stats['avg_time']:.4f}s, errors={stats['errors']}")
    
    # Show real-time metrics
    rt_metrics = rt_pipeline.get_real_time_metrics()
    print(f"   Real-time FPS: {rt_metrics.get('processing_fps', 0):.2f}")
    print(f"   Avg processing time: {rt_metrics.get('avg_processing_time', 0):.4f}s")
    
    # Stop monitoring
    print("\n8. Stopping monitoring systems...")
    pipeline_monitor.stop_monitoring()
    safety_monitor.stop_monitoring()
    rt_pipeline.stop_pipeline()
    print("   ✓ All monitoring systems stopped")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_complete_vla_pipeline_demo()
```

## Summary

In this chapter, we've built a complete Vision-Language-Action (VLA) pipeline architecture for humanoid robotics:

1. **Core Components**: We created modular perception, language processing, reasoning, and control components with standardized interfaces.

2. **Real-Time Processing**: We implemented real-time processing capabilities with optimized queues, performance monitoring, and callback systems.

3. **Safety and Monitoring**: We developed comprehensive safety monitoring systems with constraint checking and violation reporting.

4. **Scalability**: We designed the system to scale from single robots to multi-robot coordination.

5. **Performance Optimization**: We included mechanisms for measuring and optimizing performance across different components.

The complete VLA pipeline provides a robust foundation for building cognitive robots that can understand and act upon both visual and linguistic inputs in real-world environments. The modular architecture allows for easy customization and extension, while the real-time processing capabilities make it suitable for dynamic environments.

The system includes comprehensive safety measures, performance monitoring, and error handling to ensure reliable operation in complex scenarios. This architecture serves as the foundation for the next generation of intelligent humanoid robots.