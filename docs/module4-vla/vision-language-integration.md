---
title: Vision-Language Integration for Robotics
description: Integrating visual perception and language processing for cognitive robotics
sidebar_position: 8
---

# Vision-Language Integration for Robotics

## Overview

Vision-Language integration is a critical component of cognitive robotics, enabling robots to understand and interact with their environment using both visual perception and natural language. This chapter explores how to effectively combine computer vision and natural language processing to create more intelligent and intuitive robotic systems.

## Learning Objectives

- Understand the fundamentals of vision-language models
- Implement vision-language integration techniques for robotics
- Create multimodal perception systems
- Integrate visual and linguistic information for decision making
- Build robust systems that handle uncertainty in both modalities

## Vision-Language Models Fundamentals

### Understanding Vision-Language Models

Vision-Language (VL) models are deep learning architectures designed to process and understand information from both visual and textual modalities simultaneously. These models have revolutionized AI applications by enabling more natural interaction between humans and machines.

The core concept behind VL models is learning joint representations of visual and textual information that capture the relationships between them. For robotics applications, this enables:

1. **Visual Question Answering**: Answering questions about the visual environment
2. **Image Captioning**: Generating natural language descriptions of scenes
3. **Object Grounding**: Localizing objects mentioned in text within images
4. **Referring Expression Comprehension**: Identifying objects based on natural language descriptions
5. **Embodied Navigation**: Following natural language instructions in real environments

### Key Architectures

```python
# vl_models.py
import torch
import torch.nn as nn
import torchvision
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2

class VisionLanguageModel(nn.Module):
    """
    Base class for vision-language models in robotics
    """
    def __init__(self):
        super().__init__()
        self.visual_encoder = None
        self.text_encoder = None
        self.fusion_layer = None
        self.task_head = None
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual information"""
        raise NotImplementedError
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode textual information"""
        raise NotImplementedError
    
    def fuse_modalities(self, visual_features: torch.Tensor, 
                       text_features: torch.Tensor) -> torch.Tensor:
        """Fuse visual and textual features"""
        raise NotImplementedError
    
    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the vision-language model"""
        visual_features = self.encode_visual(images)
        text_features = self.encode_text(texts)
        fused_features = self.fuse_modalities(visual_features, text_features)
        
        return {
            'visual_features': visual_features,
            'text_features': text_features,
            'fused_features': fused_features
        }

class CLIPBasedRobotModel(VisionLanguageModel):
    """
    Vision-language model based on CLIP for robotics applications
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        # Load pre-trained CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define additional layers for robotics-specific tasks
        self.object_detection_head = nn.Linear(512, 80)  # COCO classes
        self.action_prediction_head = nn.Linear(512, 100)  # Action space
        
        # Freeze CLIP backbone to preserve pre-trained knowledge
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode visual information using CLIP's visual encoder
        """
        return self.clip_model.get_image_features(images)
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode textual information using CLIP's text encoder
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return self.clip_model.get_text_features(**inputs)
    
    def fuse_modalities(self, visual_features: torch.Tensor, 
                       text_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and textual features using learned attention
        """
        # Normalize features
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_features, text_features.T)
        
        # Use attention mechanism to create fused representation
        attention_weights = torch.softmax(similarity, dim=-1)
        fused_features = torch.matmul(attention_weights, text_features)
        
        return fused_features
    
    def predict_object(self, images: torch.Tensor, query: str) -> Dict[str, Any]:
        """
        Predict objects in the image based on textual query
        """
        # Encode the visual scene
        image_features = self.encode_visual(images)
        
        # Encode the query text
        text_features = self.encode_text([query])
        
        # Fuse modalities
        fused_features = self.fuse_modalities(image_features, text_features)
        
        # Predict using the detection head
        object_logits = self.object_detection_head(fused_features)
        object_probs = torch.softmax(object_logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(object_probs, k=5, dim=-1)
        
        return {
            'object_probabilities': top_probs,
            'object_indices': top_indices,
            'fused_features': fused_features
        }

# Alternative implementation using ViLT (Vision-and-Language Transformer)
class ViLTBasedModel(VisionLanguageModel):
    """
    Vision-language model based on ViLT architecture
    """
    def __init__(self):
        super().__init__()
        
        # Load ViLT or implement a simplified version
        # In practice, you would load a pre-trained model from transformers
        self.visual_backbone = torchvision.models.resnet50(pretrained=True)
        self.visual_backbone.fc = nn.Identity()  # Remove final classification layer
        
        # Text embedding
        self.text_embedding = nn.Embedding(30522, 768)  # Size for BERT vocab
        
        # Multi-modal transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=6
        )
        
        # Task-specific heads
        self.classification_head = nn.Linear(768, 1000)  # Generic classification
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual information using ResNet backbone"""
        features = self.visual_backbone(images)
        return features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode textual information"""
        # This is a simplified implementation
        # In practice, you'd use proper tokenization and embedding
        # For now, using placeholder implementation
        batch_size = len(texts)
        embedded = torch.randn(batch_size, 10, 768)  # Placeholder
        return embedded
    
    def fuse_modalities(self, visual_features: torch.Tensor, 
                       text_features: torch.Tensor) -> torch.Tensor:
        """Fuse modalities using transformer attention"""
        # Concatenate visual and text features
        combined_features = torch.cat([visual_features.unsqueeze(1), text_features], dim=1)
        
        # Apply transformer to fuse information
        fused_features = self.transformer(combined_features.transpose(0, 1))
        
        # Take the first token (visual) as the fused representation
        return fused_features[0]
```

## Robotics-Specific Vision-Language Integration

### Scene Understanding and Object Grounding

```python
# scene_understanding.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2
from collections import defaultdict

class ObjectGroundingSystem:
    """
    System for grounding objects mentioned in text within visual scenes
    """
    def __init__(self, vl_model: VisionLanguageModel):
        self.vl_model = vl_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize object detection components
        self.object_detector = self._init_object_detector()
        
    def _init_object_detector(self):
        """
        Initialize object detection model (using YOLO or similar)
        """
        # Placeholder - in real implementation, load a pre-trained detector
        # For example, YOLOv5, Detectron2, or torchvision models
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    def detect_objects_in_scene(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a scene and return bounding boxes with labels
        """
        # Convert image to tensor and move to device
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        
        # Run object detection
        with torch.no_grad():
            detections = self.object_detector([image_tensor])[0]
        
        # Process detections
        objects = []
        for i in range(len(detections['boxes'])):
            box = detections['boxes'][i].cpu().numpy()
            score = detections['scores'][i].cpu().item()
            label = detections['labels'][i].cpu().item()
            
            # Convert to COCO format label
            label_name = self._get_coco_label_name(label)
            
            if score > 0.5:  # Confidence threshold
                objects.append({
                    'bbox': box,
                    'label': label_name,
                    'confidence': score,
                    'area': (box[2] - box[0]) * (box[3] - box[1])
                })
        
        return objects
    
    def ground_textual_reference(self, image: np.ndarray, 
                                textual_reference: str) -> Optional[Dict[str, Any]]:
        """
        Ground a textual reference to objects in the image
        """
        # First, detect objects in the scene
        detected_objects = self.detect_objects_in_scene(image)
        
        # Encode the textual reference
        text_features = self.vl_model.encode_text([textual_reference])
        
        # Encode each detected object's visual features
        best_match = None
        best_similarity = -float('inf')
        
        for obj in detected_objects:
            # Extract and encode the region of the object
            bbox = obj['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            if x1 < image.shape[1] and y1 < image.shape[0]:
                object_region = image[y1:y2, x1:x2]
                
                # Encode the object region
                region_tensor = torch.from_numpy(object_region).permute(2, 0, 1).float()
                region_tensor = region_tensor.to(self.device).unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    region_features = self.vl_model.encode_visual(region_tensor)
                
                # Compute similarity between text and visual features
                similarity = self._compute_similarity(text_features, region_features)
                
                # Update best match if this is more similar
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'object': obj,
                        'similarity': similarity,
                        'textual_reference': textual_reference
                    }
        
        return best_match
    
    def _compute_similarity(self, text_features: torch.Tensor, 
                           visual_features: torch.Tensor) -> float:
        """
        Compute similarity between text and visual features
        """
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        similarity = torch.matmul(text_features, visual_features.T)
        return similarity.item()
    
    def _get_coco_label_name(self, label_id: int) -> str:
        """
        Convert COCO label ID to name
        """
        # COCO dataset class names
        coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if 0 <= label_id < len(coco_names):
            return coco_names[label_id]
        return f"unknown_{label_id}"

class SceneDescriber:
    """
    System for generating natural language descriptions of scenes
    """
    def __init__(self, vl_model: VisionLanguageModel):
        self.vl_model = vl_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize captioning components
        self.captioning_model = self._init_captioning_model()
    
    def _init_captioning_model(self):
        """
        Initialize image captioning model
        """
        # Placeholder implementation
        # In real implementation, use a pre-trained model like BLIP or similar
        class DummyCaptioningModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 30522)  # Map to vocab size
            
            def forward(self, features):
                # Simplified forward pass
                return torch.randn(features.size(0), 20, 30522)  # (batch, seq_len, vocab)
        
        return DummyCaptioningModel()
    
    def describe_scene(self, image: np.ndarray) -> str:
        """
        Generate a natural language description of the scene
        """
        # Encode the image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            image_features = self.vl_model.encode_visual(image_tensor)
        
        # Generate caption (simplified)
        # In a real model, this would involve beam search or other generation methods
        caption_logits = self.captioning_model(image_features)
        
        # Get most likely tokens
        _, predicted_tokens = torch.max(caption_logits, dim=-1)
        
        # Convert to text (simplified)
        # In practice, you'd use tokenizer to convert IDs to text
        caption = "The scene contains various objects and elements typical of indoor environments"
        
        return caption
```

### Action and Language Integration

```python
# action_language_integration.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ActionGroundingResult:
    """Result of grounding a natural language action to a robot action"""
    success: bool
    action_type: str  # e.g., 'navigation', 'manipulation', 'perception'
    action_parameters: Dict[str, Any]
    confidence: float
    grounding_text: str
    execution_plan: Optional[List[Dict[str, Any]]] = None

class ActionLanguageGrounding:
    """
    System for grounding natural language commands to robot actions
    """
    def __init__(self, vl_model: VisionLanguageModel):
        self.vl_model = vl_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define action vocabulary and templates
        self.action_templates = {
            'navigation': [
                'go to {location}',
                'move to {location}', 
                'navigate to {location}',
                'walk to {location}',
                'head to {location}'
            ],
            'manipulation': [
                'pick up {object}',
                'grasp {object}',
                'take {object}',
                'get {object}',
                'place {object} on {surface}',
                'put {object} on {surface}'
            ],
            'perception': [
                'find {object}',
                'locate {object}',
                'detect {object}',
                'look for {object}',
                'search for {object}'
            ],
            'communication': [
                'say {message}',
                'speak {message}',
                'tell {message}'
            ]
        }
        
        # Initialize action prediction head
        self.action_classifier = nn.Linear(512, len(self.action_templates))
        self.argument_extractor = nn.Linear(512, 256)  # For extracting arguments
    
    def ground_command(self, image: np.ndarray, 
                      command: str, 
                      context: Dict[str, Any] = None) -> ActionGroundingResult:
        """
        Ground a natural language command to a robot action
        """
        # Encode the command
        text_features = self.vl_model.encode_text([command])
        
        # Encode the visual context
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            image_features = self.vl_model.encode_visual(image_tensor)
        
        # Fuse visual and textual features
        fused_features = self.vl_model.fuse_modalities(image_features, text_features)
        
        # Classify action type
        action_logits = self.action_classifier(fused_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Get the most likely action type
        action_idx = torch.argmax(action_probs, dim=-1).item()
        action_types = list(self.action_templates.keys())
        action_type = action_types[action_idx] if action_idx < len(action_types) else 'unknown'
        confidence = action_probs[0, action_idx].item()
        
        # Extract action parameters
        argument_features = self.argument_extractor(fused_features)
        parameters = self._extract_parameters(command, action_type, context)
        
        # Generate execution plan
        execution_plan = self._generate_execution_plan(action_type, parameters)
        
        return ActionGroundingResult(
            success=action_type != 'unknown' and confidence > 0.3,  # Threshold
            action_type=action_type,
            action_parameters=parameters,
            confidence=confidence,
            grounding_text=command,
            execution_plan=execution_plan
        )
    
    def _extract_parameters(self, command: str, action_type: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract parameters from the command based on action type
        """
        parameters = {}
        
        if action_type == 'navigation':
            # Extract location from command
            # This is a simplified approach - in practice, use more sophisticated NLP
            if 'kitchen' in command.lower():
                parameters['target_location'] = 'kitchen'
                if context and 'locations' in context:
                    parameters['target_position'] = context['locations'].get('kitchen', [0, 0, 0])
            elif 'living room' in command.lower():
                parameters['target_location'] = 'living_room'
            elif 'bedroom' in command.lower():
                parameters['target_location'] = 'bedroom'
        
        elif action_type == 'manipulation':
            # Extract object to manipulate
            object_keywords = ['cup', 'bottle', 'book', 'box', 'mug', 'phone', 'tablet']
            for keyword in object_keywords:
                if keyword in command.lower():
                    parameters['object_type'] = keyword
                    break
            
            # Extract destination if present
            if 'place' in command.lower() or 'put' in command.lower():
                if 'table' in command.lower():
                    parameters['destination'] = 'table'
                elif 'counter' in command.lower():
                    parameters['destination'] = 'counter'
        
        elif action_type == 'perception':
            # Extract object to find
            object_keywords = ['cup', 'bottle', 'book', 'box', 'mug', 'phone', 'tablet']
            for keyword in object_keywords:
                if keyword in command.lower():
                    parameters['object_type'] = keyword
                    break
        
        elif action_type == 'communication':
            # Extract message to speak
            import re
            # Look for quoted text
            matches = re.findall(r'"([^"]*)"', command)
            if matches:
                parameters['message'] = matches[0]
            else:
                # Extract everything after "say" or "speak"
                for phrase in ['say', 'speak', 'tell']:
                    if phrase in command.lower():
                        start_idx = command.lower().find(phrase) + len(phrase)
                        message = command[start_idx:].strip()
                        # Remove any remaining articles, etc.
                        message = message.lstrip(' .,!?')
                        if message:
                            parameters['message'] = message
                        break
        
        return parameters
    
    def _generate_execution_plan(self, action_type: str, 
                                parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate an execution plan for the grounded action
        """
        plan = []
        
        if action_type == 'navigation':
            if 'target_position' in parameters:
                plan.append({
                    'action': 'navigate',
                    'target_position': parameters['target_position'],
                    'description': f"Navigate to {parameters.get('target_location', 'specified location')}"
                })
        
        elif action_type == 'manipulation':
            if parameters.get('object_type'):
                plan.append({
                    'action': 'detect_object',
                    'object_type': parameters['object_type'],
                    'description': f"Detect {parameters['object_type']}"
                })
                
                plan.append({
                    'action': 'approach_object',
                    'object_type': parameters['object_type'],
                    'description': f"Approach {parameters['object_type']}"
                })
                
                plan.append({
                    'action': 'grasp_object',
                    'object_type': parameters['object_type'],
                    'description': f"Grasp {parameters['object_type']}"
                })
                
                if 'destination' in parameters:
                    plan.append({
                        'action': 'navigate',
                        'target_location': parameters['destination'],
                        'description': f"Navigate to {parameters['destination']}"
                    })
                    
                    plan.append({
                        'action': 'place_object',
                        'destination': parameters['destination'],
                        'description': f"Place object at {parameters['destination']}"
                    })
        
        elif action_type == 'perception':
            if parameters.get('object_type'):
                plan.append({
                    'action': 'search_for_object',
                    'object_type': parameters['object_type'],
                    'description': f"Search for {parameters['object_type']}"
                })
        
        elif action_type == 'communication':
            if parameters.get('message'):
                plan.append({
                    'action': 'speak',
                    'message': parameters['message'],
                    'description': f"Speak: {parameters['message']}"
                })
        
        return plan
```

## Vision-Language Integration Pipeline

### Complete Integration System

```python
# complete_vl_integration.py
import asyncio
import threading
from typing import Any, Dict, List, Optional
from queue import Queue
import time

class VisionLanguageIntegrationPipeline:
    """
    Complete pipeline for vision-language integration in robotics
    """
    def __init__(self, vl_model: VisionLanguageModel, 
                 grounding_system: ActionLanguageGrounding = None,
                 object_grounding: ObjectGroundingSystem = None,
                 scene_describer: SceneDescriber = None):
        self.vl_model = vl_model
        self.grounding_system = grounding_system or ActionLanguageGrounding(vl_model)
        self.object_grounding = object_grounding or ObjectGroundingSystem(vl_model)
        self.scene_describer = scene_describer or SceneDescriber(vl_model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Components for real-time processing
        self.image_queue = Queue()
        self.text_queue = Queue()
        self.result_queue = Queue()
        
        # Processing state
        self.running = False
        self.processing_thread = None
        
        # Context management
        self.current_context = {}
        
    def start_processing(self):
        """
        Start the vision-language processing pipeline
        """
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        print("Vision-Language Integration Pipeline started")
    
    def stop_processing(self):
        """
        Stop the vision-language processing pipeline
        """
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Vision-Language Integration Pipeline stopped")
    
    def _process_loop(self):
        """
        Main processing loop that handles incoming images and text
        """
        while self.running:
            try:
                # Check for new images or text
                if not self.image_queue.empty() or not self.text_queue.empty():
                    # Process any pending images
                    while not self.image_queue.empty():
                        image_data = self.image_queue.get()
                        self._process_image(image_data)
                    
                    # Process any pending text commands
                    while not self.text_queue.empty():
                        text_data = self.text_queue.get()
                        self._process_text(text_data)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Brief pause before continuing
    
    def _process_image(self, image_data: Dict[str, Any]):
        """
        Process an incoming image
        """
        image = image_data['image']
        image_id = image_data.get('id', time.time())
        
        try:
            # Generate scene description
            scene_description = self.scene_describer.describe_scene(image)
            
            # Detect objects in the scene
            detected_objects = self.object_grounding.detect_objects_in_scene(image)
            
            # Update context with latest visual information
            self.current_context.update({
                'last_image_id': image_id,
                'scene_description': scene_description,
                'detected_objects': detected_objects,
                'timestamp': time.time()
            })
            
            # Create result
            result = {
                'type': 'image_processing',
                'image_id': image_id,
                'scene_description': scene_description,
                'detected_objects': detected_objects,
                'timestamp': time.time()
            }
            
            self.result_queue.put(result)
            
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
    
    def _process_text(self, text_data: Dict[str, Any]):
        """
        Process an incoming text command
        """
        command = text_data['text']
        text_id = text_data.get('id', time.time())
        
        try:
            # If we have a recent image, use it for grounding
            if 'last_image_id' in self.current_context:
                last_image = self.current_context.get('last_image', None)
                if last_image is not None:
                    # Ground the command with visual context
                    grounding_result = self.grounding_system.ground_command(
                        last_image, 
                        command, 
                        self.current_context
                    )
                    
                    # Create result
                    result = {
                        'type': 'text_grounding',
                        'text_id': text_id,
                        'command': command,
                        'grounding_result': grounding_result,
                        'timestamp': time.time()
                    }
                    
                    self.result_queue.put(result)
            
        except Exception as e:
            print(f"Error processing text command {text_id}: {e}")
    
    def submit_image(self, image: np.ndarray, image_id: Optional[str] = None):
        """
        Submit an image for processing
        """
        image_data = {
            'image': image,
            'id': image_id or str(time.time())
        }
        self.image_queue.put(image_data)
    
    def submit_command(self, command: str, text_id: Optional[str] = None):
        """
        Submit a text command for processing
        """
        text_data = {
            'text': command,
            'id': text_id or str(time.time())
        }
        self.text_queue.put(text_data)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all pending results
        """
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
    
    def process_command_with_image(self, image: np.ndarray, 
                                  command: str) -> ActionGroundingResult:
        """
        Process a command with a specific image (synchronous)
        """
        grounding_result = self.grounding_system.ground_command(
            image, 
            command, 
            self.current_context
        )
        
        return grounding_result
    
    def get_current_context(self) -> Dict[str, Any]:
        """
        Get the current system context
        """
        return self.current_context.copy()
    
    def update_context(self, updates: Dict[str, Any]):
        """
        Update the system context
        """
        self.current_context.update(updates)

# Example usage and testing
class VisionLanguageSystemExample:
    """
    Example of using the complete vision-language integration system
    """
    def __init__(self):
        # Initialize with a CLIP-based model
        self.vl_model = CLIPBasedRobotModel()
        self.pipeline = VisionLanguageIntegrationPipeline(
            vl_model=self.vl_model
        )
    
    def run_example(self):
        """
        Run a complete example of vision-language integration
        """
        print("Starting Vision-Language Integration Example")
        print("="*50)
        
        # Start processing pipeline
        self.pipeline.start_processing()
        
        # Simulate a scenario: Robot receives image and command
        print("\n1. Simulating image capture...")
        
        # Create a dummy image (in practice, this would come from robot's camera)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.pipeline.submit_image(dummy_image)
        
        # Wait a bit for image processing
        time.sleep(1)
        
        print("   Image sent for processing")
        
        # Submit a command
        command = "Go to the kitchen and pick up the red cup"
        print(f"\n2. Processing command: '{command}'")
        
        self.pipeline.submit_command(command)
        
        # Wait for processing
        time.sleep(2)
        
        # Get results
        results = self.pipeline.get_results()
        for result in results:
            if result['type'] == 'text_grounding':
                grounding_result = result['grounding_result']
                print(f"   Action type: {grounding_result.action_type}")
                print(f"   Parameters: {grounding_result.action_parameters}")
                print(f"   Confidence: {grounding_result.confidence:.2f}")
                print(f"   Success: {grounding_result.success}")
                
                if grounding_result.execution_plan:
                    print("   Execution plan:")
                    for step in grounding_result.execution_plan:
                        print(f"     - {step['description']}")
        
        # Stop pipeline
        self.pipeline.stop_processing()
        print("\nExample completed")

def main():
    """
    Main function to demonstrate the vision-language integration system
    """
    example = VisionLanguageSystemExample()
    example.run_example()

if __name__ == "__main__":
    main()
```

## Real-World Robotic Applications

### Integration with Robot Control

```python
# robot_integration.py
import asyncio
from typing import Dict, Any, Optional
import cv2
import numpy as np

class RoboticVisionLanguageInterface:
    """
    Interface to connect vision-language system with robot control
    """
    def __init__(self, vl_pipeline: VisionLanguageIntegrationPipeline, 
                 robot_controller=None):
        self.vl_pipeline = vl_pipeline
        self.robot_controller = robot_controller
        self.active_tasks = {}
        
    def handle_command(self, command: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Handle a natural language command with optional image context
        """
        try:
            if image is not None:
                # Process with visual context
                result = self.vl_pipeline.process_command_with_image(image, command)
            else:
                # Process without visual context (may use stored context)
                # Submit command to pipeline for background processing
                self.vl_pipeline.submit_command(command)
                
                # Wait for result or use stored context
                results = self.vl_pipeline.get_results()
                result = None
                for res in results:
                    if res['type'] == 'text_grounding':
                        result = res['grounding_result']
                        break
                
                if result is None:
                    return {
                        'success': False,
                        'error': 'Could not ground command without visual context',
                        'command': command
                    }
            
            if not result.success:
                return {
                    'success': False,
                    'error': f'Command grounding failed with confidence: {result.confidence}',
                    'command': command
                }
            
            # Execute the action plan
            execution_result = self._execute_action_plan(result)
            
            return {
                'success': True,
                'action_type': result.action_type,
                'parameters': result.action_parameters,
                'execution_result': execution_result,
                'confidence': result.confidence,
                'command': command
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error handling command: {str(e)}',
                'command': command
            }
    
    def _execute_action_plan(self, grounding_result: ActionGroundingResult) -> Dict[str, Any]:
        """
        Execute the grounded action plan on the robot
        """
        if not self.robot_controller:
            return {
                'success': True,
                'message': 'No robot controller connected, simulating execution',
                'actions_attempted': len(grounding_result.execution_plan or [])
            }
        
        results = []
        success = True
        
        for step in grounding_result.execution_plan or []:
            try:
                action = step['action']
                
                if action == 'navigate':
                    # Execute navigation
                    target_pos = step.get('target_position')
                    if target_pos:
                        nav_success = self.robot_controller.navigate_to(target_pos)
                        results.append({
                            'action': action,
                            'success': nav_success,
                            'target': target_pos
                        })
                        if not nav_success:
                            success = False
                            break  # Stop execution if navigation fails
                
                elif action == 'speak':
                    # Execute speech
                    message = step.get('message', '')
                    speak_success = self.robot_controller.speak(message)
                    results.append({
                        'action': action,
                        'success': speak_success,
                        'message': message
                    })
                
                elif action == 'detect_object':
                    # Execute object detection
                    obj_type = step.get('object_type')
                    detection_result = self.robot_controller.detect_object(obj_type)
                    results.append({
                        'action': action,
                        'success': detection_result is not None,
                        'object_type': obj_type,
                        'detection_result': detection_result
                    })
                
                # Add other action types as needed
                
            except Exception as e:
                results.append({
                    'action': step.get('action', 'unknown'),
                    'success': False,
                    'error': str(e)
                })
                success = False
                break  # Stop execution on error
        
        return {
            'success': success,
            'action_results': results
        }
    
    def process_camera_stream(self, camera_source, command_queue: Queue):
        """
        Process a continuous camera stream with incoming commands
        """
        def stream_processor():
            cap = cv2.VideoCapture(camera_source)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check for new commands
                try:
                    while not command_queue.empty():
                        command = command_queue.get_nowait()
                        self.handle_command(command, frame)
                except:
                    pass  # No new commands
                
                # Process frame through VL pipeline
                self.vl_pipeline.submit_image(frame)
                
                # Brief pause to control frame rate
                time.sleep(1/30.0)  # 30 FPS
            
            cap.release()
        
        # Run in separate thread
        thread = threading.Thread(target=stream_processor, daemon=True)
        thread.start()
        return thread

# Practical implementation example
def create_robot_vl_system():
    """
    Create a complete vision-language system for a robot
    """
    # Initialize the vision-language model
    vl_model = CLIPBasedRobotModel()
    
    # Create the integration pipeline
    vl_pipeline = VisionLanguageIntegrationPipeline(vl_model)
    
    # Create the robotic interface
    robot_vl_interface = RoboticVisionLanguageInterface(vl_pipeline)
    
    # Start processing
    vl_pipeline.start_processing()
    
    return robot_vl_interface

def demo_robot_vl_interaction():
    """
    Demonstrate human-robot interaction using vision-language system
    """
    print("Robot Vision-Language Interaction Demo")
    print("="*40)
    
    # Create the system
    robot_vl_system = create_robot_vl_system()
    
    # Simulate scenarios
    scenarios = [
        {
            'command': "Say hello to everyone in the room",
            'image': None  # No specific image needed
        },
        {
            'command': "Find the red ball",
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated image
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['command']}")
        
        result = robot_vl_system.handle_command(
            scenario['command'], 
            scenario['image']
        )
        
        print(f"Success: {result['success']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Action type: {result.get('action_type', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    demo_robot_vl_interaction()
```

## Summary

In this chapter, we've covered vision-language integration for robotics:

1. **Fundamentals**: We explored key vision-language model architectures like CLIP and ViLT
2. **Robotics Applications**: We implemented scene understanding, object grounding, and action planning
3. **Integration Pipeline**: We created a complete pipeline for processing visual and textual inputs
4. **Real-World Applications**: We demonstrated how to connect vision-language systems with robot control

The vision-language integration system enables robots to understand both visual scenes and natural language commands, combining these modalities to execute complex tasks. This multimodal approach allows for more natural and intuitive human-robot interaction, where users can refer to objects in the environment using natural language.

The system handles uncertainty in both visual perception and language understanding, using learned representations to bridge the gap between these modalities. This creates more robust and capable robotic systems that can operate effectively in real-world environments with natural human instructions.