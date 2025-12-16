---
title: Multi-Modal Interaction
description: Voice, gesture, and vision integration for humanoid robotics
sidebar_position: 12
---

# Multi-Modal Interaction: Voice, Gesture, and Vision Integration

## Overview

Multi-modal interaction represents the convergence of multiple sensory and communication channels - voice, gesture, and vision - to create more natural and intuitive human-robot interfaces. This chapter explores the integration of these modalities to enable sophisticated interaction patterns for humanoid robots.

## Learning Objectives

- Understand the principles of multi-modal interaction
- Implement voice, gesture, and vision integration
- Create unified representation spaces for multi-modal data
- Handle temporal synchronization across modalities
- Design intuitive multi-modal interaction paradigms

## Multi-Modal Integration Architecture

### Core Multi-Modal Components

```python
# multimodal_components.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import logging
import threading
import queue

@dataclass
class VoiceInput:
    """Voice input data"""
    audio_stream: np.ndarray
    text_transcript: str
    confidence: float
    speaker_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class GestureInput:
    """Gesture input data"""
    hand_landmarks: np.ndarray  # Coordinates of hand landmarks
    gesture_type: str
    gesture_confidence: float
    gesture_duration: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class VisionInput:
    """Vision input data"""
    image: np.ndarray
    camera_intrinsics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

@dataclass
class MultiModalOutput:
    """Output from multi-modal integration"""
    integrated_features: torch.Tensor
    interpreted_intent: str
    confidence: float
    attention_weights: Dict[str, float]  # Attention given to each modality
    timestamp: float = field(default_factory=time.time)

class MultiModalComponent(nn.Module):
    """Base class for multi-modal components"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__()
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"Multimodal.{name}")
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs"""
        raise NotImplementedError

class VoiceProcessingComponent(MultiModalComponent):
    """Voice processing component for multi-modal interaction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("voice_processing", config)
        
        # Speech-to-text model
        self.use_stt_model = config.get('use_stt', True)
        if self.use_stt_model:
            try:
                import whisper
                self.stt_model = whisper.load_model("base")
                self.use_whisper = True
            except ImportError:
                self.logger.warning("Whisper not available, using mock STT")
                self.use_whisper = False
        
        # Voice activity detection
        self.vad_threshold = config.get('vad_threshold', 0.01)
        
        # Speaker diarization (simulated)
        self.simulate_speaker_id = config.get('simulate_speaker_id', True)
    
    def process(self, audio_data: np.ndarray, sample_rate: int = 16000) -> VoiceInput:
        """Process audio data and extract voice information"""
        # Detect voice activity
        rms_energy = np.sqrt(np.mean(audio_data**2))
        has_voice_activity = rms_energy > self.vad_threshold
        
        if not has_voice_activity:
            return VoiceInput(
                audio_stream=audio_data,
                text_transcript="",
                confidence=0.0,
                timestamp=time.time()
            )
        
        # Convert audio for processing
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Get transcript using STT model
        if self.use_whisper:
            try:
                # Temporarily save audio and transcribe
                # (In a real system, use streaming transcription)
                import io
                import soundfile as sf
                
                # Create temporary buffer
                temp_buffer = io.BytesIO()
                sf.write(temp_buffer, audio_data, sample_rate, format='RAW', subtype='PCM_16')
                temp_buffer.seek(0)
                
                # Transcribe
                result = self.stt_model.transcribe(temp_buffer.getvalue())
                transcript = result['text']
                confidence = 0.8  # Simplified confidence
            except Exception as e:
                self.logger.error(f"STT error: {e}")
                transcript = "transcription_error"
                confidence = 0.0
        else:
            # Mock transcription
            transcript = self._mock_transcription(audio_data)
            confidence = 0.7
        
        # Simulate speaker identification
        speaker_id = "speaker_001" if self.simulate_speaker_id else None
        
        return VoiceInput(
            audio_stream=audio_data,
            text_transcript=transcript.strip(),
            confidence=confidence,
            speaker_id=speaker_id,
            timestamp=time.time()
        )
    
    def _mock_transcription(self, audio_data: np.ndarray) -> str:
        """Mock transcription for when STT models are not available"""
        # In a real implementation, this would use proper STT
        # For simulation, return predefined phrases
        sample_phrases = [
            "Can you come here?",
            "Look at this object",
            "Go to the kitchen",
            "Pick up that cup",
            "What do you see?"
        ]
        import random
        return random.choice(sample_phrases)

class GestureProcessingComponent(MultiModalComponent):
    """Gesture processing component for multi-modal interaction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("gesture_processing", config)
        
        # Hand landmark detection model
        self.use_hand_model = config.get('use_hand_model', True)
        if self.use_hand_model:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
            except ImportError:
                self.logger.warning("MediaPipe not available, using mock gesture detection")
                self.use_mediapipe = False
            else:
                self.use_mediapipe = True
        
        # Gesture recognition model
        self.gesture_recognizer = self._init_gesture_recognizer()
    
    def process(self, image: np.ndarray) -> GestureInput:
        """Process image to detect gestures"""
        if self.use_mediapipe:
            try:
                # Convert BGR to RGB (assuming input is OpenCV format)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process hand landmarks
                results = self.hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Extract hand landmarks
                    hand_landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.append([lm.x, lm.y, lm.z])
                        hand_landmarks.append(np.array(landmarks))
                    
                    # Recognize gesture
                    gesture_info = self._recognize_gesture(hand_landmarks)
                    
                    return GestureInput(
                        hand_landmarks=hand_landmarks[0] if hand_landmarks else np.zeros((21, 3)),
                        gesture_type=gesture_info['type'],
                        gesture_confidence=gesture_info['confidence'],
                        gesture_duration=0.0,  # Would be calculated over time
                        timestamp=time.time()
                    )
                
            except Exception as e:
                self.logger.error(f"Gesture processing error: {e}")
        
        # Return default if no gesture detected
        return GestureInput(
            hand_landmarks=np.zeros((21, 3)),
            gesture_type="unknown",
            gesture_confidence=0.0,
            gesture_duration=0.0,
            timestamp=time.time()
        )
    
    def _init_gesture_recognizer(self):
        """Initialize gesture recognition model"""
        # In a real system, this would be a trained gesture classifier
        # For now, we'll create a simple rule-based recognizer
        class MockGestureRecognizer:
            def predict(self, landmarks):
                # Simple gesture recognition based on landmark positions
                if len(landmarks) > 0:
                    # Get key landmarks
                    thumb_tip = landmarks[4]  # Thumb tip
                    index_tip = landmarks[8]  # Index finger tip
                    middle_tip = landmarks[12]  # Middle finger tip
                    
                    # Simple gesture rules
                    if self._is_fist(landmarks):
                        return {"type": "fist", "confidence": 0.9}
                    elif self._is_pointing(index_tip, landmarks):
                        return {"type": "pointing", "confidence": 0.8}
                    elif self._is_palm(landmarks):
                        return {"type": "palm", "confidence": 0.7}
                    else:
                        return {"type": "unknown", "confidence": 0.3}
            
            def _is_fist(self, landmarks):
                # Check if fingers are curled (close to palm)
                wrist = landmarks[0]
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                
                # If fingertips are close to palm, it's a fist
                return (np.linalg.norm(thumb_tip[:2] - wrist[:2]) < 0.1 and
                        np.linalg.norm(index_tip[:2] - wrist[:2]) < 0.1)
            
            def _is_pointing(self, index_tip, landmarks):
                # Check if index finger is extended and others are curled
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                
                # Index finger extended, others curled
                return (np.linalg.norm(index_tip[:2] - landmarks[0][:2]) > 0.1 and
                        np.linalg.norm(middle_tip[:2] - landmarks[0][:2]) < 0.1)
            
            def _is_palm(self, landmarks):
                # Check if all fingers are extended
                wrist = landmarks[0]
                fingers_extended = []
                
                for i in [8, 12, 16, 20]:  # Tips of fingers
                    if i < len(landmarks):
                        dist = np.linalg.norm(landmarks[i][:2] - wrist[:2])
                        fingers_extended.append(dist > 0.15)
                
                return all(fingers_extended)
        
        return MockGestureRecognizer()
    
    def _recognize_gesture(self, landmarks_list: List[np.ndarray]) -> Dict[str, Any]:
        """Recognize gesture from hand landmarks"""
        for landmarks in landmarks_list:
            if len(landmarks) == 21:  # Standard MediaPipe hand landmarks
                return self.gesture_recognizer.predict(landmarks)
        
        return {"type": "unknown", "confidence": 0.0}

class VisionProcessingComponent(MultiModalComponent):
    """Vision processing component for multi-modal interaction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("vision_processing", config)
        
        # Object detection and tracking
        self.enable_object_detection = config.get('enable_object_detection', True)
        if self.enable_object_detection:
            try:
                self.object_detector = torch.hub.load(
                    'ultralytics/yolov5', 'yolov5s', pretrained=True
                )
            except Exception as e:
                self.logger.warning(f"Failed to load YOLOv5: {e}")
                self.object_detector = None
        
        # Face detection for social interaction
        self.enable_face_detection = config.get('enable_face_detection', True)
        if self.enable_face_detection:
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except ImportError:
                self.logger.warning("MediaPipe face detection not available")
                self.face_detection = None
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Process image to extract vision information"""
        result = {
            'image': image,
            'detected_objects': [],
            'face_detected': False,
            'gaze_direction': None,
            'timestamp': time.time()
        }
        
        # Process with object detector
        if self.object_detector:
            try:
                detection_results = self.object_detector(image)
                result['detected_objects'] = self._parse_detections(detection_results)
            except Exception as e:
                self.logger.error(f"Object detection error: {e}")
        
        # Process with face detector
        if self.face_detection:
            try:
                import cv2
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.face_detection.process(rgb_image)
                
                if faces.detections:
                    result['face_detected'] = True
                    # Calculate gaze direction approximation
                    result['gaze_direction'] = self._calculate_gaze_direction(faces.detections, image.shape)
            except Exception as e:
                self.logger.error(f"Face detection error: {e}")
        
        return result
    
    def _parse_detections(self, detection_results) -> List[Dict[str, Any]]:
        """Parse object detection results"""
        detections = detection_results.xyxy[0].tolist()  # [x1, y1, x2, y2, conf, class]
        
        objects = []
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            objects.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': int(cls),
                'class_name': self.object_detector.names[int(cls)]
            })
        
        return objects
    
    def _calculate_gaze_direction(self, face_detections, image_shape) -> Optional[str]:
        """Calculate approximate gaze direction from face landmarks"""
        # This is a simplified implementation
        # In a real system, this would use detailed face landmarks
        h, w = image_shape[:2]
        
        for detection in face_detections:
            bbox = detection.location_data.relative_bounding_box
            # Approximate center of face
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            
            # Determine rough gaze direction based on position in image
            if center_x < w * 0.3:
                return 'left'
            elif center_x > w * 0.7:
                return 'right'
            elif center_y < h * 0.3:
                return 'up'
            elif center_y > h * 0.7:
                return 'down'
            else:
                return 'center'
        
        return None
```

### Multi-Modal Fusion and Integration

```python
# multimodal_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import Dict, List, Any, Optional

class MultiModalFusion(nn.Module):
    """Fusion module for combining voice, gesture, and vision features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Common feature dimension
        self.common_dim = config.get('common_dim', 512)
        
        # Modality-specific projectors
        self.voice_projector = nn.Sequential(
            nn.Linear(768, self.common_dim),  # Assuming voice features are projected to 768D
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim)
        )
        
        self.gesture_projector = nn.Sequential(
            nn.Linear(21 * 3, self.common_dim),  # 21 hand landmarks * 3 coordinates
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim)
        )
        
        self.vision_projector = nn.Sequential(
            nn.Linear(512, self.common_dim),  # CLIP vision features
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim)
        )
        
        # Cross-modal attention layers
        self.voice_vision_attention = self._create_cross_attention()
        self.voice_gesture_attention = self._create_cross_attention()
        self.vision_gesture_attention = self._create_cross_attention()
        
        # Multi-modal transformer
        self.multimodal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.common_dim,
                nhead=8,
                dim_feedforward=self.common_dim * 2,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Intent classification head
        self.intent_classifier = nn.Linear(self.common_dim, 50)  # 50 possible intents
        
        # Confidence estimation head
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.common_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _create_cross_attention(self):
        """Create cross-attention module"""
        return nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, voice_features: torch.Tensor,
                gesture_features: torch.Tensor,
                vision_features: torch.Tensor) -> MultiModalOutput:
        """Fuse features from all modalities"""
        batch_size = voice_features.size(0)
        
        # Project features to common space
        voice_proj = self.voice_projector(voice_features)
        gesture_proj = self.gesture_projector(gesture_features)
        vision_proj = self.vision_projector(vision_features)
        
        # Apply cross-modal attention
        voice_with_vision, _ = self.voice_vision_attention(
            voice_proj.unsqueeze(1), 
            vision_proj.unsqueeze(1), 
            vision_proj.unsqueeze(1)
        )
        voice_with_vision = voice_with_vision.squeeze(1)
        
        gesture_with_vision, _ = self.vision_gesture_attention(
            gesture_proj.unsqueeze(1), 
            vision_proj.unsqueeze(1), 
            vision_proj.unsqueeze(1)
        )
        gesture_with_vision = gesture_with_vision.squeeze(1)
        
        # Combine all features
        combined_features = torch.stack([
            voice_with_vision,
            gesture_with_vision,
            vision_proj
        ], dim=1)  # Shape: [batch, 3, common_dim]
        
        # Apply multi-modal transformer
        fused_features = self.multimodal_transformer(combined_features)
        
        # Use [CLS]-like token (mean across modalities) for classification
        multimodal_repr = fused_features.mean(dim=1)
        
        # Classify intent
        intent_logits = self.intent_classifier(multimodal_repr)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        _, predicted_intent_idx = torch.max(intent_probs, dim=-1)
        
        # Estimate confidence
        confidence = self.confidence_estimator(multimodal_repr).squeeze(-1)
        
        # Calculate attention weights for interpretability
        attention_weights = {
            'voice': (fused_features[:, 0, :] * multimodal_repr).sum(dim=-1) / multimodal_repr.pow(2).sum(dim=-1),
            'gesture': (fused_features[:, 1, :] * multimodal_repr).sum(dim=-1) / multimodal_repr.pow(2).sum(dim=-1),
            'vision': (fused_features[:, 2, :] * multimodal_repr).sum(dim=-1) / multimodal_repr.pow(2).sum(dim=-1)
        }
        
        return MultiModalOutput(
            integrated_features=multimodal_repr,
            interpreted_intent=self._idx_to_intent(predicted_intent_idx[0].item()),
            confidence=confidence[0].item(),
            attention_weights={k: v[0].item() for k, v in attention_weights.items()},
            timestamp=time.time()
        )
    
    def _idx_to_intent(self, idx: int) -> str:
        """Convert intent index to intent name"""
        intents = [
            'attention_requested', 'follow_me', 'pick_up_object', 
            'place_object', 'go_to_location', 'come_here', 'stop',
            'wave', 'point', 'greet', 'look_at', 'track_object',
            'answer_question', 'repeat_action', 'confirm_action',
            'cancel_action', 'increase_speed', 'decrease_speed',
            'turn_left', 'turn_right', 'move_forward', 'move_backward'
        ]
        return intents[idx % len(intents)] if idx < len(intents) else 'unknown'

class MultiModalInteractionManager:
    """Manager for coordinating multi-modal interaction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize processing components
        self.voice_processor = VoiceProcessingComponent(self.config.get('voice', {}))
        self.gesture_processor = GestureProcessingComponent(self.config.get('gesture', {}))
        self.vision_processor = VisionProcessingComponent(self.config.get('vision', {}))
        
        # Initialize fusion module
        self.fusion_module = MultiModalFusion(self.config.get('fusion', {}))
        
        # Temporal synchronization
        self.temporal_window = self.config.get('temporal_window', 1.0)  # seconds
        self.synchronization_buffer = {
            'voice': [],
            'gesture': [],
            'vision': []
        }
        
        # Context tracking
        self.context_memory = []
        self.max_context_length = 10
        
        # Logging
        self.logger = logging.getLogger("Multimodal.InteractionManager")
    
    def process_multimodal_input(self, voice_data: Optional[np.ndarray] = None,
                                gesture_image: Optional[np.ndarray] = None,
                                vision_image: Optional[np.ndarray] = None,
                                sample_rate: int = 16000) -> Optional[MultiModalOutput]:
        """Process inputs from multiple modalities"""
        current_time = time.time()
        
        # Process each modality
        voice_result = None
        if voice_data is not None:
            voice_result = self.voice_processor.process(voice_data, sample_rate)
            self.synchronization_buffer['voice'].append((current_time, voice_result))
        
        gesture_result = None
        if gesture_image is not None:
            gesture_result = self.gesture_processor.process(gesture_image)
            self.synchronization_buffer['gesture'].append((current_time, gesture_result))
        
        vision_result = None
        if vision_image is not None:
            vision_result = self.vision_processor.process(vision_image)
            self.synchronization_buffer['vision'].append((current_time, vision_result))
        
        # Attempt fusion if we have synchronized data
        synchronized_data = self._get_synchronized_data(current_time)
        
        if synchronized_data:
            # Extract features for fusion
            voice_features = self._extract_voice_features(synchronized_data['voice'])
            gesture_features = self._extract_gesture_features(synchronized_data['gesture'])
            vision_features = self._extract_vision_features(synchronized_data['vision'])
            
            # Perform fusion
            output = self.fusion_module(
                voice_features=voice_features,
                gesture_features=gesture_features,
                vision_features=vision_features
            )
            
            # Update context
            self._update_context(output)
            
            return output
        
        return None
    
    def _get_synchronized_data(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Get synchronized data from all modalities"""
        # Clean old data from buffer
        window_start = current_time - self.temporal_window
        
        for modality in self.synchronization_buffer:
            self.synchronization_buffer[modality] = [
                (t, data) for t, data in self.synchronization_buffer[modality]
                if t >= window_start
            ]
        
        # Check if we have data from all modalities within the time window
        available_modalities = []
        for modality, buffer_list in self.synchronization_buffer.items():
            if buffer_list:  # If buffer has data
                available_modalities.append(modality)
        
        # For now, return the most recent data from each modality
        # In a real implementation, this would implement more sophisticated synchronization
        synchronized_data = {}
        for modality in available_modalities:
            if self.synchronization_buffer[modality]:
                # Use the most recent data
                _, data = self.synchronization_buffer[modality][-1]
                synchronized_data[modality] = data
        
        # Check if we have all required modalities
        required_modalities = ['voice', 'gesture', 'vision']
        if all(mod in synchronized_data for mod in required_modalities):
            return synchronized_data
        else:
            # If we don't have all modalities, return best effort
            return synchronized_data or None
    
    def _extract_voice_features(self, voice_data: VoiceInput) -> torch.Tensor:
        """Extract features from voice input"""
        # In a real system, this would use more sophisticated feature extraction
        # For now, we'll create a simple embedding based on the transcript
        
        # This is a placeholder - in a real system, use proper STT embeddings
        if voice_data.text_transcript:
            # Create a simple embedding (in practice, use BERT, etc.)
            import hashlib
            text_hash = int(hashlib.md5(voice_data.text_transcript.encode()).hexdigest(), 16)
            embedding = torch.rand(1, 768)  # Random vector for demonstration
        else:
            embedding = torch.zeros(1, 768)
        
        return embedding
    
    def _extract_gesture_features(self, gesture_data: GestureInput) -> torch.Tensor:
        """Extract features from gesture input"""
        # Flatten hand landmarks for now
        landmarks = torch.from_numpy(gesture_data.hand_landmarks).float()
        landmarks_flat = landmarks.view(-1).unsqueeze(0)  # Shape: [1, 63]
        
        # Make sure we have the right size
        if landmarks_flat.size(1) < 21 * 3:
            # Pad with zeros
            pad_size = 21 * 3 - landmarks_flat.size(1)
            landmarks_flat = torch.cat([landmarks_flat, torch.zeros(1, pad_size)], dim=1)
        elif landmarks_flat.size(1) > 21 * 3:
            # Truncate if too long
            landmarks_flat = landmarks_flat[:, :(21 * 3)]
        
        return landmarks_flat
    
    def _extract_vision_features(self, vision_data: Dict[str, Any]) -> torch.Tensor:
        """Extract features from vision input"""
        # This is a placeholder - in a real system, use proper vision embeddings
        # For demonstration, return random features
        return torch.rand(1, 512)
    
    def _update_context(self, output: MultiModalOutput):
        """Update context with the latest interaction"""
        self.context_memory.append({
            'intent': output.interpreted_intent,
            'confidence': output.confidence,
            'timestamp': output.timestamp,
            'attention_weights': output.attention_weights
        })
        
        # Keep context within limits
        if len(self.context_memory) > self.max_context_length:
            self.context_memory = self.context_memory[-self.max_context_length:]
    
    def get_interaction_context(self) -> List[Dict[str, Any]]:
        """Get recent interaction context"""
        return self.context_memory.copy()

class MultiModalInteractionSystem:
    """Complete multi-modal interaction system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize interaction manager
        self.interaction_manager = MultiModalInteractionManager(self.config)
        
        # Initialize any additional components
        self.intent_handlers = self._initialize_intent_handlers()
        
        # Performance metrics
        self.performance_stats = {
            'total_interactions': 0,
            'average_fusion_time': 0.0,
            'modality_availability': {
                'voice': 0.0,
                'gesture': 0.0,
                'vision': 0.0
            }
        }
        
        # Logging
        self.logger = logging.getLogger("Multimodal.System")
    
    def _initialize_intent_handlers(self) -> Dict[str, callable]:
        """Initialize intent-specific handlers"""
        def default_handler(intent_data: Dict[str, Any]) -> str:
            return f"Processing intent: {intent_data.get('intent', 'unknown')}"
        
        # Define specific handlers for known intents
        handlers = {
            'come_here': lambda d: "Moving towards user",
            'follow_me': lambda d: "Following user movement",
            'pick_up_object': lambda d: "Attempting to pick up object",
            'go_to_location': lambda d: "Navigating to location",
            'greet': lambda d: "Greeting user",
            'stop': lambda d: "Stopping current action"
        }
        
        # Set default handler for unknown intents
        for intent in ['attention_requested', 'wave', 'point', 'look_at']:
            handlers[intent] = default_handler
        
        return handlers
    
    def process_interaction(self, voice_audio: Optional[np.ndarray] = None,
                           gesture_image: Optional[np.ndarray] = None,
                           vision_image: Optional[np.ndarray] = None,
                           sample_rate: int = 16000) -> Optional[Dict[str, Any]]:
        """Process a complete multi-modal interaction"""
        start_time = time.time()
        
        try:
            # Process multi-modal input
            fusion_result = self.interaction_manager.process_multimodal_input(
                voice_audio, gesture_image, vision_image, sample_rate
            )
            
            if fusion_result:
                # Update performance stats
                fusion_time = time.time() - start_time
                total_interactions = self.performance_stats['total_interactions']
                self.performance_stats['average_fusion_time'] = (
                    (self.performance_stats['average_fusion_time'] * total_interactions + fusion_time) /
                    (total_interactions + 1)
                )
                self.performance_stats['total_interactions'] += 1
                
                # Determine handler for the interpreted intent
                intent = fusion_result.interpreted_intent
                handler = self.intent_handlers.get(intent, self.intent_handlers['follow_me'])  # default
                
                # Execute handler
                response = handler({
                    'intent': intent,
                    'confidence': fusion_result.confidence,
                    'attention_weights': fusion_result.attention_weights
                })
                
                # Log the interaction
                self.logger.info(f"Interpreted intent: {intent} (confidence: {fusion_result.confidence:.3f})")
                self.logger.info(f"Response: {response}")
                
                return {
                    'success': True,
                    'intent': intent,
                    'confidence': fusion_result.confidence,
                    'response': response,
                    'processing_time': fusion_time,
                    'attention_weights': fusion_result.attention_weights
                }
            
            else:
                # No synchronized data available
                return {
                    'success': False,
                    'error': 'Insufficient synchronized multi-modal data',
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            self.logger.error(f"Error processing interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and performance metrics"""
        return {
            'performance_stats': self.performance_stats,
            'context_length': len(self.interaction_manager.get_interaction_context()),
            'available_intents': list(self.intent_handlers.keys()),
            'current_context': self.interaction_manager.get_interaction_context()[-3:]  # Last 3 interactions
        }

# Import cv2 for image processing if we have gesture processing
try:
    import cv2
except ImportError:
    cv2 = None
    print("OpenCV not available - some vision functions will be limited")
```

### Advanced Multi-Modal Interaction Patterns

```python
# advanced_interaction_patterns.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import asyncio
import threading

class CoordinatedMultiModalInteraction:
    """Advanced multi-modal interaction with coordinated behaviors"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Attention mechanisms for each modality
        self.voice_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.gesture_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.vision_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Coordination controller
        self.coordination_network = nn.Sequential(
            nn.Linear(512 * 3, 1024),  # Combined features
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)  # Coordination states
        )
        
        # Behavior generators
        self.behavior_generators = {
            'attention': self._generate_attention_behavior,
            'navigation': self._generate_navigation_behavior,
            'manipulation': self._generate_manipulation_behavior,
            'communication': self._generate_communication_behavior
        }
        
        # Timing coordination
        self.timing_controller = MultiModalTimingController()
        
        self.logger = logging.getLogger("Multimodal.CoordinatedInteraction")
    
    def coordinate_interaction(self, voice_features: torch.Tensor,
                              gesture_features: torch.Tensor,
                              vision_features: torch.Tensor) -> Dict[str, Any]:
        """Coordinate multi-modal interaction with timing and attention"""
        batch_size = voice_features.size(0)
        
        # Apply attention mechanisms
        voice_attended, _ = self.voice_attention(
            voice_features.unsqueeze(1),
            voice_features.unsqueeze(1),
            voice_features.unsqueeze(1)
        )
        voice_attended = voice_attended.squeeze(1)
        
        gesture_attended, _ = self.gesture_attention(
            gesture_features.unsqueeze(1),
            gesture_features.unsqueeze(1),
            gesture_features.unsqueeze(1)
        )
        gesture_attended = gesture_attended.squeeze(1)
        
        vision_attended, _ = self.vision_attention(
            vision_features.unsqueeze(1),
            vision_features.unsqueeze(1),
            vision_features.unsqueeze(1)
        )
        vision_attended = vision_attended.squeeze(1)
        
        # Combine attended features
        combined_features = torch.cat([
            voice_attended, gesture_attended, vision_attended
        ], dim=-1)
        
        # Generate coordination state
        coordination_state = self.coordination_network(combined_features)
        
        # Determine interaction type based on coordination
        interaction_type = self._classify_interaction_type(coordination_state)
        
        # Generate coordinated behavior
        behavior = self.behavior_generators[interaction_type]({
            'voice_features': voice_attended,
            'gesture_features': gesture_attended,
            'vision_features': vision_attended,
            'coordination_state': coordination_state
        })
        
        return {
            'interaction_type': interaction_type,
            'behavior': behavior,
            'coordination_state': coordination_state,
            'attention_weights': {
                'voice': torch.mean(voice_attended).item(),
                'gesture': torch.mean(gesture_attended).item(),
                'vision': torch.mean(vision_attended).item()
            }
        }
    
    def _classify_interaction_type(self, coordination_state: torch.Tensor) -> str:
        """Classify the type of interaction based on coordination state"""
        # Simple classification - in reality, this would be more sophisticated
        interaction_scores = torch.softmax(coordination_state, dim=-1)
        top_class = torch.argmax(interaction_scores, dim=-1).item()
        
        interaction_types = [
            'attention', 'navigation', 'manipulation', 'communication',
            'collaboration', 'instruction_following', 'social_interaction'
        ]
        
        return interaction_types[top_class % len(interaction_types)]
    
    def _generate_attention_behavior(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate attention-seeking behavior"""
        return {
            'type': 'attention',
            'actions': [
                {'type': 'turn_head', 'direction': 'towards_speaker', 'priority': 1},
                {'type': 'orient_body', 'direction': 'towards_speaker', 'priority': 2},
                {'type': 'make_eye_contact', 'duration': 0.5, 'priority': 3}
            ],
            'timing': self.timing_controller.schedule_attention_sequence(),
            'response': 'Attending to speaker'
        }
    
    def _generate_navigation_behavior(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate navigation behavior"""
        return {
            'type': 'navigation',
            'actions': [
                {'type': 'navigate_to', 'target': 'pointed_location', 'priority': 1},
                {'type': 'avoid_obstacles', 'priority': 2},
                {'type': 'reach_destination', 'priority': 3}
            ],
            'timing': self.timing_controller.schedule_navigation_sequence(),
            'response': 'Navigating to requested location'
        }
    
    def _generate_manipulation_behavior(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate manipulation behavior"""
        return {
            'type': 'manipulation',
            'actions': [
                {'type': 'approach_object', 'target': 'gestured_object', 'priority': 1},
                {'type': 'grasp_object', 'method': 'precision_grasp', 'priority': 2},
                {'type': 'lift_object', 'height': 0.2, 'priority': 3}
            ],
            'timing': self.timing_controller.schedule_manipulation_sequence(),
            'response': 'Manipulating requested object'
        }
    
    def _generate_communication_behavior(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate communication behavior"""
        return {
            'type': 'communication',
            'actions': [
                {'type': 'speak', 'message': 'Acknowledged', 'priority': 1},
                {'type': 'gesture_acknowledgement', 'type': 'nod', 'priority': 2},
                {'type': 'await_response', 'timeout': 5.0, 'priority': 3}
            ],
            'timing': self.timing_controller.schedule_communication_sequence(),
            'response': 'Communicating acknowledgment'
        }

class MultiModalTimingController:
    """Controller for coordinating timing across modalities"""
    
    def __init__(self):
        self.default_timings = {
            'attention_sequence': [0.0, 0.2, 0.5],  # Head turn, body orientation, eye contact
            'navigation_sequence': [0.0, 1.0, 2.0],  # Start, avoid, arrive
            'manipulation_sequence': [0.0, 0.5, 1.0],  # Approach, grasp, lift
            'communication_sequence': [0.0, 0.3, 1.0]  # Speak, gesture, wait
        }
    
    def schedule_attention_sequence(self) -> List[float]:
        """Schedule attention behavior timing"""
        return self.default_timings['attention_sequence']
    
    def schedule_navigation_sequence(self) -> List[float]:
        """Schedule navigation behavior timing"""
        return self.default_timings['navigation_sequence']
    
    def schedule_manipulation_sequence(self) -> List[float]:
        """Schedule manipulation behavior timing"""
        return self.default_timings['manipulation_sequence']
    
    def schedule_communication_sequence(self) -> List[float]:
        """Schedule communication behavior timing"""
        return self.default_timings['communication_sequence']
    
    def synchronize_modalities(self, modality_times: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Synchronize timing across modalities"""
        # Find the longest sequence to align others with
        max_len = max(len(times) for times in modality_times.values())
        
        # Pad shorter sequences with incremental times
        synchronized = {}
        for modality, times in modality_times.items():
            if len(times) < max_len:
                # Extend with incremental times based on last interval
                if len(times) >= 2:
                    interval = times[-1] - times[-2]
                else:
                    interval = 0.5  # Default interval
                
                extended_times = times.copy()
                for i in range(len(times), max_len):
                    extended_times.append(extended_times[-1] + interval)
                
                synchronized[modality] = extended_times
            else:
                synchronized[modality] = times
        
        return synchronized

class SocialMultiModalInteraction:
    """Social aspects of multi-modal interaction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Personality traits
        self.personality_traits = {
            'expressiveness': config.get('expressiveness', 0.7),
            'responsiveness': config.get('responsiveness', 0.9),
            'friendliness': config.get('friendliness', 0.8)
        }
        
        # Social rules
        self.social_rules = {
            'personal_space': config.get('personal_space', 1.0),  # meters
            'eye_contact_duration': config.get('eye_contact_duration', 0.5),  # seconds
            'response_latency': config.get('response_latency', 0.3)  # seconds
        }
        
        # Cultural adaptability
        self.cultural_adaptation = CulturalAdaptationModule()
        
        self.logger = logging.getLogger("Multimodal.SocialInteraction")
    
    def generate_social_response(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate socially appropriate response to multi-modal input"""
        # Determine interaction context
        intent = interaction_data.get('interpreted_intent', 'unknown')
        confidence = interaction_data.get('confidence', 0.0)
        attention_weights = interaction_data.get('attention_weights', {})
        
        # Apply personality modifiers
        expressiveness_factor = self.personality_traits['expressiveness']
        friendliness_factor = self.personality_traits['friendliness']
        
        # Generate social behavior
        social_behavior = self._select_social_behavior(intent, confidence, attention_weights)
        
        # Apply cultural adaptation
        culturally_adapted_behavior = self.cultural_adaptation.adapt_behavior(
            social_behavior, 
            interaction_data.get('user_context', {})
        )
        
        # Apply social rules
        socially_safe_behavior = self._apply_social_rules(culturally_adapted_behavior)
        
        return {
            'social_behavior': socially_safe_behavior,
            'personality_modifiers': {
                'expressiveness': expressiveness_factor,
                'friendliness': friendliness_factor
            },
            'cultural_adaptation_applied': True
        }
    
    def _select_social_behavior(self, intent: str, confidence: float, 
                                attention_weights: Dict[str, float]) -> Dict[str, Any]:
        """Select appropriate social behavior based on intent and context"""
        base_behaviors = {
            'greet': {
                'verbal_response': 'Hello! Nice to meet you.',
                'gesture_response': 'wave_with_smile',
                'facial_expression': 'smile',
                'proximity': 'social_distance',
                'eye_contact': True
            },
            'come_here': {
                'verbal_response': 'Coming to you now!',
                'gesture_response': 'acknowledge_and_approach',
                'facial_expression': 'attentive',
                'proximity': 'personal_space',
                'eye_contact': True
            },
            'point': {
                'verbal_response': 'I see that you\'re pointing at something.',
                'gesture_response': 'look_and_acknowledge',
                'facial_expression': 'curious',
                'proximity': 'social_distance',
                'eye_contact': True
            },
            'follow_me': {
                'verbal_response': 'I\'ll follow you, please go ahead.',
                'gesture_response': 'acknowledge_and_wait',
                'facial_expression': 'attentive',
                'proximity': 'following_distance',
                'eye_contact': intermittent
            }
        }
        
        # Default behavior for unknown intents
        if intent not in base_behaviors:
            return {
                'verbal_response': 'I\'m processing your request.',
                'gesture_response': 'neutral_posture',
                'facial_expression': 'thoughtful',
                'proximity': 'social_distance',
                'eye_contact': True
            }
        
        return base_behaviors[intent]
    
    def _apply_social_rules(self, behavior: Dict[str, Any]) -> Dict[str, Any]:
        """Apply social rules to ensure appropriate behavior"""
        # Adjust proximity based on social rules
        if behavior['proximity'] == 'personal_space':
            safe_distance = max(self.social_rules['personal_space'], 0.5)
            behavior['safe_distance'] = safe_distance
        
        # Ensure eye contact is appropriate for culture
        if behavior.get('eye_contact', False):
            behavior['max_eye_contact_duration'] = self.social_rules['eye_contact_duration']
        
        # Apply response latency
        behavior['response_delay'] = self.social_rules['response_latency']
        
        return behavior

class CulturalAdaptationModule:
    """Module for adapting behavior to different cultures"""
    
    def __init__(self):
        # Cultural norms database
        self.cultural_norms = {
            'USA': {
                'personal_space': 0.8,
                'eye_contact_duration': 0.5,
                'physical_touch': 'low',
                'formality_level': 'medium',
                'greeting_style': 'handshake'
            },
            'Japan': {
                'personal_space': 1.2,
                'eye_contact_duration': 0.3,
                'physical_touch': 'very_low',
                'formality_level': 'high',
                'greeting_style': 'bow'
            },
            'Italy': {
                'personal_space': 0.5,
                'eye_contact_duration': 0.6,
                'physical_touch': 'medium_high',
                'formality_level': 'medium',
                'greeting_style': 'handshake_or_kiss'
            }
        }
    
    def adapt_behavior(self, behavior: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt behavior based on user's cultural background"""
        culture = user_context.get('culture', 'USA')  # Default to USA
        
        if culture not in self.cultural_norms:
            return behavior  # No adaptation needed for unknown cultures
        
        norms = self.cultural_norms[culture]
        adapted_behavior = behavior.copy()
        
        # Adapt proximity
        adapted_behavior['safe_distance'] = norms['personal_space']
        
        # Adapt eye contact
        adapted_behavior['max_eye_contact_duration'] = norms['eye_contact_duration']
        
        # Adapt greeting style if applicable
        if 'greeting' in behavior.get('verbal_response', '').lower():
            adapted_behavior['greeting_style'] = norms['greeting_style']
        
        # Adapt formality
        if norms['formality_level'] == 'high':
            # Make responses more formal
            if 'verbal_response' in adapted_behavior:
                response = adapted_behavior['verbal_response']
                if 'Hi' in response or 'Hey' in response:
                    adapted_behavior['verbal_response'] = response.replace('Hi', 'Greetings').replace('Hey', 'Greetings')
        
        return adapted_behavior
```

### Real-World Multi-Modal Integration

```python
# real_world_integration.py
import asyncio
from typing import Dict, Any, Optional
import threading
import queue

class RealWorldMultiModalSystem:
    """Integration of multi-modal system with real-world components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize multi-modal system
        self.multimodal_system = MultiModalInteractionSystem(self.config)
        self.coordinated_system = CoordinatedMultiModalInteraction(self.config)
        self.social_system = SocialMultiModalInteraction(self.config)
        
        # Real-world interfaces
        self.voice_interface = self._init_voice_interface()
        self.camera_interface = self._init_camera_interface()
        self.robot_interface = self._init_robot_interface()
        
        # Processing queues
        self.voice_queue = queue.Queue(maxsize=10)
        self.gesture_queue = queue.Queue(maxsize=10)
        self.vision_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Processing threads
        self.processing_thread = None
        self.running = False
        
        # Performance monitoring
        self.interaction_count = 0
        self.avg_processing_time = 0.0
        
        self.logger = logging.getLogger("Multimodal.RealWorldSystem")
    
    def _init_voice_interface(self):
        """Initialize voice input interface"""
        # In a real system, this would connect to microphone
        class MockVoiceInterface:
            def start_listening(self):
                pass
            
            def stop_listening(self):
                pass
        
        return MockVoiceInterface()
    
    def _init_camera_interface(self):
        """Initialize camera interface"""
        # In a real system, this would connect to cameras
        class MockCameraInterface:
            def start_capture(self):
                pass
            
            def stop_capture(self):
                pass
        
        return MockCameraInterface()
    
    def _init_robot_interface(self):
        """Initialize robot control interface"""
        # In a real system, this would connect to robot hardware
        class MockRobotInterface:
            def execute_behavior(self, behavior):
                print(f"Executing behavior: {behavior}")
                return True
        
        return MockRobotInterface()
    
    def start_system(self):
        """Start the real-world multi-modal system"""
        self.running = True
        
        # Start voice and camera interfaces
        self.voice_interface.start_listening()
        self.camera_interface.start_capture()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Real-world multi-modal system started")
    
    def stop_system(self):
        """Stop the real-world multi-modal system"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Stop interfaces
        self.voice_interface.stop_listening()
        self.camera_interface.stop_capture()
        
        self.logger.info("Real-world multi-modal system stopped")
    
    def submit_voice_input(self, audio_data: np.ndarray):
        """Submit audio input to the system"""
        try:
            self.voice_queue.put(audio_data, block=False)
        except queue.Full:
            self.logger.warning("Voice queue is full, dropping input")
    
    def submit_vision_input(self, image: np.ndarray):
        """Submit vision input to the system"""
        try:
            self.vision_queue.put(image, block=False)
        except queue.Full:
            self.logger.warning("Vision queue is full, dropping input")
    
    def submit_gesture_input(self, image: np.ndarray):
        """Submit gesture input to the system"""
        try:
            self.gesture_queue.put(image, block=False)
        except queue.Full:
            self.logger.warning("Gesture queue is full, dropping input")
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the next result from the system"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop for real-world system"""
        while self.running:
            try:
                # Collect inputs from all modalities
                voice_inputs = self._collect_voice_inputs()
                gesture_inputs = self._collect_gesture_inputs()
                vision_inputs = self._collect_vision_inputs()
                
                if voice_inputs or gesture_inputs or vision_inputs:
                    # Process multi-modal interaction
                    start_time = time.time()
                    
                    # Take the most recent of each input type
                    voice_data = voice_inputs[-1] if voice_inputs else None
                    gesture_data = gesture_inputs[-1] if gesture_inputs else None
                    vision_data = vision_inputs[-1] if vision_inputs else None
                    
                    # Process through multi-modal system
                    result = self.multimodal_system.process_interaction(
                        voice_audio=voice_data,
                        gesture_image=gesture_data,
                        vision_image=vision_data
                    )
                    
                    if result and result['success']:
                        # Apply coordinated behavior
                        if gesture_data is not None and vision_data is not None:
                            coordination_result = self.coordinated_system.coordinate_interaction(
                                voice_features=torch.rand(1, 512),  # Mock features
                                gesture_features=torch.rand(1, 512),
                                vision_features=torch.rand(1, 512)
                            )
                            
                            result['coordination'] = coordination_result
                        
                        # Apply social behavior
                        social_result = self.social_system.generate_social_response(result)
                        result['social_adaptation'] = social_result
                        
                        # Execute behavior on robot
                        if 'coordination' in result:
                            behavior = result['coordination']['behavior']
                            self.robot_interface.execute_behavior(behavior)
                
                    # Update performance metrics
                    processing_time = time.time() - start_time
                    self.avg_processing_time = (
                        (self.avg_processing_time * self.interaction_count + processing_time) /
                        (self.interaction_count + 1)
                    )
                    self.interaction_count += 1
                    
                    # Put result in queue
                    if result:
                        try:
                            self.result_queue.put(result, block=False)
                        except queue.Full:
                            self.logger.warning("Result queue is full")
                
                # Small delay to prevent busy waiting
                time.sleep(0.05)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _collect_voice_inputs(self) -> List[np.ndarray]:
        """Collect available voice inputs from queue"""
        inputs = []
        while not self.voice_queue.empty():
            try:
                inputs.append(self.voice_queue.get_nowait())
            except queue.Empty:
                break
        return inputs
    
    def _collect_gesture_inputs(self) -> List[np.ndarray]:
        """Collect available gesture inputs from queue"""
        inputs = []
        while not self.gesture_queue.empty():
            try:
                inputs.append(self.gesture_queue.get_nowait())
            except queue.Empty:
                break
        return inputs
    
    def _collect_vision_inputs(self) -> List[np.ndarray]:
        """Collect available vision inputs from queue"""
        inputs = []
        while not self.vision_queue.empty():
            try:
                inputs.append(self.vision_queue.get_nowait())
            except queue.Empty:
                break
        return inputs
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the real-world system"""
        return {
            'processing_threads': 1,
            'voice_queue_size': self.voice_queue.qsize(),
            'gesture_queue_size': self.gesture_queue.qsize(),
            'vision_queue_size': self.vision_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'interaction_count': self.interaction_count,
            'avg_processing_time': self.avg_processing_time,
            'multimodal_status': self.multimodal_system.get_system_status()
        }

def run_multimodal_integration_demo():
    """Run a demo of multi-modal integration"""
    print("Multi-Modal Interaction System Demo")
    print("="*50)
    
    # Create multi-modal system
    config = {
        'temporal_window': 1.0,
        'voice': {'use_stt': False},  # Using mock for this demo
        'gesture': {'use_hand_model': False},  # Using mock
        'vision': {'enable_object_detection': False}  # Using mock
    }
    
    print("\n1. Initializing multi-modal system...")
    mm_system = MultiModalInteractionSystem(config)
    print("   ✓ Multi-modal system initialized")
    
    # Create coordinated system
    coord_system = CoordinatedMultiModalInteraction()
    print("   ✓ Coordinated interaction system ready")
    
    # Create social system
    social_system = SocialMultiModalInteraction()
    print("   ✓ Social interaction system ready")
    
    # Simulate multi-modal input
    print("\n2. Simulating multi-modal input...")
    
    # Create dummy inputs
    dummy_voice = np.random.random(16000).astype(np.float32)  # 1 sec of audio
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # 480p RGB image
    
    # Process interaction
    result = mm_system.process_interaction(
        voice_audio=dummy_voice,
        gesture_image=dummy_image,
        vision_image=dummy_image,
        sample_rate=16000
    )
    
    if result and result['success']:
        print(f"   ✓ Interaction processed successfully")
        print(f"   Intent: {result['intent']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Response: {result['response']}")
        
        # Apply coordination
        coord_result = coord_system.coordinate_interaction(
            voice_features=torch.rand(1, 512),
            gesture_features=torch.rand(1, 512),
            vision_features=torch.rand(1, 512)
        )
        
        print(f"   Coordination type: {coord_result['interaction_type']}")
        
        # Apply social adaptation
        social_input = {
            'interpreted_intent': result['intent'],
            'confidence': result['confidence'],
            'attention_weights': result['attention_weights']
        }
        social_result = social_system.generate_social_response(social_input)
        
        print(f"   Social adaptation applied: {'✓' if social_result['cultural_adaptation_applied'] else '✗'}")
    else:
        print("   ✗ Failed to process interaction")
    
    # Show system status
    status = mm_system.get_system_status()
    print(f"\n3. System Status:")
    print(f"   Total interactions: {status['performance_stats']['total_interactions']}")
    print(f"   Available intents: {len(status['available_intents'])}")
    print(f"   Context length: {status['context_length']}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_multimodal_integration_demo()
```

## Summary

In this chapter, we've implemented a comprehensive multi-modal interaction system for humanoid robots:

1. **Core Components**: We created specialized components for processing voice, gesture, and vision inputs with appropriate feature extraction and preprocessing.

2. **Fusion Architecture**: We developed a sophisticated fusion mechanism that combines features from all modalities using cross-attention and transformer architectures.

3. **Coordination Systems**: We implemented systems for coordinating behaviors across modalities with proper timing and attention mechanisms.

4. **Social Interaction**: We created social and cultural adaptation modules to ensure appropriate responses to human users.

5. **Real-World Integration**: We designed interfaces for connecting the multi-modal system with real hardware components.

The multi-modal interaction system enables humanoid robots to engage in natural, human-like communication by processing voice commands, interpreting gestures, and understanding visual context in an integrated manner. This creates much more intuitive and effective human-robot interaction compared to single-modal approaches.

The architecture is designed to be scalable, maintainable, and adaptable to different cultural contexts and interaction styles, making it suitable for deployment in diverse environments.