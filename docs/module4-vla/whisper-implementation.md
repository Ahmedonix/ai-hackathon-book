---
title: Implementing Whisper Speech-to-Text Interface
description: Practical implementation of Whisper for speech-to-text in robotics
sidebar_position: 3
---

# Implementing Whisper Speech-to-Text Interface

## Overview

This chapter provides a practical implementation guide for creating a robust Whisper-based speech-to-text interface specifically designed for humanoid robotics applications. We'll cover real-world considerations, implementation strategies, and best practices for deploying Whisper in robotic systems.

## Learning Objectives

- Implement a complete Whisper-based speech-to-text pipeline
- Create a configurable Whisper interface for robotics
- Handle real-time audio processing and buffering
- Integrate with robot control systems
- Optimize for performance and reliability in real-world scenarios

## Complete Implementation Framework

Let's create a comprehensive Whisper interface implementation:

### Core Whisper Interface Class

```python
# whisper_interface.py
import whisper
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import json
import os
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    processing_time: float
    timestamp: float

class WhisperInterface:
    """
    Comprehensive Whisper interface for robotics applications
    """
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        sample_rate: int = 16000,
        buffer_duration: float = 1.0,
        silence_threshold: float = 0.01,
        sensitivity: float = 0.5,
        vad_enabled: bool = True
    ):
        """
        Initialize the Whisper interface
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda' or 'cpu')
            sample_rate: Audio sample rate
            buffer_duration: Duration of audio buffer in seconds
            silence_threshold: Threshold for silence detection
            sensitivity: Sensitivity for voice activity detection (0.0 to 1.0)
            vad_enabled: Whether to use voice activity detection
        """
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.silence_threshold = silence_threshold
        self.sensitivity = sensitivity
        self.vad_enabled = vad_enabled
        
        # Audio parameters
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize Whisper model
        self.model = whisper.load_model(self.model_size).to(self.device)
        
        # Initialize audio system
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.command_callbacks: List[Callable[[str], None]] = []
        
        # Voice activity detection parameters
        self.vad_chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks for VAD
        self.vad_buffer = np.array([])
        self.speech_detected = False
        self.silence_start_time = None
        self.min_speech_duration = 0.3  # Minimum speech duration to consider valid
        self.silence_duration_threshold = 0.5  # Duration of silence to stop processing
        
        # Buffer management
        self.active_buffer = np.array([])
        self.min_audio_length = int(0.5 * self.sample_rate)  # Minimum 0.5 seconds
        
        # Performance monitoring
        self.transcription_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Threading
        self.processing_thread = None
        self.command_thread = None
        self.lock = threading.Lock()
        
        print(f"Whisper interface initialized on {self.device} with {self.model_size} model")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio input callback function
        """
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Add to main audio queue
        self.audio_queue.put(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def start_listening(self):
        """
        Start the audio input stream and processing
        """
        if self.is_listening:
            return False
        
        # Open audio stream
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            return False
        
        self.is_listening = True
        self.stream.start_stream()
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_audio_stream, daemon=True)
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        
        self.processing_thread.start()
        self.command_thread.start()
        
        print("Started listening for audio input")
        return True
    
    def stop_listening(self):
        """
        Stop the audio input stream and processing
        """
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("Stopped listening for audio input")
    
    def _process_audio_stream(self):
        """
        Process the continuous audio stream, detect speech, and trigger transcriptions
        """
        while self.is_listening:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)
                
                if self.vad_enabled:
                    self._process_with_vad(chunk)
                else:
                    self._process_without_vad(chunk)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio stream processing: {e}")
                continue
    
    def _process_with_vad(self, chunk):
        """
        Process audio using voice activity detection
        """
        # Add chunk to VAD buffer
        self.vad_buffer = np.concatenate([self.vad_buffer, chunk])
        
        # Process VAD when we have enough data
        if len(self.vad_buffer) >= self.vad_chunk_size:
            vad_chunk = self.vad_buffer[:self.vad_chunk_size]
            self.vad_buffer = self.vad_buffer[self.vad_chunk_size:]
            
            # Check if speech is present in this chunk
            if np.max(np.abs(vad_chunk)) > self.silence_threshold:
                # Speech detected
                self.speech_detected = True
                self.silence_start_time = None  # Reset silence timer
                
                # Add to active buffer for potential transcription
                self.active_buffer = np.concatenate([self.active_buffer, vad_chunk])
            else:
                # No speech in this chunk
                if self.speech_detected:
                    # We were in a speech segment, add to buffer
                    self.active_buffer = np.concatenate([self.active_buffer, vad_chunk])
                    
                    # Check if we have accumulated enough silence to consider speech ended
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
                    elif time.time() - self.silence_start_time >= self.silence_duration_threshold:
                        # Consider speech ended, process the buffer
                        if len(self.active_buffer) >= self.min_audio_length:
                            self._process_buffer_for_transcription()
                        
                        self.active_buffer = np.array([])
                        self.speech_detected = False
                        self.silence_start_time = None
                else:
                    # Still in silence phase, just add to silence counter
                    if self.silence_start_time is None:
                        self.silence_start_time = time.time()
    
    def _process_without_vad(self, chunk):
        """
        Process audio without voice activity detection (always buffer audio)
        """
        # Add chunk to active buffer
        self.active_buffer = np.concatenate([self.active_buffer, chunk])
        
        # If buffer is large enough, process it
        if len(self.active_buffer) >= self.min_audio_length and np.max(np.abs(self.active_buffer)) > self.silence_threshold:
            # Check amplitude to avoid processing silence
            if np.max(np.abs(self.active_buffer[-int(0.5 * self.sample_rate):])) > self.silence_threshold:
                self._process_buffer_for_transcription()
                # Keep some overlap for continuity
                keep_length = int(0.2 * self.sample_rate)  # Keep last 0.2 seconds
                self.active_buffer = self.active_buffer[-keep_length:] if len(self.active_buffer) > keep_length else np.array([])
    
    def _process_buffer_for_transcription(self):
        """
        Process the current audio buffer with Whisper
        """
        if len(self.active_buffer) < self.min_audio_length:
            return
        
        # Start transcription in a separate thread to avoid blocking
        transcribe_thread = threading.Thread(
            target=self._transcribe_and_handle_result,
            args=(self.active_buffer.copy(),),
            daemon=True
        )
        transcribe_thread.start()
    
    def _transcribe_and_handle_result(self, audio_buffer):
        """
        Perform transcription and handle the result
        """
        start_time = time.time()
        
        try:
            # Transcribe audio
            result = self.model.transcribe(audio_buffer, fp16=torch.cuda.is_available())
            
            end_time = time.time()
            
            transcription_result = TranscriptionResult(
                text=result['text'],
                language=result['language'],
                confidence=self._estimate_confidence(result),
                processing_time=end_time - start_time,
                timestamp=time.time()
            )
            
            # Add to result queue
            self.result_queue.put(transcription_result)
            
            # Update statistics
            with self.lock:
                self.transcription_count += 1
                self.total_processing_time += transcription_result.processing_time
            
            print(f"Transcribed: '{transcription_result.text}' (confidence: {transcription_result.confidence:.2f}, time: {transcription_result.processing_time:.2f}s)")
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            print(f"Transcription error: {e}")
    
    def _estimate_confidence(self, result):
        """
        Estimate confidence of transcription result
        """
        if 'segments' in result and result['segments']:
            # Average confidence across segments
            confidences = []
            for segment in result['segments']:
                if 'avg_logprob' in segment:
                    confidences.append(segment['avg_logprob'])
            
            if confidences:
                return sum(confidences) / len(confidences)
        
        return -1.0  # Default low confidence if can't determine
    
    def _process_commands(self):
        """
        Process transcription results and trigger callbacks
        """
        while self.is_listening or not self.result_queue.empty():
            try:
                result = self.result_queue.get(timeout=0.1)
                
                # Only process if confidence is acceptable and text is meaningful
                if result.confidence > -0.8 and len(result.text.strip()) > 3:  # Adjust confidence threshold as needed
                    # Trigger all registered callbacks
                    for callback in self.command_callbacks:
                        try:
                            callback(result.text, result.confidence)
                        except Exception as e:
                            print(f"Error in command callback: {e}")
                else:
                    print(f"Skipping low-confidence transcription: '{result.text}' (confidence: {result.confidence:.2f})")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in command processing: {e}")
                continue
    
    def add_command_callback(self, callback: Callable[[str, float], None]):
        """
        Add a callback function to be called when a transcription is received
        
        Args:
            callback: Function that takes (transcription_text, confidence) as arguments
        """
        self.command_callbacks.append(callback)
    
    def remove_command_callback(self, callback: Callable[[str, float], None]):
        """
        Remove a callback function
        """
        if callback in self.command_callbacks:
            self.command_callbacks.remove(callback)
    
    def get_status(self) -> Dict:
        """
        Get current status and statistics
        """
        with self.lock:
            total_requests = self.transcription_count + self.error_count
            avg_processing_time = self.total_processing_time / self.transcription_count if self.transcription_count > 0 else 0.0
            error_rate = self.error_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'is_listening': self.is_listening,
            'model_size': self.model_size,
            'device': self.device,
            'transcription_count': self.transcription_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_processing_time': avg_processing_time,
            'active_callbacks': len(self.command_callbacks)
        }
    
    def set_sensitivity(self, sensitivity: float):
        """
        Set the sensitivity for voice detection (0.0 to 1.0)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        # Adjust silence threshold based on sensitivity
        self.silence_threshold = 0.01 + (1.0 - self.sensitivity) * 0.09  # Range: 0.01 to 0.1
    
    def manual_transcribe(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """
        Manually transcribe provided audio data
        """
        start_time = time.time()
        
        try:
            result = self.model.transcribe(audio_data, fp16=torch.cuda.is_available())
            
            end_time = time.time()
            
            return TranscriptionResult(
                text=result['text'],
                language=result['language'],
                confidence=self._estimate_confidence(result),
                processing_time=end_time - start_time,
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Manual transcription error: {e}")
            return None
    
    def __del__(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
```

### ROS 2 Integration Example

```python
# whisper_ros_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import AudioData
from whisper_interface import WhisperInterface, TranscriptionResult
import threading

class WhisperROSIntegration(Node):
    def __init__(self):
        super().__init__('whisper_ros_integration')
        
        # Initialize Whisper interface
        self.whisper_interface = WhisperInterface(
            model_size="base",
            device="cuda" if self._cuda_available() else "cpu"
        )
        
        # ROS 2 publishers
        self.text_publisher = self.create_publisher(String, 'whisper/text', 10)
        self.confidence_publisher = self.create_publisher(Float32, 'whisper/confidence', 10)
        self.status_publisher = self.create_publisher(String, 'whisper/status', 10)
        
        # Add callback to process transcriptions
        self.whisper_interface.add_command_callback(self._transcription_callback)
        
        # Start listening
        self.get_logger().info("Starting Whisper interface...")
        self.whisper_interface.start_listening()
        
        # Timer for publishing status
        self.status_timer = self.create_timer(5.0, self._publish_status)
        
        self.get_logger().info('Whisper ROS Integration node initialized')
    
    def _cuda_available(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _transcription_callback(self, text: str, confidence: float):
        """
        Callback for processing Whisper transcriptions
        """
        self.get_logger().info(f"Received transcription: '{text}' (conf: {confidence:.2f})")
        
        # Publish text
        text_msg = String()
        text_msg.data = text
        self.text_publisher.publish(text_msg)
        
        # Publish confidence
        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_publisher.publish(confidence_msg)
        
        # Process the command (in a real system, this would parse and execute robot commands)
        self._process_voice_command(text)
    
    def _process_voice_command(self, command: str):
        """
        Process the voice command and execute appropriate robot behavior
        """
        command_lower = command.lower()
        
        # Simple command recognition (in a real system, this would use more sophisticated NLP)
        if any(word in command_lower for word in ["hello", "hi", "hey"]):
            self.get_logger().info("Robot greeting activated")
            # In a real robot, this would trigger a greeting behavior
        elif any(word in command_lower for word in ["move", "forward", "go"]):
            self.get_logger().info("Robot movement command received")
            # In a real robot, this would trigger movement
        elif any(word in command_lower for word in ["stop", "halt", "pause"]):
            self.get_logger().info("Robot stop command received")
            # In a real robot, this would stop movements
        elif any(word in command_lower for word in ["dance", "dancing"]):
            self.get_logger().info("Robot dance command received")
            # In a real robot, this would trigger dance behavior
        else:
            self.get_logger().info(f"Unknown command: {command}")
    
    def _publish_status(self):
        """
        Publish status information
        """
        status = self.whisper_interface.get_status()
        status_msg = String()
        status_msg.data = str(status)
        self.status_publisher.publish(status_msg)
    
    def destroy_node(self):
        """
        Clean up resources when node is destroyed
        """
        self.whisper_interface.stop_listening()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperROSIntegration()
    
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

### Advanced Configuration and Testing

```python
# whisper_configurator.py
import json
import os
from pathlib import Path
from typing import Dict, Any

class WhisperConfigurator:
    """
    Configuration manager for Whisper interfaces
    """
    DEFAULT_CONFIG = {
        "model_size": "base",
        "device": "cuda",
        "sample_rate": 16000,
        "buffer_duration": 1.0,
        "silence_threshold": 0.01,
        "sensitivity": 0.5,
        "vad_enabled": True,
        "min_audio_length": 8000,  # 0.5 seconds at 16kHz
        "processing_timeout": 10.0,
        "command_callbacks": []
    }
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "whisper_config.json"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self.DEFAULT_CONFIG.copy()
            self.save_config(config)
        
        # Validate and merge with defaults
        for key, default_value in self.DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = default_value
        
        return config
    
    def save_config(self, config: Dict[str, Any] = None):
        """
        Save configuration to file
        """
        config = config or self.config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        """
        for key, value in updates.items():
            if key in self.DEFAULT_CONFIG:
                self.config[key] = value
            else:
                print(f"Warning: Unknown configuration key '{key}'")
        
        self.save_config()
    
    def get_interface_params(self) -> Dict[str, Any]:
        """
        Get parameters suitable for WhisperInterface constructor
        """
        params = self.config.copy()
        # Remove configuration-specific keys that aren't for the interface
        for key in ['command_callbacks']:  # Add other non-interface params as needed
            params.pop(key, None)
        return params

# Configuration testing script
def test_whisper_configuration():
    """
    Test configuration loading and interface creation
    """
    # Create configurator
    config = WhisperConfigurator("test_config.json")
    
    print("Initial config:", config.config)
    
    # Update some parameters
    config.update_config({
        "model_size": "small",
        "sensitivity": 0.7,
        "silence_threshold": 0.02
    })
    
    print("Updated config:", config.config)
    
    # Get interface parameters
    interface_params = config.get_interface_params()
    print("Interface params:", interface_params)
    
    return config

if __name__ == "__main__":
    config = test_whisper_configuration()
```

## Performance Optimization Strategies

### Model Optimization for Real-time Processing

```python
# whisper_optimization.py
import whisper
import torch
import time
import numpy as np

class OptimizedWhisperProcessor:
    """
    Optimized Whisper processor for real-time applications
    """
    def __init__(self, model_size="base"):
        # Load model in evaluation mode
        self.model = whisper.load_model(model_size).eval()
        
        # Use torch.jit to trace the model for faster inference (optional)
        self.use_jit = False
        if self.use_jit:
            # Example of JIT optimization (simplified)
            dummy_input = torch.randn(1, 80, 3000)  # Example input shape
            self.model = torch.jit.trace(self.model, dummy_input)
            self.model = torch.jit.freeze(self.model)
    
    def preprocess_audio(self, audio_np: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio for Whisper model
        """
        # Ensure audio is in the right format
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Pad or trim to appropriate length (Whisper expects 0-30 seconds)
        if len(audio_np) > 30 * 16000:  # Max 30 seconds
            audio_np = audio_np[:30 * 16000]
        elif len(audio_np) < 16000:  # Min 1 second
            # Pad with zeros
            padding = np.zeros(16000 - len(audio_np), dtype=np.float32)
            audio_np = np.concatenate([audio_np, padding])
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_np).to(self.model.device)
        
        return audio_tensor
    
    def transcribe_optimized(self, audio_np: np.ndarray) -> dict:
        """
        Optimized transcription method
        """
        # Preprocess
        audio_tensor = self.preprocess_audio(audio_np)
        
        # Transcribe (with fp16 if available)
        result = self.model.transcribe(
            audio_tensor,
            fp16=torch.cuda.is_available(),
            without_timestamps=True  # Faster processing
        )
        
        return result
```

## Practical Usage Examples

### Example 1: Basic Implementation

```python
# example_basic.py
from whisper_interface import WhisperInterface

def simple_command_handler(text: str, confidence: float):
    """
    Simple command handler that just prints the command
    """
    print(f"Command received: {text}")
    print(f"Confidence: {confidence:.2f}")

def main():
    # Create Whisper interface
    whisper = WhisperInterface(model_size="base")
    
    # Add command handler
    whisper.add_command_callback(simple_command_handler)
    
    # Start listening
    if whisper.start_listening():
        print("Whisper is listening. Press Ctrl+C to stop.")
        
        try:
            # Keep the program running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            whisper.stop_listening()
    
    # Print final status
    status = whisper.get_status()
    print(f"Final status: {status}")

if __name__ == "__main__":
    main()
```

### Example 2: Robot Command Processing

```python
# example_robot_command.py
from whisper_interface import WhisperInterface
import json

class RobotCommandProcessor:
    """
    Process voice commands for a robot
    """
    def __init__(self):
        # Define robot commands and their handlers
        self.command_handlers = {
            'move_forward': self._handle_move_forward,
            'move_backward': self._handle_move_backward,
            'turn_left': self._handle_turn_left,
            'turn_right': self._handle_turn_right,
            'stop': self._handle_stop,
            'dance': self._handle_dance,
            'hello': self._handle_hello
        }
    
    def process_command(self, text: str, confidence: float):
        """
        Process a voice command
        """
        if confidence < 0.5:  # Skip low-confidence commands
            print(f"Skipping low-confidence command: {text} (conf: {confidence:.2f})")
            return
        
        text_lower = text.lower()
        
        # Simple command matching (in a real system, use NLP)
        if 'move forward' in text_lower or 'go forward' in text_lower:
            self.command_handlers['move_forward']()
        elif 'move backward' in text_lower or 'go backward' in text_lower:
            self.command_handlers['move_backward']()
        elif 'turn left' in text_lower:
            self.command_handlers['turn_left']()
        elif 'turn right' in text_lower:
            self.command_handlers['turn_right']()
        elif 'stop' in text_lower:
            self.command_handlers['stop']()
        elif 'dance' in text_lower:
            self.command_handlers['dance']()
        elif 'hello' in text_lower or 'hi' in text_lower:
            self.command_handlers['hello']()
        else:
            print(f"Unknown command: {text}")
    
    def _handle_move_forward(self):
        print("Robot moving forward")
        # In a real robot, send movement command via ROS or other interface
    
    def _handle_move_backward(self):
        print("Robot moving backward")
        # In a real robot, send movement command
    
    def _handle_turn_left(self):
        print("Robot turning left")
        # In a real robot, send turning command
    
    def _handle_turn_right(self):
        print("Robot turning right")
        # In a real robot, send turning command
    
    def _handle_stop(self):
        print("Robot stopping")
        # In a real robot, send stop command
    
    def _handle_dance(self):
        print("Robot dancing")
        # In a real robot, execute dance routine
    
    def _handle_hello(self):
        print("Robot says hello!")
        # In a real robot, execute greeting routine

def main():
    # Create robot command processor
    processor = RobotCommandProcessor()
    
    # Create Whisper interface
    whisper = WhisperInterface(model_size="base")
    
    # Add command handler
    whisper.add_command_callback(processor.process_command)
    
    # Start listening
    if whisper.start_listening():
        print("Robot voice controller is listening. Say 'hello', 'move forward', etc. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping robot voice controller...")
            whisper.stop_listening()
    
    # Print final status
    status = whisper.get_status()
    print(f"Final status: {status}")

if __name__ == "__main__":
    main()
```

## Testing and Validation

### Unit Tests

```python
# test_whisper_interface.py
import unittest
import numpy as np
from whisper_interface import WhisperInterface, TranscriptionResult

class TestWhisperInterface(unittest.TestCase):
    def setUp(self):
        # Create a minimal interface for testing without starting full audio processing
        self.interface = WhisperInterface(model_size="tiny")  # Use tiny for tests
    
    def test_manual_transcription(self):
        """Test manual transcription with sample audio data"""
        # Create a simple sample audio (silence with some signal)
        sample_audio = np.random.normal(0, 0.01, 16000).astype(np.float32)  # 1 second of noise
        
        result = self.interface.manual_transcribe(sample_audio)
        
        self.assertIsInstance(result, TranscriptionResult)
        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.processing_time, float)
    
    def test_configurable_parameters(self):
        """Test that parameters can be set correctly"""
        # Test setting sensitivity
        initial_threshold = self.interface.silence_threshold
        self.interface.set_sensitivity(0.8)
        
        # Sensitivity affects silence threshold
        self.assertNotEqual(initial_threshold, self.interface.silence_threshold)
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        status = self.interface.get_status()
        
        self.assertIn('is_listening', status)
        self.assertIn('model_size', status)
        self.assertIn('device', status)
        self.assertIn('transcription_count', status)
        self.assertIn('error_count', status)
    
    def test_callback_management(self):
        """Test adding and removing callbacks"""
        def dummy_callback(text, confidence):
            pass
        
        initial_count = len(self.interface.command_callbacks)
        
        # Add callback
        self.interface.add_command_callback(dummy_callback)
        self.assertEqual(len(self.interface.command_callbacks), initial_count + 1)
        
        # Remove callback
        self.interface.remove_command_callback(dummy_callback)
        self.assertEqual(len(self.interface.command_callbacks), initial_count)

if __name__ == '__main__':
    unittest.main()
```

## Summary

In this chapter, we've implemented a comprehensive Whisper speech-to-text interface for humanoid robotics:

1. **Core Interface**: Built a robust Whisper interface class with audio processing, buffering, and voice activity detection
2. **ROS Integration**: Created a ROS 2 node that integrates Whisper with robotics communication
3. **Configuration System**: Developed a flexible configuration system for tuning the interface
4. **Optimization Techniques**: Implemented performance optimizations for real-time processing
5. **Practical Examples**: Provided ready-to-use examples for basic and robot-specific applications
6. **Testing Framework**: Created unit tests to validate the implementation

This implementation provides a solid foundation for voice command processing in humanoid robotics, capable of handling real-world audio conditions and integrating with robot control systems. The next step is to connect this speech interface with language models for higher-level command understanding and task execution.