---
title: Whisper Speech Interface Setup
description: Setting up OpenAI Whisper for speech recognition in humanoid robotics
sidebar_position: 2
---

# Whisper Speech Interface Setup

## Overview

OpenAI's Whisper is a state-of-the-art speech recognition model that can convert spoken language into text. In this chapter, we'll explore how to integrate Whisper into humanoid robotics applications, creating a robust speech-to-text interface that allows robots to understand and respond to voice commands.

## Learning Objectives

- Understand Whisper's architecture and capabilities
- Set up Whisper for real-time speech recognition
- Integrate Whisper with ROS 2 communication systems
- Optimize Whisper for humanoid robotics applications
- Implement voice command processing pipelines

## Whisper Architecture and Capabilities

Whisper is a transformer-based model trained on a large dataset of audio and text pairs. It can handle multiple languages, accents, and background noise, making it ideal for robotic applications where acoustic conditions might vary.

### Key Features:
- Multilingual support (99 languages)
- Robust to accents and background noise
- End-to-end speech recognition
- Time-stamped outputs
- Speaker identification (in some models)

## Environment Setup

### Prerequisites

Before setting up Whisper, ensure you have the necessary dependencies installed:

```bash
# Install Python dependencies
pip install torch torchvision torchaudio
pip install openai-whisper
pip install sounddevice numpy
pip install speech-recognition
pip install pyaudio
pip install vosk  # Alternative lightweight option
```

### Docker-Based Setup for Consistency

For deployment consistency, especially on Jetson devices, consider using a Docker container:

```dockerfile
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html

# Install Whisper and related packages
RUN pip3 install openai-whisper
RUN pip3 install sounddevice numpy speech-recognition pyaudio

# Install additional dependencies for robotics
RUN pip3 install rclpy

WORKDIR /workspace

# Add your application code here
COPY . /workspace

CMD ["python3", "whisper_ros_node.py"]
```

## Whisper Installation and Model Selection

### Installing Whisper

Install Whisper using pip:

```bash
pip install -U openai-whisper
```

### Model Selection

Whisper comes in different sizes with trade-offs between accuracy and performance:

```python
import whisper

# Available models (smallest to largest)
models = {
    'tiny': 'Fastest, least accurate - suitable for real-time applications',
    'base': 'Good balance of speed and accuracy',
    'small': 'Better accuracy, slower than base',
    'medium': 'High accuracy, slower',
    'large': 'Highest accuracy, slowest - comes in two versions: large and large-v2'
}

# Load the model - for humanoid robots, 'base' or 'small' often provide the best balance
model = whisper.load_model("base")
```

For humanoid robots with resource constraints, consider the `tiny` or `base` models. For deployment on NVIDIA Jetson platforms, you might need to optimize the model further or consider using a lighter alternative like VOSK.

## Basic Whisper Implementation

### Simple Audio Processing with Whisper

```python
import whisper
import torch
import numpy as np
import pyaudio
import wave
import time
from threading import Thread, Event

class WhisperSpeechProcessor:
    def __init__(self, model_size="base", device="cuda"):
        """
        Initialize Whisper speech processor
        
        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.model = whisper.load_model(self.model_size).to(self.device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Event for controlling recording
        self.recording_event = Event()
        self.listening_event = Event()
        
        print(f"Whisper model loaded on {self.device}")
    
    def record_audio(self, duration=5):
        """
        Record audio for a specified duration
        
        Args:
            duration: Recording duration in seconds
        """
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_np
    
    def transcribe_audio(self, audio_np):
        """
        Transcribe audio using Whisper
        
        Args:
            audio_np: Audio data as numpy array
        """
        start_time = time.time()
        
        # Transcribe audio
        result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
        
        end_time = time.time()
        
        return {
            'text': result['text'],
            'language': result['language'],
            'processing_time': end_time - start_time
        }
    
    def record_and_transcribe(self, duration=5):
        """
        Record audio and transcribe it using Whisper
        """
        audio_np = self.record_audio(duration)
        result = self.transcribe_audio(audio_np)
        
        return result

# Example usage
if __name__ == "__main__":
    processor = WhisperSpeechProcessor(model_size="base")
    result = processor.record_and_transcribe(duration=5)
    print(f"Transcribed: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
```

## Real-time Speech Recognition

For robotics applications, we often need real-time or near real-time speech recognition:

```python
import queue
import threading
import time
import numpy as np

class RealTimeWhisperProcessor:
    def __init__(self, model_size="base", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.model = whisper.load_model(self.model_size).to(self.device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Create queues for audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Processing flags
        self.is_processing = False
        self.is_listening = False
        
        # Callback function for voice commands
        self.command_callback = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening for audio input"""
        if self.is_listening:
            return
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.is_listening = True
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Started listening for audio input")
    
    def stop_listening(self):
        """Stop listening for audio input"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        self.stream.stop_stream()
        self.stream.close()
        
        print("Stopped listening for audio input")
    
    def process_audio(self):
        """Process audio data from queue and perform transcription"""
        audio_buffer = np.array([])
        silence_threshold = 0.01  # Adjust based on your environment
        min_audio_length = 0.5 * self.sample_rate  # Minimum 0.5 seconds of audio
        
        while self.is_listening:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Add chunk to buffer
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Check if buffer has enough audio data
                if len(audio_buffer) >= min_audio_length:
                    # Check if audio is not just silence
                    if np.max(np.abs(audio_buffer)) > silence_threshold:
                        # Process the buffered audio
                        result = self.transcribe_audio(audio_buffer)
                        
                        if result and len(result['text'].strip()) > 0:
                            print(f"Transcribed: {result['text']}")
                            
                            # Call the command callback if set
                            if self.command_callback:
                                self.command_callback(result['text'])
                        
                        # Clear buffer after processing
                        audio_buffer = np.array([])
                    else:
                        # If the buffer is mostly silence, reduce its size
                        # to only keep recent data
                        if len(audio_buffer) > min_audio_length * 2:
                            audio_buffer = audio_buffer[-int(min_audio_length):]
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
    
    def transcribe_audio(self, audio_np):
        """
        Transcribe audio using Whisper with error handling
        """
        try:
            # Transcribe audio
            result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
            
            return {
                'text': result['text'],
                'language': result['language'],
                'confidence': self.estimate_transcription_confidence(result)
            }
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def estimate_transcription_confidence(self, result):
        """
        Estimate confidence of transcription (simplified approach)
        """
        # This is a simplified confidence estimation
        # A full implementation would use more sophisticated methods
        if 'segments' in result:
            # Average confidence across segments
            confidences = [seg.get('avg_logprob', -1.0) for seg in result['segments']]
            if confidences:
                return sum(confidences) / len(confidences)
        
        return -1.0
    
    def set_command_callback(self, callback):
        """
        Set callback function for processing recognized commands
        """
        self.command_callback = callback

# Example usage of real-time processor
def command_processor(text):
    """Handle processed voice commands"""
    print(f"Processing command: {text}")
    
    # In a real application, you would parse the command and
    # execute the corresponding robot action
    if "hello" in text.lower():
        print("Robot says: Hello! How can I help you?")
    elif "move" in text.lower():
        print("Robot is moving...")
    elif "stop" in text.lower():
        print("Robot is stopping...")

if __name__ == "__main__":
    processor = RealTimeWhisperProcessor(model_size="base")
    processor.set_command_callback(command_processor)
    
    try:
        processor.start_listening()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        processor.stop_listening()
```

## ROS 2 Integration

Now let's integrate Whisper with ROS 2 for robotics applications:

```python
# whisper_ros_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import json

class WhisperROSNode(Node):
    def __init__(self):
        super().__init__('whisper_ros_node')
        
        # Initialize Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base").to(self.device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # ROS 2 publishers and subscribers
        self.text_publisher = self.create_publisher(String, 'whisper/text', 10)
        self.status_publisher = self.create_publisher(String, 'whisper/status', 10)
        
        # Create parameter for sensitivity
        self.declare_parameter('sensitivity', 0.02)  # Silence threshold
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('continuous_listening', True)
        
        self.sensitivity = self.get_parameter('sensitivity').value
        model_size = self.get_parameter('model_size').value
        
        # Load Whisper model
        if model_size != 'base':  # Default to base if parameter is invalid
            try:
                self.model = whisper.load_model(model_size).to(self.device)
            except:
                self.get_logger().warn(f"Invalid model size: {model_size}, using 'base'")
                self.model = whisper.load_model("base").to(self.device)
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.continuous_listening = self.get_parameter('continuous_listening').value
        
        # Start audio processing
        self.start_listening()
        
        self.get_logger().info('Whisper ROS Node initialized')
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening for audio input"""
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.is_listening = True
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.get_logger().info("Started listening for audio input")
    
    def stop_listening(self):
        """Stop listening for audio input"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        self.stream.stop_stream()
        self.stream.close()
        
        self.get_logger().info("Stopped listening for audio input")
    
    def process_audio(self):
        """Process audio data and perform transcription"""
        audio_buffer = np.array([])
        min_audio_length = 0.5 * self.sample_rate  # Minimum 0.5 seconds of audio
        
        while self.is_listening:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Add chunk to buffer
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # Check if buffer has enough audio data
                if len(audio_buffer) >= min_audio_length:
                    # Check if audio is not just silence
                    if np.max(np.abs(audio_buffer)) > self.sensitivity:
                        # Process the buffered audio
                        result = self.transcribe_audio(audio_buffer)
                        
                        if result and len(result['text'].strip()) > 0:
                            self.get_logger().info(f"Transcribed: {result['text']}")
                            
                            # Publish the transcription
                            text_msg = String()
                            text_msg.data = result['text']
                            self.text_publisher.publish(text_msg)
                            
                            # Publish status
                            status_msg = String()
                            status_data = {
                                'transcription': result['text'],
                                'language': result['language'],
                                'confidence': result['confidence'],
                                'processing_time': result['processing_time']
                            }
                            status_msg.data = json.dumps(status_data)
                            self.status_publisher.publish(status_msg)
                        
                        # Clear buffer after processing
                        audio_buffer = np.array([])
                    else:
                        # If the buffer is mostly silence, reduce its size
                        # to only keep recent data
                        if len(audio_buffer) > min_audio_length * 2:
                            audio_buffer = audio_buffer[-int(min_audio_length):]
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in audio processing: {e}")
                continue
    
    def transcribe_audio(self, audio_np):
        """Transcribe audio using Whisper"""
        start_time = time.time()
        
        try:
            # Transcribe audio
            result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
            
            end_time = time.time()
            
            return {
                'text': result['text'],
                'language': result['language'],
                'confidence': self.estimate_transcription_confidence(result),
                'processing_time': end_time - start_time
            }
        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            return None
    
    def estimate_transcription_confidence(self, result):
        """Estimate confidence of transcription"""
        if 'segments' in result:
            # Average confidence across segments
            confidences = [seg.get('avg_logprob', -1.0) for seg in result['segments']]
            if confidences:
                return sum(confidences) / len(confidences)
        return -1.0

def main(args=None):
    rclpy.init(args=args)
    node = WhisperROSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_listening()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization for Robotics

### Model Quantization for Edge Deployment

For deployment on resource-constrained platforms like the NVIDIA Jetson, model quantization can help:

```python
import whisper
import torch

def quantize_model():
    """
    Demonstration of model quantization techniques for deployment
    Note: This is for illustration; actual quantization may require
    specialized tools like TensorRT for best results
    """
    
    # Load the original model
    model = whisper.load_model("base")
    
    # Convert to float16 (reduces memory usage)
    model = model.half()
    
    # Or use PyTorch's quantization (simplified example)
    model_quant = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return model_quant
```

### Audio Preprocessing for Noise Reduction

In robotics environments, audio preprocessing is crucial:

```python
import numpy as np
import scipy.signal as signal

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Design a bandpass filter for human voice (300Hz to 3400Hz)
        low_freq = 300
        high_freq = 3400
        self.b, self.a = signal.butter(
            6, [low_freq / (sample_rate / 2), high_freq / (sample_rate / 2)], 
            btype='band'
        )
    
    def bandpass_filter(self, audio_data):
        """Apply bandpass filter to isolate human voice frequencies"""
        filtered = signal.filtfilt(self.b, self.a, audio_data)
        return filtered
    
    def noise_gate(self, audio_data, threshold=0.01):
        """Apply noise gate to attenuate quiet signals"""
        amplitude = np.abs(audio_data)
        gate = (amplitude > threshold).astype(np.float32)
        return audio_data * gate
    
    def preprocess(self, audio_data):
        """Apply all preprocessing steps"""
        # Apply bandpass filter
        filtered = self.bandpass_filter(audio_data)
        
        # Apply noise gate
        gated = self.noise_gate(filtered)
        
        # Normalize
        if np.max(np.abs(gated)) > 0:
            normalized = gated / np.max(np.abs(gated))
        else:
            normalized = gated
            
        return normalized
```

## Error Handling and Robustness

Implementing robust error handling is crucial for production systems:

```python
import time
import logging
from threading import Lock

class RobustWhisperProcessor:
    def __init__(self, model_size="base", device=None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with error handling
        self.model = None
        self._load_model_with_retry()
        
        # Processing parameters
        self.sample_rate = 16000
        self.silence_threshold = 0.01
        self.min_audio_length = int(0.5 * self.sample_rate)  # 0.5 seconds
        
        # Thread safety
        self.model_lock = Lock()
        
        # Statistics for performance monitoring
        self.transcription_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
    
    def _load_model_with_retry(self, max_retries=3):
        """Load Whisper model with retry logic"""
        for attempt in range(max_retries):
            try:
                self.model = whisper.load_model(self.model_size).to(self.device)
                self.get_logger().info(f"Model {self.model_size} loaded successfully on {self.device}")
                return
            except Exception as e:
                self.get_logger().error(f"Model load attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)  # Wait before retry
    
    def transcribe_with_error_handling(self, audio_np):
        """Transcribe audio with comprehensive error handling"""
        if self.model is None:
            self.get_logger().error("Model not loaded")
            return None
        
        if len(audio_np) < self.min_audio_length:
            self.get_logger().warn(f"Audio too short: {len(audio_np)} samples, min: {self.min_audio_length}")
            return None
        
        with self.model_lock:
            try:
                start_time = time.time()
                
                # Transcribe audio
                result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
                
                end_time = time.time()
                
                # Update statistics
                self.transcription_count += 1
                self.total_processing_time += (end_time - start_time)
                
                return {
                    'text': result['text'],
                    'language': result['language'],
                    'confidence': self.estimate_transcription_confidence(result),
                    'processing_time': end_time - start_time
                }
            except Exception as e:
                self.error_count += 1
                self.get_logger().error(f"Transcription failed: {e}")
                return None
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.transcription_count == 0:
            avg_processing_time = 0.0
        else:
            avg_processing_time = self.total_processing_time / self.transcription_count
        
        return {
            'transcriptions': self.transcription_count,
            'errors': self.error_count,
            'avg_processing_time': avg_processing_time,
            'error_rate': self.error_count / (self.transcription_count + self.error_count) if (self.transcription_count + self.error_count) > 0 else 0
        }
    
    def get_logger(self):
        """Get logger - in a ROS context, this would use ROS logging"""
        return logging.getLogger(__name__)
    
    def estimate_transcription_confidence(self, result):
        """Estimate confidence of transcription"""
        if 'segments' in result:
            confidences = [seg.get('avg_logprob', -1.0) for seg in result['segments']]
            if confidences:
                return sum(confidences) / len(confidences)
        return -1.0
```

## Summary

In this chapter, we've covered:

1. **Whisper Setup**: Installation and model selection for robotics applications
2. **Real-time Processing**: Implementing continuous speech recognition
3. **ROS 2 Integration**: Creating a ROS 2 node for Whisper integration
4. **Performance Optimization**: Techniques for efficient processing on embedded systems
5. **Robustness**: Error handling and performance monitoring

The Whisper speech interface forms the foundation for the vision-language-action system, enabling humanoid robots to understand natural language commands. In the next chapter, we'll explore how to use LLMs for action planning based on these voice commands.