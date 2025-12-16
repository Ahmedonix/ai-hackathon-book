# Practical Exercise: Whisper Integration for Voice Commands

## Objective

In this exercise, you will implement an integration between OpenAI's Whisper speech recognition model and your ROS 2 humanoid robot system. By the end of this exercise, you will have a working voice command system that can convert spoken language into text that your robot can process.

## Prerequisites

Before starting this exercise, you should have:
- Completed Module 1 (ROS 2 fundamentals)
- Set up your ROS 2 Iron environment
- Installed PyAudio and OpenAI Python packages
- Access to a microphone for audio input
- Basic Python programming skills

## Time Estimate

This exercise should take approximately 3-4 hours to complete, depending on your familiarity with ROS 2 and audio processing.

## Setup

### Install Required Dependencies

First, install the required Python packages:

```bash
pip3 install openai whisper pyaudio numpy
```

Or add to your ROS package's `requirements.txt`:
```
openai>=0.27.0
openai-whisper
pyaudio
numpy
```

### Verify Audio Input

Before beginning, verify that your system can capture audio:

```python
import pyaudio

# Test audio input
p = pyaudio.PyAudio()
print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Device {i}: {info['name']}")

p.terminate()
```

## Exercise Tasks

### Task 1: Basic Audio Capture Node (45 minutes)

Create a basic ROS 2 node that can continuously listen for audio and detect when speech is occurring.

1. Create a new ROS package for audio processing:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python audio_input_node
   ```

2. Create the audio capture node in `audio_input_node/audio_input_node/audio_capture.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pyaudio
import numpy as np
from std_msgs.msg import String

class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.speech_threshold = 1000  # Amplitude threshold for speech detection
        
        # Publisher for detected speech events
        self.speech_detected_pub = self.create_publisher(String, 'speech_events', 10)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start audio monitoring
        self.monitor_audio()
        
        self.get_logger().info('Audio Capture Node initialized')
    
    def monitor_audio(self):
        """Continuously monitor audio for speech"""
        # Open audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.get_logger().info("Starting audio monitoring...")
        
        try:
            while rclpy.ok():
                # Read audio data
                data = stream.read(self.chunk)
                
                # Convert to numpy array to analyze
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.mean(np.abs(audio_data))
                
                # Check if amplitude exceeds threshold (indicating speech)
                if amplitude > self.speech_threshold:
                    # Publish speech detection event
                    msg = String()
                    msg.data = "SPEECH_DETECTED"
                    self.speech_detected_pub.publish(msg)
                    self.get_logger().info(f"Speech detected! Amplitude: {amplitude:.2f}")
        
        except KeyboardInterrupt:
            self.get_logger().info("Audio monitoring stopped by user")
        
        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
    
    try:
        # Since this runs in a loop, we don't need to spin
        # Instead, we can run the monitoring in the main thread
        pass
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. Make the file executable and test it:
   ```bash
   chmod +x audio_input_node/audio_input_node/audio_capture.py
   ros2 run audio_input_node audio_capture.py
   ```

### Task 2: Whisper Integration Node (60 minutes)

Create a node that integrates with Whisper to transcribe speech:

1. Create `audio_input_node/audio_input_node/whisper_transcriber.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import whisper
import pyaudio
import numpy as np
import threading
import queue
from std_msgs.msg import String

class WhisperTranscriberNode(Node):
    def __init__(self):
        super().__init__('whisper_transcriber_node')
        
        # Initialize Whisper model (using 'base' model for efficiency)
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")
        
        # Publishers and subscribers
        self.transcription_pub = self.create_publisher(String, 'transcriptions', 10)
        self.speech_events_sub = self.create_subscription(
            String, 'speech_events', self.speech_event_callback, 10
        )
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 3  # How long to record when speech is detected
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # State management
        self.is_recording = False
        self.recording_queue = queue.Queue()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.recording_worker, daemon=True)
        self.recording_thread.start()
        
        self.get_logger().info('Whisper Transcriber Node initialized')
    
    def speech_event_callback(self, msg):
        """Handle speech detection events"""
        if msg.data == "SPEECH_DETECTED" and not self.is_recording:
            self.get_logger().info("Starting recording for transcription...")
            self.start_recording()
    
    def start_recording(self):
        """Start recording audio for transcription"""
        if self.is_recording:
            return  # Already recording
        
        self.is_recording = True
        self.recording_queue.put("RECORD")
    
    def recording_worker(self):
        """Background worker for audio recording"""
        while rclpy.ok():
            try:
                # Wait for recording command
                cmd = self.recording_queue.get(timeout=1.0)
                
                if cmd == "RECORD":
                    self.perform_recording()
                
                self.recording_queue.task_done()
            except queue.Empty:
                continue
    
    def perform_recording(self):
        """Perform the actual audio recording and transcription"""
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Record audio
            frames = []
            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(data)
            
            # Close stream
            stream.stop_stream()
            stream.close()
            
            # Convert recorded audio to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 in range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            self.get_logger().info("Transcribing audio with Whisper...")
            result = self.model.transcribe(audio_float)
            text = result["text"].strip()
            
            if text:
                self.get_logger().info(f"Transcription: {text}")
                
                # Publish transcription
                transcription_msg = String()
                transcription_msg.data = text
                self.transcription_pub.publish(transcription_msg)
            else:
                self.get_logger().info("No speech detected in recording")
        
        except Exception as e:
            self.get_logger().error(f"Error during recording/transcription: {e}")
        finally:
            self.is_recording = False
    
    def destroy_node(self):
        """Clean up resources"""
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperTranscriberNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

2. Make it executable:
   ```bash
   chmod +x audio_input_node/audio_input_node/whisper_transcriber.py
   ```

### Task 3: Integration and Testing (45 minutes)

1. Create a launch file to run both nodes together in `audio_input_node/launch/whisper_integration.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='audio_input_node',
            executable='audio_capture',
            name='audio_capture_node'
        ),
        Node(
            package='audio_input_node',
            executable='whisper_transcriber',
            name='whisper_transcriber_node'
        )
    ])
```

2. Test the integrated system:
   ```bash
   ros2 launch audio_input_node whisper_integration.launch.py
   ```

3. In another terminal, monitor the transcriptions:
   ```bash
   ros2 topic echo /transcriptions std_msgs/msg/String
   ```

4. Speak into your microphone and observe the transcriptions in the topic.

### Task 4: Enhancements (30 minutes)

Add improvements to your basic Whisper integration:

1. **Improve speech detection**:
   - Implement a more sophisticated voice activity detection algorithm
   - Add a "silence threshold" to stop recording when speech ends

2. **Add error handling**:
   - Handle cases where Whisper model fails
   - Implement retry logic for transcription failures

3. **Create a more sophisticated transcription interface**:
   - Modify the transcriber to handle longer conversations
   - Add confidence scores to transcriptions

Example of an enhanced transcriber with better speech detection:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import whisper
import pyaudio
import numpy as np
import threading
import queue
from std_msgs.msg import String

class EnhancedWhisperNode(Node):
    def __init__(self):
        super().__init__('enhanced_whisper_node')
        
        # Initialize Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")
        
        # Publishers and subscribers
        self.transcription_pub = self.create_publisher(String, 'enhanced_transcriptions', 10)
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.silence_threshold = 500   # Lower threshold for silence detection
        self.speech_threshold = 1500   # Higher threshold for speech detection
        self.silence_duration = 1.0    # Stop recording after 1 second of silence
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start audio processing
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()
        
        self.get_logger().info('Enhanced Whisper Node initialized')
    
    def process_audio(self):
        """Continuously process audio for speech detection and transcription"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.get_logger().info("Starting audio processing...")
        
        recording = False
        frames = []
        silence_count = 0
        max_silence_frames = int(self.rate / self.chunk * self.silence_duration)
        
        try:
            while rclpy.ok():
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)
                
                # Convert to numpy array to analyze
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.mean(np.abs(audio_data))
                
                if not recording:
                    # Check if we should start recording (speech detected)
                    if amplitude > self.speech_threshold:
                        recording = True
                        frames = [data]  # Start with the current frame
                        silence_count = 0
                        self.get_logger().info("Speech detected, starting recording...")
                else:
                    # We're recording, add frame and check for silence
                    frames.append(data)
                    
                    if amplitude < self.silence_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0  # Reset on any sound above threshold
                    
                    # Check if we should stop recording (prolonged silence)
                    if silence_count > max_silence_frames:
                        self.get_logger().info("Prolonged silence detected, processing recording...")
                        self.process_recording(frames)
                        recording = False
                        frames = []
        
        except KeyboardInterrupt:
            self.get_logger().info("Audio processing stopped by user")
        finally:
            stream.stop_stream()
            stream.close()
    
    def process_recording(self, frames):
        """Process a complete recording with Whisper"""
        try:
            # Convert recorded audio to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 in range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            self.get_logger().info(f"Processing {len(frames)} frames with Whisper...")
            result = self.model.transcribe(audio_float)
            text = result["text"].strip()
            
            if text:
                self.get_logger().info(f"Transcription: {text}")
                
                # Publish transcription
                transcription_msg = String()
                transcription_msg.data = text
                self.transcription_pub.publish(transcription_msg)
            else:
                self.get_logger().info("No speech detected in recording")
                
        except Exception as e:
            self.get_logger().error(f"Error during transcription: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedWhisperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise Deliverables

For this exercise, create a submission that includes:

1. **Complete source code** for your Whisper integration
2. **Launch file** to start your nodes
3. **Documentation** explaining how your implementation works
4. **Test results** showing successful transcriptions
5. **Reflection** on challenges faced and lessons learned

## Evaluation Criteria

- **Functionality** (40%): Your system correctly captures audio and transcribes speech
- **Code Quality** (25%): Well-structured, documented, and maintainable code
- **ROS Integration** (20%): Proper use of ROS concepts (publishers, subscribers, nodes)
- **Problem-Solving** (15%): Effective debugging and error handling

## Troubleshooting

Common issues and solutions:

1. **PyAudio installation fails**:
   - On Ubuntu: `sudo apt-get install python3-pyaudio`
   - On other systems: Install PortAudio first, then PyAudio

2. **Whisper model loading fails**:
   - Ensure you have internet connection for initial download
   - Check available disk space
   - Try smaller models ('tiny' or 'base') if compute is limited

3. **Audio not being detected**:
   - Verify microphone permissions
   - Check if audio device is correctly selected
   - Adjust threshold values based on your microphone sensitivity

## Extensions

For advanced students, consider implementing:

1. **Noise reduction**: Apply audio filters to improve transcription quality
2. **Language detection**: Automatically detect input language and select appropriate model
3. **Real-time transcription**: Implement streaming transcription for longer conversations
4. **Keyword spotting**: Detect specific wake words before starting transcription

## Conclusion

This exercise has provided hands-on experience with integrating the Whisper speech recognition model into a ROS 2 system. You've learned how to capture audio, detect speech, and convert spoken language to text that can be processed by your robotic system. This forms a foundational component for voice-controlled robots.