# Practical Exercise: Multi-Modal Systems Integration

## Objective

In this exercise, you will implement and integrate multiple input modalities (voice, vision, gesture) into a unified system that processes information from different sources and coordinates responses across multiple output modalities. By the end of this exercise, you will have a working multi-modal system that can understand and respond to complex, multi-modal human inputs.

## Prerequisites

Before starting this exercise, you should have:
- Completed Modules 1-3 (ROS 2 fundamentals, simulation, AI perception)
- Understanding of basic computer vision techniques
- Access to audio and visual sensors (camera and microphone)
- Knowledge of ROS 2 message passing and service calls
- Basic understanding of human-robot interaction principles

## Time Estimate

This exercise should take approximately 5-6 hours to complete, depending on your familiarity with sensor integration and multi-modal processing.

## Setup

### Required Dependencies

Install the following packages:

```bash
pip3 install openai whisper pyaudio opencv-python mediapipe numpy
```

Or add to your ROS package's `requirements.txt`:
```
openai>=0.27.0
openai-whisper
pyaudio
opencv-python
mediapipe
numpy
```

## Exercise Tasks

### Task 1: Multi-Modal Input Processing (60 minutes)

Create nodes to process inputs from different modalities simultaneously and publish them as ROS messages.

1. Create a new ROS package for multi-modal processing:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python multimodal_input_node
   cd multimodal_input_node
   ```

2. Create an audio processing node in `multimodal_input_node/multimodal_input_node/audio_processor.py`:

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

class AudioProcessorNode(Node):
    def __init__(self):
        super().__init__('audio_processor_node')
        
        # Initialize Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")
        
        # Publishers
        self.audio_commands_pub = self.create_publisher(String, 'audio_commands', 10)
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.speech_threshold = 1000
        self.silence_threshold = 500
        self.silence_duration = 1.0  # Stop recording after this many seconds of silence
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # State management
        self.is_recording = False
        self.recording_queue = queue.Queue()
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.continuous_audio_processing, daemon=True)
        self.audio_thread.start()
        
        self.get_logger().info('Audio Processor Node initialized')

    def continuous_audio_processing(self):
        """Continuously process audio for speech events"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        self.get_logger().info("Starting audio processing loop...")
        
        recording = False
        frames = []
        silence_count = 0
        max_silence_frames = int(self.rate / self.chunk * self.silence_duration)
        
        try:
            while rclpy.ok():
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.mean(np.abs(audio_data))
                
                if not recording:
                    # Check if speech starts
                    if amplitude > self.speech_threshold:
                        recording = True
                        frames = [data]
                        silence_count = 0
                        self.get_logger().info("Speech detected, recording...")
                else:
                    # We're recording, add frame and check for silence
                    frames.append(data)
                    
                    if amplitude < self.silence_threshold:
                        silence_count += 1
                    else:
                        silence_count = 0
                    
                    # Check if we should stop recording
                    if silence_count > max_silence_frames:
                        self.process_recording(frames)
                        recording = False
                        frames = []
        
        except Exception as e:
            self.get_logger().error(f"Error in audio processing: {e}")
        finally:
            stream.stop_stream()
            stream.close()

    def process_recording(self, frames):
        """Process a complete recording with Whisper"""
        try:
            # Convert to numpy array
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            result = self.model.transcribe(audio_float)
            text = result["text"].strip()
            
            if text:
                self.get_logger().info(f"Transcribed: {text}")
                
                # Publish transcription
                msg = String()
                msg.data = text
                self.audio_commands_pub.publish(msg)
            else:
                self.get_logger().info("No speech recognized")
                
        except Exception as e:
            self.get_logger().error(f"Error in transcription: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AudioProcessorNode()
    
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

3. Create a vision processing node in `multimodal_input_node/multimodal_input_node/vision_processor.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from multimodal_msgs.msg import ObjectDetection, Gesture

class VisionProcessorNode(Node):
    def __init__(self):
        super().__init__('vision_processor_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe for object detection
        self.object_detector = cv2.dnn.readNetFromDarknet(
            'yolov4.cfg',  # You would need to download this model
            'yolov4.weights'  # You would need to download this model
        )
        # For this exercise, we'll use a simpler approach with OpenCV
        # In practice, you'd load a proper object detection model
        
        # Publishers
        self.object_detections_pub = self.create_publisher(ObjectDetection, 'object_detections', 10)
        self.gesture_detections_pub = self.create_publisher(Gesture, 'gesture_detections', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        
        self.get_logger().info('Vision Processor Node initialized')

    def image_callback(self, img_msg):
        """Process incoming images for object detection and gesture recognition"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            
            # Process for gesture recognition
            self.process_gestures(cv_image)
            
            # Process for object detection
            self.process_objects(cv_image)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def process_gestures(self, cv_image):
        """Process image for gesture recognition using MediaPipe"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find poses
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Get essential landmarks for gesture recognition
            left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # Example gesture recognition: check if right hand is raised
            if right_wrist.y < nose.y - 0.1:  # Hand above nose
                gesture_msg = Gesture()
                gesture_msg.header.stamp = self.get_clock().now().to_msg()
                gesture_msg.type = Gesture.RAISE_HAND
                gesture_msg.description = "Right hand raised above head level"
                self.gesture_detections_pub.publish(gesture_msg)
                self.get_logger().info("Gesture detected: Right hand raised")

    def process_objects(self, cv_image):
        """Process image for object detection"""
        # This is a simplified object detection for demonstration
        # In a real system, you would use a proper object detection model
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Detect red objects (cups, etc.) as an example
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out small detections
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Create object detection message
                    obj_msg = ObjectDetection()
                    obj_msg.header.stamp = self.get_clock().now().to_msg()
                    obj_msg.class_name = "red_object"
                    obj_msg.confidence = 0.8  # Estimated confidence
                    obj_msg.bounding_box.x_offset = x
                    obj_msg.bounding_box.y_offset = y
                    obj_msg.bounding_box.width = w
                    obj_msg.bounding_box.height = h
                    
                    self.object_detections_pub.publish(obj_msg)
                    self.get_logger().info(f"Red object detected at ({x}, {y})")

def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessorNode()
    
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

4. Create a simple gesture recognition node in `multimodal_input_node/multimodal_input_node/gesture_processor.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from multimodal_msgs.msg import Gesture

class GestureProcessorNode(Node):
    def __init__(self):
        super().__init__('gesture_processor_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Publishers
        self.gesture_pub = self.create_publisher(Gesture, 'hand_gestures', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        
        self.get_logger().info('Gesture Processor Node initialized')

    def image_callback(self, img_msg):
        """Process incoming images for hand gesture recognition"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on image (optional for visualization)
                    self.mp_drawing.draw_landmarks(
                        cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Recognize gesture based on landmarks
                    gesture = self.recognize_gesture(hand_landmarks, results.multi_handedness)
                    
                    if gesture:
                        # Publish gesture
                        gesture_msg = Gesture()
                        gesture_msg.header.stamp = self.get_clock().now().to_msg()
                        gesture_msg.type = gesture
                        gesture_msg.description = self.get_gesture_description(gesture)
                        
                        self.gesture_pub.publish(gesture_msg)
                        self.get_logger().info(f"Gesture detected: {self.get_gesture_description(gesture)}")
        
        except Exception as e:
            self.get_logger().error(f"Error processing gesture: {e}")

    def recognize_gesture(self, landmarks, handedness):
        """Recognize gesture based on hand landmarks"""
        # Extract specific landmarks
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Example: Recognize a pointing gesture (index finger extended, others folded)
        if (index_tip.y < middle_tip.y and 
            index_tip.y < ring_tip.y and 
            index_tip.y < pinky_tip.y and
            thumb_tip.x > index_tip.x):  # Thumb not extended
            return Gesture.POINTING
        
        # Example: Recognize a thumbs up
        if (thumb_tip.y < index_tip.y and 
            middle_tip.y < index_tip.y and
            ring_tip.y < index_tip.y and
            pinky_tip.y < index_tip.y):
            return Gesture.THUMBS_UP
        
        # Example: Recognize a waving gesture
        # This would require tracking movement over time
        return None

    def get_gesture_description(self, gesture_type):
        """Get description for a gesture type"""
        descriptions = {
            1: "Pointing gesture detected",
            2: "Thumbs up gesture detected",
            3: "Waving gesture detected",
            4: "Peace sign gesture detected"
        }
        return descriptions.get(gesture_type, f"Gesture type {gesture_type} detected")

def main(args=None):
    rclpy.init(args=args)
    node = GestureProcessorNode()
    
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

### Task 2: Multi-Modal Fusion Node (75 minutes)

Create a node that fuses information from different modalities:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String
from multimodal_msgs.msg import ObjectDetection, Gesture
from geometry_msgs.msg import Point
from humanoid_msgs.msg import GPTRequest, GPTResponse

class MultiModalFusionNode(Node):
    def __init__(self):
        super().__init__('multi_modal_fusion_node')
        
        # Store context from different modalities
        self.last_audio_command = ""
        self.last_gesture = None
        self.last_object_detections = []
        self.last_gesture_detection = None
        
        # Timestamps for synchronization
        self.audio_timestamp = None
        self.gesture_timestamp = None
        self.vision_timestamp = None
        
        # Publishers
        self.fused_command_pub = self.create_publisher(GPTRequest, 'fused_commands', 10)
        
        # Subscribers
        self.audio_sub = self.create_subscription(
            String, 'audio_commands', self.audio_callback, 10
        )
        self.gesture_sub = self.create_subscription(
            Gesture, 'hand_gestures', self.gesture_callback, 10
        )
        self.object_detection_sub = self.create_subscription(
            ObjectDetection, 'object_detections', self.object_detection_callback, 10
        )
        
        # Timer for fusing modalities periodically
        self.fusion_timer = self.create_timer(1.0, self.fusion_timer_callback)
        
        self.get_logger().info('Multi-Modal Fusion Node initialized')

    def audio_callback(self, msg):
        """Handle audio commands"""
        self.last_audio_command = msg.data
        self.audio_timestamp = self.get_clock().now()
        self.get_logger().info(f"Audio command received: {msg.data}")

    def gesture_callback(self, msg):
        """Handle gesture detections"""
        self.last_gesture_detection = msg
        self.gesture_timestamp = self.get_clock().now()
        self.get_logger().info(f"Gesture detected: {msg.description}")

    def object_detection_callback(self, msg):
        """Handle object detections"""
        # Add to list of recent detections
        self.last_object_detections.append(msg)
        
        # Keep only recent detections (last 5 seconds)
        current_time = self.get_clock().now()
        self.last_object_detections = [
            detection for detection in self.last_object_detections
            if (current_time - detection.header.stamp).nanoseconds < 5e9  # 5 seconds
        ]
        
        self.vision_timestamp = current_time
        self.get_logger().info(f"Object detected: {msg.class_name}")

    def fusion_timer_callback(self):
        """Periodically fuse modalities and create unified command"""
        # Check if we have recent data from multiple modalities
        current_time = self.get_clock().now()
        
        has_recent_audio = (
            self.audio_timestamp and 
            (current_time - self.audio_timestamp).nanoseconds < 5e9  # 5 seconds
        )
        has_recent_gesture = (
            self.gesture_timestamp and 
            (current_time - self.gesture_timestamp).nanoseconds < 3e9  # 3 seconds
        )
        has_recent_vision = (
            self.vision_timestamp and 
            (current_time - self.vision_timestamp).nanoseconds < 2e9  # 2 seconds
        )
        
        # Create context based on available modalities
        context_parts = []
        
        if has_recent_audio:
            context_parts.append(f"Audio command: '{self.last_audio_command}'")
        
        if has_recent_gesture and self.last_gesture_detection:
            context_parts.append(f"Gesture: {self.last_gesture_detection.description}")
        
        if has_recent_vision and self.last_object_detections:
            # Get unique object classes detected recently
            detected_objects = set(detection.class_name for detection in self.last_object_detections)
            context_parts.append(f"Detected objects: {', '.join(detected_objects)}")
        
        if context_parts:
            # Create a unified command context
            unified_context = " | ".join(context_parts)
            self.get_logger().info(f"Fused context: {unified_context}")
            
            # Create and publish fused command
            fused_command = GPTRequest()
            fused_command.header.stamp = current_time.to_msg()
            fused_command.command = self.last_audio_command if has_recent_audio else "No audio command"
            fused_command.context = unified_context
            fused_command.id = f"fusion_{current_time.nanoseconds}"
            
            self.fused_command_pub.publish(fused_command)

def main(args=None):
    rclpy.init(args=args)
    node = MultiModalFusionNode()
    
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

### Task 3: Multi-Modal Output System (60 minutes)

Create a node that can respond through multiple output modalities:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from multimodal_msgs.msg import Gesture
from humanoid_msgs.msg import GPTResponse
from geometry_msgs.msg import Twist
import time

class MultiModalOutputNode(Node):
    def __init__(self):
        super().__init__('multi_modal_output_node')
        
        # Publishers for different output modalities
        self.speech_pub = self.create_publisher(String, 'text_to_speech', 10)
        self.display_pub = self.create_publisher(String, 'display_output', 10)
        self.motion_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.gpt_response_sub = self.create_subscription(
            GPTResponse, 'llm_responses', self.response_callback, 10
        )
        
        # For demonstration purposes, we'll also respond to gestures
        self.gesture_sub = self.create_subscription(
            Gesture, 'hand_gestures', self.gesture_response_callback, 10
        )
        
        self.get_logger().info('Multi-Modal Output Node initialized')

    def response_callback(self, msg):
        """Handle LLM responses and generate multi-modal output"""
        if msg.is_error:
            self.get_logger().error(f"LLM Error: {msg.response}")
            return
        
        # Parse the structured response
        try:
            structured_data = json.loads(msg.structured_response)
            
            # Generate speech response
            if "text_response" in structured_data:
                speech_msg = String()
                speech_msg.data = structured_data["text_response"]
                self.speech_pub.publish(speech_msg)
                self.get_logger().info(f"Speaking: {speech_msg.data}")
            
            # Generate display response
            display_msg = String()
            display_msg.data = f"Response: {structured_data.get('text_response', msg.response)}"
            self.display_pub.publish(display_msg)
            
            # Generate motion response based on content
            self.generate_motion_response(structured_data)
            
        except json.JSONDecodeError:
            # If not structured, respond as text
            speech_msg = String()
            speech_msg.data = msg.response
            self.speech_pub.publish(speech_msg)
            self.get_logger().info(f"Speaking: {speech_msg.data}")

    def gesture_response_callback(self, msg):
        """Handle gesture inputs and provide appropriate responses"""
        if "pointing" in msg.description.lower():
            # Example: If someone points, acknowledge with speech and motion
            speech_msg = String()
            speech_msg.data = "I see you're pointing. How can I help you?"
            self.speech_pub.publish(speech_msg)
            
            # Turn slightly toward the person (simplified)
            motion_msg = Twist()
            motion_msg.angular.z = 0.2  # Turn slightly
            self.motion_pub.publish(motion_msg)
            
            self.get_logger().info("Responded to pointing gesture")

    def generate_motion_response(self, structured_data):
        """Generate motion responses based on the structured command"""
        # This is a simplified example
        # In a real robot, you would have more sophisticated motion patterns
        
        if "navigate" in str(structured_data).lower():
            # Create a simple motion command for navigation
            motion_msg = Twist()
            motion_msg.linear.x = 0.2  # Move forward slowly
            self.motion_pub.publish(motion_msg)
            time.sleep(1)  # Move for 1 second
            motion_msg.linear.x = 0.0  # Stop
            self.motion_pub.publish(motion_msg)
        
        elif "gesture" in str(structured_data).lower() or "wave" in str(structured_data).lower():
            # Example: If command involves waving, robot could acknowledge with motion
            # This would involve more complex motion planning in a real robot
            self.get_logger().info("Would execute waving motion in real robot")

def main(args=None):
    rclpy.init(args=args)
    node = MultiModalOutputNode()
    
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

### Task 4: Integration and Testing (45 minutes)

Create a launch file to bring up the complete multi-modal system:

```python
# multimodal_input_node/launch/multi_modal_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'openai_api_key',
            default_value='',
            description='OpenAI API key for LLM access'
        ),
        
        # Audio processing node
        Node(
            package='multimodal_input_node',
            executable='audio_processor',
            name='audio_processor_node'
        ),
        
        # Vision processing node
        Node(
            package='multimodal_input_node',
            executable='vision_processor',
            name='vision_processor_node'
        ),
        
        # Gesture processing node (uses MediaPipe for hand tracking)
        Node(
            package='multimodal_input_node',
            executable='gesture_processor',
            name='gesture_processor_node'
        ),
        
        # Multi-modal fusion node
        Node(
            package='multimodal_input_node',
            executable='multi_modal_fusion',
            name='multi_modal_fusion_node'
        ),
        
        # Multi-modal output node
        Node(
            package='multimodal_input_node',
            executable='multi_modal_output',
            name='multi_modal_output_node'
        ),
        
        # (Optional) LLM planner node using the key
        Node(
            package='llm_planning_node',  # Assuming this package exists from previous exercise
            executable='structured_llm_planner',
            name='structured_llm_planner_node',
            parameters=[{
                'openai_api_key': LaunchConfiguration('openai_api_key')
            }]
        )
    ])
```

### Task 5: Advanced Fusion Scenarios (90 minutes)

Implement more sophisticated fusion scenarios that demonstrate complex multi-modal understanding:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String
from multimodal_msgs.msg import ObjectDetection, Gesture
from humanoid_msgs.msg import GPTRequest
from builtin_interfaces.msg import Time
from collections import deque

class AdvancedMultiModalFusionNode(Node):
    def __init__(self):
        super().__init__('advanced_multi_modal_fusion_node')
        
        # Store temporal history of modalities
        self.audio_history = deque(maxlen=10)  # Last 10 audio commands
        self.gesture_history = deque(maxlen=10)  # Last 10 gestures
        self.vision_history = deque(maxlen=20)  # Last 20 object detections
        
        # Context for conversation and interaction
        self.interaction_context = {
            'current_task': None,
            'user_attention': None,
            'environment_state': {},
            'last_interaction_time': None
        }
        
        # Publishers
        self.fused_command_pub = self.create_publisher(GPTRequest, 'advanced_fused_commands', 10)
        
        # Subscribers
        self.audio_sub = self.create_subscription(
            String, 'audio_commands', self.audio_callback, 10
        )
        self.gesture_sub = self.create_subscription(
            Gesture, 'hand_gestures', self.gesture_callback, 10
        )
        self.object_detection_sub = self.create_subscription(
            ObjectDetection, 'object_detections', self.object_detection_callback, 10
        )
        
        # Timer for periodic fusion and context updates
        self.context_update_timer = self.create_timer(2.0, self.context_update_callback)
        
        self.get_logger().info('Advanced Multi-Modal Fusion Node initialized')

    def audio_callback(self, msg):
        """Handle audio commands with context"""
        audio_entry = {
            'timestamp': self.get_clock().now(),
            'command': msg.data
        }
        self.audio_history.append(audio_entry)
        
        self.get_logger().info(f"Audio command received: {msg.data}")
        
        # Process the audio in the context of other modalities
        self.process_fusion()

    def gesture_callback(self, msg):
        """Handle gesture detections"""
        gesture_entry = {
            'timestamp': msg.header.stamp,
            'type': msg.type,
            'description': msg.description
        }
        self.gesture_history.append(gesture_entry)
        
        self.get_logger().info(f"Gesture detected: {msg.description}")
        
        # Process the gesture in the context of other modalities
        self.process_fusion()

    def object_detection_callback(self, msg):
        """Handle object detections"""
        obj_entry = {
            'timestamp': msg.header.stamp,
            'class_name': msg.class_name,
            'confidence': msg.confidence,
            'position': {
                'x': msg.bounding_box.x_offset,
                'y': msg.bounding_box.y_offset,
                'w': msg.bounding_box.width,
                'h': msg.bounding_box.height
            }
        }
        self.vision_history.append(obj_entry)
        
        # Update environment state
        self.interaction_context['environment_state'][msg.class_name] = {
            'timestamp': msg.header.stamp,
            'position': obj_entry['position'],
            'confidence': msg.confidence
        }
        
        self.get_logger().info(f"Object detected: {msg.class_name}")
        
        # Process the object detection in the context of other modalities
        self.process_fusion()

    def process_fusion(self):
        """Process fusion based on recent multi-modal inputs"""
        # Get recent inputs (within a time window)
        recent_audio = self.get_recent_entries(self.audio_history, 5.0)
        recent_gestures = self.get_recent_entries(self.gesture_history, 3.0)
        recent_objects = self.get_recent_entries(self.vision_history, 2.0)
        
        # Only proceed if we have inputs from at least 2 modalities
        modality_count = 0
        if recent_audio:
            modality_count += 1
        if recent_gestures:
            modality_count += 1
        if recent_objects:
            modality_count += 1
        
        if modality_count < 2:
            return  # Not enough modalities for meaningful fusion
        
        # Create a rich context combining all modalities
        context = self.create_fusion_context(recent_audio, recent_gestures, recent_objects)
        
        # Create a fused command
        fused_command = GPTRequest()
        fused_command.header.stamp = self.get_clock().now().to_msg()
        fused_command.command = self.extract_intent_from_modalities(recent_audio, recent_gestures)
        fused_command.context = json.dumps(context)
        fused_command.id = f"adv_fusion_{self.get_clock().now().nanoseconds}"
        
        self.fused_command_pub.publish(fused_command)
        self.get_logger().info(f"Published fused command with {modality_count} modalities")

    def get_recent_entries(self, history_deque, time_window_sec):
        """Get entries from the history that are within the time window"""
        if not history_deque:
            return []
        
        current_time = self.get_clock().now()
        recent_entries = []
        
        for entry in list(history_deque):  # Convert to list to avoid iterator issues
            if hasattr(entry, 'timestamp'):
                timestamp = entry['timestamp'] if isinstance(entry['timestamp'], Time) else entry['timestamp'].to_msg()
                entry_time = self.node.get_clock().now() if hasattr(self, 'node') else self.get_clock().now()
                
                # Calculate time difference
                if isinstance(entry['timestamp'], Time):
                    time_diff = (current_time.nanoseconds - 
                                (entry['timestamp'].sec * 1e9 + entry['timestamp'].nanosec)) / 1e9
                else:
                    time_diff = float((current_time - entry['timestamp']).nanoseconds) / 1e9
                
                if time_diff <= time_window_sec:
                    recent_entries.append(entry)
            else:
                # If no timestamp, assume it's recent
                recent_entries.append(entry)
        
        return recent_entries

    def create_fusion_context(self, recent_audio, recent_gestures, recent_objects):
        """Create a rich context by fusing information from multiple modalities"""
        context = {
            'temporal_context': {
                'last_audio_time': str(recent_audio[-1]['timestamp']) if recent_audio else None,
                'last_gesture_time': str(recent_gestures[-1]['timestamp']) if recent_gestures else None,
                'last_object_time': str(recent_objects[-1]['timestamp']) if recent_objects else None
            },
            'audio_context': [entry['command'] for entry in recent_audio],
            'gesture_context': [entry['description'] for entry in recent_gestures],
            'vision_context': [
                {
                    'class': entry['class_name'],
                    'position': entry['position'],
                    'confidence': entry['confidence']
                } for entry in recent_objects
            ],
            'environment_context': self.interaction_context['environment_state'],
            'current_task': self.interaction_context['current_task']
        }
        
        # Add spatial relationships if we have object and gesture data
        if recent_objects and recent_gestures:
            context['spatial_relationships'] = self.analyze_spatial_relationships(
                recent_objects, recent_gestures
            )
        
        return context

    def analyze_spatial_relationships(self, objects, gestures):
        """Analyze spatial relationships between detected objects and gestures"""
        relationships = []
        
        # Example: If a pointing gesture was detected, relate it to nearby objects
        for gesture in gestures:
            if 'pointing' in gesture['description'].lower():
                for obj in objects:
                    # Calculate if the object is in the general direction of the pointing gesture
                    # This is simplified - in practice, you'd need pose estimation
                    relationships.append({
                        'gesture': gesture['description'],
                        'related_object': obj['class_name'],
                        'confidence': 0.7  # Estimated confidence
                    })
        
        return relationships

    def extract_intent_from_modalities(self, recent_audio, recent_gestures):
        """Extract the user's intent from multiple modalities"""
        if not recent_audio and not recent_gestures:
            return "No clear intent detected"
        
        # Start with audio command
        primary_intent = recent_audio[-1]['command'] if recent_audio else ""
        
        # Enhance with gesture context if applicable
        if recent_gestures:
            gesture_desc = recent_gestures[-1]['description']
            if 'pointing' in gesture_desc.lower() and primary_intent:
                primary_intent += f" (user is pointing at something)"
            elif 'waving' in gesture_desc.lower():
                primary_intent = f"Respond to wave: {primary_intent}" if primary_intent else "Wave back at user"
        
        return primary_intent

    def context_update_callback(self):
        """Periodically update interaction context"""
        # Update the last interaction time
        self.interaction_context['last_interaction_time'] = self.get_clock().now()
        
        # Clean up old entries in history
        current_time = self.get_clock().now()
        
        # Note: The implementation of time-based filtering would need to be completed
        # based on the specific ROS 2 version and Time handling approach

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedMultiModalFusionNode()
    
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

1. **Complete source code** for your multi-modal system
2. **Launch files** to start your complete system
3. **Documentation** explaining how your multi-modal fusion works
4. **Test results** demonstrating integration of multiple modalities
5. **Reflection** on challenges of multi-modal integration and synchronization

## Evaluation Criteria

- **Functionality** (30%): Your system correctly processes and fuses multiple input modalities
- **Integration** (25%): Effective coordination between different modalities
- **Code Quality** (20%): Well-structured, documented, and maintainable code
- **ROS Integration** (15%): Proper use of ROS concepts and message types
- **Synchronization** (10%): Appropriate handling of temporal relationships between modalities

## Troubleshooting

Common issues and solutions:

1. **Synchronization Problems**: Different modalities operate at different rates; implement temporal buffering to align inputs.

2. **Resource Competition**: Multiple nodes processing sensor data simultaneously; implement efficient processing and resource management.

3. **False Positives**: Gesture or object detection producing incorrect results; implement confidence thresholds and validation.

4. **Timing Issues**: Delays in processing leading to outdated context; implement appropriate timeouts and context decay.

## Extensions

For advanced students, consider implementing:

1. **Attention Mechanism**: Have the robot focus on the most relevant modality based on the context.
2. **Learning from Interaction**: Adapt fusion strategies based on successful/unsuccessful interactions.
3. **Cross-Modal Validation**: Verify information from one modality against another (e.g., verify audio "left" with visual spatial information).
4. **Uncertainty Handling**: Model and reason about uncertainty in different modalities.

## Conclusion

This exercise has provided hands-on experience with developing a multi-modal system for human-robot interaction. You've learned how to process inputs from different sensory modalities (audio, vision), fuse them into a coherent understanding, and respond through multiple output channels. This forms a critical foundation for creating robots that can naturally interact with humans in real-world environments.