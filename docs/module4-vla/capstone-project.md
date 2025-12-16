# Capstone Project: Autonomous Humanoid Demonstration

## Overview

The capstone project integrates all components learned in the Physical AI & Humanoid Robotics curriculum into a comprehensive autonomous humanoid demonstration. Students will implement a complete system that combines ROS 2 fundamentals, simulation environments, AI perception, and vision-language-action capabilities.

## Project Goals

1. Create an autonomous humanoid robot that can understand and execute natural language commands
2. Integrate perception, navigation, and manipulation capabilities
3. Demonstrate multimodal interaction (voice, vision, gesture)
4. Showcase sim-to-real transfer potential
5. Validate the entire robotics curriculum in one cohesive project

## System Requirements

### Hardware Requirements
- Compatible computer with NVIDIA GPU (RTX 4070 Ti or better recommended)
- Ubuntu 22.04 LTS
- (Optional) Physical humanoid robot or realistic simulation environment

### Software Requirements
- ROS 2 Iron
- NVIDIA Isaac Sim (for advanced simulation)
- Gazebo for physics simulation
- Unity Robotics Hub (for visualization)
- OpenAI API access for Whisper and GPT models
- Docusaurus for documentation

## Architecture Overview

```
User Natural Language Command
         ↓
Speech-to-Text (Whisper)
         ↓
Natural Language Understanding (GPT)
         ↓
Intent Recognition & Task Planning
         ↓
Action Sequencing (Navigate → Perceive → Manipulate → Respond)
         ↓
ROS 2 Execution Layer
         ↓
Hardware/Simulation Control
         ↓
Robot Actions
         ↓
User Feedback
```

## Implementation Phases

### Phase 1: Environment Setup and Integration

1. Set up the complete simulation environment with:
   - Humanoid robot model
   - Interactive environment (rooms, furniture, objects)
   - Sensor configurations (cameras, LiDAR, IMU)

2. Integrate ROS 2 nodes for:
   - Navigation stack (Nav2)
   - Manipulation stack
   - Perception pipeline (Isaac ROS)
   - Audio processing (Whisper)

```python
#!/usr/bin/env python3
# capstone_system_integration.py

import rospy
import actionlib
from humanoid_msgs.msg import GPTRequest, GPTResponse
from humanoid_msgs.msg import VLACommand
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import openai
import json
import uuid

class CapstoneIntegrationNode:
    def __init__(self):
        rospy.init_node('capstone_integration')
        
        # Initialize components
        self.openai_api_key = rospy.get_param('~openai_api_key', '')
        openai.api_key = self.openai_api_key
        
        # Publishers
        self.voice_command_pub = rospy.Publisher('/voice_commands', String, queue_size=10)
        self.gpt_request_pub = rospy.Publisher('/gpt_requests', GPTRequest, queue_size=10)
        
        # Subscribers
        self.voice_input_sub = rospy.Subscriber('/audio_input', String, self.voice_input_callback)
        self.gpt_response_sub = rospy.Subscriber('/gpt_responses', GPTResponse, self.gpt_response_callback)
        
        # Action clients
        self.nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()
        
        rospy.loginfo("Capstone Integration Node initialized")
    
    def voice_input_callback(self, msg):
        """Process voice input and send to GPT for understanding"""
        # Send voice command to GPT for processing
        gpt_request = GPTRequest()
        gpt_request.header.stamp = rospy.Time.now()
        gpt_request.command = msg.data
        gpt_request.context = self.get_environment_context()
        gpt_request.id = str(uuid.uuid4())
        
        self.gpt_request_pub.publish(gpt_request)
    
    def get_environment_context(self):
        """Get current environment context for GPT"""
        # This would integrate with perception systems to get context
        context = {
            "detected_objects": self.get_detected_objects(),
            "robot_pose": self.get_robot_pose(),
            "navigation_goals": self.get_navigation_goals()
        }
        return json.dumps(context)
    
    def get_detected_objects(self):
        """Get detected objects from perception system"""
        # Placeholder - in practice, this would interface with your perception system
        return ["chair", "table", "cup", "refrigerator"]
    
    def get_robot_pose(self):
        """Get current robot pose"""
        # Placeholder - in practice, this would get actual robot pose
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    def get_navigation_goals(self):
        """Get known navigation goals in environment"""
        # Known locations in the environment
        return ["kitchen", "living_room", "bedroom", "office"]
    
    def gpt_response_callback(self, msg):
        """Process GPT response and convert to robot actions"""
        try:
            # Parse GPT response for structured commands
            structured_response = json.loads(msg.structured_response)
            
            if "action_sequence" in structured_response:
                # Execute action sequence
                self.execute_action_sequence(structured_response["action_sequence"])
            elif "text_response" in structured_response:
                # Just provide text feedback
                self.provide_feedback(structured_response["text_response"])
                
        except json.JSONDecodeError:
            # If not JSON, treat as text response
            self.provide_feedback(msg.response)
    
    def execute_action_sequence(self, action_sequence):
        """Execute a sequence of robot actions"""
        for action in action_sequence:
            action_type = action.get("type")
            
            if action_type == "navigate":
                self.execute_navigation(action)
            elif action_type == "manipulate":
                self.execute_manipulation(action)
            elif action_type == "perceive":
                self.execute_perception(action)
    
    def execute_navigation(self, action):
        """Execute navigation action"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Extract position from action
        pos = action.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
        goal.target_pose.pose.position.x = pos["x"]
        goal.target_pose.pose.position.y = pos["y"]
        goal.target_pose.pose.position.z = pos["z"]
        
        # Simple orientation for now
        goal.target_pose.pose.orientation.w = 1.0
        
        # Send goal to navigation system
        self.nav_client.send_goal(goal)
        rospy.loginfo(f"Navigating to position: {pos}")
        
        # Wait for result (timeout after 60 seconds)
        finished_within_time = self.nav_client.wait_for_result(rospy.Duration(60))
        
        if not finished_within_time:
            self.nav_client.cancel_goal()
            rospy.logwarn("Navigation action timed out")
        else:
            state = self.nav_client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Navigation succeeded")
            else:
                rospy.logwarn(f"Navigation failed with state: {state}")
    
    def execute_manipulation(self, action):
        """Execute manipulation action (placeholder)"""
        object_name = action.get("object", "unknown")
        rospy.loginfo(f"Attempting to manipulate object: {object_name}")
        # In practice, this would interface with your manipulation stack
    
    def execute_perception(self, action):
        """Execute perception action (placeholder)"""
        object_name = action.get("object", "unknown")
        rospy.loginfo(f"Performing perception for object: {object_name}")
        # In practice, this would interface with your perception stack
    
    def provide_feedback(self, text):
        """Provide feedback to user"""
        # This could be speech synthesis, text display, etc.
        rospy.loginfo(f"Robot says: {text}")
    
    def run(self):
        """Run the capstone integration node"""
        rospy.spin()

if __name__ == '__main__':
    node = CapstoneIntegrationNode()
    node.run()
```

### Phase 2: Voice Command Integration

Integrate Whisper for speech-to-text and create the voice command system (T110):

```python
#!/usr/bin/env python3
# voice_command_system.py

import rospy
import whisper
import pyaudio
import numpy as np
from std_msgs.msg import String
import threading
import queue

class VoiceCommandSystem:
    def __init__(self):
        rospy.init_node('voice_command_system')
        
        # Initialize Whisper model (using 'base' model for efficiency)
        rospy.loginfo("Loading Whisper model...")
        self.model = whisper.load_model("base")
        
        # Publishers
        self.voice_command_pub = rospy.Publisher('/voice_commands', String, queue_size=10)
        
        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 5  # How long to record when triggered
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start audio recording thread
        self.recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()
        
        rospy.loginfo("Voice Command System initialized")
    
    def record_audio(self):
        """Continuously record audio and detect speech"""
        while not rospy.is_shutdown():
            if not self.recording:
                # Record a small chunk to check for speech
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )
                
                # Read initial frames to detect if speech is starting
                frames = []
                for i in range(0, int(self.rate / self.chunk * 0.5)):  # 0.5 seconds
                    data = stream.read(self.chunk)
                    frames.append(data)
                
                # Convert to numpy array to analyze volume
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                avg_amplitude = np.mean(np.abs(audio_data))
                
                # If amplitude is above threshold, start recording
                if avg_amplitude > 1000:  # Threshold for speech detection
                    rospy.loginfo("Speech detected, starting recording...")
                    self.recording = True
                    self.capture_speech(stream)
                else:
                    stream.stop_stream()
                    stream.close()
            
            rospy.sleep(0.1)  # Small delay to prevent busy waiting
    
    def capture_speech(self, stream):
        """Capture speech once detected"""
        frames = []
        silence_count = 0
        max_silence_frames = int(self.rate / self.chunk * 1.0)  # 1 second of silence to stop
        
        try:
            while self.recording and not rospy.is_shutdown():
                data = stream.read(self.chunk)
                frames.append(data)
                
                # Analyze amplitude to detect end of speech
                audio_data = np.frombuffer(data, dtype=np.int16)
                amplitude = np.mean(np.abs(audio_data))
                
                if amplitude < 500:  # Low amplitude = silence
                    silence_count += 1
                    if silence_count > max_silence_frames:
                        break
                else:
                    silence_count = 0  # Reset if we hear sound again
        
        except Exception as e:
            rospy.logerr(f"Error in audio capture: {e}")
        
        finally:
            stream.stop_stream()
            stream.close()
            
            if frames:
                # Convert to numpy array for Whisper
                audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                # Transcribe with Whisper
                result = self.model.transcribe(audio_float)
                text = result["text"].strip()
                
                if text:
                    rospy.loginfo(f"Recognized: {text}")
                    
                    # Publish the recognized text
                    command_msg = String()
                    command_msg.data = text
                    self.voice_command_pub.publish(command_msg)
            
            self.recording = False
    
    def run(self):
        """Run the voice command system"""
        rospy.spin()

if __name__ == '__main__':
    system = VoiceCommandSystem()
    system.run()
```

### Phase 3: VLA Pipeline Prototype (T111)

Create the VLA (Vision-Language-Action) pipeline prototype:

```python
#!/usr/bin/env python3
# vla_pipeline_prototype.py

import rospy
import openai
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
import json
import torch
import uuid
import threading

class VLAPipelinePrototype:
    def __init__(self):
        rospy.init_node('vla_pipeline_prototype')
        
        # Initialize components
        self.openai_api_key = rospy.get_param('~openai_api_key', '')
        openai.api_key = self.openai_api_key
        
        # Initialize CV bridge for image processing
        self.bridge = CvBridge()
        
        # Load vision model for object detection (using YOLOv5 as example)
        self.vision_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.voice_command_sub = rospy.Subscriber('/voice_commands', String, self.voice_command_callback)
        self.gpt_request_pub = rospy.Publisher('/gpt_requests', GPTRequest, queue_size=10)
        self.gpt_response_sub = rospy.Subscriber('/gpt_responses', GPTResponse, self.gpt_response_callback)
        
        # Navigation client
        self.nav_client = SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()
        
        # Internal state
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.pending_commands = {}  # Track pending voice commands
        
        rospy.loginfo("VLA Pipeline Prototype initialized")
    
    def image_callback(self, img_msg):
        """Process incoming images from camera"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            with self.image_lock:
                self.latest_image = cv_image
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def voice_command_callback(self, cmd_msg):
        """Process incoming voice commands"""
        try:
            # Get current image for context
            with self.image_lock:
                if self.latest_image is not None:
                    # Run object detection on the image
                    results = self.vision_model(self.latest_image)
                    detections = results.pandas().xyxy[0]  # Get detections in pandas format
                else:
                    detections = None
            
            # Create context for GPT including vision data
            context = {
                "detected_objects": self.process_detections(detections) if detections is not None else [],
                "environment": "indoor",  # This would come from a mapping system in practice
                "robot_capabilities": ["navigation", "manipulation", "speech_response"]
            }
            
            # Create GPT request
            gpt_request = GPTRequest()
            gpt_request.header.stamp = rospy.Time.now()
            gpt_request.command = cmd_msg.data
            gpt_request.context = json.dumps(context)
            gpt_request.id = str(uuid.uuid4())
            
            # Store as pending command
            self.pending_commands[gpt_request.id] = {
                'command': cmd_msg.data,
                'timestamp': rospy.Time.now()
            }
            
            # Publish request
            self.gpt_request_pub.publish(gpt_request)
            rospy.loginfo(f"Sent VLA request for command: {cmd_msg.data}")
            
        except Exception as e:
            rospy.logerr(f"Error processing voice command: {e}")
    
    def process_detections(self, detections):
        """Process vision model detections into structured format"""
        objects = []
        for _, detection in detections.iterrows():
            obj = {
                'name': detection['name'],
                'confidence': float(detection['confidence']),
                'bbox': {
                    'xmin': int(detection['xmin']),
                    'ymin': int(detection['ymin']),
                    'xmax': int(detection['xmax']),
                    'ymax': int(detection['ymax'])
                },
                'center': (
                    int((detection['xmin'] + detection['xmax']) / 2),
                    int((detection['ymin'] + detection['ymax']) / 2)
                )
            }
            objects.append(obj)
        return objects
    
    def gpt_response_callback(self, response_msg):
        """Process GPT response and execute actions"""
        if response_msg.request_id in self.pending_commands:
            command_data = self.pending_commands.pop(response_msg.request_id)
            
            try:
                # Attempt to parse structured response
                try:
                    structured_data = json.loads(response_msg.structured_response)
                except json.JSONDecodeError:
                    # If not JSON, try to extract from text response
                    structured_data = self.extract_structured_data(response_msg.response)
                
                # Execute actions based on GPT response
                self.execute_vla_actions(structured_data, response_msg.response)
                
            except Exception as e:
                rospy.logerr(f"Error executing VLA actions: {e}")
    
    def extract_structured_data(self, text_response):
        """Extract structured action data from text response (simplified)"""
        # This is a simplified approach - in practice, you'd want more robust parsing
        # or structure the GPT call to require JSON output
        
        # Look for common action patterns in the response
        import re
        
        # Example: "Navigate to the kitchen" -> {"action": "navigate", "target": "kitchen"}
        if "navigate" in text_response.lower() or "go to" in text_response.lower():
            # Extract location
            location_match = re.search(r"to the (\w+)", text_response)
            if location_match:
                return {
                    "action": "navigate",
                    "target": location_match.group(1)
                }
        
        # Example: "Pick up the red cup" -> {"action": "grasp", "object": "red cup"}
        elif "pick up" in text_response.lower() or "grasp" in text_response.lower():
            # Extract object
            obj_match = re.search(r"(?:pick up|grasp) the (.+)", text_response)
            if obj_match:
                return {
                    "action": "grasp",
                    "object": obj_match.group(1)
                }
        
        # Default: just return text response
        return {
            "action": "respond",
            "text": text_response
        }
    
    def execute_vla_actions(self, structured_data, text_response):
        """Execute actions based on structured data from GPT"""
        action_type = structured_data.get('action', 'respond')
        
        if action_type == 'navigate':
            target = structured_data.get('target', 'unknown')
            self.execute_navigation(target)
        elif action_type == 'grasp':
            obj = structured_data.get('object', 'unknown')
            self.execute_grasp(obj)
        elif action_type == 'respond':
            response_text = structured_data.get('text', text_response)
            self.provide_response(response_text)
        else:
            rospy.loginfo(f"Unknown action type: {action_type}, responding with: {text_response}")
            self.provide_response(text_response)
    
    def execute_navigation(self, target_location):
        """Navigate to a specific location"""
        # This would typically interface with a semantic map to get coordinates
        # For this prototype, we'll use some predefined locations
        locations = {
            'kitchen': {'x': 3.0, 'y': 1.0},
            'living room': {'x': 0.0, 'y': 0.0},
            'bedroom': {'x': -2.0, 'y': 2.0},
            'office': {'x': 1.0, 'y': -3.0}
        }
        
        if target_location.lower() in locations:
            target = locations[target_location.lower()]
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = target['x']
            goal.target_pose.pose.position.y = target['y']
            goal.target_pose.pose.orientation.w = 1.0
            
            rospy.loginfo(f"Navigating to {target_location} at ({target['x']}, {target['y']})")
            self.nav_client.send_goal(goal)
            
            # Wait for result with timeout
            finished = self.nav_client.wait_for_result(rospy.Duration(60.0))
            
            if finished:
                state = self.nav_client.get_state()
                if state == 3:  # SUCCEEDED
                    rospy.loginfo(f"Successfully reached {target_location}")
                    self.provide_response(f"I have reached the {target_location}")
                else:
                    rospy.logwarn(f"Failed to reach {target_location}, state: {state}")
                    self.provide_response(f"I couldn't reach the {target_location}")
            else:
                self.nav_client.cancel_goal()
                rospy.logwarn(f"Navigation to {target_location} timed out")
                self.provide_response(f"I'm having trouble reaching the {target_location}")
        else:
            rospy.logwarn(f"Unknown location: {target_location}")
            self.provide_response(f"I don't know where the {target_location} is located")
    
    def execute_grasp(self, target_object):
        """(Placeholder) Grasp a specific object"""
        # In a real implementation, this would interface with a manipulation stack
        # to detect the object in the environment and plan a grasping motion
        rospy.loginfo(f"Attempting to grasp: {target_object}")
        
        # Check if the object is visible in the latest image
        with self.image_lock:
            if self.latest_image is not None:
                # Run detection to see if the target object is visible
                results = self.vision_model(self.latest_image)
                detections = results.pandas().xyxy[0]
                
                # Look for the target object
                found_obj = False
                for _, detection in detections.iterrows():
                    if target_object.lower() in detection['name'].lower():
                        found_obj = True
                        rospy.loginfo(f"Found {target_object} in the image")
                        break
                
                if found_obj:
                    rospy.loginfo(f"Successfully located {target_object}, would attempt grasp in real robot")
                    self.provide_response(f"I found the {target_object} and attempted to grasp it")
                else:
                    rospy.logwarn(f"Could not find {target_object} in the current view")
                    self.provide_response(f"I couldn't find the {target_object} in my view")
            else:
                rospy.logwarn("No image available for object detection")
                self.provide_response("I need to see the object to grasp it")
    
    def provide_response(self, text):
        """Provide text response (would interface with TTS in real system)"""
        rospy.loginfo(f"Robot response: {text}")
        # In a real implementation, this would interface with a text-to-speech system
    
    def run(self):
        """Run the VLA pipeline"""
        rospy.loginfo("VLA Pipeline Prototype running...")
        rospy.spin()

if __name__ == '__main__':
    vla_pipeline = VLAPipelinePrototype()
    vla_pipeline.run()
```

### Phase 4: Testing and Validation

Create a comprehensive test suite for the capstone project:

```python
#!/usr/bin/env python3
# capstone_test_suite.py

import rospy
import unittest
from std_msgs.msg import String
from humanoid_msgs.msg import GPTResponse
from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalStatus
import time

class CapstoneTestSuite(unittest.TestCase):
    def setUp(self):
        rospy.init_node('capstone_tester', anonymous=True)
        
        # Publishers for sending test commands
        self.voice_cmd_pub = rospy.Publisher('/voice_commands', String, queue_size=10)
        
        # Subscribers for monitoring system responses
        self.gpt_response_sub = rospy.Subscriber('/gpt_responses', GPTResponse, self.gpt_response_callback)
        self.nav_result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.nav_result_callback)
        
        # Test state
        self.last_gpt_response = None
        self.last_nav_result = None
        self.response_received = False
        self.nav_completed = False
    
    def gpt_response_callback(self, msg):
        self.last_gpt_response = msg
        self.response_received = True
    
    def nav_result_callback(self, msg):
        self.last_nav_result = msg
        self.nav_completed = True
    
    def test_voice_command_processing(self):
        """Test that voice commands are properly processed"""
        # Send a test voice command
        cmd = String()
        cmd.data = "Go to the kitchen"
        self.voice_cmd_pub.publish(cmd)
        
        # Wait for response
        timeout = time.time() + 60*2  # 2 minute timeout
        while not self.response_received and time.time() < timeout:
            time.sleep(0.1)
        
        self.assertTrue(self.response_received, "GPT response not received within timeout")
        self.assertIsNotNone(self.last_gpt_response, "No GPT response received")
        self.assertGreater(len(self.last_gpt_response.response), 0, "Empty GPT response")
    
    def test_navigation_execution(self):
        """Test that navigation commands are executed"""
        # This test requires a navigation goal to be sent
        # For the purpose of this test, we assume that our VLA system
        # will trigger navigation based on voice commands
        
        # Send a navigation command
        cmd = String()
        cmd.data = "Navigate to the living room"
        self.voice_cmd_pub.publish(cmd)
        
        # Wait for navigation result
        timeout = time.time() + 60*3  # 3 minute timeout for navigation
        while not self.nav_completed and time.time() < timeout:
            time.sleep(0.1)
        
        self.assertTrue(self.nav_completed, "Navigation result not received within timeout")
        if self.last_nav_result:
            self.assertEqual(self.last_nav_result.status.status, GoalStatus.SUCCEEDED, 
                           "Navigation did not succeed")
    
    def test_vla_integration(self):
        """Test full VLA integration"""
        # Send a complex command that involves vision, language, and action
        cmd = String()
        cmd.data = "Find the red cup and bring it to me"
        self.voice_cmd_pub.publish(cmd)
        
        # Wait for GPT response
        timeout = time.time() + 60*2  # 2 minute timeout
        while not self.response_received and time.time() < timeout:
            time.sleep(0.1)
        
        self.assertTrue(self.response_received, "GPT response not received for VLA command")
        self.assertIsNotNone(self.last_gpt_response, "No GPT response for VLA command")

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('humanoid_vla', 'capstone_test_suite', CapstoneTestSuite)
```

## Launch Configuration

Create a launch file to start the complete capstone project:

```xml
<!-- launch/capstone_project.launch -->
<launch>
  <!-- Arguments -->
  <arg name="openai_api_key" default="" />
  <arg name="whisper_model" default="base" />
  <arg name="use_sim_time" default="false" />
  
  <!-- Set use_sim_time if needed -->
  <param name="/use_sim_time" value="$(arg use_sim_time)" />
  
  <!-- Capstone Integration Node -->
  <node name="capstone_integration" pkg="humanoid_capstone" type="capstone_system_integration.py" output="screen">
  </node>
  
  <!-- Voice Command System -->
  <node name="voice_command_system" pkg="humanoid_capstone" type="voice_command_system.py" output="screen">
  </node>
  
  <!-- VLA Pipeline Prototype -->
  <node name="vla_pipeline" pkg="humanoid_capstone" type="vla_pipeline_prototype.py" output="screen">
    <param name="openai_api_key" value="$(arg openai_api_key)" />
  </node>
  
  <!-- (Optional) Navigation Stack -->
  <include file="$(find nav2_bringup)/launch/navigation_launch.py">
  </include>
  
  <!-- (Optional) Simulation Environment -->
  <group if="$(arg use_sim_time)">
    <include file="$(find gazebo_ros)/launch/empty_world.launch">
      <arg name="world_name" value="worlds/capstone_environment.world"/>
      <arg name="paused" value="false"/>
      <arg name="use_sim_time" value="true"/>
      <arg name="gui" value="true"/>
      <arg name="headless" value="false"/>
      <arg name="debug" value="false"/>
    </include>
  </group>
</launch>
```

## Documentation and Tutorials

Create a step-by-step tutorial for students to understand and extend the capstone project:

### 1. Student Exercise: Extending the Capstone System

Create an exercise file that guides students through extending the system:

```markdown
# Capstone Extension Exercise

## Objective
Extend the autonomous humanoid demonstration to include an additional capability of your choice.

## Requirements
1. Choose one of the following extensions:
   - Add object recognition and classification capability
   - Implement a "memory" system that remembers past interactions
   - Add gesture recognition using camera input
   - Create a multi-room exploration capability
   - Implement safety/risk assessment for actions

2. Document your extension with:
   - Architecture diagram showing your additions
   - Code comments explaining the new functionality
   - Test results demonstrating the extension

## Example: Adding Object Recognition

If you chose to add object recognition, you would:

1. Enhance the vision component to not just detect objects but also classify them more specifically:
   ```python
   # In your extended vision module
   def classify_object(self, image, bbox):
       # Extract the object region from the image
       x, y, w, h = bbox
       obj_region = image[y:y+h, x:x+w]
       
       # Use a specialized classification model
       classification = self.classification_model.predict(obj_region)
       return classification
   ```

2. Update the context provided to the language model to include object classifications:
   ```python
   # In the VLA pipeline
   context = {
       "detected_objects": [
           {
               "name": obj['name'], 
               "classification": self.classify_object(full_image, obj['bbox']),
               "confidence": obj['confidence']
           } 
           for obj in detections
       ]
   }
   ```

3. Test the extended system with commands that leverage your new capability.

## Submission Requirements
- Updated code files
- Documentation file describing your extension
- Video demonstration of the extended system in action
- Reflection document discussing challenges faced and lessons learned
```

## Performance Validation

Define metrics to validate the success of the capstone project:

1. **Task Completion Rate**: Percentage of commands that result in successful task completion
2. **Response Accuracy**: Accuracy of object detection and language understanding
3. **System Latency**: Time from command input to action execution
4. **Robustness**: Ability to handle unexpected situations gracefully
5. **User Satisfaction**: Subjective measure of how well the system meets user expectations

## Troubleshooting Guide

Common issues and solutions:

1. **GPT API Errors**: 
   - Ensure your API key is correctly set
   - Check your internet connection
   - Verify rate limits are not exceeded

2. **Audio Input Issues**:
   - Verify microphone permissions
   - Check that the audio system is properly configured
   - Ensure appropriate noise thresholds

3. **Navigation Failures**:
   - Verify map is correctly loaded
   - Check that navigation goals are feasible
   - Ensure obstacles are properly detected

4. **Vision Detection Failures**:
   - Ensure adequate lighting in the environment
   - Verify camera calibration
   - Check detection thresholds and parameters

## Conclusion

This capstone project demonstrates the integration of all modules in the Physical AI & Humanoid Robotics curriculum. Students have implemented a system that combines ROS 2 fundamentals, simulation environment knowledge, AI perception capabilities, and vision-language-action integration to create an autonomous humanoid robot that can understand and execute natural language commands.

The project provides a foundation that can be extended with additional capabilities and serves as a comprehensive demonstration of the concepts learned throughout the curriculum.