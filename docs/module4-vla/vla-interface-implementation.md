# Implementing VLA Interface Components

## Overview

This document provides a practical implementation of the Vision-Language-Action (VLA) interface components, following the contract defined in the VLA interface specification. This implementation demonstrates how to build the integration layer between vision processing, language understanding, and action execution systems.

## Implementation Architecture

The implementation follows the VLA interface contract with the following components:

```
VLA Interface Manager
├── Vision Component Interface
├── Language Component Interface  
├── Action Component Interface
└── VLA State Manager
```

## Component Implementations

### 1. VLA Interface Manager

The main orchestrator that manages the communication between components:

```python
#!/usr/bin/env python3

import rospy
import threading
from std_msgs.msg import String
from humanoid_msgs.msg import (
    GPTRequest, GPTResponse, 
    VisionOutput, LanguageOutput, ActionOutput,
    VLAState, VLACommand
)
from humanoid_msgs.srv import VLAControl, VLAControlResponse
from actionlib import SimpleActionClient
from humanoid_msgs.msg import VLAAction, VLAActionGoal, VLAActionResult
import json
import time
import uuid
from queue import Queue

class VLAInterfaceManager:
    def __init__(self):
        # Initialize component interfaces
        self.vision_client = VisionComponentClient()
        self.language_client = LanguageComponentClient()
        self.action_client = ActionComponentClient()
        
        # State management
        self.current_state = "IDLE"
        self.active_request_id = None
        self.system_confidence = 0.0
        
        # Communication interfaces
        self.vla_command_sub = rospy.Subscriber(
            "/vla/control_input", VLACommand, self.handle_vla_command
        )
        self.vla_state_pub = rospy.Publisher(
            "/vla/state", VLAState, queue_size=10, latch=True
        )
        
        # Service interface
        self.vla_service = rospy.Service(
            "/vla/process_command", VLAControl, self.process_command_service
        )
        
        # Action server for complex tasks
        self.action_server = SimpleActionClient(
            "/vla/action_server", VLAAction
        )
        
        # Thread for processing requests
        self.request_queue = Queue()
        self.processing_thread = threading.Thread(
            target=self.process_requests, daemon=True
        )
        self.processing_thread.start()
        
        # Publish initial state
        self.publish_state()
        
        rospy.loginfo("VLA Interface Manager initialized")
    
    def handle_vla_command(self, command_msg):
        """Handle incoming VLA commands"""
        # Add to processing queue
        self.request_queue.put(command_msg)
        
        # Update state
        self.current_state = "PROCESSING"
        self.active_request_id = command_msg.id
        self.publish_state()
    
    def process_requests(self):
        """Process requests in a separate thread"""
        while not rospy.is_shutdown():
            try:
                command_msg = self.request_queue.get(timeout=1.0)
                
                # Process the VLA command
                success, message, actions, response_text = self.process_vla_command(command_msg)
                
                # Update state
                self.current_state = "IDLE"
                self.active_request_id = None
                self.publish_state()
                
                # Mark as processed
                self.request_queue.task_done()
                
            except:
                # Continue loop on timeout
                continue
    
    def process_vla_command(self, command_msg):
        """Process a complete VLA command through all components"""
        try:
            # Step 1: Process with vision component
            vision_output = self.vision_client.process(command_msg.context)
            
            # Step 2: Process with language component
            language_output = self.language_client.process(
                command_msg.command, 
                vision_output, 
                command_msg.context
            )
            
            # Step 3: Process with action component
            action_success = self.action_client.execute_sequence(
                language_output.action_sequence
            )
            
            # Update system confidence based on component performance
            self.system_confidence = self.calculate_system_confidence(
                vision_output, language_output, action_success
            )
            
            return True, "Command processed successfully", \
                   language_output.action_sequence, language_output.response_text
            
        except Exception as e:
            rospy.logerr(f"Error in VLA command processing: {str(e)}")
            return False, f"Processing error: {str(e)}", [], "I encountered an error processing your command"
    
    def process_command_service(self, req):
        """Process VLA command as a service"""
        # Create a VLACommand message from service request
        command_msg = VLACommand()
        command_msg.header.stamp = rospy.Time.now()
        command_msg.command = req.command
        command_msg.context = req.context
        command_msg.id = str(uuid.uuid4())
        
        # Process the command
        success, message, actions, response_text = self.process_vla_command(command_msg)
        
        # Create and return service response
        response = VLAControlResponse()
        response.success = success
        response.message = message
        response.response_text = response_text
        
        # Convert action sequence to service response format
        # (Implementation depends on specific action message format)
        
        return response
    
    def calculate_system_confidence(self, vision_output, language_output, action_success):
        """Calculate overall system confidence based on component performance"""
        # Placeholder implementation - in practice, this would use more sophisticated metrics
        vision_conf = getattr(vision_output, 'detection_confidence', 0.8)
        language_conf = getattr(language_output, 'intent_confidence', 0.85)
        action_conf = 1.0 if action_success else 0.5
        
        # Weighted average of component confidences
        avg_confidence = (vision_conf + language_conf + action_conf) / 3.0
        return min(avg_confidence, 1.0)  # Clamp to [0, 1]
    
    def publish_state(self):
        """Publish current VLA system state"""
        state_msg = VLAState()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.system_status = self.current_state
        state_msg.current_action = self.active_request_id or ""
        state_msg.system_confidence = self.system_confidence
        state_msg.active_interfaces = [
            "vision" if self.vision_client.is_active() else "",
            "language" if self.language_client.is_active() else "",
            "action" if self.action_client.is_active() else ""
        ]
        
        # Filter out empty strings
        state_msg.active_interfaces = [iface for iface in state_msg.active_interfaces if iface]
        
        self.vla_state_pub.publish(state_msg)

class VisionComponentClient:
    """Client interface for vision component"""
    
    def __init__(self):
        self.vision_input_pub = rospy.Publisher(
            "/vla/vision_input", String, queue_size=10
        )
        self.vision_output_sub = rospy.Subscriber(
            "/vla/vision_output", VisionOutput, self.vision_output_callback
        )
        self.last_output = None
        self.output_lock = threading.Lock()
    
    def process(self, context):
        """Process vision input and return output"""
        # Publish context for vision processing
        context_msg = String()
        context_msg.data = json.dumps(context)
        self.vision_input_pub.publish(context_msg)
        
        # Wait for response (simplified - in practice, would use callbacks)
        start_time = rospy.Time.now()
        timeout = rospy.Duration(5.0)  # 5 second timeout
        
        while self.last_output is None:
            if rospy.Time.now() - start_time > timeout:
                raise Exception("Vision component timeout")
            time.sleep(0.1)
        
        with self.output_lock:
            output = self.last_output
            self.last_output = None  # Reset for next call
            return output
    
    def vision_output_callback(self, msg):
        """Handle vision component output"""
        with self.output_lock:
            self.last_output = msg
    
    def is_active(self):
        """Check if vision component is active"""
        return self.vision_input_pub.get_num_connections() > 0

class LanguageComponentClient:
    """Client interface for language component"""
    
    def __init__(self):
        self.language_input_pub = rospy.Publisher(
            "/vla/language_input", GPTRequest, queue_size=10
        )
        self.language_output_sub = rospy.Subscriber(
            "/vla/language_output", GPTResponse, self.language_output_callback
        )
        self.last_output = None
        self.output_lock = threading.Lock()
    
    def process(self, command, vision_context, additional_context):
        """Process language input and return output"""
        # Create GPT request with vision and additional context
        request = GPTRequest()
        request.header.stamp = rospy.Time.now()
        request.command = command
        request.context = json.dumps({
            'vision_data': vision_context,
            'additional_context': additional_context
        })
        request.id = str(uuid.uuid4())
        
        # Publish request
        self.language_input_pub.publish(request)
        
        # Wait for response
        start_time = rospy.Time.now()
        timeout = rospy.Duration(10.0)  # 10 second timeout
        
        while self.last_output is None:
            if rospy.Time.now() - start_time > timeout:
                raise Exception("Language component timeout")
            time.sleep(0.1)
        
        with self.output_lock:
            output = self.last_output
            self.last_output = None  # Reset for next call
            return self.parse_language_output(output)
    
    def parse_language_output(self, gpt_response):
        """Parse GPT response into LanguageOutput format"""
        # In practice, this would convert the GPT response to the expected LanguageOutput format
        # For this example, we'll create a mock LanguageOutput
        language_output = LanguageOutput()
        language_output.intent = "mock_intent"
        language_output.intent_confidence = 0.9
        language_output.response_text = gpt_response.response
        # Additional parsing logic would go here
        
        return language_output
    
    def language_output_callback(self, msg):
        """Handle language component output"""
        with self.output_lock:
            self.last_output = msg
    
    def is_active(self):
        """Check if language component is active"""
        return self.language_input_pub.get_num_connections() > 0

class ActionComponentClient:
    """Client interface for action component"""
    
    def __init__(self):
        self.action_input_pub = rospy.Publisher(
            "/vla/action_input", ActionOutput, queue_size=10
        )
        self.action_output_sub = rospy.Subscriber(
            "/vla/action_output", ActionOutput, self.action_output_callback
        )
        self.last_output = None
        self.output_lock = threading.Lock()
    
    def execute_sequence(self, action_sequence):
        """Execute a sequence of actions"""
        success = True
        
        for action in action_sequence:
            try:
                # Convert action to appropriate format and publish
                action_msg = ActionOutput()
                action_msg.header.stamp = rospy.Time.now()
                action_msg.action_id = str(uuid.uuid4())
                # Additional action properties would be set here
                
                self.action_input_pub.publish(action_msg)
                
                # Wait for response
                start_time = rospy.Time.now()
                timeout = rospy.Duration(30.0)  # 30 second timeout per action
                
                while self.last_output is None:
                    if rospy.Time.now() - start_time > timeout:
                        raise Exception(f"Action {action_msg.action_id} timeout")
                    time.sleep(0.1)
                
                with self.output_lock:
                    action_result = self.last_output
                    self.last_output = None  # Reset for next call
                
                # Check if action was successful
                if action_result.status != "completed":
                    success = False
                    rospy.logwarn(f"Action {action_msg.action_id} failed with status: {action_result.status}")
                
            except Exception as e:
                rospy.logerr(f"Error executing action: {str(e)}")
                success = False
                break
        
        return success
    
    def action_output_callback(self, msg):
        """Handle action component output"""
        with self.output_lock:
            self.last_output = msg
    
    def is_active(self):
        """Check if action component is active"""
        return self.action_input_pub.get_num_connections() > 0

def main():
    rospy.init_node('vla_interface_manager')
    
    try:
        # Initialize and start the VLA interface manager
        vla_manager = VLAInterfaceManager()
        
        # Spin to handle callbacks
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("VLA Interface Manager interrupted")
    except Exception as e:
        rospy.logerr(f"VLA Interface Manager error: {str(e)}")
    finally:
        rospy.loginfo("VLA Interface Manager shutting down")

if __name__ == '__main__':
    main()
```

### 2. Component-Specific Implementation Examples

#### Vision Component Implementation
```python
#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from humanoid_msgs.msg import VisionOutput
import torch
import torchvision.transforms as T
from std_msgs.msg import String
import json

class VisionComponent:
    def __init__(self):
        # Initialize computer vision models
        # Using a pre-trained model like YOLO or similar for object detection
        self.bridge = CvBridge()
        self.model = self.load_vision_model()
        
        # Subscribers and publishers
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.image_callback
        )
        self.vision_input_sub = rospy.Subscriber(
            "/vla/vision_input", String, self.vision_input_callback
        )
        self.vision_output_pub = rospy.Publisher(
            "/vla/vision_output", VisionOutput, queue_size=10
        )
        
        # Store latest image and processing context
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        rospy.loginfo("Vision Component initialized")
    
    def load_vision_model(self):
        # Load a pre-trained model for object detection, pose estimation, etc.
        # Example using torchvision model (in practice would use a more specific model)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model
    
    def image_callback(self, img_msg):
        """Process incoming images from camera"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            with self.image_lock:
                self.latest_image = cv_image
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def vision_input_callback(self, input_msg):
        """Process vision input request"""
        try:
            # Get current image
            with self.image_lock:
                if self.latest_image is None:
                    raise Exception("No image available")
                image = self.latest_image.copy()
            
            # Perform vision processing (object detection, scene understanding, etc.)
            vision_output = self.process_vision(image)
            
            # Publish the result
            self.vision_output_pub.publish(vision_output)
            
        except Exception as e:
            rospy.logerr(f"Error in vision processing: {str(e)}")
    
    def process_vision(self, image):
        """Perform vision processing on the provided image"""
        # Run object detection
        results = self.model(image)
        
        # Create VisionOutput message
        output = VisionOutput()
        output.header.stamp = rospy.Time.now()
        output.header.frame_id = "camera_rgb_optical_frame"
        
        # Convert detection results to VisionOutput format
        detections = results.pandas().xyxy[0]  # Get detections in pandas format
        
        for _, detection in detections.iterrows():
            # Create object detection message
            obj_detection = ObjectDetection()
            obj_detection.id = str(uuid.uuid4())
            obj_detection.class = detection['name']
            obj_detection.confidence = float(detection['confidence'])
            
            # Set bounding box
            obj_detection.bbox.x = int(detection['xmin'])
            obj_detection.bbox.y = int(detection['ymin'])
            obj_detection.bbox.width = int(detection['xmax'] - detection['xmin'])
            obj_detection.bbox.height = int(detection['ymax'] - detection['ymin'])
            
            # In a full implementation, also estimate 3D pose
            # This is simplified - would require depth information for full 3D pose
            obj_detection.pose.position.x = (detection['xmin'] + detection['xmax']) / 2.0
            obj_detection.pose.position.y = (detection['ymin'] + detection['ymax']) / 2.0
            obj_detection.pose.position.z = 1.0  # Placeholder depth
            
            output.detected_objects.append(obj_detection)
        
        # Generate scene description (in practice, this could use a vision-language model)
        output.scene_description = self.generate_scene_description(detections)
        
        return output
    
    def generate_scene_description(self, detections):
        """Generate a textual description of the scene"""
        # Count objects by class
        obj_counts = {}
        for _, detection in detections.iterrows():
            obj_class = detection['name']
            obj_counts[obj_class] = obj_counts.get(obj_class, 0) + 1
        
        # Create description
        if not obj_counts:
            return "An empty scene"
        
        obj_list = [f"{count} {obj_class}" for obj_class, count in obj_counts.items()]
        obj_str = ", ".join(obj_list)
        
        return f"A scene containing {obj_str}"

def main():
    rospy.init_node('vision_component')
    
    try:
        vision_component = VisionComponent()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Vision Component interrupted")
    finally:
        rospy.loginfo("Vision Component shutting down")

if __name__ == '__main__':
    main()
```

#### Language Component Implementation
```python
#!/usr/bin/env python3

import rospy
import openai
from humanoid_msgs.msg import GPTRequest, GPTResponse
from humanoid_msgs.msg import LanguageOutput
from std_msgs.msg import String
import json

class LanguageComponent:
    def __init__(self):
        # Initialize OpenAI API key
        self.api_key = rospy.get_param('~openai_api_key', '')
        if not self.api_key:
            rospy.logerr("OpenAI API key not set!")
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        
        # Subscribers and publishers
        self.language_input_sub = rospy.Subscriber(
            "/vla/language_input", GPTRequest, self.language_input_callback
        )
        self.language_output_pub = rospy.Publisher(
            "/vla/language_output", GPTResponse, queue_size=10
        )
        
        # Context management
        self.context_history = []
        
        rospy.loginfo("Language Component initialized")
    
    def language_input_callback(self, request_msg):
        """Process language input request"""
        try:
            # Build context for the language model
            messages = self.build_context(request_msg)
            
            # Call GPT API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Could be configured via parameter
                messages=messages,
                max_tokens=rospy.get_param('~max_tokens', 500),
                temperature=rospy.get_param('~temperature', 0.7),
                functions=[
                    {
                        "name": "execute_navigation",
                        "description": "Navigate the robot to a specific location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number", "description": "X coordinate"},
                                "y": {"type": "number", "description": "Y coordinate"},
                                "z": {"type": "number", "description": "Z coordinate"}
                            },
                            "required": ["x", "y"]
                        }
                    },
                    {
                        "name": "grasp_object",
                        "description": "Grasp an object with the robot's manipulator",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "object_id": {"type": "string", "description": "ID of the object to grasp"},
                                "object_name": {"type": "string", "description": "Name of the object to grasp"}
                            },
                            "required": ["object_id", "object_name"]
                        }
                    },
                    # Additional functions would be added here
                ],
                function_call="auto"
            )
            
            # Process response
            message = response.choices[0].message
            
            # Handle function calls if present
            if 'function_call' in message:
                function_call = message['function_call']
                # Execute function based on GPT's suggestion
                self.execute_function(function_call)
            
            # Create and publish response
            response_msg = GPTResponse()
            response_msg.header.stamp = rospy.Time.now()
            response_msg.header.frame_id = "base_link"
            response_msg.request_id = request_msg.id
            response_msg.response = message['content'] if message.get('content') else "Function called"
            
            # Update context history
            self.context_history.append({"role": "user", "content": request_msg.command})
            self.context_history.append({"role": "assistant", "content": response_msg.response})
            
            # Limit context history size
            max_history = rospy.get_param('~max_context_history', 10)
            if len(self.context_history) > max_history * 2:
                self.context_history = self.context_history[-max_history*2:]
            
            self.language_output_pub.publish(response_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in language processing: {str(e)}")
    
    def build_context(self, request_msg):
        """Build context for LLM with conversation history and current request"""
        # Start with system message
        messages = [
            {"role": "system", "content": rospy.get_param(
                '~system_prompt', 
                """You are an assistant for a humanoid robot. 
                Understand natural language commands and convert them to robot actions.
                When possible, respond with structured function calls that the robot can execute.
                Be aware of the robot's capabilities and limitations."""
            )}
        ]
        
        # Add conversation history
        messages.extend(self.context_history)
        
        # Add current request
        messages.append({"role": "user", "content": request_msg.command})
        
        return messages
    
    def execute_function(self, function_call):
        """Execute the function suggested by the LLM"""
        rospy.loginfo(f"Executing function: {function_call['name']}")
        # In practice, this would translate the function call to a ROS service call or action
        # Example: if function_call['name'] == 'execute_navigation', call navigation service

def main():
    rospy.init_node('language_component')
    
    try:
        language_component = LanguageComponent()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Language Component interrupted")
    finally:
        rospy.loginfo("Language Component shutting down")

if __name__ == '__main__':
    main()
```

#### Action Component Implementation
```python
#!/usr/bin/env python3

import rospy
from humanoid_msgs.msg import ActionOutput
from humanoid_msgs.msg import ActionFeedback
from actionlib import SimpleActionClient
from humanoid_msgs.msg import NavigationAction, NavigationGoal
from humanoid_msgs.msg import ManipulationAction, ManipulationGoal
from humanoid_msgs.msg import SpeechAction, SpeechGoal
import threading

class ActionComponent:
    def __init__(self):
        # Initialize action clients for different robot capabilities
        self.nav_client = SimpleActionClient('navigation_server', NavigationAction)
        self.manip_client = SimpleActionClient('manipulation_server', ManipulationAction)
        self.speech_client = SimpleActionClient('speech_server', SpeechAction)
        
        # Wait for action servers to become available
        rospy.loginfo("Waiting for action servers...")
        self.nav_client.wait_for_server()
        self.manip_client.wait_for_server()
        self.speech_client.wait_for_server()
        rospy.loginfo("Action servers ready")
        
        # Subscribers and publishers
        self.action_input_sub = rospy.Subscriber(
            "/vla/action_input", ActionOutput, self.action_input_callback
        )
        self.action_output_pub = rospy.Publisher(
            "/vla/action_output", ActionOutput, queue_size=10
        )
        
        rospy.loginfo("Action Component initialized")
    
    def action_input_callback(self, action_msg):
        """Process action input request"""
        try:
            # Execute the action based on its type
            result = self.execute_action(action_msg)
            
            # Create response message
            response_msg = ActionOutput()
            response_msg.header.stamp = rospy.Time.now()
            response_msg.header.frame_id = "base_link"
            response_msg.action_id = action_msg.action_id
            response_msg.status = result['status']
            
            # Add feedback
            feedback = ActionFeedback()
            feedback.object_grasped = result.get('object_grasped', False)
            feedback.navigation_success = result.get('navigation_success', False)
            feedback.execution_errors = result.get('errors', [])
            response_msg.feedback = feedback
            
            # Publish response
            self.action_output_pub.publish(response_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in action execution: {str(e)}")
    
    def execute_action(self, action_msg):
        """Execute a specific action based on its type"""
        action_type = action_msg.action_type
        
        if action_type == "navigate":
            return self.execute_navigation_action(action_msg)
        elif action_type == "grasp":
            return self.execute_manipulation_action(action_msg)
        elif action_type == "speak":
            return self.execute_speech_action(action_msg)
        else:
            return {
                'status': 'failed',
                'errors': [f"Unknown action type: {action_type}"]
            }
    
    def execute_navigation_action(self, action_msg):
        """Execute navigation action"""
        try:
            # Create navigation goal
            goal = NavigationGoal()
            goal.target_pose = action_msg.target_pose
            
            # Send goal to navigation server
            self.nav_client.send_goal(goal)
            
            # Wait for result
            finished_before_timeout = self.nav_client.wait_for_result(rospy.Duration(30.0))
            
            if finished_before_timeout:
                state = self.nav_client.get_state()
                result = self.nav_client.get_result()
                
                if state == 3:  # GoalStatus.SUCCEEDED
                    return {
                        'status': 'completed',
                        'navigation_success': True,
                        'errors': []
                    }
                else:
                    return {
                        'status': 'failed',
                        'navigation_success': False,
                        'errors': [f"Navigation failed with state: {state}"]
                    }
            else:
                # Cancel goal if timeout
                self.nav_client.cancel_goal()
                return {
                    'status': 'failed',
                    'navigation_success': False,
                    'errors': ["Navigation timeout"]
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'navigation_success': False,
                'errors': [f"Navigation error: {str(e)}"]
            }
    
    def execute_manipulation_action(self, action_msg):
        """Execute manipulation/grasping action"""
        try:
            # Create manipulation goal
            goal = ManipulationGoal()
            goal.object_id = action_msg.object_id
            goal.object_pose = action_msg.object_pose
            
            # Send goal to manipulation server
            self.manip_client.send_goal(goal)
            
            # Wait for result
            finished_before_timeout = self.manip_client.wait_for_result(rospy.Duration(30.0))
            
            if finished_before_timeout:
                state = self.manip_client.get_state()
                
                if state == 3:  # GoalStatus.SUCCEEDED
                    return {
                        'status': 'completed',
                        'object_grasped': True,
                        'errors': []
                    }
                else:
                    return {
                        'status': 'failed',
                        'object_grasped': False,
                        'errors': [f"Manipulation failed with state: {state}"]
                    }
            else:
                # Cancel goal if timeout
                self.manip_client.cancel_goal()
                return {
                    'status': 'failed',
                    'object_grasped': False,
                    'errors': ["Manipulation timeout"]
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'object_grasped': False,
                'errors': [f"Manipulation error: {str(e)}"]
            }
    
    def execute_speech_action(self, action_msg):
        """Execute speech action"""
        try:
            # Create speech goal
            goal = SpeechGoal()
            goal.text = action_msg.text
            
            # Send goal to speech server
            self.speech_client.send_goal(goal)
            
            # Wait for result
            finished_before_timeout = self.speech_client.wait_for_result(rospy.Duration(10.0))
            
            if finished_before_timeout:
                state = self.speech_client.get_state()
                
                if state == 3:  # GoalStatus.SUCCEEDED
                    return {
                        'status': 'completed',
                        'errors': []
                    }
                else:
                    return {
                        'status': 'failed',
                        'errors': [f"Speech failed with state: {state}"]
                    }
            else:
                # Cancel goal if timeout
                self.speech_client.cancel_goal()
                return {
                    'status': 'failed',
                    'errors': ["Speech timeout"]
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'errors': [f"Speech error: {str(e)}"]
            }

def main():
    rospy.init_node('action_component')
    
    try:
        action_component = ActionComponent()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Action Component interrupted")
    finally:
        rospy.loginfo("Action Component shutting down")

if __name__ == '__main__':
    main()
```

### 3. Launch File for VLA System

Create a launch file to start all VLA components together:

In `launch/vla_system.launch`:
```xml
<launch>
  <!-- Arguments -->
  <arg name="openai_api_key" default="" />
  <arg name="use_sim_time" default="false" />
  
  <!-- Set use_sim_time if needed -->
  <param name="/use_sim_time" value="$(arg use_sim_time)" />
  
  <!-- VLA Interface Manager -->
  <node name="vla_interface_manager" pkg="humanoid_vla" type="vla_interface_manager.py" output="screen">
  </node>
  
  <!-- Vision Component -->
  <node name="vision_component" pkg="humanoid_vla" type="vision_component.py" output="screen">
  </node>
  
  <!-- Language Component -->
  <node name="language_component" pkg="humanoid_vla" type="language_component.py" output="screen">
    <param name="openai_api_key" value="$(arg openai_api_key)" />
    <param name="max_tokens" value="500" />
    <param name="temperature" value="0.7" />
    <param name="max_context_history" value="10" />
    <param name="system_prompt" value="You are an assistant for a humanoid robot. Understand natural language commands and convert them to robot actions. When possible, respond with structured function calls that the robot can execute. Be aware of the robot's capabilities and limitations." />
  </node>
  
  <!-- Action Component -->
  <node name="action_component" pkg="humanoid_vla" type="action_component.py" output="screen">
  </node>
  
  <!-- For simulation, you may need to include a camera simulator -->
  <group if="$(arg use_sim_time)">
    <include file="$(find gazebo_ros)/launch/empty_world.launch">
      <arg name="world_name" value="worlds/empty.world"/>
      <arg name="paused" value="false"/>
      <arg name="use_sim_time" value="true"/>
      <arg name="gui" value="true"/>
      <arg name="headless" value="false"/>
      <arg name="debug" value="false"/>
    </include>
  </group>
</launch>
```

## Testing the Implementation

### Unit Tests

```python
#!/usr/bin/env python3

import unittest
import rospy
from humanoid_msgs.msg import VLACommand
from humanoid_msgs.srv import VLAControl

class TestVLAInterface(unittest.TestCase):
    def setUp(self):
        rospy.init_node('vla_tester', anonymous=True)
    
    def test_vla_service_call(self):
        """Test VLA service functionality"""
        # Wait for service
        rospy.wait_for_service('/vla/process_command')
        
        try:
            # Create service proxy
            vla_service = rospy.ServiceProxy('/vla/process_command', VLAControl)
            
            # Call service
            response = vla_service("Navigate to the kitchen", "Current position: living room")
            
            # Check response
            self.assertTrue(response.success)
            self.assertIsNotNone(response.response_text)
            
        except rospy.ServiceException as e:
            self.fail(f"VLA service call failed: {e}")
    
    def test_vla_state_monitoring(self):
        """Test VLA state monitoring"""
        # Subscribe to state topic
        state_msg = None
        def state_callback(msg):
            nonlocal state_msg
            state_msg = msg
        
        sub = rospy.Subscriber('/vla/state', VLAState, state_callback)
        
        # Wait for message
        timeout = rospy.Time.now() + rospy.Duration(5.0)
        while state_msg is None and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        # Check that we received a state message
        self.assertIsNotNone(state_msg)
        self.assertIn(state_msg.system_status, ["IDLE", "PROCESSING", "EXECUTING", "ERROR"])

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun('humanoid_vla', 'test_vla_interface', TestVLAInterface)
```

## Conclusion

This implementation provides a complete set of VLA interface components that adhere to the defined contract. The modular architecture allows for independent development and testing of each component while maintaining a consistent interface for integration. The system handles communication, state management, error recovery, and performance monitoring as specified in the contract.