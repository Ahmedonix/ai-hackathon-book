# Practical Exercise: Complete VLA (Vision-Language-Action) System Integration

## Objective

In this comprehensive exercise, you will integrate all components of the Vision-Language-Action (VLA) system into a complete, end-to-end humanoid robot system. You will connect perception, language understanding, and action execution to create a robot that can understand natural language commands, perceive its environment, and execute complex tasks.

## Prerequisites

Before starting this exercise, you should have:
- Completed all previous modules (ROS 2 fundamentals, simulation, AI perception)
- Successfully completed the Whisper integration, LLM planning, and multi-modal exercises
- Developed understanding of computer vision, natural language processing, and robot control
- Access to computational resources capable of running vision models and LLMs
- A simulated or physical humanoid robot platform

## Time Estimate

This exercise should take approximately 8-10 hours to complete, as it requires integrating components developed in previous exercises.

## Setup Requirements

### Hardware Requirements
- Computer with NVIDIA GPU (RTX 4070 Ti or better recommended) for running vision models
- Microphone for audio input
- Camera for visual input
- Either a physical humanoid robot or simulation environment

### Software Dependencies
```bash
pip3 install openai whisper pyaudio opencv-python mediapipe torch torchvision
```

## Exercise Tasks

### Task 1: System Architecture Review (30 minutes)

Before implementation, review the complete VLA system architecture and ensure all components from previous exercises are available:

1. **Vision Component**: Object detection, gesture recognition, scene understanding
2. **Language Component**: Speech-to-text (Whisper), LLM-based planning
3. **Action Component**: Navigation, manipulation, communication
4. **Fusion Component**: Multi-modal integration and coordination

Create a system diagram showing how these components will be connected:

```
User Command
    ↓ (voice)
Speech-to-Text (Whisper)
    ↓ (text)
Language Understanding (GPT)
    ↓ (parsed intent + context)
Action Planner (Structured Output)
    ↓ (action sequence)
Action Execution
    ↓ (robot movements)
Environment → Sensors → Perception → Feedback
    ↑ (visual observations)
```

### Task 2: VLA Main Controller Implementation (90 minutes)

Create the main controller that orchestrates the entire VLA system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from multimodal_msgs.msg import Gesture
from humanoid_msgs.msg import GPTRequest, GPTResponse
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from cv_bridge import CvBridge
import threading

class VLAControllerNode(Node):
    def __init__(self):
        super().__init__('vla_controller_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # State management
        self.current_state = "IDLE"  # IDLE, LISTENING, PROCESSING, EXECUTING
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.pending_command = None
        
        # Action clients
        self.nav_client = SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()
        
        # Publishers
        self.speech_pub = self.create_publisher(String, 'text_to_speech', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.voice_command_sub = self.create_subscription(
            String, 'voice_commands', self.voice_command_callback, 10
        )
        self.gpt_response_sub = self.create_subscription(
            GPTResponse, 'structured_llm_responses', self.gpt_response_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.gesture_sub = self.create_subscription(
            Gesture, 'gesture_detections', self.gesture_callback, 10
        )
        
        # Timer for state management
        self.state_timer = self.create_timer(0.1, self.state_timer_callback)
        
        self.get_logger().info('VLA Controller Node initialized')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands"""
        if self.current_state != "IDLE":
            self.get_logger().warn(f"Ignoring command, system busy with {self.current_state}")
            return
        
        self.get_logger().info(f"Received voice command: {msg.data}")
        
        # Get current context (image and detected objects)
        with self.image_lock:
            if self.latest_image is not None:
                # Process the image to get context
                context = self.analyze_current_scene(self.latest_image)
            else:
                context = {"environment": "unknown", "objects": [], "gestures": []}
        
        # Add any recent gestures to context
        context["recent_gestures"] = [self.recent_gesture] if hasattr(self, 'recent_gesture') else []
        
        # Send to language understanding system
        gpt_request = GPTRequest()
        gpt_request.header.stamp = self.get_clock().now().to_msg()
        gpt_request.command = msg.data
        gpt_request.context = json.dumps(context)
        gpt_request.id = f"vla_{time.time()}"
        
        # Publish to LLM system
        llm_pub = self.create_publisher(GPTRequest, 'structured_llm_requests', 10)
        llm_pub.publish(gpt_request)
        
        # Update state
        self.current_state = "PROCESSING"
        self.pending_command = msg.data

    def image_callback(self, img_msg):
        """Store latest image for context"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            with self.image_lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def gesture_callback(self, msg):
        """Store latest gesture for context"""
        self.recent_gesture = msg

    def analyze_current_scene(self, cv_image):
        """Analyze the current scene to provide context to the LLM"""
        # This is a simplified example
        # In practice, this would use a more sophisticated vision system
        # that can detect and classify objects, understand spatial relationships, etc.
        
        # For this example, we'll return a dummy context
        # In a real implementation, you would:
        # 1. Run object detection on the image
        # 2. Extract spatial relationships
        # 3. Identify significant features
        
        context = {
            "environment": "indoor",
            "objects": [
                {"name": "table", "position": [1.0, 0.5], "confidence": 0.9},
                {"name": "cup", "position": [1.2, 0.6], "confidence": 0.8}
            ],
            "spatial_relationships": [
                {"subject": "cup", "relation": "on", "object": "table"}
            ]
        }
        
        return context

    def gpt_response_callback(self, msg):
        """Handle LLM response and execute actions"""
        if msg.is_error:
            self.get_logger().error(f"LLM Error: {msg.response}")
            self.speak_response("I encountered an error processing your request")
            self.current_state = "IDLE"
            return
        
        try:
            # Parse structured response
            structured_data = json.loads(msg.structured_response)
            
            if "action_sequence" in structured_data:
                action_sequence = structured_data["action_sequence"]
                self.get_logger().info(f"Executing action sequence with {len(action_sequence)} actions")
                
                # Update state to EXECUTING
                self.current_state = "EXECUTING"
                
                # Execute the action sequence
                self.execute_action_sequence(action_sequence)
                
                # After execution, return to IDLE
                self.current_state = "IDLE"
                self.speak_response("Task completed")
            else:
                # If no action sequence, just speak the response
                self.speak_response(structured_data.get("text_response", msg.response))
                self.current_state = "IDLE"
                
        except json.JSONDecodeError:
            self.get_logger().error("Could not parse structured response")
            self.speak_response("I couldn't understand the task properly")
            self.current_state = "IDLE"
        except Exception as e:
            self.get_logger().error(f"Error executing action sequence: {e}")
            self.speak_response("An error occurred while executing the task")
            self.current_state = "IDLE"

    def execute_action_sequence(self, action_sequence):
        """Execute a sequence of actions"""
        for i, action in enumerate(action_sequence):
            self.get_logger().info(f"Executing action {i+1}/{len(action_sequence)}: {action.get('action_type', 'unknown')}")
            
            success = self.execute_single_action(action)
            
            if not success:
                self.get_logger().error(f"Action failed: {action}")
                self.speak_response("I couldn't complete the task as requested")
                break

    def execute_single_action(self, action):
        """Execute a single action from the plan"""
        action_type = action.get('action_type', 'unknown')
        params = action.get('parameters', {})
        
        if action_type == 'navigate_to':
            return self.execute_navigation(params)
        elif action_type == 'grasp_object':
            return self.execute_grasp(params)
        elif action_type == 'speak':
            return self.execute_speech(params)
        elif action_type == 'detect_object':
            return self.execute_detection(params)
        elif action_type == 'turn_towards':
            return self.execute_turn(params)
        else:
            self.get_logger().warn(f"Unknown action type: {action_type}")
            return False

    def execute_navigation(self, params):
        """Execute navigation action"""
        try:
            target_location = params.get('location', 'unknown')
            
            # In a real implementation, you would have a map of named locations to coordinates
            # For this example, we'll use a simple mapping
            location_map = {
                'kitchen': {'x': 2.0, 'y': 1.0},
                'living room': {'x': 0.0, 'y': 0.0},
                'bedroom': {'x': -1.0, 'y': 2.0},
                'office': {'x': 1.5, 'y': -1.0}
            }
            
            if target_location in location_map:
                coords = location_map[target_location]
                
                # Create navigation goal
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = self.get_clock().now().to_msg()
                goal.target_pose.pose.position.x = coords['x']
                goal.target_pose.pose.position.y = coords['y']
                goal.target_pose.pose.orientation.w = 1.0
                
                # Send goal to navigation server
                self.nav_client.send_goal(goal)
                
                # Wait for result (simplified - in practice, use callbacks)
                finished = self.nav_client.wait_for_result(timeout=60.0)
                
                if finished:
                    result = self.nav_client.get_result()
                    if result:
                        self.get_logger().info(f"Navigation to {target_location} completed")
                        return True
                    else:
                        self.get_logger().error(f"Navigation to {target_location} failed")
                        return False
                else:
                    self.nav_client.cancel_goal()
                    self.get_logger().error(f"Navigation to {target_location} timed out")
                    return False
            else:
                self.get_logger().warn(f"Unknown location: {target_location}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error in navigation: {e}")
            return False

    def execute_grasp(self, params):
        """Execute grasp action (simulated)"""
        try:
            obj_name = params.get('object', 'unknown')
            self.get_logger().info(f"Attempting to grasp: {obj_name}")
            
            # In a real robot, this would interface with the manipulation system
            # For simulation, we'll just return success
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in grasp: {e}")
            return False

    def execute_speech(self, params):
        """Execute speech action"""
        try:
            text = params.get('text', 'Hello')
            self.speak_response(text)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in speech: {e}")
            return False

    def execute_detection(self, params):
        """Execute detection action (simulated)"""
        try:
            obj_type = params.get('object', 'object')
            self.get_logger().info(f"Attempting to detect: {obj_type}")
            
            # In a real implementation, this would interface with the vision system
            # to locate a specific object
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")
            return False

    def execute_turn(self, params):
        """Execute turn action"""
        try:
            direction = params.get('direction', 'unknown')
            
            twist_msg = Twist()
            if direction == 'left':
                twist_msg.angular.z = 0.5  # Turn left
            elif direction == 'right':
                twist_msg.angular.z = -0.5  # Turn right
            else:
                self.get_logger().warn(f"Unknown direction: {direction}")
                return False
                
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(1)  # Turn for 1 second
            twist_msg.angular.z = 0.0  # Stop turning
            self.cmd_vel_pub.publish(twist_msg)
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error in turning: {e}")
            return False

    def speak_response(self, text):
        """Publish text for speech synthesis"""
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)
        self.get_logger().info(f"Speaking: {text}")

    def state_timer_callback(self):
        """Periodic state management"""
        # This timer helps manage system state and can be used for timeouts
        pass

def main(args=None):
    rclpy.init(args=args)
    node = VLAControllerNode()
    
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

### Task 3: Advanced VLA Integration (120 minutes)

Create an enhanced VLA system with better perception-action feedback loops:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import time
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from multimodal_msgs.msg import Gesture
from humanoid_msgs.msg import GPTRequest, GPTResponse
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from cv_bridge import CvBridge
import threading
from collections import deque

class AdvancedVLANode(Node):
    def __init__(self):
        super().__init__('advanced_vla_node')
        
        # Initialize components
        self.bridge = CvBridge()
        
        # State and context management
        self.current_task = None
        self.task_history = deque(maxlen=10)
        self.interaction_context = {
            'user_attention': 'unknown',
            'environment_map': {},
            'object_memory': {},
            'task_progress': {}
        }
        
        # Perception buffers
        self.image_buffer = deque(maxlen=5)
        self.gesture_buffer = deque(maxlen=10) 
        self.speech_buffer = deque(maxlen=5)
        
        # Action execution tracking
        self.active_action = None
        self.action_feedback = None
        
        # Action clients
        self.nav_client = SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()
        
        # Publishers
        self.speech_pub = self.create_publisher(String, 'text_to_speech', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.vla_status_pub = self.create_publisher(String, 'vla_status', 10)
        
        # Subscribers
        self.voice_command_sub = self.create_subscription(
            String, 'voice_commands', self.voice_command_callback, 10
        )
        self.gpt_response_sub = self.create_subscription(
            GPTResponse, 'structured_llm_responses', self.gpt_response_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.gesture_sub = self.create_subscription(
            Gesture, 'gesture_detections', self.gesture_callback, 10
        )
        self.action_feedback_sub = self.create_subscription(
            String, 'action_feedback', self.action_feedback_callback, 10
        )
        
        # Timers
        self.perception_timer = self.create_timer(0.5, self.perception_processing_callback)
        self.context_timer = self.create_timer(1.0, self.context_update_callback)
        
        self.get_logger().info('Advanced VLA Node initialized')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands with context awareness"""
        self.get_logger().info(f"Received voice command: {msg.data}")
        
        # Create comprehensive context for the LLM
        context = self.create_comprehensive_context()
        context['new_command'] = msg.data
        
        # Send to language understanding system
        gpt_request = GPTRequest()
        gpt_request.header.stamp = self.get_clock().now().to_msg()
        gpt_request.command = msg.data
        gpt_request.context = json.dumps(context)
        gpt_request.id = f"adv_vla_{time.time()}"
        
        # Publish to LLM system
        llm_pub = self.create_publisher(GPTRequest, 'advanced_llm_requests', 10)
        llm_pub.publish(gpt_request)
        
        # Update task tracking
        self.current_task = {
            'command': msg.data,
            'start_time': self.get_clock().now(),
            'status': 'processing'
        }
        
        status_msg = String()
        status_msg.data = f"Processing: {msg.data}"
        self.vla_status_pub.publish(status_msg)

    def image_callback(self, img_msg):
        """Buffer images for perception processing"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            
            # Add timestamp and store
            image_entry = {
                'image': cv_image,
                'timestamp': self.get_clock().now()
            }
            self.image_buffer.append(image_entry)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def gesture_callback(self, msg):
        """Buffer gestures for context"""
        gesture_entry = {
            'gesture': msg,
            'timestamp': self.get_clock().now()
        }
        self.gesture_buffer.append(gesture_entry)

    def action_feedback_callback(self, msg):
        """Handle action execution feedback"""
        self.action_feedback = msg
        if self.current_task:
            self.current_task['feedback'] = msg.data

    def create_comprehensive_context(self):
        """Create comprehensive context from all available information"""
        context = {
            'task_history': list(self.task_history),
            'current_task': self.current_task,
            'environment_state': self.interaction_context['environment_map'],
            'object_memory': self.interaction_context['object_memory'],
            'gesture_context': self.get_recent_gestures(5),
            'temporal_context': {
                'session_start': str(self.get_clock().now()),
                'last_interaction': str(self.interaction_context.get('last_interaction', self.get_clock().now()))
            }
        }
        
        # Add recent perception data if available
        recent_images = list(self.image_buffer)[-1:]  # Just the latest
        if recent_images:
            # Analyze the latest image for current scene
            latest_img_data = self.analyze_image_for_context(recent_images[0]['image'])
            context['current_scene'] = latest_img_data
        
        return context

    def get_recent_gestures(self, seconds=10):
        """Get gestures from the last N seconds"""
        recent_gestures = []
        current_time = self.get_clock().now()
        
        for entry in list(self.gesture_buffer):
            time_diff = (current_time - entry['timestamp']).nanoseconds / 1e9
            if time_diff <= seconds:
                recent_gestures.append({
                    'type': entry['gesture'].type,
                    'description': entry['gesture'].description,
                    'time_ago': time_diff
                })
        
        return recent_gestures

    def analyze_image_for_context(self, cv_image):
        """Analyze image to extract context-relevant information"""
        # This is a simplified example - in practice, you would use more sophisticated vision models
        # that can detect objects, people, spatial relationships, etc.
        
        # For this example, we'll just return some basic information
        height, width = cv_image.shape[:2]
        
        # Detect if there are people in the image (simplified)
        # In practice, you'd use a person detection model
        people_detected = False  # Simplified
        
        # Detect basic shapes or colors that might be relevant
        # For example, detect if there's a red area (possibly a cup)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_area = cv2.countNonZero(mask)
        
        has_red_object = red_area > 1000  # Threshold for significant red area
        
        return {
            'image_size': [width, height],
            'people_detected': people_detected,
            'has_red_object': has_red_object,
            'red_pixel_count': red_area
        }

    def gpt_response_callback(self, msg):
        """Process LLM response and execute complex action sequences"""
        if msg.is_error:
            self.get_logger().error(f"LLM Error: {msg.response}")
            self.speak_response("I encountered an error processing your request")
            if self.current_task:
                self.current_task['status'] = 'error'
            return
        
        try:
            # Parse structured response
            structured_data = json.loads(msg.structured_response)
            
            if "action_sequence" in structured_data:
                action_sequence = structured_data["action_sequence"]
                self.get_logger().info(f"Executing complex action sequence with {len(action_sequence)} actions")
                
                # Execute the action sequence with monitoring
                success = self.execute_complex_action_sequence(action_sequence)
                
                if success:
                    self.get_logger().info("Action sequence completed successfully")
                    self.speak_response("I have completed the requested task")
                    
                    # Add to task history
                    if self.current_task:
                        self.current_task['status'] = 'completed'
                        self.current_task['end_time'] = self.get_clock().now()
                        self.task_history.append(self.current_task)
                else:
                    self.get_logger().error("Action sequence execution failed")
                    self.speak_response("I couldn't complete the task as requested")
                    
            else:
                # If no action sequence, just speak the response
                self.speak_response(structured_data.get("text_response", msg.response))
                
        except json.JSONDecodeError:
            self.get_logger().error("Could not parse structured response")
            self.speak_response("I couldn't understand the task properly")
        except Exception as e:
            self.get_logger().error(f"Error in complex execution: {e}")
            self.speak_response("An error occurred while executing the task")

    def execute_complex_action_sequence(self, action_sequence):
        """Execute a complex action sequence with monitoring and adaptation"""
        for i, action in enumerate(action_sequence):
            self.get_logger().info(f"Executing action {i+1}/{len(action_sequence)}: {action.get('action_type', 'unknown')}")
            
            # Update current action tracking
            self.active_action = action
            
            # Execute the action
            success = self.execute_monitored_action(action)
            
            if not success:
                self.get_logger().error(f"Action failed: {action}")
                
                # Check if we should continue or abort
                if action.get('critical', False):
                    self.get_logger().error("Critical action failed, aborting sequence")
                    return False
                else:
                    self.get_logger().warn(f"Non-critical action failed, continuing: {action.get('description', '')}")
            
            # Check for interruption from user
            if self.should_abort_sequence():
                self.get_logger().info("Action sequence interrupted by user")
                return False
        
        return True

    def execute_monitored_action(self, action):
        """Execute an action with monitoring and safety checks"""
        # Set a timeout for the action
        start_time = self.get_clock().now()
        timeout_duration = 60.0  # 60 seconds default timeout
        
        # Execute the action
        success = self.execute_single_action(action)
        
        # For actions that need monitoring, we would wait for feedback
        # For now, we'll just return the initial success
        return success

    def should_abort_sequence(self):
        """Check if current action sequence should be aborted"""
        # Check for new voice commands that might indicate interruption
        # In practice, this would be more sophisticated
        return False

    def execute_single_action(self, action):
        """Execute a single action with error handling"""
        action_type = action.get('action_type', 'unknown')
        params = action.get('parameters', {})
        description = action.get('description', '')
        
        self.get_logger().info(f"Executing: {description}")
        
        try:
            if action_type == 'navigate_to':
                return self.execute_navigation(params)
            elif action_type == 'grasp_object':
                return self.execute_grasp(params)
            elif action_type == 'speak':
                return self.execute_speech(params)
            elif action_type == 'detect_object':
                return self.execute_detection(params)
            elif action_type == 'wait':
                return self.execute_wait(params)
            elif action_type == 'verify_action':
                return self.execute_verification(params)
            else:
                self.get_logger().warn(f"Unknown action type: {action_type}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error executing action {action_type}: {e}")
            return False

    def execute_navigation(self, params):
        """Execute navigation action with path planning"""
        try:
            target_location = params.get('location', 'unknown')
            
            # Define location map
            location_map = {
                'kitchen': {'x': 2.0, 'y': 1.0},
                'living room': {'x': 0.0, 'y': 0.0},
                'bedroom': {'x': -1.0, 'y': 2.0},
                'office': {'x': 1.5, 'y': -1.0}
            }
            
            if target_location in location_map:
                coords = location_map[target_location]
                
                # Create navigation goal
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = self.get_clock().now().to_msg()
                goal.target_pose.pose.position.x = coords['x']
                goal.target_pose.pose.position.y = coords['y']
                goal.target_pose.pose.orientation.w = 1.0
                
                # Send goal to navigation server
                self.nav_client.send_goal(goal)
                
                # Wait for result with timeout
                finished = self.nav_client.wait_for_result(timeout=60.0)
                
                if finished:
                    result = self.nav_client.get_result()
                    if result:
                        self.get_logger().info(f"Navigation to {target_location} completed")
                        
                        # Update environment map
                        self.interaction_context['environment_map'][target_location] = {
                            'visited': True,
                            'timestamp': self.get_clock().now().nanoseconds
                        }
                        
                        return True
                    else:
                        self.get_logger().error(f"Navigation to {target_location} failed")
                        return False
                else:
                    self.nav_client.cancel_goal()
                    self.get_logger().error(f"Navigation to {target_location} timed out")
                    return False
            else:
                self.get_logger().warn(f"Unknown location: {target_location}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error in navigation: {e}")
            return False

    def execute_grasp(self, params):
        """Execute grasp action with verification"""
        try:
            obj_name = params.get('object', 'unknown')
            self.get_logger().info(f"Attempting to grasp: {obj_name}")
            
            # In a real robot, this would:
            # 1. Locate the object in 3D space
            # 2. Plan grasp trajectory
            # 3. Execute grasp motion
            # 4. Verify grasp success
            # For simulation, we'll just return success
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in grasp: {e}")
            return False

    def execute_speech(self, params):
        """Execute speech action"""
        try:
            text = params.get('text', 'Hello')
            self.speak_response(text)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in speech: {e}")
            return False

    def execute_detection(self, params):
        """Execute detection action with perception system"""
        try:
            # This would interface with the perception system to detect a specific object
            obj_name = params.get('object', 'object')
            self.get_logger().info(f"Attempting to detect: {obj_name}")
            
            # In a real implementation, this would call the perception system
            # and wait for detection results
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")
            return False

    def execute_wait(self, params):
        """Execute wait action"""
        try:
            duration = params.get('duration', 1.0)  # Default to 1 second
            self.get_logger().info(f"Waiting for {duration} seconds")
            time.sleep(duration)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in wait: {e}")
            return False

    def execute_verification(self, params):
        """Execute verification action to confirm a condition"""
        try:
            condition = params.get('condition', 'unknown')
            self.get_logger().info(f"Verifying condition: {condition}")
            
            # In a real implementation, this would check if a condition is met
            # For example: "verify_object_grasped", "verify_at_location", etc.
            return True  # Simplified
            
        except Exception as e:
            self.get_logger().error(f"Error in verification: {e}")
            return False

    def speak_response(self, text):
        """Publish text for speech synthesis"""
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)
        self.get_logger().info(f"Speaking: {text}")

    def perception_processing_callback(self):
        """Periodic perception processing"""
        # Process recent images for scene understanding
        if self.image_buffer:
            latest_entry = list(self.image_buffer)[-1]
            # Note: In a real implementation, perception processing should be
            # done in a separate thread to avoid blocking

    def context_update_callback(self):
        """Periodically update interaction context"""
        # Update last interaction time
        self.interaction_context['last_interaction'] = self.get_clock().now()
        
        # Update status if we have an active task
        if self.current_task:
            status_msg = String()
            status_msg.data = f"Working on: {self.current_task['command'][:50]}..."
            self.vla_status_pub.publish(status_msg)
        else:
            status_msg = String()
            status_msg.data = "Ready for command"
            self.vla_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedVLANode()
    
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

### Task 4: VLA System Testing and Validation (90 minutes)

Create comprehensive tests for the VLA system:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse
import time
import json

class VLAValidationNode(Node):
    def __init__(self):
        super().__init__('vla_validation_node')
        
        # Publishers for sending test commands
        self.test_command_pub = self.create_publisher(String, 'test_commands', 10)
        self.llm_request_pub = self.create_publisher(GPTRequest, 'structured_llm_requests', 10)
        
        # Subscribers for monitoring system responses
        self.system_response_sub = self.create_subscription(
            String, 'text_to_speech', self.response_callback, 10
        )
        self.vla_status_sub = self.create_subscription(
            String, 'vla_status', self.status_callback, 10
        )
        
        # Test tracking
        self.current_test = None
        self.test_results = []
        self.responses_received = []
        
        # Timer for running validation tests
        self.test_timer = self.create_timer(5.0, self.run_next_test)
        self.test_phase = 0  # Which test to run next
        
        self.get_logger().info('VLA Validation Node initialized')
    
    def run_next_test(self):
        """Run the next validation test in the sequence"""
        tests = [
            self.test_simple_navigation,
            self.test_object_interaction,
            self.test_gesture_integration,
            self.test_context_awareness,
            self.test_multi_step_task
        ]
        
        if self.test_phase < len(tests):
            test_func = tests[self.test_phase]
            self.get_logger().info(f"Running validation test {self.test_phase + 1}: {test_func.__name__}")
            
            # Clear previous responses
            self.responses_received = []
            
            # Run the test
            test_func()
            
            # Update to next test
            self.test_phase += 1
        else:
            # All tests completed, report results
            self.report_results()
            self.test_timer.cancel()
    
    def test_simple_navigation(self):
        """Test 1: Simple navigation command"""
        self.current_test = "Simple Navigation"
        
        # Create a simple navigation command
        command = String()
        command.data = "Go to the kitchen"
        
        # Send the command
        self.test_command_pub.publish(command)
        self.get_logger().info("Sent navigation command: 'Go to the kitchen'")
    
    def test_object_interaction(self):
        """Test 2: Object detection and interaction"""
        self.current_test = "Object Interaction"
        
        # Create an object interaction command
        command = String()
        command.data = "Find the red cup and bring it to me"
        
        # Send the command
        self.test_command_pub.publish(command)
        self.get_logger().info("Sent object command: 'Find the red cup and bring it to me'")
    
    def test_gesture_integration(self):
        """Test 3: Gesture integration"""
        self.current_test = "Gesture Integration"
        
        # To test gesture integration, we would need to publish a gesture message
        # For this test, we'll send a command that should respond to gestures
        command = String()
        command.data = "Wave back if you see me waving"
        
        # Send the command
        self.test_command_pub.publish(command)
        self.get_logger().info("Sent gesture command: 'Wave back if you see me waving'")
    
    def test_context_awareness(self):
        """Test 4: Context awareness"""
        self.current_test = "Context Awareness"
        
        # Test the system's ability to maintain context
        command = String()
        command.data = "Remember that the blue book is on the table"
        
        # Send the command
        self.test_command_pub.publish(command)
        self.get_logger().info("Sent context command: 'Remember that the blue book is on the table'")
    
    def test_multi_step_task(self):
        """Test 5: Multi-step task execution"""
        self.current_test = "Multi-Step Task"
        
        # Create a complex multi-step command
        command = String()
        command.data = "Go to the kitchen, find a cup, bring it to the living room, and place it on the coffee table"
        
        # Send the command
        self.test_command_pub.publish(command)
        self.get_logger().info("Sent multi-step command: 'Go to the kitchen, find a cup, bring it to the living room, and place it on the coffee table'")
    
    def response_callback(self, msg):
        """Handle system responses during testing"""
        self.responses_received.append({
            'test': self.current_test,
            'response': msg.data,
            'timestamp': self.get_clock().now()
        })
        
        self.get_logger().info(f"Received response during {self.current_test}: {msg.data}")
    
    def status_callback(self, msg):
        """Handle system status during testing"""
        self.get_logger().info(f"System status: {msg.data}")
    
    def report_results(self):
        """Report validation test results"""
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("VLA SYSTEM VALIDATION RESULTS")
        self.get_logger().info("="*50)
        
        for i, response in enumerate(self.responses_received):
            self.get_logger().info(f"  {i+1}. Test: {response['test']}")
            self.get_logger().info(f"     Response: {response['response']}")
            self.get_logger().info("")
        
        # Calculate success metrics
        total_responses = len(self.responses_received)
        meaningful_responses = sum(1 for r in self.responses_received 
                                  if len(r['response']) > 5 and 
                                  not r['response'].lower().startswith('i don'))
        
        success_rate = (meaningful_responses / total_responses * 100) if total_responses > 0 else 0
        
        self.get_logger().info(f"Total responses: {total_responses}")
        self.get_logger().info(f"Meaningful responses: {meaningful_responses}")
        self.get_logger().info(f"Success rate: {success_rate:.1f}%")
        
        self.get_logger().info("="*50)

def main(args=None):
    rclpy.init(args=args)
    node = VLAValidationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Validation interrupted, reporting results...")
        node.report_results()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Task 5: Integration Launch and Documentation (60 minutes)

Create launch files to bring up the entire VLA system:

```python
# multimodal_input_node/launch/vla_system.launch.py
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
        DeclareLaunchArgument(
            'whisper_model',
            default_value='base',
            description='Whisper model to use (tiny, base, small, medium, large)'
        ),
        
        # Audio processing (Whisper)
        Node(
            package='audio_input_node',
            executable='audio_processor',
            name='audio_processor_node'
        ),
        
        # Vision processing
        Node(
            package='multimodal_input_node',
            executable='vision_processor',
            name='vision_processor_node'
        ),
        
        # Gesture processing
        Node(
            package='multimodal_input_node',
            executable='gesture_processor',
            name='gesture_processor_node'
        ),
        
        # Multi-modal fusion
        Node(
            package='multimodal_input_node',
            executable='multi_modal_fusion',
            name='multi_modal_fusion_node'
        ),
        
        # LLM Planner
        Node(
            package='llm_planning_node',
            executable='structured_llm_planner',
            name='structured_llm_planner_node',
            parameters=[{
                'openai_api_key': LaunchConfiguration('openai_api_key')
            }]
        ),
        
        # Advanced VLA Controller
        Node(
            package='multimodal_input_node',
            executable='advanced_vla_controller',
            name='advanced_vla_controller_node'
        ),
        
        # Multi-modal output
        Node(
            package='multimodal_input_node',
            executable='multi_modal_output',
            name='multi_modal_output_node'
        ),
        
        # Validation node (for testing)
        Node(
            package='multimodal_input_node',
            executable='vla_validation',
            name='vla_validation_node'
        )
    ])
```

## Complete System Demonstration

Create a demonstration script showing the VLA system in action:

```bash
#!/bin/bash
# vla_demo.sh - Demonstration script for the complete VLA system

echo "Starting VLA (Vision-Language-Action) System Demonstration"

# The complete VLA system would be started with:
echo "ros2 launch multimodal_input_node vla_system.launch.py openai_api_key:='your-api-key-here'"

echo "Once the system is running, you can interact with it using voice commands like:"
echo " - 'Take the red cup to the kitchen'"
echo " - 'Pointed at the book, bring it to me'"
echo " - 'Go to the living room and wait for me'"
echo " - 'What objects do you see?'"

echo ""
echo "System architecture:"
echo "1. Audio Input → Whisper → Text Commands"
echo "2. Visual Input → Object Detection/Gesture Recognition → Context"
echo "3. Multi-Modal Fusion → LLM Planning → Action Sequences"
echo "4. Action Execution → Robot Control → Feedback"
echo "5. Validation → Performance Metrics → System Status"
```

## Exercise Deliverables

For this exercise, create a submission that includes:

1. **Complete source code** for the integrated VLA system
2. **Launch files** to start the complete system
3. **Documentation** explaining your system architecture and design decisions
4. **Test results** from the validation exercises
5. **Demonstration video** showing the system in action
6. **Reflection** on challenges of system integration and performance

## Evaluation Criteria

- **System Integration** (30%): All components work together seamlessly
- **Performance** (25%): System responds appropriately to multi-modal inputs
- **Complex Task Execution** (20%): Ability to execute complex, multi-step tasks
- **Code Quality** (15%): Well-structured, documented, and maintainable code
- **Validation Results** (10%): Successful completion of validation tests

## Troubleshooting

Common integration issues and solutions:

1. **Timing Issues**: Different components operate at different frequencies; implement appropriate buffering and synchronization mechanisms.

2. **Resource Competition**: Multiple nodes using the same resources (like camera); implement resource sharing or time-slicing.

3. **Communication Failures**: Messages not getting through; verify ROS graph, topic names, and message types.

4. **Performance Degradation**: System too slow with all components active; optimize individual components and consider task prioritization.

## Extensions

For advanced students, consider implementing:

1. **Learning Mechanisms**: Adapt system behavior based on interaction outcomes
2. **Memory Systems**: Remember object locations, user preferences, and past interactions
3. **Error Recovery**: Handle and recover from failed actions gracefully
4. **Safety Mechanisms**: Implement safety checks and fail-safes for physical robots

## Conclusion

This exercise has provided comprehensive experience with integrating a complete Vision-Language-Action system. You've learned to connect perception, language understanding, and action execution into a unified system that can understand natural language commands, perceive its environment, and execute complex tasks. This represents a sophisticated level of cognitive robotics integration.