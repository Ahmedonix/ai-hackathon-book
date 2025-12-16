# Practical Exercise: LLM-Based Action Planning for Humanoid Robots

## Objective

In this exercise, you will implement a system that uses a Large Language Model (LLM) to create action plans for a humanoid robot based on natural language commands. You will learn how to structure prompts for LLMs, parse structured outputs, and convert high-level commands into executable robot actions.

## Prerequisites

Before starting this exercise, you should have:
- Completed Module 1 (ROS 2 fundamentals)
- Set up your ROS 2 Iron environment
- Access to an LLM API (OpenAI GPT preferred)
- Basic knowledge of ROS 2 action servers and clients
- Understanding of robot navigation and manipulation concepts

## Time Estimate

This exercise should take approximately 4-5 hours to complete, depending on your familiarity with LLM APIs and robot planning concepts.

## Setup

### Install Required Dependencies

First, install the required Python packages:

```bash
pip3 install openai
```

### LLM API Access

For this exercise, you'll need access to an LLM API. We'll use OpenAI's GPT as an example, but you can adapt it to other LLMs.

## Exercise Tasks

### Task 1: Basic LLM Interface (45 minutes)

Create a basic ROS 2 node that can send commands to an LLM and receive responses:

1. Create a new ROS package for LLM planning:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python llm_planning_node
   cd llm_planning_node/llm_planning_node
   ```

2. Create the basic LLM interface in `llm_planner.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import openai
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse

class LLMPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_planner_node')
        
        # Get LLM API key from parameters
        self.declare_parameter('openai_api_key', '')
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value
        
        if not self.api_key:
            self.get_logger().error("OpenAI API key is required!")
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        
        # Publishers and subscribers
        self.request_sub = self.create_subscription(
            GPTRequest, 'llm_requests', self.request_callback, 10
        )
        self.response_pub = self.create_publisher(
            GPTResponse, 'llm_responses', 10
        )
        
        self.get_logger().info('LLM Planner Node initialized')

    def request_callback(self, request_msg):
        """Handle incoming LLM requests"""
        try:
            # Prepare the message for the LLM
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a planner for a humanoid robot. Respond with clear, executable actions when possible. Format responses as structured JSON when appropriate."},
                    {"role": "user", "content": request_msg.command}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # Extract and publish the response
            gpt_response = response.choices[0].message['content'].strip()
            
            response_msg = GPTResponse()
            response_msg.header.stamp = self.get_clock().now().to_msg()
            response_msg.header.frame_id = "base_link"
            response_msg.request_id = request_msg.id
            response_msg.response = gpt_response
            
            self.response_pub.publish(response_msg)
            self.get_logger().info(f"Processed request: {request_msg.command[:50]}...")
            
        except Exception as e:
            self.get_logger().error(f"Error processing LLM request: {e}")
            
            error_msg = GPTResponse()
            error_msg.header.stamp = self.get_clock().now().to_msg()
            error_msg.header.frame_id = "base_link"
            error_msg.request_id = request_msg.id
            error_msg.response = f"Error: {str(e)}"
            error_msg.is_error = True
            
            self.response_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    
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

### Task 2: Structured LLM Planner (60 minutes)

Enhance the planner to produce structured outputs that can be directly used for robot action planning:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import openai
import json
import re
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse
from humanoid_msgs.msg import ActionSequence, RobotAction

class StructuredLLMPlannerNode(Node):
    def __init__(self):
        super().__init__('structured_llm_planner_node')
        
        # Get LLM API key from parameters
        self.declare_parameter('openai_api_key', '')
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value
        
        if not self.api_key:
            self.get_logger().error("OpenAI API key is required!")
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        
        # Publishers and subscribers
        self.request_sub = self.create_subscription(
            GPTRequest, 'structured_llm_requests', self.request_callback, 10
        )
        self.response_pub = self.create_publisher(
            GPTResponse, 'structured_llm_responses', 10
        )
        
        self.get_logger().info('Structured LLM Planner Node initialized')

    def request_callback(self, request_msg):
        """Handle incoming requests with structured planning"""
        try:
            # Create a detailed prompt that guides the LLM to produce structured output
            prompt = f"""
            You are a task planner for a humanoid robot. Convert the following command into a sequence of specific robot actions.
            
            Command: {request_msg.command}
            
            Context: {request_msg.context}
            
            Available Actions:
            - navigate_to: Move robot to a specific location
            - detect_object: Locate a specific object in the environment
            - grasp_object: Pick up an object
            - place_object: Put down an object at a location
            - speak: Say something to a person
            - find_person: Locate a person in the environment
            - follow_person: Follow a person
            - wait: Pause for a specified time
            
            Respond ONLY with a JSON array of actions in this format:
            {{
                "action_sequence": [
                    {{
                        "action_type": "navigate_to",
                        "parameters": {{"location": "kitchen"}},
                        "description": "Move to the kitchen"
                    }},
                    {{
                        "action_type": "detect_object", 
                        "parameters": {{"object": "red cup"}},
                        "description": "Look for the red cup"
                    }}
                ]
            }}
            
            Be specific about locations and objects. If the command is unclear, ask for clarification.
            """
            
            # Call the LLM with the structured prompt
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise task planner for a humanoid robot. Always respond with properly formatted JSON as specified."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2,
                stop=None
            )
            
            # Extract the response
            raw_response = response.choices[0].message['content'].strip()
            self.get_logger().info(f"Raw LLM response: {raw_response}")
            
            # Try to extract JSON from the response
            structured_data = self.extract_json_from_response(raw_response)
            
            # Create response message
            response_msg = GPTResponse()
            response_msg.header.stamp = self.get_clock().now().to_msg()
            response_msg.header.frame_id = "base_link"
            response_msg.request_id = request_msg.id
            response_msg.response = raw_response
            
            if structured_data and "action_sequence" in structured_data:
                # Convert to structured format
                try:
                    response_msg.structured_response = json.dumps(structured_data)
                    self.get_logger().info(f"Generated action sequence with {len(structured_data['action_sequence'])} actions")
                except Exception as e:
                    self.get_logger().error(f"Error creating structured response: {e}")
                    response_msg.structured_response = json.dumps({"action_sequence": []})
            else:
                self.get_logger().warn("Could not extract action sequence from LLM response")
                response_msg.structured_response = json.dumps({"action_sequence": [], "error": "No valid action sequence generated"})
            
            self.response_pub.publish(response_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing structured LLM request: {e}")
            
            error_msg = GPTResponse()
            error_msg.header.stamp = self.get_clock().now().to_msg()
            error_msg.header.frame_id = "base_link"
            error_msg.request_id = request_msg.id
            error_msg.response = f"Error: {str(e)}"
            error_msg.is_error = True
            error_msg.structured_response = json.dumps({"action_sequence": []})
            
            self.response_pub.publish(error_msg)
    
    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response, handling various formats"""
        # Look for JSON blocks in the response
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON structure directly in the response
        try:
            # Find the first { and the last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If no JSON found, return None
        return None

def main(args=None):
    rclpy.init(args=args)
    node = StructuredLLMPlannerNode()
    
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

### Task 3: Robot Action Execution Interface (60 minutes)

Create an interface that can take the planned actions and execute them using ROS 2 action servers:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from humanoid_msgs.msg import GPTResponse
from humanoid_msgs.msg import NavigationAction, NavigationGoal
from humanoid_msgs.msg import ManipulationAction, ManipulationGoal
from std_msgs.msg import String

class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution_node')
        
        # Initialize action clients for different robot capabilities
        self.nav_client = SimpleActionClient('navigation_server', NavigationAction)
        self.manip_client = SimpleActionClient('manipulation_server', ManipulationAction)
        
        # Wait for action servers (with warning if not available)
        nav_connected = self.nav_client.wait_for_server(timeout_sec=5.0)
        manip_connected = self.manip_client.wait_for_server(timeout_sec=5.0)
        
        if not nav_connected:
            self.get_logger().warn("Navigation server not available")
        if not manip_connected:
            self.get_logger().warn("Manipulation server not available")
        
        # Subscribers
        self.planning_response_sub = self.create_subscription(
            GPTResponse, 'structured_llm_responses', self.planning_response_callback, 10
        )
        
        # Publishers
        self.execution_status_pub = self.create_publisher(
            String, 'action_execution_status', 10
        )
        
        self.get_logger().info('Action Execution Node initialized')

    def planning_response_callback(self, response_msg):
        """Process the structured planning response and execute actions"""
        if response_msg.is_error:
            self.get_logger().error(f"Planning error: {response_msg.response}")
            return
        
        try:
            # Parse the structured response
            structured_data = json.loads(response_msg.structured_response)
            
            if "action_sequence" in structured_data:
                action_sequence = structured_data["action_sequence"]
                self.get_logger().info(f"Executing action sequence with {len(action_sequence)} actions")
                
                # Execute action sequence
                self.execute_action_sequence(action_sequence, response_msg.request_id)
            else:
                self.get_logger().warn("No action sequence found in structured response")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Error parsing structured response: {e}")
        except Exception as e:
            self.get_logger().error(f"Error executing action sequence: {e}")

    def execute_action_sequence(self, action_sequence, request_id):
        """Execute a sequence of planned actions"""
        for i, action in enumerate(action_sequence):
            self.get_logger().info(f"Executing action {i+1}/{len(action_sequence)}: {action.get('action_type', 'unknown')}")
            
            success = self.execute_single_action(action)
            
            if not success:
                self.get_logger().error(f"Action failed: {action}")
                # In a real system, you might want to have failure recovery strategies
                break

    def execute_single_action(self, action):
        """Execute a single planned action"""
        action_type = action.get('action_type', 'unknown')
        
        if action_type == 'navigate_to':
            return self.execute_navigation_action(action)
        elif action_type == 'grasp_object':
            return self.execute_manipulation_action(action)  
        elif action_type == 'speak':
            return self.execute_speech_action(action)
        elif action_type == 'detect_object':
            return self.execute_detection_action(action)
        else:
            self.get_logger().warn(f"Unknown action type: {action_type}")
            return False

    def execute_navigation_action(self, action):
        """Execute a navigation action"""
        try:
            # Extract parameters
            params = action.get('parameters', {})
            location = params.get('location', 'unknown')
            
            # In a real implementation, you would have a map of location names to coordinates
            # For this example, we'll use a simple mapping
            location_coords = {
                'kitchen': {'x': 2.0, 'y': 1.0},
                'living room': {'x': 0.0, 'y': 0.0}, 
                'bedroom': {'x': -1.0, 'y': 2.0}
            }
            
            if location.lower() in location_coords:
                coords = location_coords[location.lower()]
                
                # Create navigation goal
                goal = NavigationGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = self.get_clock().now().to_msg()
                goal.target_pose.pose.position.x = coords['x']
                goal.target_pose.pose.position.y = coords['y']
                goal.target_pose.pose.orientation.w = 1.0
                
                # Send goal to navigation server
                self.nav_client.send_goal(goal)
                
                # Wait for result (simplified - in practice, you'd want a non-blocking approach)
                self.nav_client.wait_for_result()
                result = self.nav_client.get_result()
                
                if result:
                    self.get_logger().info(f"Navigation to {location} completed")
                    return True
                else:
                    self.get_logger().error(f"Navigation to {location} failed")
                    return False
            else:
                self.get_logger().error(f"Unknown location: {location}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error executing navigation action: {e}")
            return False

    def execute_manipulation_action(self, action):
        """Execute a manipulation action (placeholder)"""
        try:
            params = action.get('parameters', {})
            obj_name = params.get('object', 'unknown')
            
            self.get_logger().info(f"Attempting to manipulate object: {obj_name}")
            
            # In a real implementation, this would interface with the manipulation stack
            # For now, we'll just simulate success
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error executing manipulation action: {e}")
            return False

    def execute_speech_action(self, action):
        """Execute a speech action"""
        try:
            params = action.get('parameters', {})
            text = params.get('text', 'Hello')
            
            # In a real implementation, this would interface with a TTS system
            self.get_logger().info(f"Robot would speak: {text}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error executing speech action: {e}")
            return False

    def execute_detection_action(self, action):
        """Execute a detection action (placeholder)"""
        try:
            params = action.get('parameters', {})
            obj_type = params.get('object', 'object')
            
            self.get_logger().info(f"Attempting to detect: {obj_type}")
            
            # In a real implementation, this would interface with the perception stack
            # For now, we'll just simulate success
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error executing detection action: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = ActionExecutionNode()
    
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

Create a launch file to bring up the complete LLM planning system:

```python
# llm_planning_node/launch/llm_planning_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'openai_api_key',
            description='OpenAI API key for LLM access'
        ),
        
        # LLM Planner Node
        Node(
            package='llm_planning_node',
            executable='structured_llm_planner',
            name='structured_llm_planner_node',
            parameters=[{
                'openai_api_key': LaunchConfiguration('openai_api_key')
            }]
        ),
        
        # Action Execution Node
        Node(
            package='llm_planning_node',
            executable='action_execution',
            name='action_execution_node'
        )
    ])
```

Test your system with sample commands:

```bash
# Terminal 1: Start the planning system
ros2 launch llm_planning_node llm_planning_system.launch.py openai_api_key:="your-api-key-here"

# Terminal 2: Send a test command
ros2 topic pub /structured_llm_requests humanoid_msgs/msg/GPTRequest "header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: 'test'
id: 'test1'
command: 'Go to the kitchen and bring me a cup'
context: 'The kitchen is 2m away in the positive x direction, there is a cup on the table'" --once
```

### Task 5: Advanced Planning Scenarios (60 minutes)

Implement more sophisticated planning scenarios by creating specific planning strategies for different robot behaviors:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import openai
import json
import re
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse

class AdvancedLLMPlannerNode(Node):
    def __init__(self):
        super().__init__('advanced_llm_planner_node')
        
        # Get LLM API key from parameters
        self.declare_parameter('openai_api_key', '')
        self.api_key = self.get_parameter('openai_api_key').get_parameter_value().string_value
        
        if not self.api_key:
            self.get_logger().error("OpenAI API key is required!")
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        
        # Publishers and subscribers
        self.request_sub = self.create_subscription(
            GPTRequest, 'advanced_llm_requests', self.request_callback, 10
        )
        self.response_pub = self.create_publisher(
            GPTResponse, 'advanced_llm_responses', 10
        )
        
        self.get_logger().info('Advanced LLM Planner Node initialized')

    def request_callback(self, request_msg):
        """Handle incoming requests with advanced planning strategies"""
        try:
            # Determine the type of request and apply appropriate planning strategy
            command = request_msg.command.lower()
            
            if any(word in command for word in ["navigate", "go to", "move to", "walk to"]):
                # Navigation-focused planning
                strategy = self.navigation_strategy(request_msg)
            elif any(word in command for word in ["grasp", "pick up", "get", "bring"]):
                # Manipulation-focused planning
                strategy = self.manipulation_strategy(request_msg)
            elif any(word in command for word in ["follow", "accompany", "escort"]):
                # Social interaction planning
                strategy = self.social_strategy(request_msg)
            else:
                # General purpose planning
                strategy = self.general_strategy(request_msg)
            
            # Execute the appropriate planning strategy
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": strategy['system_prompt']},
                    {"role": "user", "content": strategy['user_prompt']}
                ],
                max_tokens=500,
                temperature=0.2,
                functions=strategy.get('functions', []),
                function_call="auto" if strategy.get('functions') else None
            )
            
            raw_response = response.choices[0].message['content'].strip()
            
            # Handle function calls if present
            if 'function_call' in response.choices[0].message:
                function_call = response.choices[0].message['function_call']
                self.get_logger().info(f"LLM suggested function: {function_call['name']}")
                # In a real implementation, you would execute the function
                raw_response = f"Function {function_call['name']} called"
            
            # Create response message
            response_msg = GPTResponse()
            response_msg.header.stamp = self.get_clock().now().to_msg()
            response_msg.header.frame_id = "base_link"
            response_msg.request_id = request_msg.id
            response_msg.response = raw_response
            
            # Try to parse structured response
            try:
                structured_data = self.extract_json_from_response(raw_response)
                if structured_data:
                    response_msg.structured_response = json.dumps(structured_data)
                else:
                    response_msg.structured_response = json.dumps({"text_response": raw_response})
            except:
                response_msg.structured_response = json.dumps({"text_response": raw_response})
            
            self.response_pub.publish(response_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in advanced planning: {e}")
            
            error_msg = GPTResponse()
            error_msg.header.stamp = self.get_clock().now().to_msg()
            error_msg.header.frame_id = "base_link"
            error_msg.request_id = request_msg.id
            error_msg.response = f"Error: {str(e)}"
            error_msg.is_error = True
            error_msg.structured_response = json.dumps({})
            
            self.response_pub.publish(error_msg)

    def navigation_strategy(self, request_msg):
        """Create a planning strategy for navigation tasks"""
        return {
            "system_prompt": "You are a navigation planner for a humanoid robot. Break down navigation tasks into specific, measurable steps. Consider obstacles, doorways, and room layout. Always provide GPS coordinates or relative positions.",
            "user_prompt": f"""
            Plan navigation for this command: 
            '{request_msg.command}'
            
            Context: {request_msg.context}
            
            Provide your plan as a JSON object with:
            1. A sequence of waypoints (x, y coordinates)
            2. Potential obstacles to look out for
            3. Landmarks for localization
            4. Expected travel time
            
            Example format:
            {{
                "waypoints": [
                    {{"x": 1.0, "y": 0.0, "description": "Exit current room"}},
                    {{"x": 2.5, "y": 1.0, "description": "Turn right at hallway intersection"}},
                    {{"x": 3.0, "y": 2.0, "description": "Destination reached"}}
                ],
                "obstacles": ["open doors", "furniture"],
                "landmarks": ["red chair", "painting on wall"],
                "estimated_time_minutes": 3
            }}
            """
        }

    def manipulation_strategy(self, request_msg):
        """Create a planning strategy for manipulation tasks"""
        return {
            "system_prompt": "You are a manipulation planner for a humanoid robot. Consider object properties (size, weight, fragility), grasp types, collision avoidance, and workspace constraints. Provide detailed steps for grasping and placing objects.",
            "user_prompt": f"""
            Plan manipulation for this command: 
            '{request_msg.command}'
            
            Context: {request_msg.context}
            
            Provide your plan as a JSON object with:
            1. Object detection requirements
            2. Suitable grasp types
            3. Collision-free trajectory
            4. Placement requirements
            5. Safety considerations
            
            Example format:
            {{
                "object_detection": {{
                    "object_name": "cup",
                    "features_to_look_for": ["cylindrical shape", "handle", "top opening"]
                }},
                "grasp_plan": {{
                    "grasp_type": "cylindrical grasp",
                    "grasp_position": "opposite side of handle",
                    "gripper_width": 80
                }},
                "trajectory": [
                    "approach object from top",
                    "align gripper with object axis",
                    "close gripper softly",
                    "lift object 10cm"
                ],
                "placement": {{
                    "target_location": "kitchen counter",
                    "orientation": "upright"
                }},
                "safety_considerations": ["fragile object", "avoid tilting"]
            }}
            """
        }

    def social_strategy(self, request_msg):
        """Create a planning strategy for social interaction tasks"""
        return {
            "system_prompt": "You are a social behavior planner for a humanoid robot. Consider human comfort zones, appropriate following distance, eye contact, and social norms during interaction.",
            "user_prompt": f"""
            Plan social interaction for this command: 
            '{request_msg.command}'
            
            Context: {request_msg.context}
            
            Provide your plan as a JSON object with:
            1. Personal space management
            2. Movement patterns
            3. Communication strategies
            4. Safety considerations regarding human proximity
            
            Example format:
            {{
                "personal_space": {{
                    "following_distance": 1.5,
                    "interaction_distance": 1.0,
                    "respect": true
                }},
                "movement_pattern": "keep pace with human, stop when human stops",
                "communication": {{
                    "acknowledge_commands": true,
                    "provide_status_updates": true
                }},
                "safety": {{
                    "maintain_clear_path": true,
                    "avoid_sudden_movements": true
                }}
            }}
            """
        }

    def general_strategy(self, request_msg):
        """General purpose planning strategy"""
        return {
            "system_prompt": "You are a task planner for a humanoid robot. Break down complex tasks into simple, actionable steps. Consider robot capabilities, environmental constraints, and safety.",
            "user_prompt": f"""
            Plan actions for this command: 
            '{request_msg.command}'
            
            Context: {request_msg.context}
            
            Provide your plan as a JSON array of actions in the following format:
            [
                {{
                    "action_type": "navigate_to",
                    "parameters": {{"location": "kitchen"}},
                    "description": "Move to the kitchen"
                }},
                {{
                    "action_type": "detect_object", 
                    "parameters": {{"object": "red cup"}},
                    "description": "Look for the red cup"
                }}
            ]
            """
        }

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response, handling various formats"""
        # Look for JSON blocks in the response
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON structure directly in the response
        try:
            # Find the first { and the last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If no JSON found, return None
        return None

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedLLMPlannerNode()
    
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

1. **Complete source code** for your LLM planning system
2. **Launch files** to start your nodes
3. **Documentation** explaining how your planning system works
4. **Test results** showing successful planning for various commands
5. **Reflection** on challenges faced and lessons learned

## Evaluation Criteria

- **Functionality** (35%): Your system correctly plans robot actions from natural language commands
- **LLM Integration** (25%): Effective use of LLM for generating structured outputs
- **Code Quality** (20%): Well-structured, documented, and maintainable code
- **ROS Integration** (15%): Proper use of ROS concepts and message types
- **Problem-Solving** (5%): Effective debugging and error handling

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**:
   - Ensure your OpenAI API key is valid and has sufficient credits
   - Check that the key is properly formatted and not expired

2. **Structured Output Problems**:
   - Make your prompts more specific about the expected JSON format
   - Use function calling if available in your LLM API

3. **Performance Issues**:
   - LLM calls can be slow; consider caching for common commands
   - Implement timeout handling for LLM requests

## Extensions

For advanced students, consider implementing:

1. **Plan Validation**: Verify that generated plans are feasible with the robot's capabilities
2. **Multi-step Feedback**: Allow the robot to report progress and adjust plans
3. **Learning from Experience**: Improve planning based on successful/failed executions
4. **Collaborative Planning**: Allow humans to correct or modify generated plans

## Conclusion

This exercise has provided hands-on experience with using LLMs for robot action planning. You've learned how to structure prompts, parse structured outputs, and connect high-level commands to low-level robot actions. This forms a crucial component of cognitive robotics systems that can understand and act on natural language instructions.