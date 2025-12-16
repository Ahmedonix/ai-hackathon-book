# Implementing GPT Models with ROS 2 Communication

## Overview

This guide demonstrates the practical implementation of GPT models integrated with ROS 2 communication for creating intelligent robotic responses to natural language commands. This implementation builds upon the documentation created in the previous task.

## Implementation Steps

### Step 1: Set Up Dependencies

First, install the required Python packages:

```bash
pip3 install openai rospy
```

Or add to your `requirements.txt` or `package.xml`:

```xml
<exec_depend>python3-openai</exec_depend>
<exec_depend>rospy</exec_depend>
```

### Step 2: Define Custom Message Types

Create the custom message definitions needed for GPT communication in your package's `msg/` directory:

In `msg/GPTRequest.msg`:
```
Header header
string id
string command
string context
geometry_msgs/PoseStamped robot_pose
string[] detected_objects
string current_task
```

In `msg/GPTResponse.msg`:
```
Header header
string request_id
string response
string structured_response
bool is_error
string error_message
float32 confidence
```

### Step 3: Implement the GPT Interface Node

Create the complete implementation of the GPT interface:

```python
#!/usr/bin/env python3

import rospy
import openai
from std_msgs.msg import String
from sensor_msgs.msg import Image
from humanoid_msgs.msg import GPTRequest, GPTResponse
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import json
import time
import threading
from queue import Queue

class GPTInterface:
    def __init__(self):
        # Initialize OpenAI API key from ROS parameter server
        self.api_key = rospy.get_param('~openai_api_key', '')
        if not self.api_key:
            rospy.logerr("OpenAI API key not set! Use: rosparam set /gpt_interface/openai_api_key <your_key>")
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        
        # Initialize class variables
        self.context_history = []
        self.request_queue = Queue()
        self.response_callbacks = {}
        
        # Publishers and subscribers
        self.request_sub = rospy.Subscriber("/gpt_requests", GPTRequest, self.request_callback)
        self.response_pub = rospy.Publisher("/gpt_responses", GPTResponse, queue_size=10)
        self.text_to_speech_pub = rospy.Publisher("/text_to_speech", String, queue_size=10)
        self.marker_pub = rospy.Publisher("/gpt_markers", MarkerArray, queue_size=10)
        
        # Thread for processing requests
        self.processing_thread = threading.Thread(target=self.process_requests, daemon=True)
        self.processing_thread.start()
        
        rospy.loginfo("GPT Interface initialized with ROS 2 communication")
    
    def request_callback(self, request_msg):
        """Add incoming request to the processing queue"""
        self.request_queue.put(request_msg)
    
    def process_requests(self):
        """Process requests from the queue in a separate thread"""
        while not rospy.is_shutdown():
            try:
                # Wait for a request with timeout
                request_msg = self.request_queue.get(timeout=1.0)
                
                # Process the request
                self.process_gpt_request(request_msg)
                
                # Mark as processed
                self.request_queue.task_done()
                
            except:
                # Timeout when queue is empty, continue loop
                continue
    
    def process_gpt_request(self, request_msg):
        """Process a single GPT request"""
        try:
            # Build context for the GPT model
            messages = self.build_context(request_msg)
            
            # Call GPT API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Can be configured via parameter
                messages=messages,
                max_tokens=rospy.get_param('~max_tokens', 500),
                temperature=rospy.get_param('~temperature', 0.7),
                top_p=rospy.get_param('~top_p', 1.0)
            )
            
            # Extract the response
            gpt_response = response.choices[0].message['content'].strip()
            
            # Create and publish response message
            response_msg = self.create_response_msg(request_msg, gpt_response)
            self.response_pub.publish(response_msg)
            
            # Optionally publish text-to-speech output
            if rospy.get_param('~enable_tts', True):
                tts_msg = String()
                tts_msg.data = self.extract_speech_content(gpt_response)
                self.text_to_speech_pub.publish(tts_msg)
            
            # Update context history for follow-up conversations
            self.update_context_history(request_msg.command, gpt_response)
            
        except Exception as e:
            rospy.logerr(f"Error processing GPT request {request_msg.id}: {str(e)}")
            error_msg = self.create_error_response(request_msg.id, str(e))
            self.response_pub.publish(error_msg)
    
    def build_context(self, request_msg):
        """Build context for GPT with conversation history and current request"""
        # Start with system message to guide behavior
        messages = [
            {"role": "system", "content": rospy.get_param(
                '~system_prompt', 
                """You are an assistant for a humanoid robot. 
                Respond with clear, executable actions when possible. 
                Format responses as structured JSON for robotic tasks. 
                Respond to questions concisely. If asked to perform tasks, break them into simple steps. 
                Be aware of current robot state, detected objects, and environment."""
            )}
        ]
        
        # Add conversation history if available
        messages.extend(self.context_history)
        
        # Create the user message with context
        user_content = f"Command: {request_msg.command}\n"
        
        if request_msg.context:
            user_content += f"Context: {request_msg.context}\n"
        
        if request_msg.detected_objects:
            user_content += f"Detected objects: {', '.join(request_msg.detected_objects)}\n"
        
        if request_msg.robot_pose:
            user_content += f"Robot pose: ({request_msg.robot_pose.pose.position.x:.2f}, {request_msg.robot_pose.pose.position.y:.2f})\n"
        
        user_content += f"Current task: {request_msg.current_task}\n"
        user_content += "Please respond appropriately."
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def create_response_msg(self, request_msg, gpt_response):
        """Create a ROS message from GPT response"""
        response_msg = GPTResponse()
        response_msg.header.stamp = rospy.Time.now()
        response_msg.header.frame_id = "base_link"
        response_msg.request_id = request_msg.id
        response_msg.response = gpt_response
        
        # Try to parse structured response from GPT
        try:
            structured_data = self.parse_structured_response(gpt_response)
            response_msg.structured_response = json.dumps(structured_data)
        except:
            # If parsing fails, just store the raw response
            response_msg.structured_response = json.dumps({"text": gpt_response})
        
        response_msg.is_error = False
        response_msg.confidence = 0.9  # Default confidence
        
        return response_msg
    
    def parse_structured_response(self, gpt_response):
        """Parse structured response from GPT output"""
        import re
        
        # Look for JSON blocks in the response
        json_match = re.search(r'```json\n(.*?)\n```', gpt_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Look for JSON objects
        obj_match = re.search(r'\{.*\}', gpt_response, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))
        
        # If no JSON found, return the text response
        return {"text": gpt_response}
    
    def extract_speech_content(self, gpt_response):
        """Extract speech content from GPT response, removing action blocks"""
        import re
        
        # Remove code blocks and JSON structures from speech output
        clean_response = re.sub(r'```.*?```', '', gpt_response, flags=re.DOTALL)
        clean_response = re.sub(r'\{.*?\}', '', clean_response, flags=re.DOTALL)
        
        return clean_response.strip()
    
    def update_context_history(self, user_command, assistant_response):
        """Update the conversation history"""
        self.context_history.append({"role": "user", "content": user_command})
        self.context_history.append({"role": "assistant", "content": assistant_response})
        
        # Keep history to a reasonable size to manage token usage
        max_history = rospy.get_param('~max_context_history', 10)
        if len(self.context_history) > max_history * 2:  # Each exchange has 2 messages
            self.context_history = self.context_history[-max_history*2:]
    
    def create_error_response(self, request_id, error_msg):
        """Create an error response message"""
        response_msg = GPTResponse()
        response_msg.header.stamp = rospy.Time.now()
        response_msg.header.frame_id = "base_link"
        response_msg.request_id = request_id
        response_msg.response = f"Error: {error_msg}"
        response_msg.is_error = True
        response_msg.error_message = error_msg
        response_msg.confidence = 0.0
        
        return response_msg

def main():
    rospy.init_node('gpt_interface')
    
    try:
        # Initialize the GPT interface
        gpt_interface = GPTInterface()
        
        # Wait for shutdown
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("GPT Interface node interrupted")
    except Exception as e:
        rospy.logerr(f"GPT Interface error: {str(e)}")
    finally:
        rospy.loginfo("GPT Interface node shutting down")

if __name__ == '__main__':
    main()
```

### Step 4: Create a Launch File

Create a launch file to start the GPT interface with proper parameters:

In `launch/gpt_interface.launch`:
```xml
<launch>
  <!-- GPT Interface Node -->
  <node name="gpt_interface" pkg="humanoid_gpt" type="gpt_interface.py" output="screen">
    <!-- API Key - Set this appropriately -->
    <param name="openai_api_key" value="$(arg openai_api_key)" unless="$(arg use_param_server)" />
    <param name="openai_api_key" value="$(arg openai_api_key)" if="$(arg use_param_server)" />
    
    <!-- Model parameters -->
    <param name="model" value="gpt-3.5-turbo" />
    <param name="max_tokens" value="500" />
    <param name="temperature" value="0.7" />
    <param name="top_p" value="1.0" />
    
    <!-- Context parameters -->
    <param name="max_context_history" value="10" />
    <param name="system_prompt" value="You are an assistant for a humanoid robot. Respond with clear, executable actions when possible. Format responses as structured JSON for robotic tasks. Respond to questions concisely. If asked to perform tasks, break them into simple steps. Be aware of current robot state, detected objects, and environment." />
    <param name="enable_tts" value="true" />
  </node>
  
  <!-- Arguments -->
  <arg name="openai_api_key" default="" />
  <arg name="use_param_server" default="false" />
</launch>
```

### Step 5: Create a Simple Test Script

Create a test script to verify the integration:

```python
#!/usr/bin/env python3

import rospy
from humanoid_msgs.msg import GPTRequest, GPTResponse
from std_msgs.msg import String
import uuid

class GPTTestClient:
    def __init__(self):
        rospy.init_node('gpt_test_client')
        
        # Publisher for requests
        self.request_pub = rospy.Publisher('/gpt_requests', GPTRequest, queue_size=10)
        
        # Subscriber for responses
        self.response_sub = rospy.Subscriber('/gpt_responses', GPTResponse, self.response_callback)
        
        # Store pending requests
        self.pending_requests = {}
        
        rospy.loginfo("GPT Test Client initialized")
    
    def send_request(self, command, context=""):
        """Send a request to the GPT interface"""
        request_msg = GPTRequest()
        request_msg.header.stamp = rospy.Time.now()
        request_msg.header.frame_id = "base_link"
        request_msg.id = str(uuid.uuid4())
        request_msg.command = command
        request_msg.context = context
        
        self.pending_requests[request_msg.id] = rospy.Time.now()
        self.request_pub.publish(request_msg)
        
        rospy.loginfo(f"Sent request: {command}")
    
    def response_callback(self, response_msg):
        """Handle responses from the GPT interface"""
        if response_msg.request_id in self.pending_requests:
            del self.pending_requests[response_msg.request_id]
            
            if response_msg.is_error:
                rospy.logerr(f"GPT Error: {response_msg.error_message}")
            else:
                rospy.loginfo(f"GPT Response: {response_msg.response}")
                rospy.loginfo(f"Structured: {response_msg.structured_response}")
    
    def run_tests(self):
        """Run a series of test requests"""
        rospy.sleep(2.0)  # Allow connections to establish
        
        # Test 1: Simple question
        self.send_request("What is the capital of France?")
        
        rospy.sleep(5.0)
        
        # Test 2: Task planning
        self.send_request("Plan how to navigate to the kitchen and bring me a cup", "The kitchen is to the left of the living room. There's a table with cups in the kitchen.")
        
        rospy.sleep(5.0)
        
        # Test 3: Context-aware request
        self.send_request("Can you tell me about the objects you see?", "Current detected objects: chair, table, cup")
        
        rospy.sleep(10.0)  # Wait for responses

def main():
    client = GPTTestClient()
    client.run_tests()
    
    # Keep the node alive to receive responses
    rospy.spin()

if __name__ == '__main__':
    main()
```

### Step 6: Create a Service Integration

For more synchronous communication, create a ROS service:

In `srv/GPTService.srv`:
```
string command
string context
---
string response
string structured_response
bool success
string error_message
```

In `scripts/gpt_service.py`:
```python
#!/usr/bin/env python3

import rospy
from humanoid_msgs.srv import GPTService, GPTServiceResponse
from humanoid_msgs.msg import GPTRequest, GPTResponse
import uuid
import threading
import time

class GPTServiceWrapper:
    def __init__(self):
        # Initialize the service
        self.service = rospy.Service('gpt_service', GPTService, self.handle_service_request)
        
        # Publisher for GPT requests
        self.request_pub = rospy.Publisher('/gpt_requests', GPTRequest, queue_size=10)
        
        # Subscriber for responses
        self.response_sub = rospy.Subscriber('/gpt_responses', GPTResponse, self.response_callback)
        
        # Store pending service requests
        self.pending_services = {}
        self.lock = threading.Lock()
        
        rospy.loginfo("GPT Service Wrapper initialized")
    
    def handle_service_request(self, req):
        """Handle a service request by forwarding to the main GPT node"""
        request_id = str(uuid.uuid4())
        
        # Create and send GPT request
        gpt_request = GPTRequest()
        gpt_request.header.stamp = rospy.Time.now()
        gpt_request.header.frame_id = "base_link"
        gpt_request.id = request_id
        gpt_request.command = req.command
        gpt_request.context = req.context
        
        # Store the service request waiting for response
        with self.lock:
            self.pending_services[request_id] = {
                'request_time': rospy.Time.now(),
                'service_request': req,
                'response': None,
                'event': threading.Event()
            }
        
        # Publish the request
        self.request_pub.publish(gpt_request)
        
        # Wait for response (with timeout)
        service_data = self.pending_services[request_id]
        timeout = rospy.Duration(30.0)  # 30 second timeout
        
        if service_data['event'].wait(timeout.to_sec()):
            # Get the response
            gpt_response = service_data['response']
            
            # Create service response
            service_response = GPTServiceResponse()
            if gpt_response.is_error:
                service_response.success = False
                service_response.error_message = gpt_response.error_message
                service_response.response = ""
                service_response.structured_response = "{}"
            else:
                service_response.success = True
                service_response.error_message = ""
                service_response.response = gpt_response.response
                service_response.structured_response = gpt_response.structured_response
            
            # Clean up
            with self.lock:
                if request_id in self.pending_services:
                    del self.pending_services[request_id]
            
            return service_response
        else:
            # Timeout occurred
            with self.lock:
                if request_id in self.pending_services:
                    del self.pending_services[request_id]
            
            service_response = GPTServiceResponse()
            service_response.success = False
            service_response.error_message = "GPT service request timed out"
            service_response.response = ""
            service_response.structured_response = "{}"
            
            return service_response
    
    def response_callback(self, response_msg):
        """Handle responses from the GPT interface"""
        request_id = response_msg.request_id
        
        with self.lock:
            if request_id in self.pending_services:
                service_data = self.pending_services[request_id]
                service_data['response'] = response_msg
                service_data['event'].set()  # Signal that response is ready

def main():
    rospy.init_node('gpt_service_wrapper')
    
    wrapper = GPTServiceWrapper()
    
    rospy.spin()

if __name__ == '__main__':
    main()
```

## Testing the Implementation

### Unit Tests
- Test the GPT request/response flow
- Verify context history management
- Validate structured response parsing
- Check error handling

### Integration Tests
- Test end-to-end communication from request to response
- Verify the service wrapper works correctly
- Test with different types of natural language commands
- Validate conversation context maintenance

## Security Considerations

1. **API Key Management**: Store the OpenAI API key securely, preferably using ROS parameters or environment variables
2. **Rate Limiting**: Implement rate limiting to avoid exceeding OpenAI's usage limits
3. **Input Sanitization**: Validate and sanitize inputs before sending to GPT
4. **Response Validation**: Validate GPT responses before executing any actions

## Performance Optimization

1. **Caching**: Cache repetitive requests to reduce API calls and response time
2. **Context Management**: Limit conversation history to manage token usage and costs
3. **Connection Pooling**: Reuse connections to the OpenAI API when possible
4. **Asynchronous Processing**: Use threading for non-blocking request processing

## Conclusion

This implementation provides a robust foundation for integrating GPT models with ROS 2 communication. The modular design allows for easy extension and modification while maintaining reliable communication between the natural language processing and robotic action systems.