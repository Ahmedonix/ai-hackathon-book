# Integrating GPT Models with ROS 2

## Overview

This document describes how to integrate OpenAI GPT models with ROS 2 for creating intelligent responses to natural language commands. The integration enables robots to understand complex instructions, plan multi-step tasks, and respond appropriately to changing situations.

## Architecture

The GPT-ROS 2 integration follows this architecture:

```
Voice Command
      ↓
Speech-to-Text (Whisper)
      ↓
Natural Language Command
      ↓
GPT Interface Node
      ↓
Intent Recognition & Task Planning
      ↓
Action Sequencing
      ↓
ROS 2 Action Execution
```

## Components

### 1. GPT Interface Node

The core component that communicates with the OpenAI API:

```python
import rospy
import openai
from std_msgs.msg import String
from humanoid_msgs.msg import GPTRequest, GPTResponse
import json
import time

class GPTInterfaceNode:
    def __init__(self):
        # Initialize OpenAI API key
        openai.api_key = rospy.get_param('~openai_api_key', '')
        
        # Subscribe to commands requiring GPT processing
        self.request_sub = rospy.Subscriber("/gpt_requests", GPTRequest, self.handle_request)
        
        # Publish GPT responses
        self.response_pub = rospy.Publisher("/gpt_responses", GPTResponse, queue_size=10)
        
        # Store context for conversation memory
        self.context_history = []
        
        rospy.loginfo("GPT Interface Node initialized")
    
    def handle_request(self, request_msg):
        try:
            # Prepare the message for GPT with context
            messages = self.build_context(request_msg.command, request_msg.context)
            
            # Call GPT API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Or "gpt-4" for more advanced capabilities
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Extract the response
            gpt_response = response.choices[0].message['content'].strip()
            
            # Create and publish response message
            response_msg = GPTResponse()
            response_msg.header.stamp = rospy.Time.now()
            response_msg.header.frame_id = "base_link"
            response_msg.request_id = request_msg.id
            response_msg.response = gpt_response
            response_msg.structured_response = self.parse_structured_response(gpt_response)
            
            # Update context history
            self.context_history.append({"role": "user", "content": request_msg.command})
            self.context_history.append({"role": "assistant", "content": gpt_response})
            
            # Keep history to a reasonable size
            if len(self.context_history) > 20:
                self.context_history = self.context_history[-20:]
                
            self.response_pub.publish(response_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing GPT request: {str(e)}")
            self.publish_error_response(request_msg.id, str(e))
    
    def build_context(self, user_command, additional_context):
        # Add system context to guide GPT behavior
        messages = [
            {"role": "system", "content": """You are an assistant for a humanoid robot. 
            Respond with clear, executable actions. Format responses as structured JSON when possible.
            Respond to questions concisely. If asked to perform tasks, break them into simple steps."""}
        ]
        
        # Add conversation history for context
        messages.extend(self.context_history)
        
        # Add the current user command
        messages.append({"role": "user", "content": f"{user_command} {additional_context}"})
        
        return messages
    
    def parse_structured_response(self, gpt_response):
        # Attempt to parse GPT response as structured data (JSON)
        try:
            # Look for JSON blocks in the response
            import re
            json_match = re.search(r'```json\n(.*?)\n```', gpt_response, re.DOTALL)
            if json_match:
                return json.dumps(json.loads(json_match.group(1)))
            
            # Look for standalone JSON
            return json.dumps(json.loads(gpt_response))
        except:
            # If not JSON, return original response
            return json.dumps({"text": gpt_response})
    
    def publish_error_response(self, request_id, error_msg):
        response_msg = GPTResponse()
        response_msg.header.stamp = rospy.Time.now()
        response_msg.header.frame_id = "base_link"
        response_msg.request_id = request_id
        response_msg.response = f"Error: {error_msg}"
        response_msg.is_error = True
        self.response_pub.publish(response_msg)

def main():
    rospy.init_node('gpt_interface')
    
    # Initialize and start the GPT interface
    gpt_node = GPTInterfaceNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("GPT Interface Node shutting down")

if __name__ == '__main__':
    main()