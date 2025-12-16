#!/usr/bin/env python3
# rule_based_ai_agent.py

"""
Rule-Based AI Agent for Humanoid Robot
This node implements a simple rule-based AI that interfaces with ROS 2 nodes
to make decisions based on sensor data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math


class RuleBasedAIAgent(Node):

    def __init__(self):
        super().__init__('rule_based_ai_agent')
        
        # Create subscribers to listen to robot data
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        # Create publishers to send commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'ai_agent_status', 10)
        
        # Set up a timer to run the AI decision loop at 10Hz
        self.timer = self.create_timer(0.1, self.ai_decision_loop)
        
        # Robot state tracking
        self.joint_positions = {}
        self.joint_velocities = {}
        self.last_command_time = self.get_clock().now()
        
        self.get_logger().info('Rule-Based AI Agent node initialized')

    def joint_state_callback(self, msg):
        """Callback to process joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def ai_decision_loop(self):
        """Main AI decision loop"""
        # Get current time
        current_time = self.get_clock().now()
        
        # Simple rule-based decision making
        cmd_msg = Twist()
        status_msg = String()
        
        # Check if we have joint data
        if 'left_knee_joint' in self.joint_positions and 'right_knee_joint' in self.joint_positions:
            left_knee_pos = self.joint_positions['left_knee_joint']
            right_knee_pos = self.joint_positions['right_knee_joint']
            
            # Simple rule: if knees are bent beyond certain angle, move forward
            if abs(left_knee_pos) > 0.8 or abs(right_knee_pos) > 0.8:
                cmd_msg.linear.x = 0.2  # Move forward slowly
                status_msg.data = 'Moving forward: knees are bent'
            else:
                cmd_msg.linear.x = 0.0  # Stop
                status_msg.data = 'Standing still: knees not bent enough'
        else:
            # Default behavior if no joint data
            cmd_msg.linear.x = 0.0
            status_msg.data = 'No joint data available'
        
        # Also add a simple periodic behavior
        time_diff = (current_time - self.last_command_time).nanoseconds / 1e9  # Convert to seconds
        if time_diff > 5.0:  # Every 5 seconds, try to move
            cmd_msg.linear.x = 0.1
            status_msg.data = 'Periodic movement command'
            self.last_command_time = current_time
        
        # Publish commands and status
        self.cmd_vel_pub.publish(cmd_msg)
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f'AI Status: {status_msg.data}')


def main(args=None):
    rclpy.init(args=args)

    ai_agent = RuleBasedAIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()