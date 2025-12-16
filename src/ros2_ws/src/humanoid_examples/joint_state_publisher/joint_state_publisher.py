#!/usr/bin/env python3
# joint_state_publisher.py

"""
Joint State Publisher Node
This node publishes joint state messages for the humanoid robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math


class JointStatePublisher(Node):

    def __init__(self):
        super().__init__('joint_state_publisher')
        
        # Create publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Set up a timer to publish joint states at 50Hz
        self.timer = self.create_timer(0.02, self.publish_joint_states)
        
        # Initialize joint positions
        self.time = 0.0
        
        self.get_logger().info('Joint State Publisher node initialized')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Simulate simple joint movements
        self.time += 0.02  # Increment time based on timer rate
        
        msg.position = [
            math.sin(self.time) * 0.3,      # left_hip_joint
            -math.sin(self.time) * 0.5,     # left_knee_joint
            math.sin(self.time) * 0.2,      # left_ankle_joint
            math.sin(self.time) * 0.3,      # right_hip_joint
            -math.sin(self.time) * 0.5,     # right_knee_joint
            math.sin(self.time) * 0.2       # right_ankle_joint
        ]
        
        msg.velocity = [0.0] * len(msg.position)  # For simplicity, zero velocity
        msg.effort = [0.0] * len(msg.position)    # For simplicity, zero effort
        
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    joint_publisher = JointStatePublisher()

    try:
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        joint_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()