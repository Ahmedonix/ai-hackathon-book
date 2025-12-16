#!/usr/bin/env python3
# robot_controller.py

"""
Simple Robot Controller Node
This node demonstrates controlling a robot by publishing joint commands
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import math


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Timer for publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # 10Hz
        
        # Robot state
        self.joint_positions = {
            'left_hip_joint': 0.0,
            'left_knee_joint': 0.0,
            'left_ankle_joint': 0.0,
            'right_hip_joint': 0.0,
            'right_knee_joint': 0.0,
            'right_ankle_joint': 0.0
        }
        
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        self.get_logger().info('Robot Controller node initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z
        
        # Update joint positions based on velocity commands
        # This is a simplified kinematic model
        self.joint_positions['left_hip_joint'] = math.sin(
            self.get_clock().now().nanoseconds / 1e9
        ) * 0.3
        
        self.joint_positions['right_hip_joint'] = math.sin(
            self.get_clock().now().nanoseconds / 1e9
        ) * 0.3
        
        self.joint_positions['left_knee_joint'] = -math.sin(
            self.get_clock().now().nanoseconds / 1e9
        ) * 0.5
        
        self.joint_positions['right_knee_joint'] = -math.sin(
            self.get_clock().now().nanoseconds / 1e9
        ) * 0.5

    def publish_joint_states(self):
        """Publish joint states message"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()