#!/usr/bin/env python3
# joint_state_subscriber.py

"""
Joint State Subscriber Node
This node subscribes to joint state messages and logs the values
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateSubscriber(Node):

    def __init__(self):
        super().__init__('joint_state_subscriber')
        
        # Create subscriber for joint states
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info('Joint State Subscriber node initialized')

    def joint_state_callback(self, msg):
        # Log joint positions
        self.get_logger().info('Received joint states:')
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.get_logger().info(f'  {name}: {msg.position[i]:.3f}')


def main(args=None):
    rclpy.init(args=args)

    joint_subscriber = JointStateSubscriber()

    try:
        rclpy.spin(joint_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        joint_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()