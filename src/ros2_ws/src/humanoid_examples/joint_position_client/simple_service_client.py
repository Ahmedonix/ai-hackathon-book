#!/usr/bin/env python3
# simple_service_client.py

"""
Simple Service Client Node
This node calls the joint position service to set joint positions
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetFloat64
import time


class JointPositionClient(Node):

    def __init__(self):
        super().__init__('joint_position_client')
        
        # Create a client for the service
        self.cli = self.create_client(SetFloat64, 'set_joint_position')
        
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = SetFloat64.Request()
        
        # Set up a timer to periodically call the service
        self.timer = self.create_timer(2.0, self.call_service)
        self.call_count = 0
        
        self.get_logger().info('Joint Position Client node initialized')

    def call_service(self):
        """Call the service with a new position"""
        # Alternate between different positions
        positions = [0.0, 0.5, -0.3, 0.8, 0.0]
        target_pos = positions[self.call_count % len(positions)]
        
        self.req.data = target_pos
        self.call_count += 1
        
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.service_response_callback)
        
        self.get_logger().info(f'Requesting to set joint position to {target_pos}')

    def service_response_callback(self, future):
        """Handle the response from the service"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully set joint position: {response.message}')
            else:
                self.get_logger().error(f'Failed to set joint position: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)

    client_node = JointPositionClient()

    try:
        rclpy.spin(client_node)
    except KeyboardInterrupt:
        pass
    finally:
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()