#!/usr/bin/env python3
# simple_service_server.py

"""
Simple Service Server Node
This node provides a service to set joint positions for the humanoid robot
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetFloat64


class JointPositionService(Node):

    def __init__(self):
        super().__init__('joint_position_service')
        
        # Create a service
        self.srv = self.create_service(
            SetFloat64, 
            'set_joint_position', 
            self.set_joint_position_callback
        )
        
        # Track joint positions
        self.joint_positions = {}
        
        self.get_logger().info('Joint Position Service server initialized')

    def set_joint_position_callback(self, request, response):
        """Callback to handle joint position requests"""
        # In a real robot, this would interface with hardware
        # Here we just simulate setting the position
        joint_name = "target_joint"  # In a real implementation, you'd have joint name parameter
        target_position = request.data
        
        self.joint_positions[joint_name] = target_position
        
        response.success = True
        response.message = f'Set {joint_name} position to {target_position:.3f}'
        
        self.get_logger().info(f'Service called: {response.message}')
        
        return response


def main(args=None):
    rclpy.init(args=args)

    service_node = JointPositionService()

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        pass
    finally:
        service_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()