#!/usr/bin/env python3
# fibonacci_action_client.py

"""
Fibonacci Action Client
This node calls the Fibonacci action server to compute a sequence of Fibonacci numbers
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')
        
        # Set up a timer to send goals periodically
        self.timer = self.create_timer(5.0, self.send_goal)
        self.goal_count = 0

    def send_goal(self):
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        # Create a goal
        goal_msg = Fibonacci.Goal()
        # Vary the order to demonstrate different computations
        goal_msg.order = 5 + (self.goal_count % 3)  # 5, 6, or 7
        self.goal_count += 1
        
        self.get_logger().info(f'Sending goal: order = {goal_msg.order}')
        
        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        # Add a callback to handle the result
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected')
            return

        self.get_logger().info('Goal accepted')
        
        # Get the result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_client = FibonacciActionClient()

    try:
        rclpy.spin(fibonacci_action_client)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()