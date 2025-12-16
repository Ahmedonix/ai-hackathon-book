#!/usr/bin/env python3
# fibonacci_action_server.py

"""
Fibonacci Action Server
This node implements an action server that computes a sequence of Fibonacci numbers
as an example of a long-running task with feedback
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        # Initialize the result and feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        # Publish initial feedback
        goal_handle.publish_feedback(feedback_msg)
        
        # Compute the Fibonacci sequence up to the requested order
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal was cancelled')
                goal_handle.canceled()
                return Fibonacci.Result()

            # Calculate the next Fibonacci number
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            
            # Log the progress
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            
            # Sleep to simulate computation time
            from time import sleep
            sleep(0.5)

        # Set success and return the result
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')
        
        return result


def main(args=None):
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()