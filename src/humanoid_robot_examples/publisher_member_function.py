# Copyright 2023 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Basic publisher example for humanoid robot
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HumanoidPublisher(Node):

    def __init__(self):
        super().__init__('humanoid_publisher')
        self.publisher_ = self.create_publisher(String, 'humanoid_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Humanoid Robot Status Update: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    humanoid_publisher = HumanoidPublisher()

    rclpy.spin(humanoid_publisher)

    # Destroy the node explicitly
    humanoid_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()