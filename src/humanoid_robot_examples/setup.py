from setuptools import setup

package_name = 'humanoid_robot_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Author Name',
    maintainer_email='author@example.com',
    description='ROS 2 packages for humanoid robot examples in the Physical AI & Humanoid Robotics Book',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = humanoid_robot_examples.publisher_member_function:main',
            'listener = humanoid_robot_examples.subscriber_member_function:main',
        ],
    },
)