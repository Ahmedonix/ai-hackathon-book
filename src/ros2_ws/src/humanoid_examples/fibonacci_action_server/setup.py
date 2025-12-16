from setuptools import setup
import os
from glob import glob

package_name = 'fibonacci_action_server'

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
    maintainer='maintainer',
    maintainer_email='maintainer@todo.com',
    description='Fibonacci action server for humanoid examples',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fibonacci_action_server = fibonacci_action_server.fibonacci_action_server:main',
        ],
    },
)