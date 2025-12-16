from setuptools import setup
import os
from glob import glob

package_name = 'ai_agent'

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
    description='Rule-based AI agent for humanoid robot examples',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rule_based_ai_agent = ai_agent.rule_based_ai_agent:main',
        ],
    },
)