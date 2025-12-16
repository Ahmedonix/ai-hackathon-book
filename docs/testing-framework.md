# Testing Framework for ROS 2 Nodes and Documentation Validation

## Overview

This document outlines the testing framework for ROS 2 nodes and documentation validation in the Physical AI & Humanoid Robotics Book project. The framework ensures that all code examples work correctly and that documentation meets quality standards.

## ROS 2 Node Testing Framework

### 1. Unit Testing with pytest

For ROS 2 Python nodes, we use pytest with the `rclpy` testing utilities:

#### Installation and Setup
```bash
pip install pytest pytest-cov
```

#### Basic Test Structure
```python
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from humanoid_robot_examples.publisher_member_function import HumanoidPublisher
from humanoid_robot_examples.subscriber_member_function import HumanoidSubscriber


class TestHumanoidPublisher:
    
    def setup_method(self):
        """Setup method that runs before each test."""
        rclpy.init()
        self.node = HumanoidPublisher()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def teardown_method(self):
        """Teardown method that runs after each test."""
        self.node.destroy_node()
        rclpy.shutdown()

    def test_publisher_creation(self):
        """Test that the publisher node is created correctly."""
        assert self.node.publisher_ is not None
        assert self.node.publisher_.topic_name == 'humanoid_status'

    def test_message_published(self):
        """Test that messages are published."""
        initial_count = self.node.i
        # Trigger the timer callback
        self.node.timer_callback()
        # Check that counter increased
        assert self.node.i > initial_count
```

### 2. Integration Testing

For testing multiple nodes together:

```python
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from humanoid_robot_examples.publisher_member_function import HumanoidPublisher
from humanoid_robot_examples.subscriber_member_function import HumanoidSubscriber


class TestPublisherSubscriberIntegration:
    
    def setup_method(self):
        """Setup method that runs before each test."""
        rclpy.init()
        self.publisher_node = HumanoidPublisher()
        self.subscriber_node = HumanoidSubscriber()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.publisher_node)
        self.executor.add_node(self.subscriber_node)

    def teardown_method(self):
        """Teardown method that runs after each test."""
        self.publisher_node.destroy_node()
        self.subscriber_node.destroy_node()
        rclpy.shutdown()

    def test_publisher_subscriber_integration(self):
        """Test that publisher and subscriber work together."""
        # This test would require more sophisticated setup to actually
        # check if messages published by one node are received by another
        # In practice, this might involve using mock subscribers or message buffers
        assert self.publisher_node.publisher_ is not None
        assert self.subscriber_node.subscription is not None
```

### 3. Launch File Testing

Testing launch files using `launch_testing`:

```python
import unittest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_testing.actions import ReadyToTest

import pytest
import launch
import launch_ros
import launch_testing


def generate_test_description():
    """Generate launch description for testing."""
    return LaunchDescription([
        Node(
            package='humanoid_robot_examples',
            executable='talker',
            name='humanoid_publisher_test'
        ),
        Node(
            package='humanoid_robot_examples',
            executable='listener',
            name='humanoid_subscriber_test'
        ),
        ReadyToTest()
    ])


@pytest.mark.launch_test
def generate_test_description():
    """Test that the launch file starts without errors."""
    return LaunchDescription([
        Node(
            package='humanoid_robot_examples',
            executable='talker',
            name='humanoid_publisher_test'
        ),
        Node(
            package='humanoid_robot_examples',
            executable='listener',
            name='humanoid_subscriber_test'
        ),
        ReadyToTest()
    ])


class TestNodeStartup(unittest.TestCase):
    
    def test_node_startup(self, proc_info, humanoid_publisher_test, humanoid_subscriber_test):
        """Test that nodes start without errors."""
        proc_info.assertWaitForStartup(humanoid_publisher_test, timeout=5)
        proc_info.assertWaitForStartup(humanoid_subscriber_test, timeout=5)
```

## Documentation Validation Framework

### 1. Documentation Build Testing

Ensure that the documentation builds correctly without errors:

```bash
# Test documentation build
npm run build

# For Docusaurus projects
npm run serve &  # Serve locally to test functionality
```

### 2. Link Validation

Use automated tools to validate internal and external links:

```javascript
// In package.json
{
  "scripts": {
    "test:links": "lychee --verbose --config .lycheeignore **/*.md"
  }
}
```

### 3. Content Quality Testing

Use automated tools for content quality checks:

```javascript
// Accessibility checking with pa11y
{
  "scripts": {
    "test:accessibility": "pa11y-ci --sitemap http://localhost:3000/sitemap.xml"
  }
}
```

## Test Organization Structure

### 1. Test Directory Structure
```
tests/
├── unit/
│   ├── test_publisher.py
│   └── test_subscriber.py
├── integration/
│   ├── test_publisher_subscriber_integration.py
│   └── test_launch_files.py
├── system/
│   ├── test_full_system.py
│   └── test_performance.py
└── documentation/
    ├── test_build.py
    ├── test_links.py
    └── test_content_quality.py
```

### 2. CI/CD Integration

Include testing in GitHub Actions workflows:

```yaml
name: Test Documentation and Code

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup ROS 2
      uses: ros-tooling/setup-ros@v0.7
      with:
        required-ros-distributions: iron
    
    - name: Install dependencies
      run: |
        sudo apt update
        rosdep update
        rosdep install --from-paths src --ignore-src -r -y
    
    - name: Build packages
      run: colcon build --packages-select humanoid_robot_examples
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: 18
        cache: npm
    
    - name: Install Node dependencies
      run: npm install
    
    - name: Run Python unit tests
      run: |
        source install/setup.bash
        colcon test --packages-select humanoid_robot_examples
        colcon test-result --all --verbose
    
    - name: Run documentation build test
      run: npm run build
    
    - name: Run documentation link checker
      run: npx markdown-link-check "**/*.md"
```

## Code Example Validation Framework

### 1. Automated Code Example Testing

Create a script to validate all code examples in the documentation:

```python
# validate_code_examples.py
import os
import tempfile
import subprocess
from pathlib import Path


def extract_code_examples_from_docs():
    """Extract code examples from documentation files."""
    code_examples = []
    
    # Find all markdown files in docs
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    
                    # Extract code blocks with language specification
                    lines = content.split('\n')
                    in_code_block = False
                    current_block = []
                    lang = None
                    
                    for line in lines:
                        if line.startswith('```python'):
                            in_code_block = True
                            lang = 'python'
                            current_block = []
                        elif line.startswith('```') and in_code_block:
                            in_code_block = False
                            if lang == 'python':
                                code_examples.append('\n'.join(current_block))
                            current_block = []
                        elif in_code_block:
                            current_block.append(line)
    
    return code_examples


def test_code_example(code_example):
    """Test a single code example."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code_example)
        temp_file = f.name
    
    try:
        # Run the code example with a timeout
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = ""
        error = "Code example timed out"
    except Exception as e:
        success = False
        output = ""
        error = str(e)
    finally:
        os.remove(temp_file)
    
    return success, output, error


def validate_all_examples():
    """Validate all code examples in the documentation."""
    examples = extract_code_examples_from_docs()
    results = []
    
    for i, example in enumerate(examples):
        success, output, error = test_code_example(example)
        results.append({
            'example_id': i,
            'success': success,
            'output': output,
            'error': error
        })
        
        if not success:
            print(f"Failed example {i}: {error}")
        else:
            print(f"Passed example {i}")
    
    return results


if __name__ == "__main__":
    results = validate_all_examples()
    failed_count = sum(1 for r in results if not r['success'])
    
    if failed_count > 0:
        print(f"Validation failed: {failed_count} out of {len(results)} examples failed")
        exit(1)
    else:
        print(f"All {len(results)} examples passed validation")
        exit(0)
```

### 2. Test Coverage Requirements

Set minimum test coverage requirements:

```yaml
# In package.xml for ROS 2 package
<test_depend>python3-pytest-cov</test_depend>
```

```python
# In setup.cfg or pyproject.toml
[tool.coverage.run]
source = .
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    */site-packages/*

[tool.coverage.report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Module-Specific Testing Considerations

### Module 1 (ROS 2 Fundamentals) - Basic Testing
- Unit tests for individual nodes
- Integration tests for node communication
- System tests for launch files
- Validation of basic communication patterns

### Module 2 (Digital Twin Simulation) - Simulation Testing
- Tests for URDF model validity
- Gazebo simulation tests
- Sensor data validation
- Physics simulation verification

### Module 3 (AI-Robot Brain) - AI Component Testing
- Perception pipeline tests
- Navigation system tests
- AI model input/output validation
- Performance tests for AI components

### Module 4 (Vision-Language-Action) - Cognitive System Testing
- Voice processing accuracy tests
- Command interpretation tests
- Multi-modal integration tests
- Safety constraint validation

## Running Tests

### 1. Unit Tests
```bash
# Run all unit tests
python3 -m pytest tests/unit/ -v

# Run with coverage
python3 -m pytest tests/unit/ --cov=src/ --cov-report=html
```

### 2. Integration Tests
```bash
# Run integration tests
python3 -m pytest tests/integration/ -v

# For ROS 2 specific tests
source install/setup.bash
colcon test --packages-select humanoid_robot_examples
colcon test-result --all --verbose
```

### 3. Documentation Tests
```bash
# Build documentation
npm run build

# Check for broken links
npm run test:links

# Run code example validation
python3 tests/documentation/validate_code_examples.py
```

This testing framework ensures that all code examples in the Physical AI & Humanoid Robotics Book work correctly and that the documentation meets quality standards for educational use.