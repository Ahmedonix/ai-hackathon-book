# Simulation Debugging Techniques for Humanoid Robotics

## Overview

Debugging humanoid robot simulations presents unique challenges due to the complex interplay between physics, sensors, control systems, and multiple software components. This section covers comprehensive debugging techniques for identifying, diagnosing, and resolving issues in Gazebo and Unity simulation environments.

## Understanding Simulation Debugging Categories

### 1. Physics-Related Issues

**Symptoms:**
- Robot falls unexpectedly
- Joint limits not respected
- Unrealistic movements or behaviors
- High computational load causing physics inaccuracies
- Collision detection problems

**Common Causes:**
- Invalid joint limits or motor parameters
- Incorrect mass/inertia properties
- Physics engine parameters too aggressive
- Inconsistent geometry definitions

### 2. Sensor Simulation Issues

**Symptoms:**
- No sensor data published
- Inaccurate sensor readings
- Data published at wrong rate
- Sensor noise not matching specifications

**Common Causes:**
- Incorrect sensor plugin configurations
- Invalid frame ID assignments
- Physics update rate mismatches
- Gazebo model definition problems

### 3. Communication Issues

**Symptoms:**
- ROS topics not connecting
- High latency between components
- Intermittent disconnections
- Missing messages

**Common Causes:**
- Network configuration problems
- QoS policy mismatches
- Firewall blocking communication
- Memory leaks in high-frequency topics

### 4. Visualization Issues

**Symptoms:**
- Incorrect robot state visualization
- Unity not rendering properly
- Performance degradation
- Model misalignment between simulators

## Essential Debugging Tools

### 1. ROS 2 Debugging Tools

```bash
# Check topic connections and message rates
ros2 topic list
ros2 topic info /topic_name
ros2 topic echo /topic_name --field field_name --rate 1

# Monitor bandwidth usage
ros2 topic bw /topic_name

# Check service availability
ros2 service list
ros2 service call /service_name type_package/type

# Monitor TF tree
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo frame1 frame2

# Check node health
ros2 node list
ros2 run rqt_graph rqt_graph
```

### 2. Gazebo Debugging Tools

```bash
# Run Gazebo with verbose output
gz sim -v 4 world_name.sdf

# List all models in simulation
gz model -m

# Check physics properties
gz topic -e -t /world/world_name/physics

# Monitor simulation performance
gz topic -e -t /world/world_name/stats

# Check for missing models
gz model -m world_name
```

### 3. System Monitoring Tools

```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor Gazebo processes specifically
htop -p $(pgrep -f gazebo)

# Check GPU usage (for rendering)
nvidia-smi  # For NVIDIA cards
```

## Diagnostic Techniques

### 1. Systematic Problem Isolation

Create a diagnostic node to help isolate problems:

```python
# scripts/diagnostic_monitor.py
#!/usr/bin/env python3

"""
Comprehensive diagnostic monitor for humanoid simulation debugging.
Identifies and reports various simulation issues.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
from builtin_interfaces.msg import Time
import time
import math
import psutil


class SimulationDiagnosticMonitor(Node):
    def __init__(self):
        super().__init__('simulation_diagnostic_monitor')
        
        # Subscribers for common topics
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Publishers for diagnostics
        self.diag_pub = self.create_publisher(
            String,
            '/diagnostics',
            10
        )
        
        self.health_pub = self.create_publisher(
            Float64MultiArray,
            '/health_metrics',
            10
        )
        
        # Internal state tracking
        self.joint_state_received = False
        self.joint_state_time = None
        self.joint_state_count = 0
        
        self.imu_received = False
        self.imu_time = None
        self.imu_count = 0
        
        self.scan_received = False
        self.scan_time = None
        self.scan_count = 0
        
        # Diagnostic thresholds
        self.message_timeout = 1.0  # seconds
        self.low_frequency_threshold = 10.0  # Hz
        self.high_frequency_threshold = 200.0  # Hz
        
        # Timer for diagnostics
        self.diagnostic_timer = self.create_timer(2.0, self.run_diagnostics)
        
        self.get_logger().info('Simulation Diagnostic Monitor Initialized')

    def joint_state_callback(self, msg):
        """Track joint state message"""
        self.joint_state_received = True
        self.joint_state_time = self.get_clock().now()
        self.joint_state_count += 1

    def imu_callback(self, msg):
        """Track IMU message"""
        self.imu_received = True
        self.imu_time = self.get_clock().now()
        self.imu_count += 1

    def scan_callback(self, msg):
        """Track laser scan message"""
        self.scan_received = True
        self.scan_time = self.get_clock().now()
        self.scan_count += 1

    def cmd_vel_callback(self, msg):
        """Track velocity command message"""
        self.cmd_vel_received = True
        self.cmd_vel_time = self.get_clock().now()

    def odom_callback(self, msg):
        """Track odometry message"""
        self.odom_received = True
        self.odom_time = self.get_clock().now()

    def run_diagnostics(self):
        """Run comprehensive diagnostic checks"""
        diagnostics = []
        health_metrics = []
        
        # Check joint states
        joint_freq = self.calculate_frequency(
            self.joint_state_count,
            self.joint_state_time
        ) if self.joint_state_time else 0
        
        joint_status = self.check_message_status(
            self.joint_state_received, 
            self.joint_state_time, 
            joint_freq
        )
        
        if joint_status != "OK":
            diagnostics.append(f"JOINT_STATES: {joint_status} (freq: {joint_freq:.2f}Hz)")
        
        health_metrics.extend([float(joint_status == "OK"), joint_freq])
        
        # Check IMU
        imu_freq = self.calculate_frequency(
            self.imu_count,
            self.imu_time
        ) if self.imu_time else 0
        
        imu_status = self.check_message_status(
            self.imu_received, 
            self.imu_time, 
            imu_freq
        )
        
        if imu_status != "OK":
            diagnostics.append(f"IMU: {imu_status} (freq: {imu_freq:.2f}Hz)")
        
        health_metrics.extend([float(imu_status == "OK"), imu_freq])
        
        # Check laser scan
        scan_freq = self.calculate_frequency(
            self.scan_count,
            self.scan_time
        ) if self.scan_time else 0
        
        scan_status = self.check_message_status(
            self.scan_received, 
            self.scan_time, 
            scan_freq
        )
        
        if scan_status != "OK":
            diagnostics.append(f"LASER_SCAN: {scan_status} (freq: {scan_freq:.2f}Hz)")
        
        health_metrics.extend([float(scan_status == "OK"), scan_freq])
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 90.0:
            diagnostics.append(f"HIGH_CPU_USAGE: {cpu_usage:.1f}%")
        
        if memory_usage > 90.0:
            diagnostics.append(f"HIGH_MEMORY_USAGE: {memory_usage:.1f}%")
        
        health_metrics.extend([cpu_usage, memory_usage])
        
        # Publish diagnostics
        if diagnostics:
            diag_msg = String()
            diag_msg.data = " | ".join(diagnostics)
            self.diag_pub.publish(diag_msg)
            for diag_item in diagnostics:
                self.get_logger().warn(f'Diagnostic Issue: {diag_item}')
        else:
            diag_msg = String()
            diag_msg.data = "ALL_SYSTEMS_NORMAL"
            self.diag_pub.publish(diag_msg)
            # Only log normal status occasionally to avoid spam
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                self.get_logger().info('All systems normal')
        
        # Publish health metrics
        health_msg = Float64MultiArray()
        health_msg.data = health_metrics
        self.health_pub.publish(health_msg)

    def calculate_frequency(self, count, last_time):
        """Calculate message frequency"""
        if last_time is None:
            return 0.0
        
        # Calculate time since start of a 10-second window
        # In a real implementation, you'd track multiple timestamps
        # This is a simplified version
        return count / 10.0 if count > 10 else 0.0

    def check_message_status(self, received, last_time, frequency):
        """Check if a message stream is healthy"""
        if not received:
            return "NOT_RECEIVED"
        
        if last_time is None:
            return "NO_DATA"
        
        # Check for timeouts
        time_since_last = (self.get_clock().now() - last_time).nanoseconds / 1e9
        if time_since_last > self.message_timeout:
            return "TIMED_OUT"
        
        # Check frequency boundaries
        if frequency < self.low_frequency_threshold:
            return f"LOW_FREQ ({frequency:.1f}Hz)"
        
        if frequency > self.high_frequency_threshold:
            return f"HIGH_FREQ ({frequency:.1f}Hz)"
        
        return "OK"

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Simulation Diagnostic Monitor Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    monitor = SimulationDiagnosticMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Node interrupted by user')
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. State Validator for Robot Position

Create a tool to validate robot state and detect anomalies:

```python
# scripts/robot_state_validator.py
#!/usr/bin/env python3

"""
Robot state validator for detecting anomalous robot behavior.
Identifies potential issues with robot state in simulation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import String, Float64MultiArray
import numpy as np
import math


class RobotStateValidator(Node):
    def __init__(self):
        super().__init__('robot_state_validator')
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Publishers
        self.anomaly_report_pub = self.create_publisher(
            String,
            '/anomaly_reports',
            10
        )
        
        self.state_health_pub = self.create_publisher(
            Float64MultiArray,
            '/state_health',
            10
        )
        
        # Internal state tracking
        self.joint_history = {}
        self.imu_history = []
        
        # Validation parameters
        self.max_joint_velocity = 10.0  # rad/s
        self.max_joint_effort = 200.0  # N*m
        self.max_angular_velocity = 5.0  # rad/s
        self.max_linear_acceleration = 20.0  # m/s²
        
        # Timer for validation
        self.validation_timer = self.create_timer(0.1, self.validate_state)
        
        self.get_logger().info('Robot State Validator Initialized')

    def joint_state_callback(self, msg):
        """Validate joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                pos = msg.position[i]
            else:
                pos = 0.0
                
            if i < len(msg.velocity):
                vel = msg.velocity[i]
            else:
                vel = 0.0
                
            if i < len(msg.effort):
                effort = msg.effort[i]
            else:
                effort = 0.0
            
            # Store joint data
            if name not in self.joint_history:
                self.joint_history[name] = {
                    'positions': [],
                    'velocities': [],
                    'efforts': []
                }
            
            # Update histories
            hist = self.joint_history[name]
            hist['positions'].append(pos)
            hist['velocities'].append(vel)
            hist['efforts'].append(effort)
            
            # Keep only last 100 values
            for key in ['positions', 'velocities', 'efforts']:
                if len(hist[key]) > 100:
                    hist[key].pop(0)

    def imu_callback(self, msg):
        """Validate IMU states"""
        self.imu_history.append({
            'timestamp': self.get_clock().now(),
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        })
        
        # Keep only last 100 values
        if len(self.imu_history) > 100:
            self.imu_history.pop(0)

    def validate_state(self):
        """Validate current robot state"""
        anomalies = []
        health_scores = []
        
        # Validate joint states
        for joint_name, history in self.joint_history.items():
            if len(history['velocities']) > 1:
                # Check velocity limits
                avg_vel = np.mean(np.abs(history['velocities']))
                if avg_vel > self.max_joint_velocity:
                    anomalies.append(f"EXCESSIVE_VELOCITY_{joint_name}: {avg_vel:.2f} rad/s")
                
                # Check effort limits
                max_effort = max(abs(e) for e in history['efforts']) if history['efforts'] else 0.0
                if max_effort > self.max_joint_effort:
                    anomalies.append(f"EXCESSIVE_EFFORT_{joint_name}: {max_effort:.2f} N*m")
            
            # Check for NaN or inf values
            for i, pos in enumerate(history['positions']):
                if not np.isfinite(pos):
                    anomalies.append(f"INVALID_POSITION_{joint_name}[{i}]: {pos}")
            
            for i, vel in enumerate(history['velocities']):
                if not np.isfinite(vel):
                    anomalies.append(f"INVALID_VELOCITY_{joint_name}[{i}]: {vel}")
        
        # Validate IMU states
        if self.imu_history:
            latest_imu = self.imu_history[-1]
            
            # Check angular velocity
            ang_vel = latest_imu['angular_velocity']
            ang_vel_mag = math.sqrt(ang_vel.x**2 + ang_vel.y**2 + ang_vel.z**2)
            if ang_vel_mag > self.max_angular_velocity:
                anomalies.append(f"EXCESSIVE_ANGULAR_VELOCITY: {ang_vel_mag:.2f} rad/s")
            
            # Check linear acceleration
            lin_acc = latest_imu['linear_acceleration']
            lin_acc_mag = math.sqrt(lin_acc.x**2 + lin_acc.y**2 + lin_acc.z**2)
            if lin_acc_mag > self.max_linear_acceleration:
                anomalies.append(f"EXCESSIVE_LINEAR_ACCELERATION: {lin_acc_mag:.2f} m/s²")
            
            # Check orientation validity
            orientation = latest_imu['orientation']
            orient_norm = math.sqrt(
                orientation.x**2 + 
                orientation.y**2 + 
                orientation.z**2 + 
                orientation.w**2
            )
            
            # Orientation should be normalized (magnitude ≈ 1.0)
            if abs(orient_norm - 1.0) > 0.1:
                anomalies.append(f"INVALID_ORIENTATION_NORM: {orient_norm}")
        
        # Calculate overall health score
        if not anomalies:
            overall_health = 1.0  # Perfect health
        elif len(anomalies) <= 2:
            overall_health = 0.7   # Minor issues
        elif len(anomalies) <= 5:
            overall_health = 0.4   # Several issues
        else:
            overall_health = 0.1   # Many issues
        
        # Publish anomaly reports
        if anomalies:
            report_msg = String()
            report_msg.data = " | ".join(anomalies)
            self.anomaly_report_pub.publish(report_msg)
            
            for anomaly in anomalies:
                self.get_logger().warn(f'State Anomaly Detected: {anomaly}')
        else:
            report_msg = String()
            report_msg.data = "STATE_NORMAL"
            self.anomaly_report_pub.publish(report_msg)
            
            # Only report normal status periodically to avoid spam
            if int(self.get_clock().now().nanoseconds / 1e9) % 10 == 0:
                self.get_logger().info('Robot state validated as normal')
        
        # Publish health metrics
        health_msg = Float64MultiArray()
        health_msg.data = [overall_health, len(anomalies)] + [float(overall_health > 0.5)]
        self.state_health_pub.publish(health_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Robot State Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = RobotStateValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Node interrupted by user')
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Debugging Techniques

### 1. Physics Debugging

Debug physics-related issues in Gazebo:

```xml
<!-- Add physics debugging to your world file -->
<world name="debug_world">
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    
    <ode>
      <solver>
        <type>quick</type>
        <iters>100</iters>  <!-- Increase for stability -->
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0001</cfm>    <!-- Lower CFM for more stability -->
        <erp>0.2</erp>       <!-- Higher ERP for more constraint enforcement -->
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
  
  <!-- Add debugging visuals -->
  <include>
    <uri>model://ground_plane</uri>
  </include>
  <include>
    <uri>model://sun</uri>
  </include>
  
  <!-- Robot model with debugging -->
</world>
```

### 2. Sensor Data Validation

Create sensor validation nodes:

```python
# scripts/sensor_validator.py
#!/usr/bin/env python3

"""
Sensor data validator for detecting invalid sensor readings.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, JointState
from std_msgs.msg import String
import numpy as np


class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')
        
        # Subscribers for different sensor types
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        
        # Publisher for sensor validation reports
        self.validation_pub = self.create_publisher(
            String,
            '/sensor_validation_reports',
            10
        )
        
        self.get_logger().info('Sensor Validator Initialized')

    def scan_callback(self, msg):
        """Validate laser scan data"""
        anomalies = []
        
        # Check for proper range values
        for i, range_val in enumerate(msg.ranges):
            if np.isnan(range_val):
                anomalies.append(f"SCAN_NAN_RANGE[{i}]")
            elif np.isinf(range_val):
                # Infinite ranges might be acceptable in some cases
                if range_val < 0:  # Negative infinity is never OK
                    anomalies.append(f"SCAN_NEG_INFINITY[{i}]")
            elif range_val < msg.range_min or range_val > msg.range_max:
                if np.isfinite(range_val):  # Only report if not infinite
                    anomalies.append(f"SCAN_OUT_OF_RANGE[{i}]: {range_val}")
        
        # Check consistency among adjacent readings
        valid_ranges = [r for r in msg.ranges if np.isfinite(r)]
        if len(valid_ranges) > 10:
            mean_range = np.mean(valid_ranges)
            std_range = np.std(valid_ranges)
            # Look for extreme outliers
            extreme_threshold = 3 * std_range  # 3-sigma rule
            
            for i, range_val in enumerate(msg.ranges):
                if np.isfinite(range_val) and abs(range_val - mean_range) > extreme_threshold:
                    anomalies.append(f"SCAN_EXTREME_OUTLIER[{i}]: {range_val}")
        
        # Check scan timing consistency
        if msg.time_increment <= 0:
            anomalies.append("SCAN_INVALID_TIME_INCREMENT")
        
        if anomalies:
            report = f"LASER_SCAN_ISSUES: {' | '.join(anomalies)}"
            self.publish_validation_report(report)
        elif int(self.get_clock().now().nanoseconds / 1e9) % 30 == 0:
            # Periodically report normal status
            self.publish_validation_report("LASER_SCAN_NORMAL")

    def camera_callback(self, msg):
        """Validate camera data"""
        anomalies = []
        
        # Check image dimensions
        expected_size = msg.width * msg.height * 3  # assuming RGB
        actual_size = len(msg.data)
        
        if actual_size != expected_size:
            anomalies.append(f"CAMERA_SIZE_MISMATCH: expected {expected_size}, got {actual_size}")
        
        # Check for uniform images (potential sensor issues)
        if len(msg.data) > 0:
            # Sample a portion of the image to check for uniformity
            sample_size = min(100, len(msg.data) // 3)
            sample_data = msg.data[:sample_size]
            
            # If more than 90% of pixels are identical, it might be stuck
            unique_pixels = len(set(sample_data))
            if unique_pixels < 2:  # Very uniform image
                anomalies.append(f"CAMERA_STUCK_IMAGE: only {unique_pixels} unique pixel values in sample")
        
        if anomalies:
            report = f"CAMERA_ISSUES: {' | '.join(anomalies)}"
            self.publish_validation_report(report)
        elif int(self.get_clock().now().nanoseconds / 1e9) % 30 == 0:
            self.publish_validation_report("CAMERA_NORMAL")

    def imu_callback(self, msg):
        """Validate IMU data"""
        anomalies = []
        
        # Check if orientation is normalized
        orient_norm = np.sqrt(
            msg.orientation.x**2 +
            msg.orientation.y**2 +
            msg.orientation.z**2 +
            msg.orientation.w**2
        )
        
        if abs(orient_norm - 1.0) > 0.001:  # Allow small numerical errors
            anomalies.append(f"IMU_ORIENTATION_NOT_NORMALIZED: norm={orient_norm}")
        
        # Check if angular velocity is reasonable
        ang_vel_mag = np.sqrt(
            msg.angular_velocity.x**2 +
            msg.angular_velocity.y**2 +
            msg.angular_velocity.z**2
        )
        
        if ang_vel_mag > 10.0:  # 10 rad/s is quite high for stable operation
            anomalies.append(f"IMU_EXCESSIVE_ANGULAR_VELOCITY: {ang_vel_mag}")
        
        # Check linear acceleration magnitude
        lin_acc_mag = np.sqrt(
            msg.linear_acceleration.x**2 +
            msg.linear_acceleration.y**2 +
            msg.linear_acceleration.z**2
        )
        
        # Normal acceleration should be close to gravity (9.81 m/s²)
        # But allow for some movement
        if abs(lin_acc_mag - 9.81) > 15.0:  # Allow for dynamic acceleration up to 15 m/s²
            anomalies.append(f"IMU_UNUSUAL_ACCELERATION_MAGNITUDE: {lin_acc_mag}")
        
        if anomalies:
            report = f"IMU_ISSUES: {' | '.join(anomalies)}"
            self.publish_validation_report(report)
        elif int(self.get_clock().now().nanoseconds / 1e9) % 30 == 0:
            self.publish_validation_report("IMU_NORMAL")

    def publish_validation_report(self, report):
        """Publish validation report"""
        report_msg = String()
        report_msg.data = report
        self.validation_pub.publish(report_msg)
        
        # Log based on content
        if "ISSUES" in report or "ANOMALIES" in report:
            self.get_logger().warn(f'Sensor Validation: {report}')
        else:
            self.get_logger().info(f'Sensor Validation: {report}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Sensor Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = SensorValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Node interrupted by user')
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Unity-Specific Debugging

### 1. Unity-ROS Communication Debugging

Debug issues between Unity and ROS:

```csharp
// Scripts/UnityROSDebugger.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using System.Collections.Generic;
using System.Linq;

public class UnityROSDebugger : MonoBehaviour
{
    [Header("Debug Settings")]
    public bool enableDebugLogging = true;
    public bool enablePerformanceLogging = true;
    public float debugLogInterval = 5.0f;  // seconds
    
    [Header("Connection Monitoring")]
    public float connectionCheckInterval = 1.0f;
    private float lastConnectionCheckTime;
    
    [Header("Performance Monitoring")]
    public float performanceLogInterval = 30.0f;  // seconds
    private float lastPerformanceLogTime;
    
    private ROSConnection m_ROSConnection;
    private float m_LastLogTime;
    
    // Message statistics
    private Dictionary<string, MessageStats> m_MessageStats = new Dictionary<string, MessageStats>();
    private int m_TotalMessagesReceived = 0;
    private int m_TotalMessagesPublished = 0;
    
    [System.Serializable]
    public class MessageStats
    {
        public int receivedCount = 0;
        public int publishedCount = 0;
        public float lastReceiveTime = 0;
        public float lastPublishTime = 0;
        public List<float> intervals = new List<float>();  // time between messages
    }

    void Start()
    {
        m_ROSConnection = ROSConnection.instance;
        if (m_ROSConnection == null)
        {
            m_ROSConnection = gameObject.AddComponent<ROSConnection>();
            m_ROSConnection.Initialize("127.0.0.1", 10000);
        }
        
        m_LastLogTime = Time.time;
        lastConnectionCheckTime = Time.time;
        lastPerformanceLogTime = Time.time;
        
        Debug.Log("Unity ROS Debugger Started");
    }
    
    void Update()
    {
        // Check for connection issues periodically
        if (Time.time - lastConnectionCheckTime > connectionCheckInterval)
        {
            CheckConnectionStatus();
            lastConnectionCheckTime = Time.time;
        }
        
        // Log debug info periodically
        if (enableDebugLogging && Time.time - m_LastLogTime > debugLogInterval)
        {
            LogDebugInfo();
            m_LastLogTime = Time.time;
        }
        
        // Log performance info periodically
        if (enablePerformanceLogging && Time.time - lastPerformanceLogTime > performanceLogInterval)
        {
            LogPerformanceInfo();
            lastPerformanceLogTime = Time.time;
        }
    }
    
    public void RecordMessageReceived(string topic)
    {
        if (!m_MessageStats.ContainsKey(topic))
        {
            m_MessageStats[topic] = new MessageStats();
        }
        
        var stats = m_MessageStats[topic];
        stats.receivedCount++;
        m_TotalMessagesReceived++;
        
        // Track interval between messages
        if (stats.lastReceiveTime > 0)
        {
            float interval = Time.time - stats.lastReceiveTime;
            stats.intervals.Add(interval);
            
            // Keep only last 50 intervals to prevent memory issues
            if (stats.intervals.Count > 50)
                stats.intervals.RemoveAt(0);
        }
        
        stats.lastReceiveTime = Time.time;
    }
    
    public void RecordMessagePublished(string topic)
    {
        if (!m_MessageStats.ContainsKey(topic))
        {
            m_MessageStats[topic] = new MessageStats();
        }
        
        var stats = m_MessageStats[topic];
        stats.publishedCount++;
        m_TotalMessagesPublished++;
        
        stats.lastPublishTime = Time.time;
    }
    
    void CheckConnectionStatus()
    {
        // This is a basic check; in practice you might ping ROS to verify connectivity
        if (m_ROSConnection == null)
        {
            Debug.LogError("ROS Connection is null!");
            return;
        }
        
        // Check if recent messages have been received
        if (m_TotalMessagesReceived == 0 && Time.time > 10.0f)  // Give time for startup
        {
            Debug.LogWarning("No messages received - connection may not be working properly");
        }
    }
    
    void LogDebugInfo()
    {
        Debug.Log($"=== UNITY ROS DEBUG INFO ===");
        Debug.Log($"Total Messages Received: {m_TotalMessagesReceived}");
        Debug.Log($"Total Messages Published: {m_TotalMessagesPublished}");
        Debug.Log($"Active Topics: {m_MessageStats.Count}");
        Debug.Log($"Connected: {(m_ROSConnection != null ? "YES" : "NO")}");
        
        // Log stats for each topic
        foreach (var kvp in m_MessageStats)
        {
            var stats = kvp.Value;
            float avgInterval = stats.intervals.Count > 0 ? stats.intervals.Average() : 0;
            float freq = avgInterval > 0 ? 1.0f / avgInterval : 0;
            
            Debug.Log($"  Topic: {kvp.Key}, " +
                     $"Received: {stats.receivedCount}, " +
                     $"Published: {stats.publishedCount}, " +
                     $"Avg Freq: {freq:F2}Hz");
        }
    }
    
    void LogPerformanceInfo()
    {
        Debug.Log($"=== UNITY PERFORMANCE INFO ===");
        Debug.Log($"Unity Frame Rate: {1.0f / Time.unscaledDeltaTime:F2} FPS");
        Debug.Log($"Unity Memory Usage: {UnityEngine.Profiling.Profiler.usedHeapSizeLong / 1024 / 1024:F2} MB");
        
        // Monitor ROS connection performance
        if (m_ROSConnection != null)
        {
            // In a real implementation, you might track network metrics here
        }
    }
    
    // Example of how to wrap ROS publications with debugging
    public void PublishWithDebug<T>(string topic, T message) where T : Message
    {
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Publish(topic, message);
            RecordMessagePublished(topic);
            
            if (enableDebugLogging && Time.time - m_LastLogTime < 1.0f) // Throttle logging
            {
                Debug.Log($"Published to {topic}");
            }
        }
        else
        {
            Debug.LogError($"Cannot publish to {topic} - ROS connection not established");
        }
    }
    
    // Example of how to register subscriptions with debugging
    public void SubscribeWithDebug<T>(string topic, System.Action<T> callback) where T : Message
    {
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<T>(topic, (msg) => {
                RecordMessageReceived(topic);
                callback(msg);
            });
            
            Debug.Log($"Subscribed to {topic}");
        }
        else
        {
            Debug.LogError($"Cannot subscribe to {topic} - ROS connection not established");
        }
    }
}
```

### 2. Visualization Debugging

Debug visualization issues in Unity:

```csharp
// Scripts/VisualizationDebugger.cs
using UnityEngine;
using System.Collections.Generic;

public class VisualizationDebugger : MonoBehaviour
{
    [Header("Visualization Monitoring")]
    public bool monitorVisualization = true;
    public bool validateTransforms = true;
    public float monitorInterval = 2.0f;
    
    private Dictionary<Transform, TransformInfo> m_TransformHistory = new Dictionary<Transform, TransformInfo>();
    private float m_LastMonitorTime = 0;
    
    [System.Serializable]
    public class TransformInfo
    {
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 scale;
        public float lastCheckTime;
    }
    
    [Header("Debug Visualization")]
    public bool showVelocityVectors = false;
    public bool showAccelerationVectors = false;
    public Color velocityColor = Color.blue;
    public Color accelerationColor = Color.red;
    
    void Update()
    {
        if (monitorVisualization && Time.time - m_LastMonitorTime > monitorInterval)
        {
            MonitorVisualization();
            m_LastMonitorTime = Time.time;
        }
    }
    
    void MonitorVisualization()
    {
        CheckForVisualizationIssues();
        ValidateTransforms();
    }
    
    void CheckForVisualizationIssues()
    {
        // Check for common visualization problems
        Renderer[] renderers = FindObjectsOfType<Renderer>();
        
        foreach (Renderer renderer in renderers)
        {
            // Check if renderer is visible but shouldn't be
            if (renderer.isVisible && !renderer.enabled)
            {
                Debug.LogWarning($"Renderer {renderer.name} is visible but not enabled");
            }
            
            // Check for missing materials
            if (renderer.sharedMaterials.Length == 0)
            {
                Debug.LogWarning($"Renderer {renderer.name} has no materials assigned");
            }
            
            // Check for null materials
            foreach (var material in renderer.sharedMaterials)
            {
                if (material == null)
                {
                    Debug.LogWarning($"Renderer {renderer.name} has null material");
                }
            }
        }
        
        // Check for excessive draw calls
        int triangleCount = 0;
        foreach (Renderer renderer in renderers)
        {
            if (renderer.TryGetComponent<MeshFilter>(out MeshFilter meshFilter))
            {
                if (meshFilter.mesh != null)
                {
                    triangleCount += meshFilter.mesh.triangles.Length / 3;
                }
            }
        }
        
        if (triangleCount > 100000)  // Arbitrary threshold
        {
            Debug.LogWarning($"High triangle count: {triangleCount:N0} triangles. Consider optimization.");
        }
    }
    
    void ValidateTransforms()
    {
        if (!validateTransforms) return;
        
        Transform[] allTransforms = FindObjectsOfType<Transform>();
        
        foreach (Transform t in allTransforms)
        {
            if (t == transform) continue; // Skip the debugger's own transform
            
            // Check for transforms with infinite values
            if (float.IsInfinity(t.position.x) || float.IsInfinity(t.position.y) || float.IsInfinity(t.position.z))
            {
                Debug.LogError($"Transform {t.name} has infinite position");
            }
            
            if (float.IsNaN(t.position.x) || float.IsNaN(t.position.y) || float.IsNaN(t.position.z))
            {
                Debug.LogError($"Transform {t.name} has NaN position");
            }
            
            if (float.IsInfinity(t.rotation.x) || float.IsInfinity(t.rotation.y) || 
                float.IsInfinity(t.rotation.z) || float.IsInfinity(t.rotation.w))
            {
                Debug.LogError($"Transform {t.name} has infinite rotation");
            }
            
            // Check for extremely large positions (could indicate simulation drift)
            if (t.position.magnitude > 10000f)  // 10km seems excessive for most robot scenarios
            {
                Debug.LogWarning($"Transform {t.name} has extremely large position: {t.position}");
            }
            
            // Track transform changes to detect excessive movement
            if (!m_TransformHistory.ContainsKey(t))
            {
                m_TransformHistory[t] = new TransformInfo();
            }
            
            var info = m_TransformHistory[t];
            float distMoved = Vector3.Distance(info.position, t.position);
            
            if (distMoved > 100f)  // Moved more than 100m in 2 seconds
            {
                Debug.LogWarning($"Transform {t.name} moved {distMoved:F2}m in {monitorInterval} seconds - possible physics glitch?");
            }
            
            info.position = t.position;
            info.rotation = t.rotation;
            info.scale = t.localScale;
            info.lastCheckTime = Time.time;
        }
    }
    
    void OnRenderObject()
    {
        if (!showVelocityVectors && !showAccelerationVectors) return;
        
        Transform[] allTransforms = FindObjectsOfType<Transform>();
        
        foreach (Transform t in allTransforms)
        {
            if (!m_TransformHistory.ContainsKey(t)) continue;
            
            var info = m_TransformHistory[t];
            float deltaTime = Time.time - info.lastCheckTime;
            
            if (deltaTime > 0)
            {
                // Calculate velocity (change in position over time)
                Vector3 velocity = (t.position - info.position) / deltaTime;
                
                if (showVelocityVectors && velocity.magnitude > 0.01f)
                {
                    DrawArrow(t.position, t.position + velocity.normalized * 0.5f, velocityColor);
                }
                
                // Calculate acceleration (change in velocity over time)
                // Previous velocity is stored as the velocity from the previous check
                if (deltaTime > 0.01f)  // Avoid division by very small numbers
                {
                    Vector3 previousVelocity = (info.position - t.position) / deltaTime; // From last check
                    Vector3 acceleration = (velocity - previousVelocity) / deltaTime;
                    
                    if (showAccelerationVectors && acceleration.magnitude > 0.01f)
                    {
                        DrawArrow(t.position + Vector3.up * 0.2f, 
                                 t.position + Vector3.up * 0.2f + acceleration.normalized * 0.3f, 
                                 accelerationColor);
                    }
                }
            }
        }
    }
    
    void DrawArrow(Vector3 start, Vector3 end, Color color)
    {
        GL.PushMatrix();
        GL.Begin(GL.LINES);
        GL.Color(color);
        
        // Main line
        GL.Vertex(start);
        GL.Vertex(end);
        
        // Arrowhead
        Vector3 direction = (end - start).normalized;
        Vector3 right = Vector3.Cross(direction, Vector3.up).normalized;
        Vector3 arrowEnd = end - direction * 0.1f;
        
        GL.Vertex(end);
        GL.Vertex(arrowEnd + right * 0.05f);
        GL.Vertex(end);
        GL.Vertex(arrowEnd - right * 0.05f);
        
        GL.End();
        GL.PopMatrix();
    }
}
```

## Debugging Workflows

### 1. Systematic Debugging Approach

Follow a systematic approach to debugging simulation issues:

#### Step 1: Reproduce the Issue
```bash
# 1. Document the exact reproduction steps
# 2. Note system specifications
# 3. Record the ROS graph state
ros2 run rqt_graph rqt_graph

# 4. Check for running processes
ps aux | grep -E "gazebo|ros|unity"
```

#### Step 2: Isolate the Problem
```bash
# Test individual components
# 1. Test ROS network communication independently
ros2 topic echo /joint_states --timeout 10

# 2. Test Gazebo without ROS
gz sim -v 4 --iterations 1000 empty.sdf

# 3. Test robot model in isolation
gz model -m simple_humanoid --file path/to/model.sdf
```

#### Step 3: Examine System Logs
```bash
# Check ROS logs
tail -f ~/.ros/log/latest/*.log

# Check Gazebo logs
tail -f ~/.gz/sim/log/*/log.txt

# Monitor system resources
htop
iotop
dmesg | tail -20
```

### 2. Common Debugging Scenarios

#### Scenario 1: Robot Falls Immediately
**Problem:** Robot falls over immediately after spawning in Gazebo

**Diagnosis Steps:**
1. Check URDF for proper mass and inertial values
2. Verify joint limits and positions
3. Validate that the robot model is properly structured

**Solution:**
```xml
<!-- Example of correct inertial setup -->
<link name="base_link">
  <inertial>
    <!-- Mass should be realistic -->
    <mass value="10.0"/>
    <!-- Inertia values should be physically plausible -->
    <inertia 
      ixx="0.1" ixy="0" ixz="0" 
      iyy="0.2" iyz="0" 
      izz="0.15"/>
  </inertial>
</link>
```

#### Scenario 2: Sensor Data Not Publishing
**Problem:** Sensor topics show no data

**Diagnosis Steps:**
1. Check sensor plugin configuration in URDF
2. Verify Gazebo sensor topics are being created
3. Confirm ROS bridge connection

**Solution:**
```bash
# Check if Gazebo topics exist
gz topic -l | grep -i scan

# Check ROS topic connection
ros2 topic info /scan

# Verify plugin loading
gz model -m simple_humanoid --info
```

## Performance Debugging

### 1. Profiling Tools

Use profiling tools to identify performance bottlenecks:

```python
# scripts/performance_profiler.py
#!/usr/bin/env python3

"""
Performance profiler for simulation debugging.
Monitors computational performance and identifies bottlenecks.
"""

import rclpy
from rclpy.node import Node
import time
import psutil
import threading
from std_msgs.msg import Float64MultiArray, String


class PerformanceProfiler(Node):
    def __init__(self):
        super().__init__('performance_profiler')
        
        # Publishers
        self.perf_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/performance_metrics',
            10
        )
        
        self.perf_report_pub = self.create_publisher(
            String,
            '/performance_report',
            10
        )
        
        # Internal tracking
        self.process_times = {}
        self.cpu_history = []
        self.memory_history = []
        self.topic_rates = {}
        
        # Performance monitoring timer
        self.perf_timer = self.create_timer(1.0, self.monitor_performance)
        
        self.get_logger().info('Performance Profiler Initialized')

    def monitor_performance(self):
        """Monitor system performance"""
        # Get current system stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Track historical data
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        
        # Keep only recent data (last 60 seconds)
        if len(self.cpu_history) > 60:
            self.cpu_history.pop(0)
        if len(self.memory_history) > 60:
            self.memory_history.pop(0)
        
        # Calculate moving averages
        avg_cpu = sum(self.cpu_history[-10:]) / len(self.cpu_history[-10:])
        avg_memory = sum(self.memory_history[-10:]) / len(self.memory_history[-10:])
        
        # Check for performance issues
        issues = []
        if avg_cpu > 80:
            issues.append(f"HIGH_CPU: {avg_cpu:.1f}%")
        if avg_memory > 85:
            issues.append(f"HIGH_MEMORY: {avg_memory:.1f}%")
        if cpu_percent > 95:
            issues.append(f"CPU_SPIKE: {cpu_percent:.1f}%")
        if memory_percent > 95:
            issues.append(f"MEMORY_SPIKE: {memory_percent:.1f}%")
        
        # Publish metrics
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            cpu_percent,
            memory_percent,
            avg_cpu,
            avg_memory,
            0.0,  # placeholder for other metrics
            len(issues)
        ]
        self.perf_metrics_pub.publish(metrics_msg)
        
        # Publish report
        report_msg = String()
        if issues:
            report_msg.data = f"PERFORMANCE_ISSUES: {' | '.join(issues)}"
            self.perf_report_pub.publish(report_msg)
            for issue in issues:
                self.get_logger().warn(f'Performance Issue: {issue}')
        elif int(time.time()) % 60 == 0:  # Every minute
            report_msg.data = f"PERFORMANCE_NORMAL - CPU: {avg_cpu:.1f}%, MEM: {avg_memory:.1f}%"
            self.perf_report_pub.publish(report_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Performance Profiler Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    profiler = PerformanceProfiler()
    
    try:
        rclpy.spin(profiler)
    except KeyboardInterrupt:
        profiler.get_logger().info('Node interrupted by user')
    finally:
        profiler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Debugging Best Practices

### 1. Logging Best Practices

When debugging, implement structured logging:

```python
# Best practice: Use appropriate log levels
logger.debug("Detailed information for diagnosing problems")
logger.info("General information about the system operation")
logger.warn("Something unexpected happened, but system can continue")
logger.error("Functionality is impaired or broken")
logger.fatal("Critical error that stops the system")
```

### 2. Reproduction Steps

Document everything when reporting issues:
1. System specifications (OS, ROS version, Gazebo version, etc.)
2. Exact steps to reproduce the problem
3. Expected vs. actual behavior
4. Relevant log snippets
5. Any configuration files involved

### 3. Incremental Testing

Test components incrementally:
1. Start with the simplest possible setup
2. Add complexity gradually
3. Verify functionality at each step
4. Isolate which addition caused the issue

## Troubleshooting Common Issues

### 1. Physics Instability
**Issue:** Robot shakes violently or explodes
**Solution:**
- Reduce physics timestep
- Increase solver iterations
- Check mass/inertia values
- Verify joint limits

### 2. Sensor Noise
**Issue:** Sensor readings are unrealistic
**Solution:**
- Check sensor plugin configuration
- Verify noise parameters
- Ensure proper coordinate frames

### 3. Communication Delays
**Issue:** Delay between command and response
**Solution:**
- Check network configuration
- Verify topic QoS settings
- Monitor system resources

## Next Steps

With comprehensive debugging techniques in place, you're now prepared to create practical hands-on exercises that will help users apply these debugging methods in real-world scenarios. The debugging framework you've learned provides the foundation for maintaining stable and reliable humanoid robot simulations.