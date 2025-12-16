# Sim-to-Real Transfer Techniques for Humanoid Robotics

## Overview

Sim-to-real transfer is the process of taking skills, behaviors, or models developed in simulation and successfully deploying them on real robots. For humanoid robotics, this process is particularly critical as it bridges the gap between safe, low-cost development in simulation and actual deployment on expensive hardware. This section explores techniques to maximize the effectiveness of transferring your humanoid robot behaviors from simulation to reality.

## Understanding the Reality Gap

### 1. Sources of the Reality Gap

The reality gap consists of differences between simulation and real environments:

- **Physics Fidelity**: Simulations approximate real physics but may not capture all nuances
- **Sensor Noise**: Real sensors have noise, delays, and imperfections not fully modeled in simulation
- **Actuator Dynamics**: Real actuators have different response characteristics than simulated ones
- **Environmental Factors**: Lighting, terrain variations, air resistance, and other factors
- **Modeling Imperfections**: Inaccuracies in robot and environment modeling
- **Temporal Discrepancies**: Differences in timing and reaction speeds

### 2. Sim-to-Real Transfer Strategies

There are several approaches to handling the reality gap:

1. **Domain Randomization**: Training policies in highly varied simulated environments
2. **System Identification**: Calibrating simulation parameters based on real-world data
3. **Transfer Learning**: Fine-tuning simulated models with minimal real-world data
4. **Robust Control Design**: Designing controllers that work across different domains
5. **Domain Adaptation**: Adapting simulation to match real-world statistics

## Domain Randomization for Humanoid Robotics

### 1. Implementing Domain Randomization

Domain randomization involves varying simulation parameters to make the learned behavior robust to differences in reality:

```python
# scripts/domain_randomizer.py
#!/usr/bin/env python3

"""
Domain Randomizer for humanoid robot simulation training.
Randomizes physics parameters, sensor noise, and environmental conditions.
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import random
from threading import Lock


class DomainRandomizerNode(Node):
    """
    Node to implement domain randomization techniques in simulation
    to improve sim-to-real transfer of humanoid behaviors.
    """
    
    def __init__(self):
        super().__init__('domain_randomizer_node')
        
        # Declare parameters
        self.declare_parameter('randomization_rate', 30.0)  # Randomize every 30 seconds
        self.declare_parameter('physics_variance', 0.1)   # 10% variance in physics params
        self.declare_parameter('sensor_noise_variance', 0.2)  # 20% variance in sensor noise
        self.declare_parameter('actuator_variance', 0.15)     # 15% variance in actuator params
        
        # Get parameters
        self.randomization_rate = self.get_parameter('randomization_rate').value
        self.physics_variance = self.get_parameter('physics_variance').value
        self.sensor_noise_variance = self.get_parameter('sensor_noise_variance').value
        self.actuator_variance = self.get_parameter('actuator_variance').value
        
        # Create clients for Gazebo services
        self.get_physics_client = self.create_client(
            GetPhysicsProperties, 
            '/gazebo/get_physics_properties'
        )
        self.set_physics_client = self.create_client(
            SetPhysicsProperties, 
            '/gazebo/set_physics_properties'
        )
        
        # Wait for services to become available
        while not self.get_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get physics service...')
        
        while not self.set_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set physics service...')
        
        # Timer for randomization
        self.randomization_timer = self.create_timer(
            self.randomization_rate, 
            self.randomize_domain
        )
        
        # Publishers for logging randomization parameters
        self.param_log_pub = self.create_publisher(
            Float64MultiArray,
            '/domain_randomization_params',
            10
        )
        
        # Store original physics parameters
        self.original_physics_params = None
        self.domain_lock = Lock()
        
        self.get_logger().info('Domain Randomizer Node Initialized')

    def randomize_domain(self):
        """Apply domain randomization to simulation parameters"""
        self.get_logger().info('Applying domain randomization...')
        
        # Get original physics properties if not already stored
        if self.original_physics_params is None:
            self.store_original_physics()
        
        # Randomize physics parameters
        randomized_params = self.randomize_physics_params()
        
        # Apply randomized physics
        self.apply_physics_changes(randomized_params)
        
        # Randomize sensor noise parameters
        self.randomize_sensor_noise()
        
        # Randomize actuator parameters
        self.randomize_actuator_params()
        
        # Log applied parameters
        self.log_randomization(randomized_params)
        
        self.get_logger().info('Domain randomization applied')

    def store_original_physics(self):
        """Store original physics parameters for reference"""
        try:
            request = GetPhysicsProperties.Request()
            future = self.get_physics_client.call_async(request)
            
            # Wait for response
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            
            self.original_physics_params = {
                'max_step_size': response.max_step_size,
                'real_time_factor': response.real_time_factor,
                'real_time_update_rate': response.real_time_update_rate,
                'gravity': response.gravity,
                'ode_config': response.ode_config
            }
            
            self.get_logger().info('Stored original physics parameters')
            
        except Exception as e:
            self.get_logger().error(f'Error getting original physics: {str(e)}')

    def randomize_physics_params(self):
        """Randomize physics parameters within reasonable bounds"""
        if not self.original_physics_params:
            self.get_logger().error('Original physics parameters not available')
            return None
        
        # Calculate randomized values
        randomized_params = {}
        
        # Randomize max step size (affects integration accuracy)
        step_var = self.physics_variance * self.original_physics_params['max_step_size']
        randomized_params['max_step_size'] = max(
            0.0001,  # Minimum safe step size
            self.original_physics_params['max_step_size'] + random.uniform(-step_var, step_var)
        )
        
        # Randomize real time factor (affects simulation speed)
        rt_var = self.physics_variance * self.original_physics_params['real_time_factor']
        randomized_params['real_time_factor'] = max(
            0.1,  # Minimum RTF
            min(10.0, self.original_physics_params['real_time_factor'] + random.uniform(-rt_var, rt_var))
        )
        
        # Randomize gravity (to account for different environments)
        gravity_variance = self.physics_variance * 9.8
        randomized_params['gravity'] = [
            random.uniform(-0.1, 0.1),  # x variation
            random.uniform(-0.1, 0.1),  # y variation
            random.uniform(-(9.8 + gravity_variance), -(9.8 - gravity_variance))  # z (account for gravity variations)
        ]
        
        # Randomize ODE solver parameters
        randomized_params['ode_config'] = self.original_physics_params['ode_config']
        
        # Randomize solver iterations
        iter_variation = int(self.physics_variance * randomized_params['ode_config'].solver_iterations)
        randomized_params['ode_config'].solver_iterations = max(
            10,  # Minimum iterations
            randomized_params['ode_config'].solver_iterations + random.randint(-iter_variation, iter_variation)
        )
        
        return randomized_params

    def apply_physics_changes(self, params):
        """Apply randomized physics parameters to Gazebo"""
        if not params:
            return
            
        try:
            request = SetPhysicsProperties.Request()
            request.time_step = params['max_step_size']
            request.max_update_rate = params['real_time_update_rate']
            
            # Set gravity
            from geometry_msgs.msg import Vector3
            request.gravity = Vector3(
                x=params['gravity'][0],
                y=params['gravity'][1], 
                z=params['gravity'][2]
            )
            
            # Set ODE config
            request.ode_config = params['ode_config']
            
            future = self.set_physics_client.call_async(request)
            
            # Wait for response
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            
            if response.success:
                self.get_logger().info(
                    f'Physics updated: step={params["max_step_size"]:.4f}, '
                    f'RTF={params["real_time_factor"]:.2f}, '
                    f'gravity=({params["gravity"][0]:.2f},{params["gravity"][1]:.2f},{params["gravity"][2]:.2f})'
                )
            else:
                self.get_logger().error(f'Failed to update physics: {response.status_message}')
                
        except Exception as e:
            self.get_logger().error(f'Error applying physics changes: {str(e)}')

    def randomize_sensor_noise(self):
        """Randomize sensor noise parameters"""
        # In a real implementation, this would dynamically update noise parameters
        # for sensors in the simulation. For this example, we'll log what would be changed.
        
        # This would typically involve modifying sensor plugins dynamically,
        # which might require custom services to communicate with Gazebo plugins
        sensor_noise_adjustments = {
            'lidar_noise': random.uniform(0.8, 1.2) * self.sensor_noise_variance,
            'imu_noise': random.uniform(0.8, 1.2) * self.sensor_noise_variance,
            'camera_noise': random.uniform(0.8, 1.2) * self.sensor_noise_variance,
            'joint_encoder_noise': random.uniform(0.8, 1.2) * self.sensor_noise_variance
        }
        
        self.get_logger().info(f'Sensor noise adjusted: {sensor_noise_adjustments}')

    def randomize_actuator_params(self):
        """Randomize actuator parameters"""
        # This would update actuator dynamics like friction, damping, and response characteristics
        actuator_changes = {
            'friction_coefficients': [random.uniform(0.8, 1.2) for _ in range(12)],  # For 12 joints
            'position_precision': random.uniform(0.8, 1.2) * self.actuator_variance,
            'response_delay': random.uniform(0.0, 0.05)  # 0-50ms delay
        }
        
        self.get_logger().info(f'Actuator parameters adjusted: {actuator_changes}')

    def log_randomization(self, params):
        """Log randomization parameters for tracking"""
        if not params:
            return
            
        log_msg = Float64MultiArray()
        log_msg.data = [
            params['max_step_size'],
            params['real_time_factor'],
            params['gravity'][0],
            params['gravity'][1],
            params['gravity'][2],
            params['ode_config'].solver_iterations,
            self.get_clock().now().nanoseconds / 1e9  # Timestamp
        ]
        
        self.param_log_pub.publish(log_msg)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Domain Randomizer Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    randomizer = DomainRandomizerNode()
    
    try:
        rclpy.spin(randomizer)
    except KeyboardInterrupt:
        randomizer.get_logger().info('Node interrupted by user')
    finally:
        randomizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## System Identification Approach

### 1. Parameter Calibration for Realistic Simulation

System identification involves adjusting simulation parameters to match real-world behavior:

```python
# scripts/system_identification.py
#!/usr/bin/env python3

"""
System Identification Node for improving sim-to-real transfer.
Calibrates simulation parameters based on real-world data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float64MultiArray, String
import numpy as np
from scipy.optimize import minimize
from threading import Lock
import pickle


class SystemIdentificationNode(Node):
    """
    Node to perform system identification to calibrate simulation
    parameters based on real robot data to reduce sim-to-real gap.
    """
    
    def __init__(self):
        super().__init__('system_identification_node')
        
        # Declare parameters
        self.declare_parameter('identification_frequency', 1.0)  # Hz
        self.declare_parameter('calibration_window', 10.0)      # seconds
        self.declare_parameter('max_iterations', 100)           # optimization iterations
        self.declare_parameter('output_calibration_file', 'calibration_results.pkl')
        
        # Get parameters
        self.identification_frequency = self.get_parameter('identification_frequency').value
        self.calibration_window = self.get_parameter('calibration_window').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.output_cal_file = self.get_parameter('output_calibration_file').value
        
        # Data storage for system identification
        self.simulator_data = []  # Data from simulation
        self.real_robot_data = []  # Placeholder for real robot data (would come from actual robot or dataset)
        self.calibration_targets = {}  # Parameters we want to calibrate
        
        # Subscriptions
        self.sim_joint_state_sub = self.create_subscription(
            JointState,
            '/sim/joint_states',
            self.sim_joint_state_callback,
            10
        )
        
        self.sim_imu_sub = self.create_subscription(
            Imu,
            '/sim/imu',
            self.sim_imu_callback,
            10
        )
        
        # Real robot data subscription (would connect to real robot or dataset)
        self.real_joint_state_sub = self.create_subscription(
            JointState,
            '/real_robot/joint_states',  # This would come from real robot
            self.real_joint_state_callback,
            10
        )
        
        self.real_imu_sub = self.create_subscription(
            Imu,
            '/real_robot/imu',  # This would come from real robot
            self.real_imu_callback,
            10
        )
        
        # Publishers
        self.calibration_status_pub = self.create_publisher(
            String,
            '/calibration_status',
            10
        )
        
        self.optimized_params_pub = self.create_publisher(
            Float64MultiArray,
            '/optimized_simulation_params',
            10
        )
        
        # Timer for calibration process
        self.calibration_timer = self.create_timer(
            1.0 / self.identification_frequency,
            self.run_calibration_cycle
        )
        
        # Initialize calibration target parameters
        self.initialize_calibration_targets()
        
        self.get_logger().info('System Identification Node Initialized')

    def initialize_calibration_targets(self):
        """Initialize parameters we want to calibrate"""
        # Physics parameters that significantly impact sim-to-real transfer
        self.calibration_targets = {
            # Joint-level parameters
            'left_hip_damping': {
                'initial_value': 0.5,
                'bounds': (0.01, 5.0),
                'type': 'joint_specific',
                'joints': ['left_hip_joint']
            },
            'left_knee_damping': {
                'initial_value': 1.0,
                'bounds': (0.01, 10.0),
                'type': 'joint_specific',
                'joints': ['left_knee_joint']
            },
            'left_ankle_damping': {
                'initial_value': 0.3,
                'bounds': (0.01, 5.0),
                'type': 'joint_specific',
                'joints': ['left_ankle_joint']
            },
            'right_hip_damping': {
                'initial_value': 0.5,
                'bounds': (0.01, 5.0),
                'type': 'joint_specific',
                'joints': ['right_hip_joint']
            },
            'right_knee_damping': {
                'initial_value': 1.0,
                'bounds': (0.01, 10.0),
                'type': 'joint_specific',
                'joints': ['right_knee_joint']
            },
            'right_ankle_damping': {
                'initial_value': 0.3,
                'bounds': (0.01, 5.0),
                'type': 'joint_specific',
                'joints': ['right_ankle_joint']
            },
            # Global parameters
            'gravity_compensation': {
                'initial_value': 0.98,  # Multiplier for gravity (9.62 to 9.90 m/s^2 range)
                'bounds': (0.97, 1.03),
                'type': 'global'
            },
            'friction_coefficient': {
                'initial_value': 0.8,
                'bounds': (0.1, 2.0),
                'type': 'global'
            },
            'contact_stiffness': {
                'initial_value': 1000000.0,
                'bounds': (100000.0, 10000000.0),
                'type': 'global'
            }
        }
        
        # Initialize optimization parameters
        self.param_names = list(self.calibration_targets.keys())
        self.param_bounds = [
            self.calibration_targets[name]['bounds'] 
            for name in self.param_names
        ]
        self.initial_params = [
            self.calibration_targets[name]['initial_value'] 
            for name in self.param_names
        ]

    def sim_joint_state_callback(self, msg):
        """Store simulator joint state data"""
        self.simulator_data.append({
            'type': 'joint_state',
            'timestamp': msg.header.stamp,
            'data': msg,
            'source': 'simulation'
        })

    def sim_imu_callback(self, msg):
        """Store simulator IMU data"""
        self.simulator_data.append({
            'type': 'imu',
            'timestamp': msg.header.stamp,
            'data': msg,
            'source': 'simulation'
        })

    def real_joint_state_callback(self, msg):
        """Store real robot joint state data"""
        self.real_robot_data.append({
            'type': 'joint_state',
            'timestamp': msg.header.stamp,
            'data': msg,
            'source': 'real_robot'
        })

    def real_imu_callback(self, msg):
        """Store real robot IMU data"""
        self.real_robot_data.append({
            'type': 'imu',
            'timestamp': msg.header.stamp,
            'data': msg,
            'source': 'real_robot'
        })

    def run_calibration_cycle(self):
        """Run the calibration cycle"""
        # Only run if we have sufficient data
        if len(self.simulator_data) < 100 or len(self.real_robot_data) < 100:
            status_msg = String()
            status_msg.data = "INSUFFICIENT_DATA_FOR_CALIBRATION"
            self.calibration_status_pub.publish(status_msg)
            return
        
        # Perform system identification
        try:
            optimized_params = self.perform_system_identification()
            
            # Apply optimized parameters to simulation
            self.apply_calibration_results(optimized_params)
            
            # Publish results
            self.publish_calibration_results(optimized_params)
            
            status_msg = String()
            status_msg.data = f"CALIBRATION_COMPLETE: {len(optimized_params)} parameters adjusted"
            self.calibration_status_pub.publish(status_msg)
            
            self.get_logger().info(f'Calibration completed: {optimized_params}')
            
        except Exception as e:
            self.get_logger().error(f'Error during calibration: {str(e)}')

    def perform_system_identification(self):
        """Perform system identification using optimization"""
        def cost_function(params):
            """
            Calculate cost between simulation and real robot behavior
            based on current parameters
            """
            # Apply parameters to simulator (this would involve calling Gazebo services)
            self.apply_temporary_params(params)
            
            # Calculate similarity between sim and real data
            sim_subset = self.extract_matching_data('simulation', len(self.real_robot_data))
            real_subset = self.extract_matching_data('real_robot', len(self.simulator_data))
            
            # Calculate cost as sum of squared differences
            cost = 0.0
            min_len = min(len(sim_subset), len(real_subset))
            
            if min_len == 0:
                return float('inf')  # Return high cost if no matching data
            
            for i in range(min_len):
                sim_dat = sim_subset[i]
                real_dat = real_subset[i]
                
                # Compare joint positions for similar timestamps
                if sim_dat['type'] == 'joint_state' and real_dat['type'] == 'joint_state':
                    if abs(
                        (sim_dat['timestamp'].sec + sim_dat['timestamp'].nanosec/1e9) - 
                        (real_dat['timestamp'].sec + real_dat['timestamp'].nanosec/1e9)
                    ) < 0.1:  # If timestamps are within 100ms
                        # Calculate difference in joint positions
                        for j in range(min(len(sim_dat['data'].position), len(real_dat['data'].position))):
                            position_diff = sim_dat['data'].position[j] - real_dat['data'].position[j]
                            cost += position_diff**2
            
            # Apply regularization to prevent extreme parameter values
            regularization = 0.1 * sum((p - init_p)**2 for p, init_p in zip(params, self.initial_params))
            total_cost = cost + regularization
            
            return total_cost

        # Perform optimization
        result = minimize(
            cost_function,
            self.initial_params,
            method='L-BFGS-B',
            bounds=self.param_bounds,
            options={'maxiter': self.max_iterations}
        )
        
        return result.x

    def apply_temporary_params(self, params):
        """
        Apply parameters to the simulation temporarily for evaluation.
        In a real implementation, this would modify Gazebo physics parameters.
        """
        # This is a simulation of applying parameters
        # In reality, we'd call Gazebo services to change physics properties
        param_dict = dict(zip(self.param_names, params))
        
        # In real implementation:
        # - Modify physics parameters via Gazebo services
        # - Update joint dynamics in URDF/SDF
        # - Adjust control parameters
        pass

    def extract_matching_data(self, source, max_count):
        """Extract data from specified source, limited to max_count entries"""
        data = [d for d in self.simulator_data if d['source'] == source]
        return data[:max_count]

    def apply_calibration_results(self, optimized_params):
        """Apply the optimized parameters to the simulation"""
        calibrated_params = dict(zip(self.param_names, optimized_params))
        
        # In a real implementation, this would update Gazebo physics parameters
        # through appropriate services/parameters
        
        # Log the calibrated parameters
        for name, value in calibrated_params.items():
            self.get_logger().info(f'Calibrated {name}: {value:.4f}')

    def publish_calibration_results(self, optimized_params):
        """Publish the optimized parameters"""
        params_msg = Float64MultiArray()
        params_msg.data = list(optimized_params)
        
        self.optimized_params_pub.publish(params_msg)
        
        # Save calibration results to file
        with open(self.output_cal_file, 'wb') as f:
            pickle.dump(dict(zip(self.param_names, optimized_params)), f)
        
        self.get_logger().info(f'Calibration results saved to {self.output_cal_file}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('System Identification Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    ident_node = SystemIdentificationNode()
    
    try:
        rclpy.spin(ident_node)
    except KeyboardInterrupt:
        ident_node.get_logger().info('Node interrupted by user')
    finally:
        ident_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Robust Control Design

### 1. Implementing Robust Controllers

Robust controllers can handle variations between simulation and reality:

```python
# scripts/robust_controller.py
#!/usr/bin/env python3

"""
Robust Controller for Humanoid Robots.
Implements controllers that maintain performance across different domains.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
from builtin_interfaces.msg import Duration
import numpy as np
import math


class RobustController(Node):
    """
    Robust controller for humanoid robots that maintains stability
    across simulation-to-reality variations.
    """
    
    def __init__(self):
        super().__init__('robust_controller')
        
        # Declare parameters
        self.declare_parameter('control_frequency', 100.0)  # Hz
        self.declare_parameter('pid_kp_range', [0.8, 1.2])  # Range for Kp randomization
        self.declare_parameter('pid_ki_range', [0.5, 1.5])  # Range for Ki randomization
        self.declare_parameter('pid_kd_range', [0.9, 1.1])  # Range for Kd randomization
        self.declare_parameter('disturbance_magnitude', 0.1)  # Magnitude of injected disturbances
        
        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.pid_kp_range = self.get_parameter('pid_kp_range').value
        self.pid_ki_range = self.get_parameter('pid_ki_range').value
        self.pid_kd_range = self.get_parameter('pid_kd_range').value
        self.disturbance_magnitude = self.get_parameter('disturbance_magnitude').value
        
        # Controller state
        self.joint_states = None
        self.command_buffer = []  # Buffer for command smoothing
        self.controller_gains = {}  # Per-joint adaptive gains
        
        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.command_sub = self.create_subscription(
            JointState,
            '/joint_commands',
            self.command_callback,
            10
        )
        
        # Publishers
        self.command_pub = self.create_publisher(
            JointState,
            '/position_commands',  # For position-controlled robots
            10
        )
        
        self.effort_pub = self.create_publisher(
            JointState,
            '/effort_commands',  # For effort-controlled robots
            10
        )
        
        self.controller_status_pub = self.create_publisher(
            String,
            '/robust_controller_status',
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_cycle)
        
        # Initialize adaptive PID controllers for each joint
        self.initialize_controllers()
        
        self.get_logger().info('Robust Controller Node Initialized')

    def initialize_controllers(self):
        """Initialize PID controllers with adaptive parameters"""
        # Define humanoid-specific joints
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]
        
        # Initialize PID controllers with randomizable gains
        self.pid_controllers = {}
        for joint_name in self.joint_names:
            # Randomize gains within range to make controllers robust to gain variations
            kp = np.random.uniform(self.pid_kp_range[0], self.pid_kp_range[1])
            ki = np.random.uniform(self.pid_ki_range[0], self.pid_ki_range[1])
            kd = np.random.uniform(self.pid_kd_range[0], self.pid_kd_range[1])
            
            self.pid_controllers[joint_name] = {
                'kp': kp,
                'ki': ki,
                'kd': kd,
                'integral': 0.0,
                'previous_error': 0.0,
                'previous_time': None,
                'command_history': [],
                'state_history': []
            }
            
            self.get_logger().info(f'Initialized PID for {joint_name}: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}')

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.joint_states = msg

    def command_callback(self, msg):
        """Store desired joint commands"""
        self.desired_joint_states = msg

    def control_cycle(self):
        """Main control cycle"""
        if not self.joint_states or not hasattr(self, 'desired_joint_states'):
            return
        
        try:
            # Calculate control commands using robust control algorithms
            control_commands = self.compute_robust_controls()
            
            # Publish commands
            self.command_pub.publish(control_commands)
            
            # Publish controller status
            status_msg = String()
            status_msg.data = f"CONTROLLING {len(self.joint_names)} JOINTS"
            self.controller_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in control cycle: {str(e)}')

    def compute_robust_controls(self):
        """Compute robust control commands for all joints"""
        # Create output message
        output_cmd = JointState()
        output_cmd.header.stamp = self.get_clock().now().to_msg()
        output_cmd.name = self.joint_names[:]
        output_cmd.position = [0.0] * len(self.joint_names)
        output_cmd.velocity = [0.0] * len(self.joint_names)
        output_cmd.effort = [0.0] * len(self.joint_names)
        
        # Process each joint
        for i, joint_name in enumerate(self.joint_names):
            # Find joint in current state
            current_pos = self.get_joint_position(joint_name)
            desired_pos = self.get_desired_position(joint_name)
            
            if current_pos is not None and desired_pos is not None:
                # Apply PID control with adaptive gains
                control_effort = self.adaptive_pid_control(
                    joint_name, 
                    desired_pos, 
                    current_pos
                )
                
                # Add disturbance to make controller robust to real-world variations
                disturbance = np.random.uniform(-self.disturbance_magnitude, self.disturbance_magnitude)
                control_effort += disturbance
                
                # Apply safety limits
                control_effort = max(min(control_effort, 100.0), -100.0)  # Effort limits
                
                output_cmd.effort[i] = control_effort
                # For position control, add to current position
                output_cmd.position[i] = current_pos + control_effort * 0.001  # Small increment for stability
        
        return output_cmd

    def adaptive_pid_control(self, joint_name, desired, current):
        """Adaptive PID control with gain scheduling"""
        if joint_name not in self.pid_controllers:
            return 0.0
        
        controller = self.pid_controllers[joint_name]
        
        # Calculate error
        error = desired - current
        
        # Get current time
        current_time = self.get_clock().now()
        if controller['previous_time'] is None:
            controller['previous_time'] = current_time
            return 0.0
        
        # Calculate dt
        dt = (current_time - controller['previous_time']).nanoseconds / 1e9
        if dt <= 0:
            return 0.0
        
        # Update integral term (with anti-windup)
        controller['integral'] += error * dt
        integral_limit = 10.0  # Limit integral windup
        controller['integral'] = max(min(controller['integral'], integral_limit), -integral_limit)
        
        # Calculate derivative
        derivative = 0.0
        if dt > 0:
            derivative = (error - controller['previous_error']) / dt
            # Add noise filtering to make derivative more robust
            derivative *= 0.7  # Simple first-order filter
            derivative += controller['previous_derivative'] * 0.3 if 'previous_derivative' in controller else 0.0
        
        # Calculate PID output
        output = (
            controller['kp'] * error + 
            controller['ki'] * controller['integral'] + 
            controller['kd'] * derivative
        )
        
        # Update stored values
        controller['previous_error'] = error
        controller['previous_derivative'] = derivative
        controller['previous_time'] = current_time
        
        return output

    def get_joint_position(self, joint_name):
        """Get current position for specified joint"""
        if not self.joint_states:
            return None
        
        try:
            idx = self.joint_states.name.index(joint_name)
            return self.joint_states.position[idx] if idx < len(self.joint_states.position) else None
        except ValueError:
            return None

    def get_desired_position(self, joint_name):
        """Get desired position for specified joint"""
        if not hasattr(self, 'desired_joint_states') or not self.desired_joint_states:
            return None
        
        try:
            idx = self.desired_joint_states.name.index(joint_name)
            return self.desired_joint_states.position[idx] if idx < len(self.desired_joint_states.position) else None
        except ValueError:
            return None

    def inject_disturbances(self, control_signal, joint_name):
        """Inject realistic disturbances to make controller robust"""
        # Add random disturbances that simulate real-world variations
        disturbance = 0.0
        
        # Position-dependent disturbances
        if joint_name in ['left_hip_joint', 'right_hip_joint']:
            # Hip joints have more external forces
            disturbance = np.random.normal(0, self.disturbance_magnitude * 1.5)
        elif joint_name in ['left_ankle_joint', 'right_ankle_joint']:
            # Ankle joints experience ground reaction forces
            disturbance = np.random.normal(0, self.disturbance_magnitude * 1.2)
        else:
            # Other joints
            disturbance = np.random.normal(0, self.disturbance_magnitude)
        
        # Add time-varying components
        time_factor = math.sin(self.get_clock().now().nanoseconds / 1e9)  # Slow oscillation
        disturbance += time_factor * self.disturbance_magnitude * 0.1
        
        return control_signal + disturbance

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Robust Controller Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobustController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Node interrupted by user')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Domain Adaptation Techniques

### 1. Neural Network Adaptation

For AI-based systems, domain adaptation can help neural networks work across domains:

```python
# scripts/domain_adaptation.py
#!/usr/bin/env python3

"""
Domain Adaptation Node for AI models in humanoid robotics.
Adapts simulation-trained models to work better in reality.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
import numpy as np
import tensorflow as tf  # Using TensorFlow as example, but PyTorch could be used
from sklearn.preprocessing import StandardScaler


class DomainAdaptationNode(Node):
    """
    Implements domain adaptation techniques to adjust simulation-trained
    models for real-world use in humanoid robotics.
    """
    
    def __init__(self):
        super().__init__('domain_adaptation_node')
        
        # Declare parameters
        self.declare_parameter('adaptation_frequency', 1.0)  # Hz
        self.declare_parameter('adaptation_method', 'statistical_matching')  # Options: statistical_matching, fine_tuning, normalization
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('adaptation_threshold', 0.1)  # Threshold for triggering adaptation
        
        # Get parameters
        self.adaptation_frequency = self.get_parameter('adaptation_frequency').value
        self.adaptation_method = self.get_parameter('adaptation_method').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.adaptation_threshold = self.get_parameter('adaptation_threshold').value
        
        # Data buffers for domain adaptation
        self.sim_image_buffer = []
        self.real_image_buffer = []
        self.sim_lidar_buffer = []
        self.real_lidar_buffer = []
        
        # Adaptation parameters
        self.scaler = StandardScaler()
        self.adaptation_active = True
        self.last_adaptation_time = self.get_clock().now()
        
        # Subscriptions (simulated real-world data)
        self.sim_image_sub = self.create_subscription(
            Image,
            '/sim/camera/image_raw',
            self.sim_image_callback,
            10
        )
        
        self.real_image_sub = self.create_subscription(
            Image,
            '/real_robot/camera/image_raw',  # Would come from real robot
            self.real_image_callback,
            10
        )
        
        self.sim_lidar_sub = self.create_subscription(
            LaserScan,
            '/sim/scan',
            self.sim_lidar_callback,
            10
        )
        
        self.real_lidar_sub = self.create_subscription(
            LaserScan,
            '/real_robot/scan',  # Would come from real robot
            self.real_lidar_callback,
            10
        )
        
        # Publishers
        self.adaptation_status_pub = self.create_publisher(
            String,
            '/domain_adaptation_status',
            10
        )
        
        self.adapted_commands_pub = self.create_publisher(
            Twist,
            '/adapted_cmd_vel',
            10
        )
        
        # Timer for adaptation
        self.adaptation_timer = self.create_timer(
            1.0 / self.adaptation_frequency,
            self.run_adaptation_cycle
        )
        
        self.get_logger().info('Domain Adaptation Node Initialized')

    def sim_image_callback(self, msg):
        """Store simulation image data"""
        if len(self.sim_image_buffer) >= self.buffer_size:
            self.sim_image_buffer.pop(0)
        
        # Convert image to usable format for statistics
        img_stats = self.extract_image_statistics(msg)
        self.sim_image_buffer.append(img_stats)

    def real_image_callback(self, msg):
        """Store real robot image data"""
        if len(self.real_image_buffer) >= self.buffer_size:
            self.real_image_buffer.pop(0)
        
        # Convert image to usable format for statistics
        img_stats = self.extract_image_statistics(msg)
        self.real_image_buffer.append(img_stats)

    def sim_lidar_callback(self, msg):
        """Store simulation LiDAR data"""
        if len(self.sim_lidar_buffer) >= self.buffer_size:
            self.sim_lidar_buffer.pop(0)
        
        # Extract statistics from LiDAR scan
        lidar_stats = self.extract_lidar_statistics(msg)
        self.sim_lidar_buffer.append(lidar_stats)

    def real_lidar_callback(self, msg):
        """Store real robot LiDAR data"""
        if len(self.real_lidar_buffer) >= self.buffer_size:
            self.real_lidar_buffer.pop(0)
        
        # Extract statistics from LiDAR scan
        lidar_stats = self.extract_lidar_statistics(msg)
        self.real_lidar_buffer.append(lidar_stats)

    def extract_image_statistics(self, image_msg):
        """Extract statistical features from image for domain adaptation"""
        # This is a simplified approach - in practice, we'd use more sophisticated features
        if not image_msg.data:
            return np.array([0.0, 0.0, 0.0, 0.0])  # default values
        
        # Convert to numpy array (simplified)
        try:
            # In a real implementation, we'd properly decode the image message
            # For now, we'll just calculate some basic statistics on the raw data
            data_sample = np.frombuffer(image_msg.data[:min(len(image_msg.data), 1000)], dtype=np.uint8)
            
            if len(data_sample) == 0:
                return np.array([0.0, 0.0, 0.0, 0.0])
            
            mean_intensity = float(np.mean(data_sample))
            std_intensity = float(np.std(data_sample))
            min_intensity = float(np.min(data_sample))
            max_intensity = float(np.max(data_sample))
            
            return np.array([mean_intensity, std_intensity, min_intensity, max_intensity])
        except:
            return np.array([0.0, 0.0, 0.0, 0.0])

    def extract_lidar_statistics(self, scan_msg):
        """Extract statistical features from LiDAR data for domain adaptation"""
        valid_ranges = [r for r in scan_msg.ranges if np.isfinite(r)]
        
        if not valid_ranges:
            return np.array([0.0, 0.0, 0.0, 0.0])  # defaults
        
        mean_range = float(np.mean(valid_ranges))
        std_range = float(np.std(valid_ranges))
        min_range = float(np.min(valid_ranges))
        max_range = float(np.max(valid_ranges))
        
        return np.array([mean_range, std_range, min_range, max_range])

    def run_adaptation_cycle(self):
        """Main adaptation cycle"""
        if not (self.sim_image_buffer and self.real_image_buffer):
            status_msg = String()
            status_msg.data = "NO_DATA_AVAILABLE"
            self.adaptation_status_pub.publish(status_msg)
            return
        
        try:
            # Calculate domain discrepancy
            discrepancy = self.calculate_domain_discrepancy()
            
            # Check if adaptation is needed
            if discrepancy > self.adaptation_threshold:
                self.perform_adaptation()
                self.get_logger().info(f'Domain adaptation performed. Discrepancy: {discrepancy:.4f}')
            else:
                self.get_logger().debug(f'Domain discrepancy acceptable: {discrepancy:.4f}')
            
            # Publish status
            status_msg = String()
            status_msg.data = f"DOMAIN_ADAPTION: Discrepancy={discrepancy:.4f}, Active={self.adaptation_active}"
            self.adaptation_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in adaptation cycle: {str(e)}')

    def calculate_domain_discrepancy(self):
        """Calculate discrepancy between simulation and real domains"""
        if not (self.sim_image_buffer and self.real_image_buffer):
            return 0.0
        
        # Calculate discrepancy using statistical moments
        sim_stats = np.array(self.sim_image_buffer)
        real_stats = np.array(self.real_image_buffer)
        
        # Calculate mean discrepancy
        sim_means = np.mean(sim_stats, axis=0)
        real_means = np.mean(real_stats, axis=0)
        
        # Normalize by range
        sim_ranges = np.max(sim_stats, axis=0) - np.min(sim_stats, axis=0)
        # Avoid division by zero
        sim_ranges = np.where(sim_ranges == 0, 1.0, sim_ranges)
        
        mean_discrepancy = np.mean(np.abs(sim_means - real_means) / sim_ranges)
        
        return mean_discrepancy

    def perform_adaptation(self):
        """Perform domain adaptation based on selected method"""
        if self.adaptation_method == 'statistical_matching':
            self.statistical_domain_matching()
        elif self.adaptation_method == 'fine_tuning':
            self.fine_tune_model()
        elif self.adaptation_method == 'normalization':
            self.feature_normalization()
        
        self.last_adaptation_time = self.get_clock().now()

    def statistical_domain_matching(self):
        """Match statistical distributions between sim and real domains"""
        if not (self.sim_image_buffer and self.real_image_buffer):
            return
        
        sim_data = np.array(self.sim_image_buffer)
        real_data = np.array(self.real_image_buffer)
        
        # Fit scaler to real data distribution
        self.scaler.fit(real_data)
        
        # Calculate transformation parameters
        sim_mean = np.mean(sim_data, axis=0)
        sim_std = np.std(sim_data, axis=0)
        real_mean = np.mean(real_data, axis=0)
        real_std = np.std(real_data, axis=0)
        
        # Store normalization parameters
        self.normalization_params = {
            'sim_to_real_scale': real_std / (sim_std + 1e-8),
            'sim_to_real_bias': real_mean - sim_mean * (real_std / (sim_std + 1e-8))
        }

    def feature_normalization(self):
        """Normalize features to match real domain statistics"""
        # This would update internal normalization parameters
        # to transform simulation features to better match real features
        pass

    def fine_tune_model(self):
        """Placeholder for neural network fine-tuning"""
        # In a real implementation, this would load a pretrained model
        # and fine-tune it with real robot data
        self.get_logger().warn('Fine-tuning implementation would go here')

    def transform_simulation_features(self, sim_features):
        """Transform simulation features to better match real domain"""
        if hasattr(self, 'normalization_params'):
            # Apply learned transformation
            scale = self.normalization_params['sim_to_real_scale']
            bias = self.normalization_params['sim_to_real_bias']
            return sim_features * scale + bias
        else:
            # No transformation if adaptation hasn't been performed yet
            return sim_features

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Domain Adaptation Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    adapter = DomainAdaptationNode()
    
    try:
        rclpy.spin(adapter)
    except KeyboardInterrupt:
        adapter.get_logger().info('Node interrupted by user')
    finally:
        adapter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Transfer Learning Implementation

### 1. Knowledge Distillation Node

Implement a knowledge distillation approach for transferring learned behaviors:

```python
# scripts/knowledge_distiller.py
#!/usr/bin/env python3

"""
Knowledge Distiller Node for transferring knowledge from 
simulation-trained models to reality-adapted models.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import numpy as np
import math


class KnowledgeDistillerNode(Node):
    """
    Implements knowledge distillation to transfer learned behaviors
    from simulation to reality with minimal additional training.
    """
    
    def __init__(self):
        super().__init__('knowledge_distiller_node')
        
        # Declare parameters
        self.declare_parameter('distillation_frequency', 0.5)  # Hz
        self.declare_parameter('temperature', 3.0)  # Temperature for softening predictions
        self.declare_parameter('distillation_loss_coeff', 0.7)  # Weight of distillation loss
        self.declare_parameter('student_learning_rate', 0.001)
        
        # Get parameters
        self.distillation_frequency = self.get_parameter('distillation_frequency').value
        self.temperature = self.get_parameter('temperature').value
        self.distillation_loss_coeff = self.get_parameter('distillation_loss_coeff').value
        self.student_learning_rate = self.get_parameter('student_learning_rate').value
        
        # Student and teacher model representations (simplified)
        self.teacher_model = {}  # Would be a neural network in practice
        self.student_model = {}  # Would be a smaller/lighter neural network
        
        # Training data buffers
        self.sim_behavior_buffer = []
        self.real_behavior_buffer = []
        
        # Subscriptions for behavior data
        self.teacher_action_sub = self.create_subscription(
            Float64MultiArray,
            '/teacher_actions',  # Actions from simulation-trained model
            self.teacher_action_callback,
            10
        )
        
        self.student_action_sub = self.create_subscription(
            Float64MultiArray,
            '/student_actions',  # Actions from reality-adapted model
            self.student_action_callback,
            10
        )
        
        self.perception_input_sub = self.create_subscription(
            Float64MultiArray,
            '/perception_features',
            self.perception_input_callback,
            10
        )
        
        # Publishers
        self.distilled_action_pub = self.create_publisher(
            Float64MultiArray,
            '/distilled_actions',
            10
        )
        
        self.distillation_status_pub = self.create_publisher(
            String,
            '/distillation_status',
            10
        )
        
        # Timer for distillation process
        self.distillation_timer = self.create_timer(
            1.0 / self.distillation_frequency,
            self.run_distillation_cycle
        )
        
        self.get_logger().info('Knowledge Distiller Node Initialized')

    def teacher_action_callback(self, msg):
        """Store actions from simulation-trained (teacher) model"""
        if len(self.sim_behavior_buffer) >= 1000:  # Limit buffer size
            self.sim_behavior_buffer.pop(0)
        
        self.sim_behavior_buffer.append({
            'timestamp': self.get_clock().now(),
            'actions': list(msg.data),
            'source': 'teacher'
        })

    def student_action_callback(self, msg):
        """Store actions from reality-adapted (student) model"""
        if len(self.real_behavior_buffer) >= 1000:  # Limit buffer size
            self.real_behavior_buffer.pop(0)
        
        self.real_behavior_buffer.append({
            'timestamp': self.get_clock().now(),
            'actions': list(msg.data),
            'source': 'student'
        })

    def perception_input_callback(self, msg):
        """Store perception inputs to associate with actions"""
        # This would be used to correlate perceptions with actions for learning
        pass

    def run_distillation_cycle(self):
        """Run knowledge distillation cycle"""
        if len(self.sim_behavior_buffer) < 10 or len(self.real_behavior_buffer) < 10:
            status_msg = String()
            status_msg.data = "INSUFFICIENT_TRAINING_DATA"
            self.distillation_status_pub.publish(status_msg)
            return
        
        try:
            # Calculate distillation loss between teacher and student
            distillation_loss = self.calculate_distillation_loss()
            
            # Update student model based on distillation loss
            self.update_student_model(distillation_loss)
            
            # Calculate similarity between teacher and student actions
            similarity = self.calculate_behavior_similarity()
            
            # Publish distilled actions
            distilled_actions = self.generate_distilled_actions()
            self.distilled_action_pub.publish(distilled_actions)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Distillation Loss: {distillation_loss:.4f}, " \
                             f"Similarity: {similarity:.4f}, " \
                             f"Student Updated: {self.get_clock().now().nanoseconds / 1e9:.2f}"
            self.distillation_status_pub.publish(status_msg)
            
            self.get_logger().info(f'Distillation cycle: Loss={distillation_loss:.4f}, ' \
                                  f'Similarity={similarity:.4f}')
            
        except Exception as e:
            self.get_logger().error(f'Error in distillation cycle: {str(e)}')

    def calculate_distillation_loss(self):
        """Calculate distillation loss between teacher and student outputs"""
        if not (self.sim_behavior_buffer and self.real_behavior_buffer):
            return 0.0
        
        # Get recent teacher and student actions
        teacher_actions = np.array([item['actions'] for item in self.sim_behavior_buffer[-10:]])
        student_actions = np.array([item['actions'] for item in self.real_behavior_buffer[-10:]])
        
        # Ensure same dimensions
        min_length = min(teacher_actions.shape[1], student_actions.shape[1])
        teacher_actions = teacher_actions[:, :min_length]
        student_actions = student_actions[:, :min_length]
        
        # Calculate soft targets (using temperature to soften probability distributions)
        soft_teacher = self.softmax(teacher_actions, self.temperature)
        soft_student = self.softmax(student_actions, self.temperature)
        
        # Calculate KL divergence loss
        distillation_loss = self.kl_divergence(soft_teacher, soft_student)
        
        return np.mean(distillation_loss)

    def softmax(self, x, temperature=1.0):
        """Softmax function with temperature parameter"""
        exp_x = np.exp((x - np.max(x, axis=1, keepdims=True)) / temperature)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def kl_divergence(self, p, q):
        """Calculate KL divergence between two probability distributions"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        
        return np.sum(p * np.log(p / q), axis=1)

    def update_student_model(self, loss):
        """Update student model based on distillation loss"""
        # In a real implementation, this would update the neural network weights
        # For this example, we'll just log the update
        self.get_logger().debug(f'Updating student model with distillation loss: {loss:.4f}')

    def calculate_behavior_similarity(self):
        """Calculate similarity between teacher and student behaviors"""
        if not (self.sim_behavior_buffer and self.real_behavior_buffer):
            return 0.0
        
        # Get recent behaviors
        recent_sim = np.array([item['actions'] for item in self.sim_behavior_buffer[-5:]]).flatten()
        recent_real = np.array([item['actions'] for item in self.real_behavior_buffer[-5:]]).flatten()
        
        if len(recent_sim) == 0 or len(recent_real) == 0:
            return 0.0
        
        # Pad shorter array if needed
        max_len = max(len(recent_sim), len(recent_real))
        sim_padded = np.pad(recent_sim, (0, max_len - len(recent_sim)))
        real_padded = np.pad(recent_real, (0, max_len - len(recent_real)))
        
        # Calculate cosine similarity
        dot_product = np.dot(sim_padded, real_padded)
        norms = np.linalg.norm(sim_padded) * np.linalg.norm(real_padded)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        return float(similarity)

    def generate_distilled_actions(self):
        """Generate distilled actions combining teacher and student knowledge"""
        if not (self.sim_behavior_buffer and self.real_behavior_buffer):
            distilled_actions = Float64MultiArray()
            distilled_actions.data = [0.0] * 6  # Default 6 actions
            return distilled_actions
        
        # Get most recent actions
        last_teacher = self.sim_behavior_buffer[-1]['actions']
        last_student = self.real_behavior_buffer[-1]['actions']
        
        # Combine with learned weighting
        # In distillation, we'd typically just guide the student's learning
        # For this example, we'll blend the outputs
        min_len = min(len(last_teacher), len(last_student))
        combined_actions = [
            0.3 * last_teacher[i] + 0.7 * last_student[i]  # Favor student but guided by teacher
            for i in range(min_len)
        ]
        
        # Pad if necessary
        if len(combined_actions) < 6:
            combined_actions.extend([0.0] * (6 - len(combined_actions)))
        
        distilled_actions = Float64MultiArray()
        distilled_actions.data = combined_actions[:6]
        
        return distilled_actions

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Knowledge Distiller Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    distiller = KnowledgeDistillerNode()
    
    try:
        rclpy.spin(distiller)
    except KeyboardInterrupt:
        distiller.get_logger().info('Node interrupted by user')
    finally:
        distiller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Validation and Testing of Sim-to-Real Transfer

### 1. Create a Transfer Validation Node

Create a comprehensive validation node to test the effectiveness of transfer techniques:

```python
# scripts/transfer_validator.py
#!/usr/bin/env python3

"""
Transfer Validation Node for evaluating sim-to-real transfer effectiveness.
Compares behavior in simulation vs. adapted behavior in reality.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    success_rate: float
    stability_score: float
    task_completion_time: float
    energy_efficiency: float
    accuracy_score: float
    transfer_gap: float  # Difference between sim and real performance


class TransferValidatorNode(Node):
    """
    Evaluates the effectiveness of sim-to-real transfer techniques
    by comparing performance metrics between simulation and reality.
    """
    
    def __init__(self):
        super().__init__('transfer_validator_node')
        
        # Declare parameters
        self.declare_parameter('validation_frequency', 1.0)
        self.declare_parameter('success_threshold', 0.8)
        self.declare_parameter('stability_threshold', 0.7)
        self.declare_parameter('results_output_file', 'transfer_validation_results.csv')
        
        # Get parameters
        self.validation_frequency = self.get_parameter('validation_frequency').value
        self.success_threshold = self.get_parameter('success_threshold').value
        self.stability_threshold = self.get_parameter('stability_threshold').value
        self.results_output_file = self.get_parameter('results_output_file').value
        
        # Data storage for validation metrics
        self.sim_performance_history = []
        self.real_performance_history = []
        self.transfer_metrics_history = []
        
        # Subscriptions
        self.sim_performance_sub = self.create_subscription(
            Float64MultiArray,
            '/simulation_performance_metrics',
            self.sim_performance_callback,
            10
        )
        
        self.real_performance_sub = self.create_subscription(
            Float64MultiArray,
            '/reality_performance_metrics',
            self.real_performance_callback,
            10
        )
        
        self.transfer_metrics_sub = self.create_subscription(
            Float64MultiArray,
            '/transfer_effectiveness',
            self.transfer_metrics_callback,
            10
        )
        
        # Publishers
        self.validation_report_pub = self.create_publisher(
            String,
            '/transfer_validation_report',
            10
        )
        
        self.effectiveness_score_pub = self.create_publisher(
            Float64MultiArray,
            '/transfer_effectiveness_score',
            10
        )
        
        # Timer for validation
        self.validation_timer = self.create_timer(
            1.0 / self.validation_frequency,
            self.run_validation_cycle
        )
        
        self.get_logger().info('Transfer Validator Node Initialized')

    def sim_performance_callback(self, msg):
        """Store simulation performance metrics"""
        if len(self.sim_performance_history) >= 1000:
            self.sim_performance_history.pop(0)
        
        metrics = {
            'timestamp': self.get_clock().now(),
            'success_rate': msg.data[0] if len(msg.data) > 0 else 0.0,
            'stability': msg.data[1] if len(msg.data) > 1 else 0.0,
            'time_efficiency': msg.data[2] if len(msg.data) > 2 else 0.0,
            'energy_efficiency': msg.data[3] if len(msg.data) > 3 else 0.0,
            'accuracy': msg.data[4] if len(msg.data) > 4 else 0.0
        }
        
        self.sim_performance_history.append(metrics)

    def real_performance_callback(self, msg):
        """Store real robot performance metrics"""
        if len(self.real_performance_history) >= 1000:
            self.real_performance_history.pop(0)
        
        metrics = {
            'timestamp': self.get_clock().now(),
            'success_rate': msg.data[0] if len(msg.data) > 0 else 0.0,
            'stability': msg.data[1] if len(msg.data) > 1 else 0.0,
            'time_efficiency': msg.data[2] if len(msg.data) > 2 else 0.0,
            'energy_efficiency': msg.data[3] if len(msg.data) > 3 else 0.0,
            'accuracy': msg.data[4] if len(msg.data) > 4 else 0.0
        }
        
        self.real_performance_history.append(metrics)

    def transfer_metrics_callback(self, msg):
        """Store transfer effectiveness metrics"""
        if len(self.transfer_metrics_history) >= 1000:
            self.transfer_metrics_history.pop(0)
        
        metrics = {
            'timestamp': self.get_clock().now(),
            'transfer_gap': msg.data[0] if len(msg.data) > 0 else 0.0,
            'adaptation_success': msg.data[1] if len(msg.data) > 1 else 0.0,
            'domain_similarity': msg.data[2] if len(msg.data) > 2 else 0.0
        }
        
        self.transfer_metrics_history.append(metrics)

    def run_validation_cycle(self):
        """Run the transfer validation cycle"""
        if not (self.sim_performance_history and self.real_performance_history):
            report_msg = String()
            report_msg.data = "INSUFFICIENT_DATA_FOR_VALIDATION"
            self.validation_report_pub.publish(report_msg)
            return
        
        try:
            # Calculate comprehensive validation metrics
            validation_metrics = self.calculate_validation_metrics()
            
            # Generate validation report
            report = self.generate_validation_report(validation_metrics)
            
            # Publish report
            report_msg = String()
            report_msg.data = report
            self.validation_report_pub.publish(report_msg)
            
            # Publish effectiveness score
            score_msg = Float64MultiArray()
            score_msg.data = [
                validation_metrics.success_rate,
                validation_metrics.stability_score,
                validation_metrics.transfer_gap,
                validation_metrics.accuracy_score
            ]
            self.effectiveness_score_pub.publish(score_msg)
            
            # Log validation results
            self.get_logger().info(f'Validation Results:\n{report}')
            
            # Save results periodically
            if int(self.get_clock().now().nanoseconds / 1e9) % 30 == 0:  # Every 30 seconds
                self.save_validation_results(validation_metrics)
                
        except Exception as e:
            self.get_logger().error(f'Error in validation cycle: {str(e)}')

    def calculate_validation_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive validation metrics"""
        if not (self.sim_performance_history and self.real_performance_history):
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        
        # Get recent metrics (last 10 samples) to reduce noise
        recent_sim = self.sim_performance_history[-10:]
        recent_real = self.real_performance_history[-10:]
        
        # Calculate averages
        sim_success = np.mean([m['success_rate'] for m in recent_sim])
        real_success = np.mean([m['success_rate'] for m in recent_real])
        
        sim_stability = np.mean([m['stability'] for m in recent_sim])
        real_stability = np.mean([m['stability'] for m in recent_real])
        
        sim_time_efficiency = np.mean([m['time_efficiency'] for m in recent_sim])
        real_time_efficiency = np.mean([m['time_efficiency'] for m in recent_real])
        
        sim_energy_efficiency = np.mean([m['energy_efficiency'] for m in recent_sim])
        real_energy_efficiency = np.mean([m['energy_efficiency'] for m in recent_real])
        
        sim_accuracy = np.mean([m['accuracy'] for m in recent_sim])
        real_accuracy = np.mean([m['accuracy'] for m in recent_real])
        
        # Calculate transfer gap (difference between sim and real performance)
        success_gap = abs(sim_success - real_success)
        stability_gap = abs(sim_stability - real_stability)
        time_gap = abs(sim_time_efficiency - real_time_efficiency)
        energy_gap = abs(sim_energy_efficiency - real_energy_efficiency)
        accuracy_gap = abs(sim_accuracy - real_accuracy)
        
        avg_transfer_gap = np.mean([success_gap, stability_gap, time_gap, energy_gap, accuracy_gap])
        
        # Calculate overall effectiveness
        effectiveness = 1.0 - avg_transfer_gap  # Higher is better
        
        return PerformanceMetrics(
            success_rate=min(real_success, sim_success),  # Use lower of sim/real
            stability_score=real_stability,
            task_completion_time=real_time_efficiency,
            energy_efficiency=real_energy_efficiency,
            accuracy_score=real_accuracy,
            transfer_gap=avg_transfer_gap  # Lower is better
        )

    def generate_validation_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a comprehensive validation report"""
        report_parts = []
        report_parts.append("=== SIM-TO-REAL TRANSFER VALIDATION REPORT ===")
        report_parts.append(f"Success Rate: {metrics.success_rate:.3f}")
        report_parts.append(f"Stability Score: {metrics.stability_score:.3f}")
        report_parts.append(f"Task Completion Time: {metrics.task_completion_time:.3f}s")
        report_parts.append(f"Energy Efficiency: {metrics.energy_efficiency:.3f}")
        report_parts.append(f"Accuracy Score: {metrics.accuracy_score:.3f}")
        report_parts.append(f"Transfer Gap: {metrics.transfer_gap:.3f}")
        
        # Assess transfer effectiveness
        if metrics.transfer_gap < 0.2:
            effectiveness = "EXCELLENT"
        elif metrics.transfer_gap < 0.4:
            effectiveness = "GOOD"
        elif metrics.transfer_gap < 0.6:
            effectiveness = "FAIR"
        else:
            effectiveness = "POOR"
        
        report_parts.append(f"Transfer Effectiveness: {effectiveness}")
        
        # Provide recommendations
        if metrics.stability_score < self.stability_threshold:
            report_parts.append("RECOMMENDATION: Improve robot stability through better control algorithms")
        
        if metrics.transfer_gap > 0.5:
            report_parts.append("RECOMMENDATION: Consider additional domain randomization or system identification")
        
        if metrics.success_rate < self.success_threshold:
            report_parts.append("RECOMMENDATION: Enhance perception or planning algorithms")
        
        return "\n".join(report_parts)

    def save_validation_results(self, metrics: PerformanceMetrics):
        """Save validation results to file"""
        import csv
        import os
        
        # Prepare results for CSV
        results_row = [
            self.get_clock().now().nanoseconds / 1e9,  # timestamp
            metrics.success_rate,
            metrics.stability_score,
            metrics.task_completion_time,
            metrics.energy_efficiency,
            metrics.accuracy_score,
            metrics.transfer_gap
        ]
        
        # Write to CSV file
        file_exists = os.path.isfile(self.results_output_file)
        
        with open(self.results_output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write header if file doesn't exist
                writer.writerow([
                    'timestamp', 'success_rate', 'stability_score', 
                    'time_efficiency', 'energy_efficiency', 
                    'accuracy_score', 'transfer_gap'
                ])
            writer.writerow(results_row)

    def create_validation_visualization(self):
        """Create visualization of validation metrics"""
        if not (self.sim_performance_history and self.real_performance_history):
            return
        
        # Prepare data for plotting
        sim_success = [m['success_rate'] for m in self.sim_performance_history]
        real_success = [m['success_rate'] for m in self.real_performance_history]
        
        # Create plots
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot success rate comparison
        ax[0, 0].plot(sim_success, label='Simulation', alpha=0.7)
        ax[0, 0].plot(real_success, label='Reality', alpha=0.7)
        ax[0, 0].set_title('Success Rate Comparison')
        ax[0, 0].set_xlabel('Time Steps')
        ax[0, 0].set_ylabel('Success Rate')
        ax[0, 0].legend()
        
        # Additional plots would go here...
        
        plt.tight_layout()
        plt.savefig('/tmp/transfer_validation_plot.png')
        self.get_logger().info('Validation plot saved to /tmp/transfer_validation_plot.png')

    def destroy_node(self):
        """Cleanup before node destruction"""
        # Create final validation report
        if self.sim_performance_history and self.real_performance_history:
            final_metrics = self.calculate_validation_metrics()
            final_report = self.generate_validation_report(final_metrics)
            self.get_logger().info(f'FINAL VALIDATION REPORT:\n{final_report}')
        
        self.get_logger().info('Transfer Validator Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    validator = TransferValidatorNode()
    
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

## Implementation Strategy Summary

### 1. Complete Implementation Plan

To implement effective sim-to-real transfer for your humanoid robot:

1. **Domain Randomization**: Train in varied simulation conditions
2. **System Identification**: Calibrate simulation to match reality
3. **Robust Control**: Design controllers that handle domain variations
4. **Domain Adaptation**: Adjust models to better match real data
5. **Transfer Learning**: Use knowledge distillation techniques
6. **Validation**: Continuously validate transfer effectiveness

### 2. Best Practices for Sim-to-Real Transfer

- **Start Simple**: Begin with basic behaviors and gradually increase complexity
- **Iterative Approach**: Use cycles of simulation → real testing → parameter adjustment → simulation
- **Multiple Sensors**: Combine multiple sensor modalities to increase robustness
- **Model Verification**: Continuously verify that your simulation model represents reality well
- **Safety First**: Always implement safety measures when transitioning to real hardware

## Next Steps

With comprehensive sim-to-real transfer techniques implemented, you're now ready to explore Jetson deployment workflows. The techniques you've learned will be essential for ensuring that your AI models trained in simulation work effectively on the physical humanoid robot hardware.

The validation and transfer techniques form the bridge between the safe development environment of simulation and the complex real world.