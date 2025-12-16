# Implementing Sim-to-Real Transfer Strategies for Humanoid Robotics

## Overview

This section provides practical implementation of sim-to-real transfer strategies for humanoid robots. We'll implement the theoretical concepts from the previous section into working code that demonstrates how to bridge the gap between simulation and reality for humanoid robotics applications.

## Implementation Prerequisites

Before implementing sim-to-real transfer strategies, ensure you have:
- Completed the perception pipeline implementation
- Set up Gazebo with Isaac ROS sensors
- Configured domain randomization settings
- Established proper system identification procedures

## Part 1: Domain Randomization Implementation

### Step 1: Creating a Domain Randomizer Node

Create a comprehensive domain randomizer that adjusts simulation parameters during training:

```python
# scripts/domain_randomizer_impl.py
#!/usr/bin/env python3

"""
Implementation of domain randomization techniques for sim-to-real transfer.
This node dynamically changes simulation parameters to improve robustness.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState, LaserScan, Imu
from geometry_msgs.msg import Vector3
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from gazebo_msgs.msg import ModelState
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
import numpy as np
import random
import time
from threading import Lock


class DomainRandomizationImpl(Node):
    """
    Implementation of domain randomization techniques to improve sim-to-real transfer.
    Dynamically adjusts physics parameters at runtime to make controllers more robust.
    """
    
    def __init__(self):
        super().__init__('domain_randomization_impl')
        
        # Declare parameters with ranges
        self.declare_parameter('randomization_frequency', 10.0)  # Hz
        self.declare_parameter('physics_variance', 0.15)       # 15% variance
        self.declare_parameter('sensor_noise_variance', 0.25)  # 25% variance 
        self.declare_parameter('actuator_variance', 0.20)      # 20% variance
        self.declare_parameter('randomization_enabled', True)
        
        # Get parameters
        self.randomization_frequency = self.get_parameter('randomization_frequency').value
        self.physics_variance = self.get_parameter('physics_variance').value
        self.sensor_noise_variance = self.get_parameter('sensor_noise_variance').value
        self.actuator_variance = self.get_parameter('actuator_variance').value
        self.randomization_enabled = self.get_parameter('randomization_enabled').value
        
        # Store original simulation parameters
        self.original_physics_params = {}
        self.original_joint_params = {}
        self.randomization_lock = Lock()
        
        # Gazebo service clients
        self.get_physics_client = self.create_client(
            GetPhysicsProperties, 
            '/gazebo/get_physics_properties'
        )
        self.set_physics_client = self.create_client(
            SetPhysicsProperties, 
            '/gazebo/set_physics_properties'
        )
        
        # Wait for Gazebo services
        while not self.get_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo physics services...')
        
        while not self.set_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo physics services...')
        
        # Store original physics parameters
        self.store_original_physics()
        
        # Publishers
        self.randomization_status_pub = self.create_publisher(
            String,
            '/domain_randomization_status',
            10
        )
        
        self.randomized_params_pub = self.create_publisher(
            Float64MultiArray,
            '/domain_randomized_parameters',
            10
        )
        
        # Timer for randomization
        self.randomization_timer = self.create_timer(
            1.0 / self.randomization_frequency,
            self.apply_domain_randomization
        )
        
        # Track current randomization values
        self.current_randomized_params = {}
        
        self.get_logger().info(
            f'Domain Randomization Implementation: Frequency={self.randomization_frequency}Hz, '
            f'Physics Variance={self.physics_variance}, '
            f'Sensor Noise Variance={self.sensor_noise_variance}, '
            f'Actuator Variance={self.actuator_variance}'
        )

    def store_original_physics(self):
        """Store original physics parameters for reference"""
        try:
            request = GetPhysicsProperties.Request()
            future = self.get_physics_client.call_async(request)
            
            # Wait for response with timeout
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            response = future.result()
            
            if response:
                self.original_physics_params = {
                    'max_step_size': response.max_step_size,
                    'real_time_factor': response.real_time_factor,
                    'real_time_update_rate': response.real_time_update_rate,
                    'gravity': {
                        'x': response.gravity.x,
                        'y': response.gravity.y,
                        'z': response.gravity.z
                    },
                    'ode_config': response.ode_config
                }
                
                self.get_logger().info('Stored original physics parameters')
            else:
                self.get_logger().error('Failed to get original physics parameters')
                
        except Exception as e:
            self.get_logger().error(f'Error storing original physics: {str(e)}')

    def apply_domain_randomization(self):
        """Apply domain randomization to simulation parameters"""
        if not self.randomization_enabled:
            return
        
        try:
            # Randomize physics parameters
            randomized_physics = self.randomize_physics_parameters()
            
            # Apply randomized physics to Gazebo
            self.apply_physics_to_gazebo(randomized_physics)
            
            # Randomize sensor noise parameters
            self.randomize_sensor_parameters()
            
            # Randomize actuator parameters
            self.randomize_actuator_parameters()
            
            # Publish randomization status
            self.publish_randomization_status(randomized_physics)
            
            # Log randomization
            self.log_randomization(randomized_physics)
            
        except Exception as e:
            self.get_logger().error(f'Error applying domain randomization: {str(e)}')

    def randomize_physics_parameters(self):
        """Generate randomized physics parameters"""
        if not self.original_physics_params:
            return None
        
        randomized = {}
        
        # Randomize max step size (affects integration accuracy)
        step_range = self.physics_variance * self.original_physics_params['max_step_size']
        randomized['max_step_size'] = max(
            0.0005,  # Minimum safe step size
            self.original_physics_params['max_step_size'] + 
            random.uniform(-step_range, step_range)
        )
        
        # Randomize real time factor (affects simulation speed)
        rt_range = self.physics_variance * self.original_physics_params['real_time_factor']
        randomized['real_time_factor'] = max(
            0.1,  # Minimum RTF
            min(5.0, self.original_physics_params['real_time_factor'] + 
                random.uniform(-rt_range, rt_range))
        )
        
        # Randomize gravity (to account for different locations/real-world variations)
        gravity_base = self.original_physics_params['gravity']
        g_range = self.physics_variance * 9.81
        randomized['gravity'] = {
            'x': random.uniform(gravity_base['x'] - 0.05, gravity_base['x'] + 0.05),  # Small x/y variations
            'y': random.uniform(gravity_base['y'] - 0.05, gravity_base['y'] + 0.05),
            'z': random.uniform(gravity_base['z'] - g_range, gravity_base['z'] + g_range)  # Z can vary more
        }
        
        # Randomize ODE solver parameters
        ode_config = self.original_physics_params['ode_config']
        randomized['ode_config'] = ode_config
        
        # Randomize solver iterations
        iter_range = int(self.physics_variance * ode_config.solver_iterations)
        randomized['ode_config'].solver_iterations = max(
            10,  # Minimum iterations
            ode_config.solver_iterations + random.randint(-iter_range, iter_range)
        )
        
        # Randomize solver sor parameter
        sor_range = self.physics_variance * ode_config.solver_sor
        randomized['ode_config'].solver_sor = max(
            1.0,  # Minimum SOR
            ode_config.solver_sor + random.uniform(-sor_range, sor_range)
        )
        
        return randomized

    def apply_physics_to_gazebo(self, params):
        """Apply randomized physics parameters to Gazebo"""
        if not params:
            return
        
        try:
            request = SetPhysicsProperties.Request()
            
            # Set physics properties
            request.time_step = params['max_step_size']
            request.max_update_rate = params['real_time_update_rate']
            
            # Set gravity
            gravity_vector = Vector3()
            gravity_vector.x = params['gravity']['x']
            gravity_vector.y = params['gravity']['y'] 
            gravity_vector.z = params['gravity']['z']
            request.gravity = gravity_vector
            
            # Set ODE config
            request.ode_config = params['ode_config']
            
            # Call service
            future = self.set_physics_client.call_async(request)
            
            # Wait for response
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            
            if response and response.success:
                self.get_logger().debug(
                    f'Physics updated: step={params["max_step_size"]:.4f}, '
                    f'RTF={params["real_time_factor"]:.2f}, '
                    f'gravity=({params["gravity"]["x"]:.3f}, {params["gravity"]["y"]:.3f}, {params["gravity"]["z"]:.3f})'
                )
            else:
                self.get_logger().error(f'Failed to update physics: {response.status_message if response else "No response"}')
                
        except Exception as e:
            self.get_logger().error(f'Error applying physics changes: {str(e)}')

    def randomize_sensor_parameters(self):
        """Randomize sensor parameters like noise characteristics"""
        # In a real implementation, this would dynamically adjust sensor plugins
        # which requires custom services or parameter updates
        
        # For demonstration, we'll log what would be changed
        sensor_noise_factors = {
            'lidar_noise_factor': 1.0 + random.uniform(-self.sensor_noise_variance, self.sensor_noise_variance),
            'camera_noise_factor': 1.0 + random.uniform(-self.sensor_noise_variance/2, self.sensor_noise_variance/2),  # Cameras less noisy
            'imu_noise_factor': 1.0 + random.uniform(-self.sensor_noise_variance, self.sensor_noise_variance)
        }
        
        self.current_randomized_params['sensor_noise'] = sensor_noise_factors

    def randomize_actuator_parameters(self):
        """Randomize actuator dynamics"""
        actuator_changes = {
            'friction_factor': 1.0 + random.uniform(-self.actuator_variance, self.actuator_variance),
            'damping_factor': 1.0 + random.uniform(-self.actuator_variance, self.actuator_variance),
            'effort_limit_factor': 1.0 + random.uniform(-self.actuator_variance/3, self.actuator_variance/3),  # Less variation in limits
            'velocity_limit_factor': 1.0 + random.uniform(-self.actuator_variance/3, self.actuator_variance/3)
        }
        
        self.current_randomized_params['actuators'] = actuator_changes

    def publish_randomization_status(self, params):
        """Publish the current randomization status"""
        if params:
            status_msg = String()
            status_msg.data = (
                f"RANDOMIZED: Step={params['max_step_size']:.4f}, "
                f"RTF={params['real_time_factor']:.2f}, "
                f"GravityZ={params['gravity']['z']:.3f}"
            )
            self.randomization_status_pub.publish(status_msg)

        # Publish numeric parameters
        params_msg = Float64MultiArray()
        params_msg.data = [
            params.get('max_step_size', 0.001) if params else 0.0,
            params.get('real_time_factor', 1.0) if params else 1.0,
            params['gravity']['z'] if params and 'gravity' in params else -9.81,
            self.current_randomized_params.get('sensor_noise', {}).get('lidar_noise_factor', 1.0),
            self.current_randomized_params.get('actuators', {}).get('friction_factor', 1.0)
        ]
        self.randomized_params_pub.publish(params_msg)

    def log_randomization(self, params):
        """Log randomization parameters"""
        if not params:
            return
        
        # Log periodically to avoid spam
        current_time = self.get_clock().now().nanoseconds / 1e9
        if int(current_time) % 10 == 0:  # Log every 10 seconds
            self.get_logger().info(
                f'Domain Randomization Applied - '
                f'Step: {params["max_step_size"]:.4f}s, '
                f'RTF: {params["real_time_factor"]:.2f}, '
                f'Gravity Z: {params["gravity"]["z"]:.3f} m/s²'
            )

    def toggle_randomization(self, enable):
        """Enable or disable domain randomization"""
        self.randomization_enabled = enable
        self.get_logger().info(f'Domain randomization { "ENABLED" if enable else "DISABLED" }')

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Domain Randomization Implementation Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    randomizer = DomainRandomizationImpl()
    
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

## Part 2: System Identification Implementation

### Step 2.1: Creating a System Identification Node

Create a system identification implementation to calibrate simulation parameters:

```python
# scripts/system_identification_impl.py
#!/usr/bin/env python3

"""
Implementation of system identification techniques for sim-to-real transfer.
Calibrates simulation parameters based on real robot behavior data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Float64MultiArray
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from gazebo_msgs.msg import ModelState
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import time
from threading import Lock


class SystemIdentificationImpl(Node):
    """
    Implementation of system identification for humanoid robots.
    Uses real robot data to calibrate simulation parameters.
    """
    
    def __init__(self):
        super().__init__('system_identification_impl')
        
        # Declare parameters
        self.declare_parameter('identification_frequency', 0.1)  # Hz (lower for performance)
        self.declare_parameter('data_window_size', 1000)        # Size of data buffer
        self.declare_parameter('calibration_tolerance', 0.01)   # Tolerance for convergence
        self.declare_parameter('max_calibration_iterations', 50)  # Max optimization iterations
        self.declare_parameter('calibration_enabled', False)    # Only enable during calibration phase
        
        # Get parameters
        self.identification_frequency = self.get_parameter('identification_frequency').value
        self.data_window_size = self.get_parameter('data_window_size').value
        self.calibration_tolerance = self.get_parameter('calibration_tolerance').value
        self.max_calibration_iterations = self.get_parameter('max_calibration_iterations').value
        self.calibration_enabled = self.get_parameter('calibration_enabled').value
        
        # Data buffers for system identification
        self.sim_data_buffer = []  # Simulation data
        self.real_data_buffer = []  # Real robot data (or reference data)
        self.param_history = []     # Track parameter evolution
        
        # Parameters to calibrate
        self.calibration_targets = {
            'joint_damping': {
                'initial': 0.5,
                'bounds': (0.01, 5.0),
                'current': 0.5,
                'affects': ['all_joints']
            },
            'friction_coefficient': {
                'initial': 0.8,
                'bounds': (0.1, 2.0),
                'current': 0.8,
                'affects': ['contact_models']
            },
            'gravity_factor': {
                'initial': 1.0,
                'bounds': (0.8, 1.2),
                'current': 1.0,
                'affects': ['global_physics']
            },
            'imu_drift_compensation': {
                'initial': 0.0,
                'bounds': (-0.1, 0.1),
                'current': 0.0,
                'affects': ['orientation_estimates']
            }
        }
        
        # Gazebo service clients
        self.get_physics_client = self.create_client(
            GetPhysicsProperties,
            '/gazebo/get_physics_properties'
        )
        self.set_physics_client = self.create_client(
            SetPhysicsProperties,
            '/gazebo/set_physics_properties'
        )
        
        # Wait for Gazebo services
        while not self.get_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo physics services...')
        
        while not self.set_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Gazebo physics services...')
        
        # Subscriptions
        self.sim_joint_state_sub = self.create_subscription(
            JointState,
            '/sim/joint_states',  # Would come from simulation
            self.sim_joint_state_callback,
            10
        )
        
        self.real_joint_state_sub = self.create_subscription(
            JointState,
            '/real/joint_states',  # Would come from real robot or recorded data
            self.real_joint_state_callback,
            10
        )
        
        self.sim_imu_sub = self.create_subscription(
            Imu,
            '/sim/imu',
            self.sim_imu_callback,
            10
        )
        
        self.real_imu_sub = self.create_subscription(
            Imu,
            '/real/imu',
            self.real_imu_callback,
            10
        )
        
        # Publishers
        self.calibration_status_pub = self.create_publisher(
            String,
            '/calibration_status',
            10
        )
        
        self.calibrated_params_pub = self.create_publisher(
            Float64MultiArray,
            '/calibrated_simulation_parameters',
            10
        )
        
        self.system_id_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/system_identification_metrics',
            10
        )
        
        # Timer for identification process
        self.identification_timer = self.create_timer(
            1.0 / self.identification_frequency,
            self.run_system_identification
        )
        
        # Storage for current robot states
        self.current_sim_joint_state = None
        self.current_real_joint_state = None
        self.current_sim_imu = None
        self.current_real_imu = None
        
        self.get_logger().info('System Identification Implementation Node Initialized')

    def sim_joint_state_callback(self, msg):
        """Store simulation joint state data"""
        if len(self.sim_data_buffer) >= self.data_window_size:
            self.sim_data_buffer.pop(0)
        
        self.sim_data_buffer.append({
            'timestamp': msg.header.stamp,
            'type': 'joint_state',
            'data': msg,
            'source': 'simulation'
        })
        
        self.current_sim_joint_state = msg

    def real_joint_state_callback(self, msg):
        """Store real robot joint state data"""
        if len(self.real_data_buffer) >= self.data_window_size:
            self.real_data_buffer.pop(0)
        
        self.real_data_buffer.append({
            'timestamp': msg.header.stamp,
            'type': 'joint_state',
            'data': msg,
            'source': 'real'
        })
        
        self.current_real_joint_state = msg

    def sim_imu_callback(self, msg):
        """Store simulation IMU data"""
        self.sim_data_buffer.append({
            'timestamp': msg.header.stamp,
            'type': 'imu',
            'data': msg,
            'source': 'simulation'
        })
        
        # Keep buffer size reasonable
        if len(self.sim_data_buffer) > self.data_window_size:
            # Keep only the most recent data
            self.sim_data_buffer = self.sim_data_buffer[-self.data_window_size:]

    def real_imu_callback(self, msg):
        """Store real robot IMU data"""
        self.real_data_buffer.append({
            'timestamp': msg.header.stamp,
            'type': 'imu',
            'data': msg,
            'source': 'real'
        })
        
        # Keep buffer size reasonable
        if len(self.real_data_buffer) > self.data_window_size:
            # Keep only the most recent data
            self.real_data_buffer = self.real_data_buffer[-self.data_window_size:]

    def run_system_identification(self):
        """Run system identification cycle"""
        if not (self.calibration_enabled and len(self.sim_data_buffer) > 10 and len(self.real_data_buffer) > 10):
            # Publish idle status
            status_msg = String()
            status_msg.data = f"SYSTEM_ID_IDLE: Sim:{len(self.sim_data_buffer)}, Real:{len(self.real_data_buffer)}"
            self.calibration_status_pub.publish(status_msg)
            return
        
        try:
            # Prepare data for identification
            sim_data = self.extract_matching_data('simulation', min(len(self.sim_data_buffer), len(self.real_data_buffer)))
            real_data = self.extract_matching_data('real', min(len(self.sim_data_buffer), len(self.real_data_buffer)))
            
            if len(sim_data) < 10 or len(real_data) < 10:
                self.get_logger().debug('Insufficient matching data for system identification')
                return
            
            # Extract current parameters
            current_params = [self.calibration_targets[key]['current'] for key in self.calibration_targets.keys()]
            
            # Optimize parameters to minimize sim-real mismatch
            result = minimize(
                fun=self.sim_real_distance,
                x0=current_params,
                args=(sim_data, real_data),
                method='L-BFGS-B',
                bounds=[self.calibration_targets[key]['bounds'] for key in self.calibration_targets.keys()],
                options={'maxiter': self.max_calibration_iterations}
            )
            
            # Update calibration targets with optimized values
            param_keys = list(self.calibration_targets.keys())
            for i, key in enumerate(param_keys):
                self.calibration_targets[key]['current'] = result.x[i]
            
            # Apply optimized parameters to simulation
            self.apply_calibration_parameters(result.x)
            
            # Publish results
            self.publish_calibration_results(result.x, result.success)
            
            # Log progress
            if int(self.get_clock().now().nanoseconds / 1e9) % 30 == 0:  # Every 30 seconds
                self.get_logger().info(
                    f'System Identification Result: Success={result.success}, '
                    f'Fun={result.fun:.4f}, Iter={result.nit}'
                )
        
        except Exception as e:
            self.get_logger().error(f'Error in system identification: {str(e)}')

    def sim_real_distance(self, params, sim_data, real_data):
        """
        Calculate the distance between simulation and real robot behavior.
        This is the objective function that optimization tries to minimize.
        """
        # Apply parameters to simulation temporarily
        self.apply_temporary_parameters(params)
        
        # Calculate distance metric between sim and real data
        distance = 0.0
        
        # Compare joint positions over time
        for sim_entry, real_entry in zip(sim_data, real_data):
            if sim_entry['type'] == 'joint_state' and real_entry['type'] == 'joint_state':
                sim_joints = sim_entry['data']
                real_joints = real_entry['data']
                
                # Calculate joint position differences
                for i in range(min(len(sim_joints.position), len(real_joints.position))):
                    diff = abs(sim_joints.position[i] - real_joints.position[i])
                    distance += diff**2  # Square for L2 norm
        
        # Compare IMU orientations
        sim_imu_data = [d for d in sim_data if d['type'] == 'imu']
        real_imu_data = [d for d in real_data if d['type'] == 'imu']
        
        for sim_imu, real_imu in zip(sim_imu_data, real_imu_data):
            # Calculate orientation difference using quaternion distance
            q_sim = sim_imu['data'].orientation
            q_real = real_imu['data'].orientation
            
            # Quaternion dot product (measure of similarity)
            dot_prod = q_sim.x*q_real.x + q_sim.y*q_real.y + q_sim.z*q_real.z + q_sim.w*q_real.w
            # Convert to angle difference
            angle_diff = 2 * np.arccos(min(abs(dot_prod), 1.0))  # Clamp to avoid numerical issues
            distance += angle_diff**2
        
        # Add regularization to prevent extreme parameter values
        regularization = 0.01 * sum((p-c)**2 for p, c in zip(params, [v['current'] for v in self.calibration_targets.values()]))
        total_distance = distance + regularization
        
        return total_distance

    def apply_temporary_parameters(self, params):
        """Apply parameters temporarily for evaluation during optimization"""
        # In a real implementation, this would modify physics parameters in simulation
        # without permanently changing them until optimization completes
        param_names = list(self.calibration_targets.keys())
        
        for i, name in enumerate(param_names):
            if i < len(params):
                self.calibration_targets[name]['current'] = params[i]
        
        # Apply the parameters to Gazebo physics
        self.apply_current_parameters_to_simulation()

    def apply_current_parameters_to_simulation(self):
        """Apply current calibration parameters to the simulation"""
        try:
            request = SetPhysicsProperties.Request()
            
            # Get current physics parameters
            current_physics_future = self.get_physics_client.call_async(GetPhysicsProperties.Request())
            rclpy.spin_until_future_complete(self, current_physics_future, timeout_sec=2.0)
            current_physics = current_physics_future.result()
            
            if not current_physics:
                self.get_logger().error('Could not get current physics parameters for calibration')
                return
            
            # Apply gravity factor
            gravity_factor = self.calibration_targets['gravity_factor']['current']
            request.gravity.x = current_physics.gravity.x * gravity_factor
            request.gravity.y = current_physics.gravity.y * gravity_factor
            request.gravity.z = current_physics.gravity.z * gravity_factor
            
            # Apply other parameters to physics settings
            # This is a simplified implementation - in reality, you'd need services to modify
            # joint damping, friction, etc. which may require custom Gazebo plugins
            request.time_step = current_physics.max_step_size
            request.max_update_rate = current_physics.real_time_update_rate
            request.ode_config = current_physics.ode_config
            
            # Call service to update physics
            future = self.set_physics_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            response = future.result()
            
            if response and response.success:
                self.get_logger().debug('Applied calibrated parameters to simulation')
            else:
                self.get_logger().error(f'Failed to apply calibrated parameters: {response.status_message if response else "No response"}')
                
        except Exception as e:
            self.get_logger().error(f'Error applying calibration: {str(e)}')

    def apply_calibration_parameters(self, params):
        """Apply final calibrated parameters to simulation"""
        param_names = list(self.calibration_targets.keys())
        
        for i, name in enumerate(param_names):
            if i < len(params):
                self.calibration_targets[name]['current'] = params[i]
        
        # Apply to simulation
        self.apply_current_parameters_to_simulation()
        
        # Log the changes
        self.param_history.append({
            'timestamp': self.get_clock().now(),
            'parameters': dict(zip(param_names, params))
        })

    def extract_matching_data(self, source, count):
        """Extract data with specified source, up to count items"""
        all_data = self.sim_data_buffer if source == 'simulation' else self.real_data_buffer
        matching = [d for d in all_data if d['source'] == source]
        return matching[-count:]

    def publish_calibration_results(self, params, success):
        """Publish calibration results"""
        # Publish status
        status_msg = String()
        if success:
            param_names = list(self.calibration_targets.keys())
            param_str = ", ".join([f"{name}:{val:.3f}" for name, val in zip(param_names, params)])
            status_msg.data = f"CALIBRATION_SUCCESS: {param_str}"
        else:
            status_msg.data = "CALIBRATION_FAILED"
        
        self.calibration_status_pub.publish(status_msg)
        
        # Publish numeric results
        results_msg = Float64MultiArray()
        results_msg.data = list(params) + [1.0 if success else 0.0]  # Include success flag
        self.calibrated_params_pub.publish(results_msg)
        
        # Publish system identification metrics
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            len(self.sim_data_buffer),  # Sim data count
            len(self.real_data_buffer), # Real data count
            len(self.param_history),    # Calibration attempts
            self.get_clock().now().nanoseconds / 1e9  # Timestamp
        ]
        self.system_id_metrics_pub.publish(metrics_msg)

    def start_calibration_process(self):
        """Start the calibration process"""
        self.calibration_enabled = True
        self.get_logger().info('Starting system identification calibration process')

    def stop_calibration_process(self):
        """Stop the calibration process"""
        self.calibration_enabled = False
        self.get_logger().info('Stopped system identification calibration process')

    def save_calibration_results(self, filename=None):
        """Save calibration results to file"""
        if not filename:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'/tmp/calibration_results_{timestamp}.yaml'
        
        try:
            with open(filename, 'w') as f:
                f.write('# System Identification Calibration Results\n')
                f.write(f'# Timestamp: {datetime.datetime.now()}\n')
                f.write('calibrated_parameters:\n')
                
                param_names = list(self.calibration_targets.keys())
                for name, value in zip(param_names, [self.calibration_targets[n]['current'] for n in param_names]):
                    f.write(f'  {name}: {value}\n')
                
                f.write(f'\ndata_points_used: {min(len(self.sim_data_buffer), len(self.real_data_buffer))}\n')
                f.write(f'calibration_iterations: {len(self.param_history)}\n')
            
            self.get_logger().info(f'Calibration results saved to {filename}')
        except Exception as e:
            self.get_logger().error(f'Error saving calibration results: {str(e)}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        # Save final calibration results
        self.save_calibration_results()
        
        self.get_logger().info('System Identification Implementation Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    sys_id = SystemIdentificationImpl()
    
    try:
        rclpy.spin(sys_id)
    except KeyboardInterrupt:
        sys_id.get_logger().info('Node interrupted by user')
    finally:
        sys_id.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 3: Robust Control Implementation

### Step 3.1: Creating a Robust Controller Node

Implement a robust controller that adapts to domain shifts:

```python
# scripts/robust_controller_impl.py
#!/usr/bin/env python3

"""
Implementation of robust control strategies for sim-to-real transfer.
Includes adaptive control and domain-aware controllers.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
from scipy import signal
import math


class RobustControllerImpl(Node):
    """
    Implementation of robust controllers designed to work across sim and real domains.
    Includes adaptive control and domain-invariant control strategies.
    """
    
    def __init__(self):
        super().__init__('robust_controller_impl')
        
        # Declare parameters
        self.declare_parameter('control_frequency', 100.0)  # Hz
        self.declare_parameter('adaptation_rate', 0.01)    # Rate of parameter adaptation
        self.declare_parameter('stability_threshold', 0.5)  # Threshold for stability detection
        self.declare_parameter('domain_invariant_gain', 0.8)  # Gain for domain-invariant control
        
        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.adaptation_rate = self.get_parameter('adaptation_rate').value
        self.stability_threshold = self.get_parameter('stability_threshold').value
        self.domain_invariant_gain = self.get_parameter('domain_invariant_gain').value
        
        # Joint information
        self.joint_names = []  # Will be populated based on robot model
        self.current_joint_states = None
        self.desired_joint_positions = {}
        self.previous_errors = {}
        self.integral_errors = {}
        self.adaptive_gains = {}
        
        # Initialize joint-specific controllers
        self.initialize_joint_controllers()
        
        # Subscriptions
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
        
        self.command_sub = self.create_subscription(
            JointState,
            '/joint_commands',
            self.command_callback,
            10
        )
        
        # Publishers
        self.control_output_pub = self.create_publisher(
            JointState,
            '/joint_control_commands',
            10
        )
        
        self.controller_status_pub = self.create_publisher(
            String,
            '/robust_controller_status',
            10
        )
        
        self.stability_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/stability_metrics',
            10
        )
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.robust_control_cycle
        )
        
        self.get_logger().info('Robust Controller Implementation Node Initialized')

    def initialize_joint_controllers(self):
        """Initialize adaptive controllers for each joint"""
        # Define humanoid-specific joints
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]
        
        # Initialize controller parameters for each joint
        for joint_name in self.joint_names:
            # Initial PID gains (will be adapted over time)
            self.adaptive_gains[joint_name] = {
                'kp': np.random.uniform(0.8, 1.2),  # Random initial gain for adaptation
                'ki': np.random.uniform(0.1, 0.3),
                'kd': np.random.uniform(0.1, 0.2)
            }
            
            # Initialize error tracking
            self.previous_errors[joint_name] = 0.0
            self.integral_errors[joint_name] = 0.0
            
            # Initialize desired positions
            self.desired_joint_positions[joint_name] = 0.0

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joint_states = msg

    def imu_callback(self, msg):
        """Process IMU data for stability assessment"""
        # Extract orientation and angular velocity for stability assessment
        self.current_orientation = msg.orientation
        self.current_angular_velocity = msg.angular_velocity
        self.current_linear_acceleration = msg.linear_acceleration

    def command_callback(self, msg):
        """Store desired joint commands"""
        for i, joint_name in enumerate(msg.name):
            if joint_name in self.desired_joint_positions:
                if i < len(msg.position):
                    self.desired_joint_positions[joint_name] = msg.position[i]

    def robust_control_cycle(self):
        """Main robust control cycle"""
        if not self.current_joint_states:
            return
        
        try:
            # Calculate control commands with robust techniques
            control_commands = self.compute_robust_controls()
            
            # Assess stability based on IMU data
            stability_metrics = self.assess_stability()
            self.publish_stability_metrics(stability_metrics)
            
            # Adapt control parameters based on stability performance
            self.adapt_control_parameters(stability_metrics)
            
            # Publish control commands
            self.control_output_pub.publish(control_commands)
            
            # Publish status
            status_msg = String()
            status_msg.data = self.generate_controller_status(stability_metrics)
            self.controller_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in robust control cycle: {str(e)}')

    def compute_robust_controls(self):
        """Compute robust control commands for all joints"""
        if not self.current_joint_states:
            # Return zero command if no joint state available
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = self.joint_names[:]
            cmd.position = [0.0] * len(self.joint_names)
            cmd.velocity = [0.0] * len(self.joint_names) 
            cmd.effort = [0.0] * len(self.joint_names)
            return cmd
        
        # Create control command message
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = []
        cmd.position = []
        cmd.velocity = []
        cmd.effort = []
        
        # Process each joint with robust control logic
        for joint_name in self.joint_names:
            # Get current position for this joint
            current_pos = 0.0
            try:
                idx = self.current_joint_states.name.index(joint_name)
                if idx < len(self.current_joint_states.position):
                    current_pos = self.current_joint_states.position[idx]
            except (ValueError, IndexError):
                pass  # Joint not found in current state, use default
            
            # Get desired position
            desired_pos = self.desired_joint_positions.get(joint_name, 0.0)
            
            # Compute robust control with adaptive gains
            control_effort = self.robust_pid_control(
                joint_name,
                desired_pos,
                current_pos
            )
            
            # Apply safety limits
            control_effort = np.clip(control_effort, -100.0, 100.0)  # Effort limits
            
            # Add to command message
            cmd.name.append(joint_name)
            cmd.effort.append(control_effort)
            # For position control, calculate next position
            next_pos = current_pos + control_effort * (1.0 / self.control_frequency) * 0.001  # Small increment
            cmd.position.append(next_pos)
            cmd.velocity.append(0.0)  # Will be calculated based on position changes
        
        return cmd

    def robust_pid_control(self, joint_name, desired, current):
        """Robust PID control with adaptation and domain-invariant features"""
        # Calculate error
        error = desired - current
        
        # Get current time
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Calculate dt (if we have previous time)
        dt = 1.0 / self.control_frequency  # Default to control frequency if first run
        if hasattr(self, 'previous_time'):
            dt = current_time - self.previous_time
            if dt <= 0:
                dt = 1.0 / self.control_frequency
        
        self.previous_time = current_time
        
        # Update integral (with anti-windup)
        self.integral_errors[joint_name] += error * dt
        # Anti-windup: limit integral term
        integral_limit = 10.0
        self.integral_errors[joint_name] = max(
            -integral_limit, 
            min(integral_limit, self.integral_errors[joint_name])
        )
        
        # Calculate derivative
        derivative = 0.0
        if hasattr(self, f'previous_error_{joint_name}'):
            if dt > 0:
                derivative = (error - self.previous_errors[joint_name]) / dt
        self.previous_errors[joint_name] = error
        
        # Apply low-pass filter to derivative to reduce noise sensitivity
        # (Important for robustness across sim-to-real transition)
        if not hasattr(self, f'derivative_filter_{joint_name}'):
            # Initialize Butterworth filter (second order, cutoff at 10Hz)
            sos = signal.butter(2, 10, 'low', fs=self.control_frequency, output='sos')
            setattr(self, f'derivative_filter_{joint_name}', sos)
            setattr(self, f'derivative_filtered_{joint_name}', 0.0)
        
        derivative_filter = getattr(self, f'derivative_filter_{joint_name}')
        derivative_filtered = signal.sosfilt(derivative_filter, [derivative])[0]
        setattr(self, f'derivative_filtered_{joint_name}', derivative_filtered)
        
        # Get adaptive gains
        gains = self.adaptive_gains[joint_name]
        
        # Compute PID output with domain-invariant gain adjustment
        output = (
            gains['kp'] * error + 
            gains['ki'] * self.integral_errors[joint_name] + 
            gains['kd'] * derivative_filtered
        ) * self.domain_invariant_gain
        
        return output

    def assess_stability(self):
        """Assess robot stability using IMU data"""
        if not hasattr(self, 'current_orientation'):
            return {'stability_score': 0.0, 'roll_angle': 0.0, 'pitch_angle': 0.0, 'angular_velocity': 0.0}
        
        # Calculate roll and pitch angles from orientation
        orientation = self.current_orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Calculate angular velocity magnitude
        ang_vel = self.current_angular_velocity
        angular_velocity_mag = math.sqrt(
            ang_vel.x**2 + ang_vel.y**2 + ang_vel.z**2
        )
        
        # Calculate stability score based on orientation deviations and angular velocity
        # Lower absolute roll/pitch angles = more stable
        # Lower angular velocity = more stable
        stability_from_orientation = max(0.0, 1.0 - (abs(roll) + abs(pitch)) / 1.57)  # 90 degrees in radians
        stability_from_velocity = max(0.0, 1.0 - angular_velocity_mag / 1.0)  # 1.0 rad/s threshold
        
        # Combined stability score
        stability_score = 0.7 * stability_from_orientation + 0.3 * stability_from_velocity
        
        return {
            'stability_score': stability_score,
            'roll_angle': roll,
            'pitch_angle': pitch,
            'yaw_angle': yaw,
            'angular_velocity': angular_velocity_mag,
            'linear_acceleration': math.sqrt(
                self.current_linear_acceleration.x**2 +
                self.current_linear_acceleration.y**2 +
                self.current_linear_acceleration.z**2
            )
        }

    def adapt_control_parameters(self, stability_metrics):
        """Adapt control parameters based on stability assessment"""
        if stability_metrics['stability_score'] < self.stability_threshold:
            # Robot is unstable, adjust gains to stabilize
            for joint_name in self.joint_names:
                # Reduce proportional gain to reduce aggression
                self.adaptive_gains[joint_name]['kp'] *= (1.0 - self.adaptation_rate * 0.5)
                # Slightly increase derivative gain to dampen oscillations
                self.adaptive_gains[joint_name]['kd'] *= (1.0 + self.adaptation_rate * 0.2)
        else:
            # Robot is stable, can be more aggressive for accuracy
            for joint_name in self.joint_names:
                # Slightly increase proportional gain for better tracking
                self.adaptive_gains[joint_name]['kp'] *= (1.0 + self.adaptation_rate * 0.1)
                # Slightly decrease derivative gain to allow more responsiveness
                self.adaptive_gains[joint_name]['kd'] *= (1.0 - self.adaptation_rate * 0.1)
        
        # Bound the gains to reasonable values
        for joint_name in self.joint_names:
            gains = self.adaptive_gains[joint_name]
            gains['kp'] = max(0.1, min(5.0, gains['kp']))
            gains['ki'] = max(0.01, min(1.0, gains['ki']))
            gains['kd'] = max(0.01, min(1.0, gains['kd']))

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def publish_stability_metrics(self, metrics):
        """Publish stability assessment metrics"""
        metrics_msg = Float64MultiArray()
        metrics_msg.data = [
            metrics['stability_score'],
            metrics['roll_angle'],
            metrics['pitch_angle'],
            metrics['angular_velocity'],
            metrics['linear_acceleration']
        ]
        self.stability_metrics_publisher.publish(metrics_msg)

    def generate_controller_status(self, stability_metrics):
        """Generate controller status string"""
        status_parts = [
            f"STABILITY:{stability_metrics['stability_score']:.2f}",
            f"ROLL:{stability_metrics['roll_angle']:.2f}",
            f"PITCH:{stability_metrics['pitch_angle']:.2f}",
            f"W:{stability_metrics['angular_velocity']:.2f}"
        ]
        
        # Add information about adaptive gains
        avg_kp = np.mean([g['kp'] for g in self.adaptive_gains.values()])
        status_parts.append(f"AVG_KP:{avg_kp:.3f}")
        
        return " | ".join(status_parts)

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Robust Controller Implementation Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobustControllerImpl()
    
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

## Part 4: Domain Adaptation Implementation

### Step 4.1: Creating a Domain Adaptation Node

Implement domain adaptation techniques to adjust models between simulation and reality:

```python
# scripts/domain_adaptation_impl.py
#!/usr/bin/env python3

"""
Implementation of domain adaptation techniques for sim-to-real transfer.
Uses statistical matching and normalization to adapt simulation outputs to reality.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DomainAdaptationImpl(Node):
    """
    Implementation of domain adaptation techniques to make simulation outputs
    more similar to real-world sensor outputs for better transfer learning.
    """
    
    def __init__(self):
        super().__init__('domain_adaptation_impl')
        
        # Declare parameters
        self.declare_parameter('adaptation_frequency', 10.0)  # Hz
        self.declare_parameter('buffer_size', 1000)
        self.declare_parameter('adaptation_method', 'statistical_matching')  # Options: statistical_matching, normalization, regression
        self.declare_parameter('model_save_path', '/tmp/domain_adaptation_model.pkl')
        
        # Get parameters
        self.adaptation_frequency = self.get_parameter('adaptation_frequency').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.adaptation_method = self.get_parameter('adaptation_method').value
        self.model_save_path = self.get_parameter('model_save_path').value
        
        # Initialize internal state
        self.sim_data_buffers = {}  # Buffers for simulation data
        self.real_data_buffers = {}  # Buffers for real robot data
        self.adaptation_models = {}  # Models for adapting sim to real
        self.scalers = {}  # Scalers for normalization
        self.cv_bridge = CvBridge()
        
        # Subscriptions for simulation data
        self.sim_image_sub = self.create_subscription(
            Image,
            '/sim/camera/image_raw',
            self.sim_image_callback,
            10
        )
        
        self.sim_scan_sub = self.create_subscription(
            LaserScan,
            '/sim/scan',
            self.sim_scan_callback,
            10
        )
        
        self.sim_imu_sub = self.create_subscription(
            Imu,
            '/sim/imu',
            self.sim_imu_callback,
            10
        )
        
        # Subscriptions for real robot data (in practice, would come from real robot)
        self.real_image_sub = self.create_subscription(
            Image,
            '/real/camera/image_raw',
            self.real_image_callback,
            10
        )
        
        self.real_scan_sub = self.create_subscription(
            LaserScan,
            '/real/scan',
            self.real_scan_callback,
            10
        )
        
        self.real_imu_sub = self.create_subscription(
            Imu,
            '/real/imu',
            self.real_imu_callback,
            10
        )
        
        # Publishers for adapted data
        self.adapted_image_pub = self.create_publisher(
            Image,
            '/adapted/camera/image_raw',
            10
        )
        
        self.adapted_scan_pub = self.create_publisher(
            LaserScan,
            '/adapted/scan',
            10
        )
        
        self.adapted_imu_pub = self.create_publisher(
            Imu,
            '/adapted/imu',
            10
        )
        
        self.adaptation_status_pub = self.create_publisher(
            String,
            '/domain_adaptation_status',
            10
        )
        
        # Timer for adaptation
        self.adaptation_timer = self.create_timer(
            1.0 / self.adaptation_frequency,
            self.run_domain_adaptation
        )
        
        self.get_logger().info('Domain Adaptation Implementation Node Initialized')

    def sim_image_callback(self, msg):
        """Store simulation camera image data"""
        if 'sim_image' not in self.sim_data_buffers:
            self.sim_data_buffers['sim_image'] = []
        
        # Extract features from image for domain adaptation
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            features = self.extract_image_features(cv_image)
            
            # Store in buffer
            self.sim_data_buffers['sim_image'].append({
                'timestamp': msg.header.stamp,
                'features': features,
                'raw_msg': msg
            })
            
            # Keep buffer size reasonable
            if len(self.sim_data_buffers['sim_image']) > self.buffer_size:
                self.sim_data_buffers['sim_image'].pop(0)
                
        except Exception as e:
            self.get_logger().error(f'Error processing sim image: {str(e)}')

    def real_image_callback(self, msg):
        """Store real robot camera image data"""
        if 'real_image' not in self.real_data_buffers:
            self.real_data_buffers['real_image'] = []
        
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            features = self.extract_image_features(cv_image)
            
            self.real_data_buffers['real_image'].append({
                'timestamp': msg.header.stamp,
                'features': features,
                'raw_msg': msg
            })
            
            if len(self.real_data_buffers['real_image']) > self.buffer_size:
                self.real_data_buffers['real_image'].pop(0)
                
        except Exception as e:
            self.get_logger().error(f'Error processing real image: {str(e)}')

    def sim_scan_callback(self, msg):
        """Store simulation LiDAR data"""
        if 'sim_scan' not in self.sim_data_buffers:
            self.sim_data_buffers['sim_scan'] = []
        
        # Extract statistical features from scan
        features = self.extract_scan_features(msg)
        
        self.sim_data_buffers['sim_scan'].append({
            'timestamp': msg.header.stamp,
            'features': features,
            'raw_msg': msg
        })
        
        if len(self.sim_data_buffers['sim_scan']) > self.buffer_size:
            self.sim_data_buffers['sim_scan'].pop(0)

    def real_scan_callback(self, msg):
        """Store real robot LiDAR data"""
        if 'real_scan' not in self.real_data_buffers:
            self.real_data_buffers['real_scan'] = []
        
        features = self.extract_scan_features(msg)
        
        self.real_data_buffers['real_scan'].append({
            'timestamp': msg.header.stamp,
            'features': features,
            'raw_msg': msg
        })
        
        if len(self.real_data_buffers['real_scan']) > self.buffer_size:
            self.real_data_buffers['real_scan'].pop(0)

    def sim_imu_callback(self, msg):
        """Store simulation IMU data"""
        if 'sim_imu' not in self.sim_data_buffers:
            self.sim_data_buffers['sim_imu'] = []
        
        features = self.extract_imu_features(msg)
        
        self.sim_data_buffers['sim_imu'].append({
            'timestamp': msg.header.stamp,
            'features': features,
            'raw_msg': msg
        })
        
        if len(self.sim_data_buffers['sim_imu']) > self.buffer_size:
            self.sim_data_buffers['sim_imu'].pop(0)

    def real_imu_callback(self, msg):
        """Store real robot IMU data"""
        if 'real_imu' not in self.real_data_buffers:
            self.real_data_buffers['real_imu'] = []
        
        features = self.extract_imu_features(msg)
        
        self.real_data_buffers['real_imu'].append({
            'timestamp': msg.header.stamp,
            'features': features,
            'raw_msg': msg
        })
        
        if len(self.real_data_buffers['real_imu']) > self.buffer_size:
            self.real_data_buffers['real_imu'].pop(0)

    def extract_image_features(self, image):
        """Extract features from image for domain adaptation"""
        # Calculate image statistics that can be matched between domains
        mean_color = np.mean(image, axis=(0,1))  # Mean color in each channel
        std_color = np.std(image, axis=(0,1))    # Std of color in each channel
        histogram = [np.histogram(image[:,:,i], bins=50)[0] for i in range(3)]  # Color histograms
        gradient_magnitude = np.mean(np.gradient(image.astype(np.float32), axis=(0,1)))  # Gradient statistics
        
        return {
            'mean_color': mean_color,
            'std_color': std_color,
            'histogram': histogram,
            'gradient_mean': gradient_magnitude
        }

    def extract_scan_features(self, scan_msg):
        """Extract statistical features from LiDAR scan"""
        # Process scan ranges to extract statistical features
        valid_ranges = [r for r in scan_msg.ranges if np.isfinite(r)]
        
        if not valid_ranges:
            return {
                'mean_range': 0.0,
                'std_range': 0.0,
                'min_range': float('inf'),
                'max_range': 0.0,
                'num_valid': 0
            }
        
        return {
            'mean_range': float(np.mean(valid_ranges)),
            'std_range': float(np.std(valid_ranges)),
            'min_range': float(np.min(valid_ranges)),
            'max_range': float(np.max(valid_ranges)),
            'num_valid': len(valid_ranges)
        }

    def extract_imu_features(self, imu_msg):
        """Extract features from IMU data"""
        return {
            'orientation': [
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w
            ],
            'angular_velocity': [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ],
            'linear_acceleration': [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z
            ]
        }

    def run_domain_adaptation(self):
        """Run domain adaptation cycle"""
        try:
            # Adapt each sensor modality
            if len(self.sim_data_buffers.get('sim_image', [])) > 10 and len(self.real_data_buffers.get('real_image', [])) > 10:
                self.adapt_image_domain()
            
            if len(self.sim_data_buffers.get('sim_scan', [])) > 10 and len(self.real_data_buffers.get('real_scan', [])) > 10:
                self.adapt_scan_domain()
            
            if len(self.sim_data_buffers.get('sim_imu', [])) > 10 and len(self.real_data_buffers.get('real_imu', [])) > 10:
                self.adapt_imu_domain()
            
            # Publish status
            self.publish_adaptation_status()
            
        except Exception as e:
            self.get_logger().error(f'Error in domain adaptation: {str(e)}')

    def adapt_image_domain(self):
        """Apply domain adaptation to image data"""
        if not ('sim_image' in self.sim_data_buffers and 'real_image' in self.real_data_buffers):
            return
        
        # Extract all features from buffers
        sim_features = [d['features'] for d in self.sim_data_buffers['sim_image']]
        real_features = [d['features'] for d in self.real_data_buffers['real_image']]
        
        # Create feature vectors for each type
        sim_means = np.array([f['mean_color'] for f in sim_features])
        real_means = np.array([f['mean_color'] for f in real_features])
        
        # Train adaptation model (for this example, we'll use a simple statistical matching)
        # In practice, this could be more sophisticated (regression, neural networks, etc.)
        
        if len(sim_means) > 0 and len(real_means) > 0:
            # Calculate mean and std for both domains
            sim_mean_stats = np.mean(sim_means, axis=0)
            sim_std_stats = np.std(sim_means, axis=0)
            
            real_mean_stats = np.mean(real_means, axis=0)
            real_std_stats = np.std(real_means, axis=0)
            
            # Store normalization parameters
            self.adaptation_models['image_color'] = {
                'sim_mean': sim_mean_stats,
                'sim_std': sim_std_stats,
                'real_mean': real_mean_stats,
                'real_std': real_std_stats
            }
            
            self.get_logger().debug(f'Updated image adaptation model')

    def adapt_scan_domain(self):
        """Apply domain adaptation to LiDAR data"""
        if not ('sim_scan' in self.sim_data_buffers and 'real_scan' in self.real_data_buffers):
            return
        
        # Extract scan statistics
        sim_stats = [d['features'] for d in self.sim_data_buffers['sim_scan']]
        real_stats = [d['features'] for d in self.real_data_buffers['real_scan']]
        
        # Calculate domain statistics
        sim_mean_ranges = np.array([s['mean_range'] for s in sim_stats if s['num_valid'] > 0])
        real_mean_ranges = np.array([s['mean_range'] for s in real_stats if s['num_valid'] > 0])
        
        if len(sim_mean_ranges) > 0 and len(real_mean_ranges) > 0:
            # Calculate transformation parameters
            sim_avg_mean = np.mean(sim_mean_ranges)
            real_avg_mean = np.mean(real_mean_ranges)
            
            sim_avg_std = np.std(sim_mean_ranges)
            real_avg_std = np.std(real_mean_ranges)
            
            # Store adaptation parameters
            self.adaptation_models['scan_range'] = {
                'scale': real_avg_std / (sim_avg_std + 1e-8),  # Add small value to prevent division by zero
                'offset': real_avg_mean - sim_avg_mean * (real_avg_std / (sim_avg_std + 1e-8))
            }
            
            self.get_logger().debug(f'Updated scan adaptation model: scale={self.adaptation_models["scan_range"]["scale"]:.3f}')

    def adapt_imu_domain(self):
        """Apply domain adaptation to IMU data"""
        if not ('sim_imu' in self.sim_data_buffers and 'real_imu' in self.real_data_buffers):
            return
        
        # Extract IMU statistics
        sim_orientations = [d['features']['orientation'] for d in self.sim_data_buffers['sim_imu']]
        real_orientations = [d['features']['orientation'] for d in self.real_data_buffers['real_imu']]
        
        sim_angular_velocities = [d['features']['angular_velocity'] for d in self.sim_data_buffers['sim_imu']]
        real_angular_velocities = [d['features']['angular_velocity'] for d in self.real_data_buffers['real_imu']]
        
        if len(sim_orientations) > 0 and len(real_orientations) > 0:
            # Calculate statistics for orientation
            sim_orient_mean = np.mean(sim_orientations, axis=0)
            real_orient_mean = np.mean(real_orientations, axis=0)
            
            # Calculate statistics for angular velocity
            sim_angvel_mean = np.mean(sim_angular_velocities, axis=0)
            real_angvel_mean = np.mean(real_angular_velocities, axis=0)
            
            sim_angvel_std = np.std(sim_angular_velocities, axis=0)
            real_angvel_std = np.std(real_angular_velocities, axis=0)
            
            # Store adaptation parameters
            self.adaptation_models['imu_orientation'] = {
                'offset': real_orient_mean - sim_orient_mean
            }
            
            self.adaptation_models['imu_angular_velocity'] = {
                'scale': real_angvel_std / (sim_angvel_std + 1e-8),
                'offset': real_angvel_mean - sim_angvel_mean * (real_angvel_std / (sim_angvel_std + 1e-8))
            }
            
            self.get_logger().debug(f'Updated IMU adaptation models')

    def adapt_message_data(self, msg_type, msg):
        """Adapt a message from simulation domain to real domain"""
        if msg_type == 'image' and 'image_color' in self.adaptation_models:
            # For image adaptation, we'd modify the image content
            # This is a simplified version - in practice would apply learned transformation
            return self.adapt_image_message(msg)
        elif msg_type == 'scan' and 'scan_range' in self.adaptation_models:
            return self.adapt_scan_message(msg)
        elif msg_type == 'imu' and 'imu_angular_velocity' in self.adaptation_models:
            return self.adapt_imu_message(msg)
        else:
            # Return original message if no adaptation is available
            return msg

    def adapt_scan_message(self, scan_msg):
        """Adapt a LiDAR scan message from sim to real domain"""
        if 'scan_range' not in self.adaptation_models:
            return scan_msg
        
        # Create a new message with adapted ranges
        adapted_msg = LaserScan()
        adapted_msg.header = scan_msg.header
        adapted_msg.angle_min = scan_msg.angle_min
        adapted_msg.angle_max = scan_msg.angle_max
        adapted_msg.angle_increment = scan_msg.angle_increment
        adapted_msg.time_increment = scan_msg.time_increment
        adapted_msg.scan_time = scan_msg.scan_time
        adapted_msg.range_min = scan_msg.range_min
        adapted_msg.range_max = scan_msg.range_max
        
        # Apply adaptation transformation to ranges
        adaptation_params = self.adaptation_models['scan_range']
        adapted_ranges = []
        
        for range_val in scan_msg.ranges:
            if np.isfinite(range_val):
                # Apply learned transformation
                adapted_range = (range_val - adaptation_params['offset']) / adaptation_params['scale']
                adapted_ranges.append(adapted_range)
            else:
                # Keep non-finite values as is
                adapted_ranges.append(range_val)
        
        adapted_msg.ranges = adapted_ranges
        return adapted_msg

    def adapt_imu_message(self, imu_msg):
        """Adapt an IMU message from sim to real domain"""
        if 'imu_angular_velocity' not in self.adaptation_models:
            return imu_msg
        
        adapted_msg = Imu()
        adapted_msg.header = imu_msg.header
        adapted_msg.orientation = imu_msg.orientation  # We won't modify orientation for now
        adapted_msg.orientation_covariance = imu_msg.orientation_covariance
        
        # Apply adaptation to angular velocity
        adaptation_params = self.adaptation_models['imu_angular_velocity']
        scale = adaptation_params['scale']
        offset = adaptation_params['offset']
        
        adapted_msg.angular_velocity.x = (
            imu_msg.angular_velocity.x * scale[0] + offset[0]
        )
        adapted_msg.angular_velocity.y = (
            imu_msg.angular_velocity.y * scale[1] + offset[1]
        )
        adapted_msg.angular_velocity.z = (
            imu_msg.angular_velocity.z * scale[2] + offset[2]
        )
        
        adapted_msg.angular_velocity_covariance = imu_msg.angular_velocity_covariance
        
        # Apply adaptation to linear acceleration
        # (for simplicity, we'll assume linear acceleration doesn't need adaptation)
        adapted_msg.linear_acceleration = imu_msg.linear_acceleration
        adapted_msg.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance
        
        return adapted_msg

    def publish_adaptation_status(self):
        """Publish domain adaptation status"""
        status_parts = []
        
        if 'image_color' in self.adaptation_models:
            status_parts.append("IMAGE_ADAPTED")
        
        if 'scan_range' in self.adaptation_models:
            status_parts.append("SCAN_ADAPTED")
        
        if 'imu_angular_velocity' in self.adaptation_models:
            status_parts.append("IMU_ADAPTED")
        
        if not status_parts:
            status_parts.append("WAITING_DATA")
        
        status_msg = String()
        status_msg.data = f"DOMAIN_ADAPTATION: {', '.join(status_parts)} | " \
                         f"SimBuf: {len(self.sim_data_buffers)} | " \
                         f"RealBuf: {len(self.real_data_buffers)}"
        self.adaptation_status_pub.publish(status_msg)
        
        self.get_logger().debug(f'Domain adaptation status: {status_msg.data}')

    def save_adaptation_model(self):
        """Save the trained adaptation model to file"""
        try:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(self.adaptation_models, f)
            self.get_logger().info(f'Saved domain adaptation model to {self.model_save_path}')
        except Exception as e:
            self.get_logger().error(f'Error saving adaptation model: {str(e)}')

    def load_adaptation_model(self):
        """Load a previously saved adaptation model"""
        if os.path.exists(self.model_save_path):
            try:
                with open(self.model_save_path, 'rb') as f:
                    self.adaptation_models = pickle.load(f)
                self.get_logger().info(f'Loaded domain adaptation model from {self.model_save_path}')
            except Exception as e:
                self.get_logger().error(f'Error loading adaptation model: {str(e)}')

    def destroy_node(self):
        """Cleanup before node destruction"""
        # Save the adaptation model before shutting down
        self.save_adaptation_model()
        self.get_logger().info('Domain Adaptation Implementation Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    adapter = DomainAdaptationImpl()
    
    try:
        # Load any existing adaptation model
        adapter.load_adaptation_model()
        
        rclpy.spin(adapter)
    except KeyboardInterrupt:
        adapter.get_logger().info('Node interrupted by user')
    finally:
        adapter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Part 5: Integration and Testing

### Step 5.1: Create a Complete Launch File

Create a launch file to bring up all the sim-to-real transfer components:

```python
# launch/sim2real_transfer.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_domain_randomization = LaunchConfiguration('enable_domain_randomization', default='false')
    enable_system_identification = LaunchConfiguration('enable_system_identification', default='false')
    enable_domain_adaptation = LaunchConfiguration('enable_domain_adaptation', default='false')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_simple_robot'),
                'worlds',
                'simple_humanoid_world.sdf'
            ]),
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_simple_robot'),
                'urdf',
                'advanced_humanoid.urdf'
            ])
        }],
        output='screen'
    )
    
    # Spawner node to load the robot into Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'advanced_humanoid',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )
    
    # Domain Randomization Node
    domain_randomizer = Node(
        package='humanoid_simple_robot',
        executable='domain_randomizer_impl',
        name='domain_randomizer_impl',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'randomization_frequency': 5.0},  # Lower frequency for less disruption
            {'physics_variance': 0.15},
            {'sensor_noise_variance': 0.25},
            {'actuator_variance': 0.20},
            {'randomization_enabled': enable_domain_randomization}
        ],
        output='screen'
    )
    
    # System Identification Node
    system_identifier = Node(
        package='humanoid_simple_robot',
        executable='system_identification_impl',
        name='system_identification_impl',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'identification_frequency': 0.2},  # Very low frequency for optimization
            {'calibration_enabled': enable_system_identification}
        ],
        output='screen'
    )
    
    # Domain Adaptation Node
    domain_adapter = Node(
        package='humanoid_simple_robot',
        executable='domain_adaptation_impl',
        name='domain_adaptation_impl',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'adaptation_frequency': 10.0},
            {'buffer_size': 500},
            {'model_save_path': '/tmp/domain_adaptation_model.pkl'}
        ],
        output='screen'
    )
    
    # Perception Integrator Node
    perception_integrator = Node(
        package='humanoid_simple_robot',
        executable='perception_integrator',
        name='perception_integrator',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'update_rate': 30.0}
        ],
        output='screen'
    )
    
    # AI Planner Node
    ai_planner = Node(
        package='humanoid_simple_robot',
        executable='ai_planner',
        name='ai_planner',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'planning_frequency': 5.0}
        ],
        output='screen'
    )
    
    # Robust Controller Node
    robust_controller = Node(
        package='humanoid_simple_robot',
        executable='robust_controller_impl',
        name='robust_controller_impl',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'control_frequency': 100.0},
            {'adaptation_rate': 0.01}
        ],
        output='screen'
    )
    
    # RViz for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d', 
            PathJoinSubstitution([
                FindPackageShare('humanoid_simple_robot'),
                'rviz',
                'sim2real_transfer.rviz'
            ])
        ],
        output='screen'
    )
    
    return LaunchDescription([
        # Set parameters
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        # Launch Gazebo
        gazebo,
        
        # Launch robot state publisher after delay
        TimerAction(
            period=2.0,
            actions=[robot_state_publisher]
        ),
        
        # Launch spawner after Gazebo is ready
        TimerAction(
            period=4.0,
            actions=[spawn_entity]
        ),
        
        # Launch sim-to-real transfer components with appropriate delays
        TimerAction(
            period=6.0,
            actions=[perception_integrator]
        ),
        
        TimerAction(
            period=7.0,
            actions=[ai_planner]
        ),
        
        TimerAction(
            period=8.0,
            actions=[robust_controller]
        ),
        
        # Launch domain adaptation components
        TimerAction(
            period=9.0,
            actions=[domain_adapter]
        ),
        
        # Optional: Launch system identification for calibration
        TimerAction(
            period=10.0,
            actions=[system_identifier]
        ),
        
        # Launch domain randomizer with appropriate delay
        TimerAction(
            period=12.0,
            actions=[domain_randomizer]
        ),
        
        # Launch RViz for visualization
        TimerAction(
            period=14.0,
            actions=[rviz]
        ),
    ])
```

### Step 5.2: Create a Validation Node

Create a validation node to test the sim-to-real transfer effectiveness:

```python
# scripts/transfer_validator.py
#!/usr/bin/env python3

"""
Transfer Validator Node for assessing sim-to-real transfer effectiveness.
Compares performance between simulation and reality (simulated reality in this case).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from std_msgs.msg import String, Float64MultiArray
from nav_msgs.msg import Odometry
import numpy as np
import math


class TransferValidatorNode(Node):
    """
    Validates the effectiveness of sim-to-real transfer techniques
    by comparing performance metrics between simulation and reality.
    """
    
    def __init__(self):
        super().__init__('transfer_validator_node')
        
        # Declare parameters
        self.declare_parameter('validation_frequency', 2.0)  # Hz
        self.declare_parameter('performance_window', 10.0)  # seconds
        self.declare_parameter('success_threshold', 0.7)  # Performance threshold
        self.declare_parameter('stability_threshold', 0.8)  # Stability threshold
        
        # Get parameters
        self.validation_frequency = self.get_parameter('validation_frequency').value
        self.performance_window = self.get_parameter('performance_window').value
        self.success_threshold = self.get_parameter('success_threshold').value
        self.stability_threshold = self.get_parameter('stability_threshold').value
        
        # Data collection
        self.sim_performance_history = []
        self.real_performance_history = []
        self.transfer_metrics_history = []
        
        # Subscriptions for simulation data
        self.sim_joint_sub = self.create_subscription(
            JointState,
            '/sim/joint_states',
            self.sim_joint_callback,
            10
        )
        
        self.sim_imu_sub = self.create_subscription(
            Imu,
            '/sim/imu',
            self.sim_imu_callback,
            10
        )
        
        self.sim_scan_sub = self.create_subscription(
            LaserScan,
            '/sim/scan',
            self.sim_scan_callback,
            10
        )
        
        # Subscriptions for reality data (or adapted simulation data)
        self.real_joint_sub = self.create_subscription(
            JointState,
            '/real/joint_states',  # In this example, would be from real robot
            self.real_joint_callback,
            10
        )
        
        self.real_imu_sub = self.create_subscription(
            Imu,
            '/real/imu',
            self.real_imu_callback,
            10
        )
        
        self.real_scan_sub = self.create_subscription(
            LaserScan,
            '/real/scan',
            self.real_scan_callback,
            10
        )
        
        # Publishers
        self.validation_report_pub = self.create_publisher(
            String,
            '/transfer_validation_report',
            10
        )
        
        self.transfer_metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/transfer_effectiveness_metrics',
            10
        )
        
        # Timer for validation
        self.validation_timer = self.create_timer(
            1.0 / self.validation_frequency,
            self.run_validation_cycle
        )
        
        self.get_logger().info('Transfer Validator Node Initialized')

    def sim_joint_callback(self, msg):
        """Process simulation joint state data"""
        self.store_performance_data('simulation', 'joint_state', msg)

    def real_joint_callback(self, msg):
        """Process reality joint state data"""
        self.store_performance_data('reality', 'joint_state', msg)

    def sim_imu_callback(self, msg):
        """Process simulation IMU data"""
        self.store_performance_data('simulation', 'imu', msg)

    def real_imu_callback(self, msg):
        """Process reality IMU data"""
        self.store_performance_data('reality', 'imu', msg)

    def sim_scan_callback(self, msg):
        """Process simulation scan data"""
        self.store_performance_data('simulation', 'scan', msg)

    def real_scan_callback(self, msg):
        """Process reality scan data"""
        self.store_performance_data('reality', 'scan', msg)

    def store_performance_data(self, domain, data_type, msg):
        """Store performance data from sim or real"""
        entry = {
            'timestamp': self.get_clock().now(),
            'domain': domain,
            'type': data_type,
            'message': msg
        }
        
        if domain == 'simulation':
            if len(self.sim_performance_history) >= 1000:  # Limit storage
                self.sim_performance_history.pop(0)
            self.sim_performance_history.append(entry)
        else:  # reality
            if len(self.real_performance_history) >= 1000:  # Limit storage
                self.real_performance_history.pop(0)
            self.real_performance_history.append(entry)

    def run_validation_cycle(self):
        """Run the validation cycle"""
        try:
            # Calculate performance metrics over the recent window
            sim_metrics = self.calculate_recent_metrics('simulation')
            real_metrics = self.calculate_recent_metrics('reality')
            
            # Calculate transfer effectiveness
            effectiveness = self.calculate_transfer_effectiveness(sim_metrics, real_metrics)
            
            # Generate validation report
            report = self.generate_validation_report(sim_metrics, real_metrics, effectiveness)
            
            # Publish effectiveness metrics
            metrics_msg = Float64MultiArray()
            metrics_msg.data = [
                effectiveness.get('stability_similarity', 0.0),
                effectiveness.get('motion_similarity', 0.0),
                effectiveness.get('sensor_similarity', 0.0),
                effectiveness.get('overall_score', 0.0),
                len(self.sim_performance_history),
                len(self.real_performance_history)
            ]
            self.transfer_metrics_publisher.publish(metrics_msg)
            
            # Publish report
            report_msg = String()
            report_msg.data = report
            self.validation_report_publisher.publish(report_msg)
            
            # Log if transfer is effective or needs attention
            if effectiveness.get('overall_score', 0.0) > self.success_threshold:
                self.get_logger().info(f'✓ Transfer Effective: {effectiveness["overall_score"]:.3f}')
            else:
                self.get_logger().warn(f'✗ Transfer Needs Attention: {effectiveness["overall_score"]:.3f}')
                self.get_logger().info(f'Details: {report}')
                
        except Exception as e:
            self.get_logger().error(f'Error in validation cycle: {str(e)}')

    def calculate_recent_metrics(self, domain):
        """Calculate metrics from recent performance history"""
        history = self.sim_performance_history if domain == 'simulation' else self.real_performance_history
        
        # Get data from the last performance_window seconds
        current_time = self.get_clock().now().nanoseconds / 1e9
        recent_data = [
            entry for entry in history 
            if (current_time - entry['timestamp'].nanoseconds/1e9) < self.performance_window
        ]
        
        if not recent_data:
            return {'empty': True}
        
        # Calculate stability metrics from IMU data
        imu_entries = [d for d in recent_data if d['type'] == 'imu']
        if imu_entries:
            stability_metrics = self.calculate_stability_metrics(imu_entries)
        else:
            stability_metrics = {'average_angular_velocity': 0.0, 'orientation_variance': 0.0}
        
        # Calculate motion metrics from joint data
        joint_entries = [d for d in recent_data if d['type'] == 'joint_state']
        if joint_entries:
            motion_metrics = self.calculate_motion_metrics(joint_entries)
        else:
            motion_metrics = {'average_velocity': 0.0, 'smoothness': 1.0}
        
        return {
            'stability': stability_metrics,
            'motion': motion_metrics,
            'sensor_count': len([d for d in recent_data if d['type'] == 'scan']),
            'data_points': len(recent_data)
        }

    def calculate_stability_metrics(self, imu_entries):
        """Calculate stability metrics from IMU data"""
        angular_velocities = []
        orientations = []
        
        for entry in imu_entries:
            msg = entry['message']
            # Calculate angular velocity magnitude
            vel_mag = math.sqrt(
                msg.angular_velocity.x**2 + 
                msg.angular_velocity.y**2 + 
                msg.angular_velocity.z**2
            )
            angular_velocities.append(vel_mag)
            
            # Store orientation quaternion
            orientations.append([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])
        
        # Calculate average angular velocity (lower is more stable)
        avg_angular_velocity = np.mean(angular_velocities) if angular_velocities else 0.0
        
        # Calculate orientation variance (lower is more stable)
        if len(orientations) > 1:
            orientations_array = np.array(orientations)
            orientation_variance = np.var(orientations_array, axis=0).sum()
        else:
            orientation_variance = 0.0
        
        return {
            'average_angular_velocity': avg_angular_velocity,
            'orientation_variance': orientation_variance
        }

    def calculate_motion_metrics(self, joint_entries):
        """Calculate motion metrics from joint state data"""
        if not joint_entries:
            return {'average_velocity': 0.0, 'smoothness': 1.0}
        
        # Calculate average joint velocity
        all_velocities = []
        for entry in joint_entries:
            msg = entry['message']
            for vel in msg.velocity:
                all_velocities.append(abs(vel))
        
        avg_velocity = np.mean(all_velocities) if all_velocities else 0.0
        
        # Calculate smoothness (simplified as inverse of velocity variance)
        if len(all_velocities) > 1:
            velocity_variance = np.var(all_velocities)
            smoothness = max(0.0, 1.0 - velocity_variance)  # Scale to [0,1]
        else:
            smoothness = 1.0
        
        return {
            'average_velocity': avg_velocity,
            'smoothness': smoothness
        }

    def calculate_transfer_effectiveness(self, sim_metrics, real_metrics):
        """Calculate how effective the transfer is between simulation and reality"""
        if sim_metrics.get('empty', False) or real_metrics.get('empty', False):
            return {
                'stability_similarity': 0.0,
                'motion_similarity': 0.0,
                'sensor_similarity': 0.0,
                'overall_score': 0.0
            }
        
        # Calculate similarity scores
        # For stability: compare angular velocities (lower is better)
        stability_sim = sim_metrics['stability']['average_angular_velocity']
        stability_real = real_metrics['stability']['average_angular_velocity']
        
        # Convert to similarity (higher = more similar)
        stability_similarity = 1.0 - abs(stability_sim - stability_real) / max(stability_sim, stability_real, 0.1)
        stability_similarity = max(0.0, min(1.0, stability_similarity))
        
        # For motion: compare smoothness and velocities
        motion_sim = sim_metrics['motion']['smoothness']
        motion_real = real_metrics['motion']['smoothness']
        
        motion_similarity = 1.0 - abs(motion_sim - motion_real)
        motion_similarity = max(0.0, min(1.0, motion_similarity))
        
        # For sensor similarity: compare observation consistency
        sensor_similarity = min(
            sim_metrics['sensor_count'], 
            real_metrics['sensor_count']
        ) / max(
            sim_metrics['sensor_count'], 
            real_metrics['sensor_count'], 
            1
        )
        
        # Calculate overall score with weighted components
        overall_score = (
            0.4 * stability_similarity + 
            0.4 * motion_similarity + 
            0.2 * sensor_similarity
        )
        
        return {
            'stability_similarity': stability_similarity,
            'motion_similarity': motion_similarity,
            'sensor_similarity': sensor_similarity,
            'overall_score': overall_score
        }

    def generate_validation_report(self, sim_metrics, real_metrics, effectiveness):
        """Generate human-readable validation report"""
        if sim_metrics.get('empty', False) or real_metrics.get('empty', False):
            return "INSUFFICIENT_DATA_FOR_VALIDATION"
        
        report_parts = []
        report_parts.append("=== SIM-TO-REAL TRANSFER VALIDATION ===")
        report_parts.append(f"Time Window: Last {self.performance_window}s")
        report_parts.append(f"Sim Data Points: {sim_metrics['data_points']}")
        report_parts.append(f"Real Data Points: {real_metrics['data_points']}")
        report_parts.append("")
        
        # Stability comparison
        report_parts.append("STABILITY COMPARISON:")
        report_parts.append(f"  Simulation: {sim_metrics['stability']['average_angular_velocity']:.3f} rad/s")
        report_parts.append(f"  Reality:    {real_metrics['stability']['average_angular_velocity']:.3f} rad/s")
        report_parts.append(f"  Similarity: {effectiveness['stability_similarity']:.3f}")
        report_parts.append("")
        
        # Motion comparison
        report_parts.append("MOTION COMPARISON:")
        report_parts.append(f"  Simulation Smoothness: {sim_metrics['motion']['smoothness']:.3f}")
        report_parts.append(f"  Reality Smoothness:    {real_metrics['motion']['smoothness']:.3f}")
        report_parts.append(f"  Similarity: {effectiveness['motion_similarity']:.3f}")
        report_parts.append("")
        
        # Overall effectiveness
        report_parts.append("TRANSFER EFFECTIVENESS:")
        report_parts.append(f"  Overall Score: {effectiveness['overall_score']:.3f}")
        
        # Interpretation
        if effectiveness['overall_score'] > 0.8:
            interpretation = "EXCELLENT - Transfer is working very well"
        elif effectiveness['overall_score'] > 0.6:
            interpretation = "GOOD - Transfer is working reasonably well"
        elif effectiveness['overall_score'] > 0.4:
            interpretation = "FAIR - Transfer has some issues but is functional"
        else:
            interpretation = "POOR - Significant transfer issues detected"
        
        report_parts.append(f"  Interpretation: {interpretation}")
        
        return "\n".join(report_parts)

    def destroy_node(self):
        """Cleanup before node destruction"""
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

## Part 6: Final Configuration and Testing

### Step 6.1: Update Package Configuration

Update the package setup to include all scripts:

```python
# In setup.py, update the entry points:
entry_points={
    'console_scripts': [
        'sensor_validator = humanoid_simple_robot.scripts.sensor_validator:main',
        'laser_processor = humanoid_simple_robot.scripts.laser_processor:main',
        'perception_integrator = humanoid_simple_robot.scripts.perception_integrator:main',
        'ai_planner = humanoid_simple_robot.scripts.ai_planner:main',
        'robust_controller_impl = humanoid_simple_robot.scripts.robust_controller_impl:main',
        'domain_randomizer_impl = humanoid_simple_robot.scripts.domain_randomizer_impl:main',
        'system_identification_impl = humanoid_simple_robot.scripts.system_identification_impl:main',
        'domain_adaptation_impl = humanoid_simple_robot.scripts.domain_adaptation_impl:main',
        'transfer_validator = humanoid_simple_robot.scripts.transfer_validator:main',
    ],
},
```

### Step 6.2: Build and Test the Complete System

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS 2
source /opt/ros/iron/setup.bash

# Build the package
colcon build --packages-select humanoid_simple_robot

# Source the workspace
source install/setup.bash

# Launch the complete sim-to-real transfer system
ros2 launch humanoid_simple_robot sim2real_transfer.launch.py
```

In a separate terminal, monitor the system:

```bash
# Check all nodes are running
ros2 lifecycle list

# Monitor transfer validation
ros2 topic echo /transfer_validation_report

# Monitor adaptation status
ros2 topic echo /domain_adaptation_status

# Check controller status
ros2 topic echo /robust_controller_status
```

## Performance Optimization

### 1. Optimized Parameter Tuning

Create an optimization script to help tune parameters for sim-to-real transfer:

```bash
# optimization_script.sh - Helper script for parameter tuning
#!/bin/bash

echo "Starting Sim-to-Real Transfer Parameter Optimization..."

# Test different noise levels
for noise_level in 0.05 0.1 0.15 0.2; do
    echo "Testing with sensor noise factor: $noise_level"
    
    # Set parameter and run for a while
    ros2 param set /domain_randomizer_impl sensor_noise_variance $noise_level
    
    sleep 30  # Run for 30 seconds
done

echo "Optimization complete. Check /tmp/transfer_validation_logs for results."
```

## Best Practices for Sim-to-Real Transfer Implementation

1. **Gradual Complexity**: Start with simple behaviors and gradually increase complexity
2. **Validation First**: Always validate that simulation matches expected behavior before transfer
3. **Parameter Sensitivity**: Test how sensitive your system is to parameter changes
4. **Monitoring**: Implement comprehensive monitoring to detect transfer failures
5. **Fallbacks**: Have robust fallback behaviors when transfer doesn't work as expected

## Next Steps

With the sim-to-real transfer strategies fully implemented, you're now ready to explore Jetson deployment workflows. The techniques you've implemented provide the foundation for deploying your simulation-tested humanoid behaviors to real hardware, bridging the gap between safe simulation development and real-world deployment.