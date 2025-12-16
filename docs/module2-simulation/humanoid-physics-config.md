# Configuring Physics Properties for Humanoid Robots in Gazebo

## Overview

In this section, we'll focus specifically on configuring physics properties tailored to humanoid robots. Humanoid robots present unique challenges in simulation due to their complex joint structure, relatively high center of gravity, and the need for dynamic balance during locomotion.

## Physics Challenges for Humanoid Robots

### 1. Balance and Stability

Humanoid robots are inherently unstable due to their design:
- High center of gravity compared to support base
- Narrow support polygon during walking
- Complex multi-link dynamics that can lead to chaotic behavior

### 2. Contact Dynamics

Humanoid robots have several unique contact requirements:
- Stable foot-ground contact during walking
- Smooth transitions between single and double support phases
- Proper friction properties for realistic walking behavior

## Inertial Properties Configuration

### 1. Realistic Mass Distribution

For a humanoid robot similar to our simple model:

```xml
<!-- Torso (main body) -->
<link name="base_link">
  <inertial>
    <!-- Higher mass for the torso which contains the main computing units -->
    <mass value="8.0"/>
    <!-- Inertia values based on box approximation -->
    <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.12" izz="0.15"/>
  </inertial>
</link>

<!-- Head -->
<link name="head">
  <inertial>
    <!-- Lower mass for head -->
    <mass value="0.8"/>
    <inertia ixx="0.0013" ixy="0.0" ixz="0.0" iyy="0.0013" izz="0.0013"/>
  </inertial>
</link>

<!-- Limbs -->
<link name="left_knee">
  <inertial>
    <!-- Medium mass for leg segments -->
    <mass value="1.5"/>
    <inertia ixx="0.006" ixy="0.0" ixz="0.0" iyy="0.006" izz="0.001"/>
  </inertial>
</link>

<link name="left_shoulder">
  <inertial>
    <!-- Lower mass for arm segments -->
    <mass value="0.7"/>
    <inertia ixx="0.0006" ixy="0.0" ixz="0.0" iyy="0.0006" izz="0.0001"/>
  </inertial>
</link>
```

### 2. Adjusting for Realism

For more realistic humanoid simulation:

```xml
<!-- Total robot mass should be in the range of 50-100kg for human-sized robots -->
<!-- For our simplified model, we'll use lighter values but keep proportions -->

<!-- Torso: should contain most of the mass -->
<inertial>
  <mass value="6.0"/>
  <!-- Higher moment of inertia around lateral axes due to width -->
  <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.15" izz="0.2"/>
</inertial>

<!-- Legs: significant mass, especially upper legs -->
<link name="left_hip">
  <inertial>
    <mass value="2.0"/>
    <!-- Approximate as cylinder -->
    <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" izz="0.001"/>
  </inertial>
</link>

<link name="left_knee">
  <inertial>
    <mass value="1.8"/>
    <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" izz="0.001"/>
  </inertial>
</link>

<!-- Feet: need appropriate mass for realistic dynamics -->
<link name="left_ankle">
  <inertial>
    <mass value="0.8"/>
    <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.003" izz="0.002"/>
  </inertial>
</link>
```

## Joint Configuration for Humanoid Locomotion

### 1. Hip Joint Configuration

```xml
<!-- Hip joint with appropriate limits for human-like movement -->
<joint name="left_hip_joint_pitch" type="revolute">
  <parent link="base_link"/>
  <child link="left_hip"/>
  <origin xyz="-0.1 0 -0.25" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <!-- Human hip can move forward/back about 45 degrees -->
  <limit lower="-0.78" upper="0.78" effort="200.0" velocity="5.0"/>
  <!-- Higher damping to simulate muscle resistance -->
  <dynamics damping="3.0" friction="0.2"/>
</joint>

<joint name="left_hip_joint_roll" type="revolute">
  <parent link="base_link"/>
  <child link="left_hip"/>
  <origin xyz="-0.1 0 -0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <!-- Hip abduction/adduction limits -->
  <limit lower="-0.5" upper="0.5" effort="150.0" velocity="3.0"/>
  <dynamics damping="2.0" friction="0.15"/>
</joint>

<joint name="left_hip_joint_yaw" type="revolute">
  <parent link="base_link"/>
  <child link="left_hip"/>
  <origin xyz="-0.1 0 -0.25" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <!-- Hip rotation limits -->
  <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
  <dynamics damping="1.5" friction="0.1"/>
</joint>
```

### 2. Knee Joint Configuration

```xml
<joint name="left_knee_joint" type="revolute">
  <parent link="left_hip"/>
  <child link="left_knee"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <!-- Human knee only flexes forward, no extension beyond straight -->
  <limit lower="0" upper="2.3" effort="250.0" velocity="6.0"/>
  <!-- High damping to simulate muscle control -->
  <dynamics damping="5.0" friction="0.3"/>
</joint>
```

### 3. Ankle Joint Configuration

```xml
<joint name="left_ankle_joint_pitch" type="revolute">
  <parent link="left_knee"/>
  <child link="left_ankle"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <!-- Ankle pitch (dorsiflexion/plantarflexion) -->
  <limit lower="-0.6" upper="0.8" effort="100.0" velocity="4.0"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>

<joint name="left_ankle_joint_roll" type="revolute">
  <parent link="left_knee"/>
  <child link="left_ankle"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <!-- Ankle roll (inversion/eversion) -->
  <limit lower="-0.4" upper="0.4" effort="75.0" velocity="3.0"/>
  <dynamics damping="0.8" friction="0.08"/>
</joint>
```

## Ground Contact Configuration

### 1. Foot-Ground Interaction

For stable humanoid locomotion, foot-ground contact is crucial:

```xml
<link name="left_ankle">
  <collision name="left_foot_collision">
    <origin xyz="0.075 0 -0.025" rpy="0 0 0"/>
    <geometry>
      <!-- Box slightly larger than foot for stable contact -->
      <box size="0.15 0.08 0.05"/>
    </geometry>
    <surface>
      <friction>
        <ode>
          <!-- High friction for stable walking -->
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <!-- Soft CFM for stable contact -->
          <soft_cfm>0.0001</soft_cfm>
          <!-- ERP for error correction -->
          <erp>0.2</erp>
          <!-- Maximum contact velocity -->
          <max_vel>100.0</max_vel>
          <!-- Minimum depth for contact detection -->
          <min_depth>0.002</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### 2. Ground Properties

In your world file, make sure the ground plane has appropriate properties:

```xml
<world>
  <include>
    <uri>model://ground_plane</uri>
    <pose>0 0 0 0 0 0</pose>
  </include>
  
  <model name='ground_plane'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <surface>
          <friction>
            <ode>
              <!-- High friction coefficient for stable humanoid walking -->
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
  </model>
</world>
```

## Center of Mass Optimization

### 1. Calculating Total Center of Mass

To maintain stability, the center of mass should be within the support polygon:

```python
# Example calculation in a ROS2 node
import numpy as np
from geometry_msgs.msg import Point

def calculate_center_of_mass(self):
    """Calculate the center of mass of the robot based on link poses and masses"""
    total_mass = 0.0
    weighted_pos = np.array([0.0, 0.0, 0.0])
    
    # This would typically access the TF transform tree and mass parameters
    for link_name, mass in self.link_masses.items():
        # Get pose from TF
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', link_name, rclpy.time.Time())
            pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            weighted_pos += mass * pos
            total_mass += mass
        except Exception as e:
            self.get_logger().warn(f'Could not get transform for {link_name}: {e}')
    
    if total_mass > 0:
        center_of_mass = weighted_pos / total_mass
        return Point(x=center_of_mass[0], y=center_of_mass[1], z=center_of_mass[2])
    else:
        return Point(x=0.0, y=0.0, z=0.0)
```

### 2. Keeping COM Low

For humanoid stability:

- Keep heavy components (batteries, computers) low in the torso
- Balance mass distribution between left and right sides
- Consider adjustable ballast for dynamic balance experiments

## Simulation Parameters for Humanoid Locomotion

### 1. Optimized Physics Settings

```xml
<world name='humanoid_world'>
  <!-- Use appropriate gravity -->
  <gravity>0 0 -9.8</gravity>
  
  <physics type='ode'>
    <!-- For humanoid robots, use smaller time steps for stability -->
    <max_step_size>0.0005</max_step_size>
    <!-- Higher update rate for accurate dynamics -->
    <real_time_update_rate>2000</real_time_update_rate>
    <real_time_factor>1.0</real_time_factor>
    
    <ode>
      <solver>
        <type>quick</type>
        <!-- More iterations for stable contact handling -->
        <iters>500</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <!-- Soft constraint mixing for stable contacts -->
        <cfm>0.0001</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>200.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

### 2. Performance vs Stability Trade-offs

For humanoid robots, you might need to adjust parameters based on your needs:

- **For stability**: Smaller time steps, more iterations, softer constraints
- **For performance**: Larger time steps, fewer iterations, harder constraints
- **For accuracy**: Smaller time steps, more iterations, precise masses

## Controller Integration

### 1. Physics-Aware Controllers

Humanoid controllers must account for physics properties:

```python
class HumanoidBalanceController(Node):
    def __init__(self):
        super().__init__('humanoid_balance_controller')
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )
        self.cmd_pub = self.create_publisher(
            JointTrajectory, 
            '/position_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Control loop
        self.control_timer = self.create_timer(0.01, self.balance_control_loop)
        
    def balance_control_loop(self):
        """Main control loop that considers physical properties"""
        # Calculate desired joint positions based on balance requirements
        # This would implement a controller like PID, LQR, or MPC
        pass
```

### 2. Safety Limits

Always implement safety limits that consider physical constraints:

```xml
<!-- In your controller configuration -->
<rosparam param="joint_limits">
  left_hip_joint:
    has_position_limits: true
    max_position: 0.78
    min_position: -0.78
    has_velocity_limits: true
    max_velocity: 5.0
    has_effort_limits: true
    max_effort: 200.0
</rosparam>
```

## Verification and Tuning

### 1. Testing Balance

```bash
# Launch the humanoid model and observe:
# 1. Does it remain standing without control inputs?
# 2. Does it fall naturally when pushed?
# 3. Do feet make stable contact with the ground?

# Use the following command to apply forces:
gz topic -t /world/humanoid_world/apply_force -m gz.msgs.EntityWrench \
  -p '{entity: {name: "simple_humanoid::base_link"}, wrench: {force: {x: 50.0, y: 0, z: 0}}}'
```

### 2. Parameter Tuning Process

1. Start with conservative damping values
2. Gradually reduce time step if unstable
3. Fine-tune friction and contact parameters for stable walking
4. Verify that the robot behaves predictably under various conditions

## Next Steps

With physics properties properly configured for humanoid robots, you're now ready to implement sensor simulation in Gazebo. This will include LiDAR, camera, and IMU sensors that are essential for humanoid robot perception and navigation tasks.

The physics configuration you've implemented provides the foundation for realistic sensor simulation, as sensors need accurate physics to generate realistic data.