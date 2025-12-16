# Physics Simulation: Gravity, Collisions, and Joints

## Overview

Physics simulation is the backbone of realistic robot simulation in Gazebo. In this section, we'll explore how to configure and fine-tune gravity, collisions, and joints for humanoid robots, ensuring they behave realistically in the simulation environment.

## Understanding Gazebo Physics

### 1. Physics Engines

Gazebo supports several physics engines:
- **ODE (Open Dynamics Engine)**: Default, good balance of speed and stability
- **Bullet**: Good for complex interactions and constraints
- **DART**: Advanced for articulated figures and contacts
- **SIMBODY**: For complex biomechanics (less common in robotics)

### 2. Key Physics Concepts

- **World Gravity**: The gravitational acceleration applied to all objects
- **Collision Detection**: How objects interact when they come into contact
- **Joint Dynamics**: How connected parts move relative to each other
- **Friction**: Resistance to sliding and rolling motion
- **Bounce**: Elasticity of collisions

## Configuring World Gravity

### 1. Setting Global Gravity

In your world file, set the global gravity:

```xml
<sdf version='1.7'>
  <world name='humanoid_world'>
    <!-- Set gravity - default is -9.8 m/s^2 in Z direction -->
    <gravity>0 0 -9.8</gravity>
    
    <!-- Physics engine configuration -->
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>
    
    <!-- Include ground plane and other elements -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### 2. Adjusting for Different Environments

For different gravity environments:

```xml
<!-- On the Moon (1/6 of Earth's gravity) -->
<gravity>0 0 -1.63</gravity>

<!-- Zero gravity for space robotics -->
<gravity>0 0 0</gravity>

<!-- Adjusted for testing (to make physics errors more apparent) -->
<gravity>0 0 -19.6</gravity>  <!-- 2x Earth's gravity -->
```

## Collision Properties

### 1. Collision Detection Parameters

Collision properties in your URDF should be optimized for humanoid robots:

```xml
<link name="left_foot">
  <collision>
    <geometry>
      <!-- Use box for simple foot collision -->
      <box size="0.15 0.08 0.05"/>
    </geometry>
    <!-- Collision properties -->
    <surface>
      <friction>
        <ode>
          <!-- Static friction coefficient -->
          <mu>0.8</mu>
          <!-- Dynamic friction coefficient -->
          <mu2>0.8</mu2>
          <!-- FDIR1 for anisotropic friction (advanced use) -->
          <fdir1>0 0 0</fdir1>
        </ode>
      </friction>
      <contact>
        <ode>
          <!-- Soft CFM (Constraint Force Mixing) for stable contacts -->
          <soft_cfm>0.001</soft_cfm>
          <!-- Error Reduction Parameter -->
          <erp>0.2</erp>
          <!-- Maximum contacts per collision -->
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
      <bounce>
        <!-- Coefficient of restitution (0 = no bounce, 1 = perfectly elastic) -->
        <restitution_coefficient>0.01</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
    </surface>
  </collision>
</link>
```

### 2. Collision Mesh Optimization

For humanoid robots, optimize collision geometry:

- **Feet**: Use box or simplified shape for stable contact
- **Limbs**: Use cylinders for good contact properties and performance
- **Torso**: Use box or capsule for simple, stable contact
- **Avoid**: Complex mesh collisions for real-time simulation

### 3. Self-Collision Prevention

In your URDF, specify which links should not collide:

```xml
<!-- Prevent adjacent links from colliding -->
<collision_checking>
  <self_collide>false</self_collide>
</collision_checking>

<!-- Or specify collisions to ignore in the model -->
<gazebo>
  <self_collide>false</self_collide>
  <enable_wind>false</enable_wind>
  <kinematic>false</kinematic>
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
  <fdir1>0 0 0</fdir1>
</gazebo>
```

## Joint Configuration

### 1. Joint Types for Humanoid Robots

Humanoid robots require specific joint configurations:

```xml
<!-- Hip joint - typically revolute with limited range -->
<joint name="left_hip_joint" type="revolute">
  <parent link="base_link"/>
  <child link="left_hip"/>
  <origin xyz="-0.1 0 -0.25" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Rotate around X-axis -->
  <limit lower="-0.5" upper="0.5" effort="100.0" velocity="2.0"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>

<!-- Knee joint - typically revolute with specific limits -->
<joint name="left_knee_joint" type="revolute">
  <parent link="left_hip"/>
  <child link="left_knee"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Rotate around X-axis (knee flexion) -->
  <limit lower="0" upper="2.5" effort="100.0" velocity="2.0"/>  <!-- Only flex forward
  <dynamics damping="2.0" friction="0.1"/>
</joint>

<!-- Ankle joint - may need multiple DOF -->
<joint name="left_ankle_joint" type="revolute">
  <parent link="left_knee"/>
  <child link="left_ankle"/>
  <origin xyz="0 0 -0.15" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Pitch axis -->
  <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.5"/>
  <dynamics damping="0.5" friction="0.05"/>
</joint>
```

### 2. Joint Dynamics Parameters

Fine-tune joint dynamics for realistic humanoid movement:

- **Damping**: Resistance to motion (simulates internal friction)
- **Friction**: Static friction that must be overcome to start motion
- **Effort limits**: Maximum force/torque the joint can apply
- **Velocity limits**: Maximum speed the joint can move

### 3. Complex Joint Configurations

For more complex humanoid joints:

```xml
<!-- Ball joint for shoulder (simplified as 3 revolute joints) -->
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="base_link"/>
  <child link="left_shoulder"/>
  <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Rotate around Y-axis -->
  <limit lower="-1.57" upper="1.57" effort="50.0" velocity="3.0"/>
  <dynamics damping="0.5" friction="0.05"/>
</joint>

<joint name="left_shoulder_roll" type="revolute">
  <parent link="left_shoulder"/>
  <child link="left_upper_arm"/>
  <origin xyz="0 0 -0.1" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Rotate around X-axis -->
  <limit lower="-2.0" upper="1.0" effort="40.0" velocity="3.0"/>
  <dynamics damping="0.4" friction="0.05"/>
</joint>
```

## Physics Tuning for Humanoid Stability

### 1. Center of Mass Considerations

For stable humanoid simulation:

- Keep the center of mass low (in the torso area)
- Ensure the center of mass stays within the support polygon during movement
- Use realistic mass distribution

### 2. Time Step and Real-Time Factor

Optimize simulation parameters for humanoid robots:

```xml
<physics type='ode'>
  <!-- Smaller time steps for more accurate physics -->
  <max_step_size>0.001</max_step_size>
  <!-- Update physics at 1000 Hz for humanoid applications -->
  <real_time_update_rate>1000</real_time_update_rate>
  <!-- Target real-time factor of 1 (can be less if performance is poor) -->
  <real_time_factor>1.0</real_time_factor>
</physics>
```

### 3. Contact Parameters for Bipedal Locomotion

For humanoid robots, especially for walking:

```xml
<world>
  <physics type='ode'>
    <!-- More accurate but slower contact model -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>100</iters>  <!-- More iterations for stable contacts -->
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0001</cfm>  <!-- Soft constraint force mixing for stable contacts -->
        <erp>0.2</erp>    <!-- Error reduction parameter -->
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>  <!-- Contact layer -->
      </constraints>
    </ode>
  </physics>
</world>
```

## Troubleshooting Physics Issues

### 1. Robot Failing to Balance

Causes and solutions:
- Mass too high/low: Adjust inertial properties
- Center of mass too high: Redistribute mass lower in the robot
- Joint damping too low: Increase damping values
- Time step too large: Reduce max_step_size

### 2. Jittery Movement

- Increase physics solver iterations
- Adjust CFM and ERP parameters
- Ensure collision geometry is properly defined
- Reduce time step if necessary

### 3. Robot Sinking Through Ground

- Check collision geometry in links
- Ensure proper mass and inertia values
- Adjust contact parameters (CFM, ERP, min_depth)

## Verifying Physics Configuration

### 1. Testing Joint Limits

```bash
# Publish commands to test joint limits
ros2 topic pub /simple_humanoid/left_hip_joint_position_controller/commands std_msgs/Float64MultiArray "data: [10.0]"
```

### 2. Testing Balance

```bash
# Check if robot maintains balance when standing
# In Gazebo GUI, observe if robot remains stable
# Use ROS2 to get robot state:
ros2 topic echo /joint_states
```

### 3. Performance Monitoring

Monitor simulation performance:
- Real Time Factor (RTF) should stay close to 1.0
- Check CPU usage
- Ensure consistent frame rate

## Next Steps

With proper physics configuration in place, you'll next learn to configure physics properties specifically for humanoid robots. This will include fine-tuning parameters for stable walking, running, and other complex movements that humanoid robots perform.

The physics simulation forms the foundation for all dynamic robot behaviors in simulation, making it critical to get right before adding sensors and other complex components.