# Environment Design & World-Building in Gazebo

## Overview

This section covers the creation and design of simulation environments for humanoid robots in Gazebo. A well-designed environment is essential for testing robot capabilities, validating algorithms, and providing realistic training scenarios for AI systems.

## Understanding Simulation Environments

### 1. Types of Environments

For humanoid robotics, we typically need several types of environments:

- **Simple Testing Environments**: Basic worlds with flat surfaces for initial validation
- **Navigation Environments**: Complex layouts with obstacles for path planning
- **Manipulation Environments**: Spaces with objects for interaction testing
- **Outdoor Environments**: Terrains with varying elevations and natural obstacles
- **Indoor Environments**: Rooms, corridors, and furniture for domestic robotics

### 2. Key Design Considerations

When designing simulation environments:
- **Realism vs. Performance**: Balance visual fidelity with simulation speed
- **Repeatability**: Design deterministic elements for consistent testing
- **Variability**: Include randomizable elements for robustness testing
- **Scalability**: Create modular components that can be combined

## Basic World Structure

### 1. World File Format

Gazebo uses SDF (Simulation Description Format) for world files:

```xml
<?xml version="1.0" ?>
<sdf version='1.7'>
  <world name="simple_humanoid_world">
    <!-- World properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>
    
    <!-- Plugins can be added here -->
    
    <!-- Models and objects -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Custom models go here -->
  </world>
</sdf>
```

### 2. Essential World Components

Every humanoid robot simulation environment should include:

```xml
<world name="humanoid_test_world">
  <!-- Physics engine configuration -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000.0</real_time_update_rate>
    <gravity>0 0 -9.8</gravity>
  </physics>
  
  <!-- Ground plane -->
  <include>
    <uri>model://ground_plane</uri>
    <pose>0 0 0 0 0 0</pose>
  </include>
  
  <!-- Lighting -->
  <include>
    <uri>model://sun</uri>
  </include>
  
  <!-- Optional: sky -->
  <scene>
    <ambient>0.4 0.4 0.4</ambient>
    <background>0.7 0.7 0.7</background>
    <shadows>true</shadows>
  </scene>
</world>
```

## Creating Custom Models

### 1. Simple Obstacle Models

Create basic geometric obstacles for navigation testing:

```xml
<!-- Cube obstacle -->
<model name="cube_obstacle_1">
  <pose>-2 0 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>1 1 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1 1 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
        <diffuse>0.8 0.2 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>100</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
</model>

<!-- Cylindrical pillar -->
<model name="cylinder_pillar">
  <pose>2 0 1.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.5</radius>
          <length>3</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.5</radius>
          <length>3</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>0.2 0.2 0.8 1</ambient>
        <diffuse>0.2 0.2 0.8 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>100</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
</model>
```

### 2. Indoor Environment Elements

Create indoor elements like walls, doors, and furniture:

```xml
<!-- Wall segment -->
<model name="wall_segment">
  <pose>0 5 1 0 0 0</pose>
  <static>true</static>  <!-- Static objects don't move -->
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>5 0.2 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>5 0.2 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Door frame (static) -->
<model name="door_frame">
  <pose>3 0 1.5 0 0 0</pose>
  <static>true</static>
  <link name="frame">
    <collision name="collision">
      <geometry>
        <box>
          <size>3 0.1 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>3 0.1 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.6 0.4 0.2 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Table -->
<model name="table">
  <pose>-3 -3 0.75 0 0 0</pose>
  <link name="base">
    <collision name="collision">
      <geometry>
        <box>
          <size>1.5 0.8 0.7</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1.5 0.8 0.7</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.2 0.1 1</ambient>
        <diffuse>0.4 0.2 0.1 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>20</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
</model>
```

## Advanced Environment Features

### 1. Terrain with Elevation Changes

Create outdoor environments with varying terrain:

```xml
<!-- Sloped terrain -->
<model name="slope_terrain">
  <pose>0 0 0 0 0.3 0</pose>  <!-- 0.3 radians ~ 17 degrees slope -->
  <static>true</static>
  <link name="terrain">
    <collision name="collision">
      <geometry>
        <box>
          <size>10 10 0.5</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>10 10 0.5</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Stairs -->
<model name="stairs">
  <pose>5 0 0 0 0 0</pose>
  <static>true</static>
  <link name="step1">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.4 0.4 1</ambient>
        <diffuse>0.4 0.4 0.4 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="step2">
    <pose>0 0 0.2 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.4 0.4 1</ambient>
        <diffuse>0.4 0.4 0.4 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="step3">
    <pose>0 0 0.4 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 1 0.2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.4 0.4 0.4 1</ambient>
        <diffuse>0.4 0.4 0.4 1</diffuse>
      </material>
    </visual>
  </link>
  <!-- Connect rigidly -->
  <joint name="step2_joint" type="fixed">
    <parent>step1</parent>
    <child>step2</child>
  </joint>
  <joint name="step3_joint" type="fixed">
    <parent>step2</parent>
    <child>step3</child>
  </joint>
</model>
```

### 2. Interactive Objects

Add objects that the humanoid robot can interact with:

```xml
<!-- Movable box -->
<model name="movable_box">
  <pose>-4 2 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.5 0.5 0.5</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.5 0.5 0.5</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.6 0.2 1</ambient>
        <diffuse>0.8 0.6 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>5</mass>  <!-- Light enough to move -->
      <inertia>
        <ixx>0.1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.1</iyy>
        <iyz>0</iyz>
        <izz>0.1</izz>
      </inertia>
    </inertial>
  </link>
</model>

<!-- Door that can be opened -->
<model name="simple_door">
  <pose>3 1 1 0 0 0</pose>
  <link name="door_frame">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.1 2 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.1 2 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.6 0.4 0.2 1</ambient>
        <diffuse>0.6 0.4 0.2 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="door_panel">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 0.05 2</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 0.05 2</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.6 0.4 1</ambient>
        <diffuse>0.8 0.6 0.4 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>10</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
  <joint name="door_hinge" type="revolute">
    <parent>door_frame</parent>
    <child>door_panel</child>
    <axis>
      <xyz>0 1 0</xyz>
      <limit>
        <lower>-1.57</lower>  <!-- -90 degrees -->
        <upper>0.0</upper>     <!-- 0 degrees (closed) -->
      </limit>
    </axis>
  </joint>
</model>
```

## Modular Environment Design

### 1. Creating Reusable Environment Components

Design components that can be easily combined:

```xml
<!-- Corridor segment -->
<model name="corridor_segment" name_in_namespace="corridor">
  <pose>0 0 0 0 0 0</pose>
  <static>true</static>
  <link name="floor">
    <collision name="collision">
      <geometry>
        <box>
          <size>10 2 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>10 2 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="left_wall">
    <pose>0 -1.1 0.5 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>10 0.2 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>10 0.2 1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
  <link name="right_wall">
    <pose>0 1.1 0.5 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <box>
          <size>10 0.2 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>10 0.2 1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>
  </link>
</model>

<!-- Room segment -->
<model name="room_segment" name_in_namespace="room">
  <pose>0 0 0 0 0 0</pose>
  <static>true</static>
  <!-- Similar to corridor but enclosed on all sides -->
  <!-- [Definition similar to corridor but with 4 walls] -->
</model>
```

### 2. Parameterized Environments

Use parameters to create variations:

```xml
<!-- Template for room with variable size -->
<sdf version='1.7'>
  <model name='parametrized_room'>
    <!-- Room dimensions as parameters -->
    <pose>{{ X_POSITION }} {{ Y_POSITION }} {{ Z_POSITION }} 0 0 0</pose>
    <static>true</static>
    
    <link name="floor">
      <collision name="collision">
        <geometry>
          <box>
            <size>{{ WIDTH }} {{ DEPTH }} 0.1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{{ WIDTH }} {{ DEPTH }} 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>{{ FLOOR_COLOR_R }} {{ FLOOR_COLOR_G }} {{ FLOOR_COLOR_B }} 1</ambient>
          <diffuse>{{ FLOOR_COLOR_R }} {{ FLOOR_COLOR_G }} {{ FLOOR_COLOR_B }} 1</diffuse>
        </material>
      </visual>
    </link>
    
    <!-- Walls would be similarly parameterized -->
    <!-- [Walls definition] -->
  </model>
</sdf>
```

## Using SDF Macros for Complex Environments

### 1. Creating SDF Macros

For complex recurring structures, use SDF macros:

```xml
<!-- In a separate file: macro_library.sdf -->
<sdf version='1.7'>
  <model name="office_furniture_set">
    <include>
      <uri>model://desk</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
    <include>
      <uri>model://chair</uri>
      <pose>0.5 -0.5 0 0 0 1.57</pose>
    </include>
    <include>
      <uri>model://bookshelf</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>
  </model>
</sdf>
```

### 2. Including Models in Worlds

```xml
<!-- In your world file -->
<world name="office_world">
  <!-- Physics, lighting, etc. -->
  
  <!-- Include pre-defined furniture set -->
  <include>
    <uri>model://office_furniture_set</uri>
    <pose>0 0 0 0 0 0</pose>
  </include>
  
  <!-- Add humanoid robot -->
  <include>
    <uri>model://simple_humanoid</uri>
    <pose>1 1 1 0 0 0</pose>
  </include>
</world>
```

## Advanced Environment Features

### 1. Dynamic Elements

Add elements that change during simulation:

```xml
<!-- Moving platform -->
<model name="moving_platform">
  <pose>0 0 1 0 0 0</pose>
  <link name="platform">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.2 0.8 0.2 1</ambient>
        <diffuse>0.2 0.8 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>50</mass>
      <inertia>
        <ixx>5</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>5</iyy>
        <iyz>0</iyz>
        <izz>10</izz>
      </inertia>
    </inertial>
  </link>
  
  <!-- Motor to move the platform -->
  <plugin name="platform_motor" filename="libgazebo_ros_p3d.so">
    <alwaysOn>true</alwaysOn>
    <updateRate>100</updateRate>
    <bodyName>platform</bodyName>
    <topicName>platform/odom</topicName>
    <gaussianNoise>0.0</gaussianNoise>
    <frameName>world</frameName>
    <xyzOffsets>0 0 0</xyzOffsets>
    <rpyOffsets>0 0 0</rpyOffsets>
  </plugin>
</model>
```

### 2. Environmental Effects

Add environmental conditions:

```xml
<!-- Add weather effects as plugins -->
<world name="outdoor_humanoid_world">
  <!-- Physics, lighting, etc. -->
  
  <!-- Wind plugin -->
  <plugin filename="libgazebo_ros_wind.so" name="gazebo_ros_wind">
    <alwaysOn>true</alwaysOn>
    <updateRate>10</updateRate>
    <windDirection>1 0 0</windDirection>
    <windForce>0.5 0 0</windForce>
    <windMean>0.5</windMean>
  </plugin>
  
  <!-- Add objects and robot -->
</world>
```

## Performance Optimization

### 1. Level of Detail (LOD)

Balance visual quality with performance:

```xml
<!-- For performance-critical simulations, use simpler models -->
<model name="simple_cube">
  <pose>-2 0 1 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>  <!-- Use box instead of complex mesh -->
          <size>1 1 1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
        <diffuse>0.8 0.2 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>10</mass>
      <inertia>
        <ixx>1</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>1</iyy>
        <iyz>0</iyz>
        <izz>1</izz>
      </inertia>
    </inertial>
  </link>
</model>
```

### 2. Spatial Organization

Organize elements to improve rendering performance:

```xml
<!-- Group objects by type and location -->
<world name="organized_humanoid_world">
  <!-- Physics, lighting -->
  
  <!-- Static obstacles close to robot start position -->
  <model name="obstacle_1">
    <!-- [Definition] -->
  </model>
  <model name="obstacle_2">
    <!-- [Definition] -->
  </model>
  
  <!-- Furniture and decorative elements further away -->
  <model name="decor_1">
    <!-- [Definition] -->
  </model>
  
  <!-- Large static elements like walls -->
  <model name="wall_1">
    <static>true</static>
    <!-- [Definition] -->
  </model>
</world>
```

## Testing and Validation

### 1. Environment Validation

Verify that environments work as expected:

```bash
# Test a world file
gz sim -r -v 4 simple_humanoid_world.sdf

# Check if all models loaded properly
gz model -m simple_humanoid

# Verify physics properties
gz topic -e -t /world/humanoid_world/physics
```

### 2. Robot-Environment Interaction Testing

```bash
# Test navigation in the environment
ros2 run nav2_test nav2_tester --world simple_humanoid_world.sdf

# Test interaction with objects
ros2 run manipulation_test manipulator_tester --world interactive_world.sdf
```

## Best Practices

### 1. Reusability

- Create modular components that can be combined
- Use common coordinate systems and units
- Document environmental parameters for reproducibility

### 2. Realism vs. Performance

- Balance visual and physical fidelity appropriately
- Use simpler collision models where visual detail isn't needed
- Consider the computational requirements of your target platform

### 3. Extensibility

- Design environments that can be easily modified
- Use common patterns for different environment types
- Include documentation for environment parameters

## Next Steps

With a solid understanding of environment design and world-building, you'll next learn how to create custom simulation environments specifically for humanoid robotics tasks. This will include more complex scenarios that test the robot's capabilities in realistic settings.

The environment design skills you've learned form the foundation for creating challenging and realistic test scenarios for your humanoid robot.