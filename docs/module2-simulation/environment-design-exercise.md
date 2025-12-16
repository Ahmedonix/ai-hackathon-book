# Practical Hands-On Exercise: Environment Design for Humanoid Robotics

## Exercise Overview

This hands-on exercise will teach you how to design and implement complex simulation environments for humanoid robotics in Gazebo. You'll learn to create both simple and complex environments that challenge your robot's navigation, perception, and manipulation capabilities.

## Learning Objectives

By the end of this exercise, you will be able to:
1. Create custom SDF world files for Gazebo
2. Design indoor and outdoor environments with obstacles
3. Model static and dynamic objects in the environment
4. Implement terrain variations and challenging surfaces
5. Validate that your environment works properly with a humanoid robot

## Prerequisites

Before starting this exercise, you should have:
- Completed the Gazebo setup exercise
- Completed the sensor simulation exercise
- Basic knowledge of XML/SDF format
- A working humanoid robot model with sensors
- Access to a ROS 2 Iron workspace

## Part 1: Understanding Environment Components

### Step 1.1: Basic Environment Structure

First, let's understand the basic structure of a Gazebo world file. Create a simple test environment:

```bash
# Navigate to your workspace
cd ~/humanoid_ws

# Create directories for our world files
mkdir -p src/humanoid_simple_robot/worlds
```

Create `src/humanoid_simple_robot/worlds/basic_environment.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_environment">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.7 0.7 0.7</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Environment objects -->
    
    <!-- 1. Simple obstacle -->
    <model name="wall_1">
      <pose>2 0 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <pose>0 0 1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 2. Another obstacle -->
    <model name="wall_2">
      <pose>0 2 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 3. A simple ramp -->
    <model name="ramp">
      <pose>-2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <pose>0 0 0.5 0 0.2 0</pose>  <!-- 0.2 radians ~ 11.4 degrees slope -->
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.2 1</ambient>
            <diffuse>0.6 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- 4. A movable box (dynamic object) -->
    <model name="movable_box">
      <pose>-3 -1 0.5 0 0 0</pose>
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
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.02</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.02</iyy>
            <iyz>0</iyz>
            <izz>0.02</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- 5. Cylindrical obstacle -->
    <model name="cylinder_obstacle">
      <pose>3 1 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>2</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.3</radius>
              <length>2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

### Step 1.2: Test the Basic Environment

```bash
# Source your workspace
source ~/humanoid_ws/install/setup.bash

# Launch Gazebo with the basic environment
gz sim -v 4 ~/humanoid_ws/src/humanoid_simple_robot/worlds/basic_environment.sdf
```

## Part 2: Creating an Indoor Environment

### Step 2.1: Create a Room Layout

Create a more complex indoor environment with multiple rooms and obstacles:

Create `src/humanoid_simple_robot/worlds/indoor_environment.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_environment">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Artificial lighting -->
    <model name="light_1">
      <pose>-3 0 3 0 0 0</pose>
      <link name="link">
        <light type="point" name="point_light">
          <pose>0 0 0 0 0 0</pose>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
          <attenuation>
            <range>10</range>
            <constant>0.9</constant>
            <linear>0.01</linear>
            <quadratic>0.001</quadratic>
          </attenuation>
          <cast_shadows>true</cast_shadows>
        </light>
      </link>
    </model>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.3 0.3 0.3</ambient>
      <background>0.5 0.5 0.5</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Environment: Indoor rooms -->
    
    <!-- Outer walls -->
    <model name="north_wall">
      <pose>0 4.5 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="south_wall">
      <pose>0 -4.5 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="east_wall">
      <pose>4.5 0 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="west_wall">
      <pose>-4.5 0 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Inner walls creating rooms -->
    <model name="inner_wall_1">
      <pose>0 1 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="inner_wall_2">
      <pose>-2 -2 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Furniture -->
    <model name="table">
      <pose>-2 2 0.75 0 0 0</pose>
      <static>true</static>
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
      </link>
    </model>
    
    <!-- Chair -->
    <model name="chair">
      <pose>-2.5 2.3 0.3 0 0 0</pose>
      <static>true</static>
      <link name="seat">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.5 1</ambient>
            <diffuse>0.3 0.3 0.5 1</diffuse>
          </material>
        </visual>
      </link>
      <link name="back">
        <pose>0 0 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.05 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.05 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.5 1</ambient>
            <diffuse>0.2 0.2 0.5 1</diffuse>
          </material>
        </visual>
      </link>
      <joint name="back_joint" type="fixed">
        <parent>seat</parent>
        <child>back</child>
      </joint>
    </model>
    
    <!-- Objects on table -->
    <model name="cup">
      <pose>-2.2 2.1 1.05 0 0 0</pose>
      <static>false</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.2</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0</iyz>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>
    
    <!-- Doorway gap in wall -->
    <!-- We'll remove a section of inner_wall_1 to create a doorway -->
    <!-- Instead of one continuous wall, we'll have two separate wall segments -->
    <model name="partial_wall_1">
      <pose>-1 1 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="partial_wall_2">
      <pose>1 1 1 0 0 1.57</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

### Step 2.2: Test the Indoor Environment

```bash
# Launch Gazebo with the indoor environment
gz sim -v 4 ~/humanoid_ws/src/humanoid_simple_robot/worlds/indoor_environment.sdf
```

## Part 3: Creating an Outdoor/Challenging Environment

### Step 3.1: Create a Complex Outdoor Environment

Create `src/humanoid_simple_robot/worlds/outdoor_challenge.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="outdoor_challenge">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane with elevation map (we'll create a simple one) -->
    <!-- First, we'll use a flat ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.8 0.8 1.0</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Environment: Outdoor challenge course -->
    
    <!-- Start area -->
    <model name="start_area">
      <pose>-6 0 0.01 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.2 1</ambient>  <!-- Yellow start area -->
            <diffuse>0.8 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Hilly terrain using multiple boxes -->
    <model name="hill_1">
      <pose>-3 0 0.5 0 0.2 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 2 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 2 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.6 0.3 1</ambient>
            <diffuse>0.3 0.6 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Stairs -->
    <model name="stairs">
      <static>true</static>
      
      <!-- Step 1 -->
      <link name="step_1">
        <pose>0 0 0.1 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 2 -->
      <link name="step_2">
        <pose>0 0 0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Step 3 -->
      <link name="step_3">
        <pose>0 0 0.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1.5 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
      
      <!-- Connect steps rigidly -->
      <joint name="step_2_joint" type="fixed">
        <parent>step_1</parent>
        <child>step_2</child>
      </joint>
      
      <joint name="step_3_joint" type="fixed">
        <parent>step_2</parent>
        <child>step_3</child>
      </joint>
    </model>
    
    <!-- Sloped path -->
    <model name="slope_path">
      <pose>3 0 0 0 0.3 0</pose>  <!-- 0.3 radians ~ 17 degrees slope -->
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>3 2 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>3 2 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.7 0.4 1</ambient>
            <diffuse>0.4 0.7 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Gap jump challenge -->
    <model name="platform_1">
      <pose>6 -1 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
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
            <ambient>0.8 0.4 0.2 1</ambient>
            <diffuse>0.8 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="platform_2">
      <pose>7.5 1 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
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
            <ambient>0.8 0.4 0.2 1</ambient>
            <diffuse>0.8 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Slalom poles -->
    <model name="slalom_pole_1">
      <pose>9 2 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="slalom_pole_2">
      <pose>9 0 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <model name="slalom_pole_3">
      <pose>9 -2 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.05</radius>
              <length>2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Final destination area -->
    <model name="finish_area">
      <pose>12 0 0.01 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 2 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>  <!-- Green finish area -->
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

### Step 3.2: Test the Challenge Environment

```bash
# Launch Gazebo with the outdoor challenge environment
gz sim -v 4 ~/humanoid_ws/src/humanoid_simple_robot/worlds/outdoor_challenge.sdf
```

## Part 4: Creating a Launch File for Environments

### Step 4.1: Update the Package with Environment Launches

Create a launch file that can spawn your robot in any of the environments: `src/humanoid_simple_robot/launch/environment_demo.launch.py`

```python
# launch/environment_demo.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    environment = LaunchConfiguration('environment', default='basic_environment')
    
    # Get package share directory
    pkg_share = get_package_share_directory('humanoid_simple_robot')
    
    # Create worlds directory if it doesn't exist
    worlds_dir = os.path.join(pkg_share, 'worlds')
    os.makedirs(worlds_dir, exist_ok=True)
    
    # Construct world file path
    world_file = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('environment')])
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('gazebo_ros'),
            '/launch/empty_world.launch.py'
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'false',
            'gui': 'true'
        }.items()
    )
    
    # Robot description (using the advanced humanoid model from previous exercise)
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('humanoid_simple_robot'),
        'urdf',
        'simple_humanoid.urdf'  # or whichever URDF file you're using
    ])
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_path
        }],
        output='screen'
    )
    
    # Joint State Publisher (for non-controlled joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )
    
    # Spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0', '-y', '0', '-z', '1.0'  # Start 1m above ground
        ],
        output='screen'
    )
    
    # Optionally, include sensor processing nodes from previous exercise
    # Uncomment these if you want full sensor processing
    '''
    sensor_validator = Node(
        package='humanoid_simple_robot',
        executable='sensor_validator',
        name='sensor_validator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    laser_processor = Node(
        package='humanoid_simple_robot',
        executable='laser_processor',
        name='laser_processor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    '''
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=use_sim_time),
        
        DeclareLaunchArgument(
            'environment',
            default_value='basic_environment.sdf',
            description='Choose one of the world files from `/humanoid_simple_robot/worlds`'
        ),
        
        # Launch Gazebo
        gazebo,
        
        # Launch robot state publisher after a delay
        TimerAction(
            period=2.0,
            actions=[robot_state_publisher]
        ),
        
        # Launch joint state publisher
        TimerAction(
            period=3.0,
            actions=[joint_state_publisher]
        ),
        
        # Launch spawn entity after more delay
        TimerAction(
            period=4.0,
            actions=[spawn_entity]
        ),
        
        # Launch sensor processors
        '''
        TimerAction(
            period=5.0,
            actions=[sensor_validator]
        ),
        
        TimerAction(
            period=6.0,
            actions=[laser_processor]
        )
        '''
    ])
```

### Step 4.2: Build and Test Environment Launches

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS
source /opt/ros/iron/setup.bash

# Build the package
colcon build --packages-select humanoid_simple_robot

# Source the workspace
source install/setup.bash

# Test each environment
ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=basic_environment.sdf

# In new terminals, test other environments:
# ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=indoor_environment.sdf
# ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=outdoor_challenge.sdf
```

## Part 5: Creating Environment-Specific Navigation Challenges

### Step 5.1: Create a Navigation Challenge Node

Create `scripts/navigation_challenges.py` that demonstrates how to navigate in different environments:

```python
#!/usr/bin/env python3

"""
Navigation Challenges Node for Humanoid Robot Simulation.
Demonstrates navigation in different environments with various obstacles.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
import numpy as np
import math


class NavigationChallengesNode(Node):
    def __init__(self):
        super().__init__('navigation_challenges_node')
        
        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/humanoid_robot/scan',
            self.scan_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/humanoid_robot/odom',
            self.odom_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid_robot/imu',
            self.imu_callback,
            10
        )
        
        # Publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/humanoid_robot/cmd_vel',
            10
        )
        
        # Publisher for challenge status
        self.challenge_status_pub = self.create_publisher(
            String,
            '/challenge_status',
            10
        )
        
        # Publisher for completion status
        self.challenge_complete_pub = self.create_publisher(
            Bool,
            '/challenge_complete',
            10
        )
        
        # Internal state
        self.current_scan = None
        self.current_odom = None
        self.current_imu = None
        self.robot_pose = Pose()
        self.challenge_active = False
        self.current_challenge = 0
        self.challenges_completed = 0
        
        # Challenge parameters
        self.challenge_goals = [
            # Challenge 0: Basic obstacle avoidance
            {'x': 2.0, 'y': 0.0, 'tolerance': 0.5, 'name': 'Basic Obstacle Avoidance'},
            # Challenge 1: Indoor navigation
            {'x': 0.0, 'y': 3.0, 'tolerance': 0.5, 'name': 'Navigate to Indoor Room'},
            # Challenge 2: Outdoor challenge course
            {'x': 12.0, 'y': 0.0, 'tolerance': 0.5, 'name': 'Complete Outdoor Course'}
        ]
        
        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.safe_distance = 0.6  # meters for obstacle avoidance
        self.obstacle_threshold = 0.7  # meters to trigger avoidance
        
        # Timer for navigation control
        self.nav_timer = self.create_timer(0.1, self.navigate)
        
        self.get_logger().info('Navigation Challenges Node Started')

    def scan_callback(self, msg):
        """Store laser scan data"""
        self.current_scan = msg

    def odom_callback(self, msg):
        """Process odometry for navigation"""
        self.current_odom = msg
        self.robot_pose = msg.pose.pose

    def imu_callback(self, msg):
        """Store IMU data for balance"""
        self.current_imu = msg

    def navigate(self):
        """Main navigation logic - executes challenges based on current environment"""
        if self.current_scan is None or self.current_odom is None:
            return
        
        # Determine which challenge to execute based on environment
        # This would typically be determined by an environment detection system
        # For now, we'll use a simple state machine approach
        
        cmd = Twist()
        
        # Get current position
        current_x = self.robot_pose.position.x
        current_y = self.robot_pose.position.y
        
        # Determine goal for current challenge
        if self.current_challenge < len(self.challenge_goals):
            goal = self.challenge_goals[self.current_challenge]
            goal_x, goal_y = goal['x'], goal['y']
            
            # Calculate distance to goal
            dist_to_goal = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
            
            if dist_to_goal < goal['tolerance']:
                # Goal reached, move to next challenge
                self.current_challenge += 1
                self.challenges_completed += 1
                self.get_logger().info(f"Challenge {goal['name']} completed!")
                
                # Publish challenge completion
                status_msg = String()
                status_msg.data = f"CHALLENGE_COMPLETED: {goal['name']}"
                self.challenge_status_pub.publish(status_msg)
                
                if self.current_challenge >= len(self.challenge_goals):
                    # All challenges completed
                    complete_msg = Bool()
                    complete_msg.data = True
                    self.challenge_complete_pub.publish(complete_msg)
                    self.get_logger().info("ALL CHALLENGES COMPLETED!")
                    return
            else:
                # Navigate toward goal
                goal_angle = math.atan2(goal_y - current_y, goal_x - current_x)
                
                # Get current orientation
                current_yaw = self.get_yaw_from_quaternion(self.robot_pose.orientation)
                
                # Calculate angle difference
                angle_diff = self.normalize_angle(goal_angle - current_yaw)
                
                # Check for obstacles
                obstacle_detected = self.check_for_obstacles()
                
                if obstacle_detected:
                    # Obstacle avoidance
                    cmd.linear.x = 0.0
                    cmd.angular.z = self.angular_speed  # Turn to avoid obstacle
                else:
                    # Move toward goal
                    if abs(angle_diff) > 0.2:  # If orientation is not aligned with goal
                        cmd.angular.z = self.angular_speed * np.sign(angle_diff)
                        cmd.linear.x = 0.0  # Don't move forward when turning
                    else:
                        cmd.linear.x = self.linear_speed  # Move forward
                        cmd.angular.z = 0.0  # Keep orientation
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
    
    def check_for_obstacles(self):
        """Check laser scan data for obstacles in front of the robot"""
        if self.current_scan is None:
            return False
        
        # Check the front 30 degrees of the scan (middle section)
        mid_point = len(self.current_scan.ranges) // 2
        sector_size = int(30 / 2 / self.current_scan.angle_increment)  # 30 degrees sector
        
        # Check ranges in the front sector
        for i in range(mid_point - sector_size, mid_point + sector_size):
            if 0 <= i < len(self.current_scan.ranges):
                distance = self.current_scan.ranges[i]
                if np.isfinite(distance) and distance < self.obstacle_threshold:
                    return True
        
        return False

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def destroy_node(self):
        """Cleanup before node destruction"""
        self.get_logger().info('Navigation Challenges Node Shutting Down')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    nav_node = NavigationChallengesNode()
    
    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        nav_node.get_logger().info('Node interrupted by user')
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5.2: Update Package Configuration

Update the `package.xml` to include navigation dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_simple_robot</name>
  <version>0.0.0</version>
  <description>Simple humanoid robot for Gazebo simulation with environments</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>cv_bridge</depend>
  <depend>message_filters</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>

  <exec_depend>gazebo_ros_pkgs</exec_depend>
  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update the setup.py to include the new script:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'humanoid_simple_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include URDF and launch files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Simple humanoid robot for Gazebo simulation with environments',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_validator = humanoid_simple_robot.scripts.sensor_validator:main',
            'laser_processor = humanoid_simple_robot.scripts.laser_processor:main',
            'navigation_challenges = humanoid_simple_robot.scripts.navigation_challenges:main',
        ],
    },
)
```

Place the navigation script in the correct location:

```bash
mkdir -p ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts
mv ~/humanoid_ws/src/humanoid_simple_robot/scripts/navigation_challenges.py \
   ~/humanoid_ws/src/humanoid_simple_robot/humanoid_simple_robot/scripts/navigation_challenges.py
```

## Part 6: Building and Testing the Environments

### Step 6.1: Build the Package

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Source ROS
source /opt/ros/iron/setup.bash

# Build the package
colcon build --packages-select humanoid_simple_robot

# Source the workspace
source install/setup.bash
```

### Step 6.2: Test Each Environment

```bash
# Test basic environment with navigation challenges
ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=basic_environment.sdf

# In another terminal, run the navigation challenges node
source install/setup.bash
ros2 run humanoid_simple_robot navigation_challenges
```

Test each environment with a different terminal:

```bash
# Terminal 1: Basic environment
source install/setup.bash
ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=basic_environment.sdf

# Terminal 2: Indoor environment  
source install/setup.bash
ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=indoor_environment.sdf

# Terminal 3: Outdoor challenge
source install/setup.bash
ros2 launch humanoid_simple_robot environment_demo.launch.py environment:=outdoor_challenge.sdf
```

Monitor the robot's performance in each environment:

```bash
# In another terminal:
source install/setup.bash
ros2 topic echo /challenge_status
```

## Part 7: Advanced Environment Design Techniques

### Step 7.1: Creating Parameterized Environments

For more complex environments, you can create parameterized models. Create `src/humanoid_simple_robot/models/parametric_room/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="parametric_room">
    <link name="floor">
      <pose>0 0 0 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 10 0.1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>10 10 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
    </link>
    
    <link name="wall_north">
      <pose>0 5 1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>10 0.2 2</size>
        </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
    
    <link name="wall_south">
      <pose>0 -5 1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
    
    <link name="wall_east">
      <pose>5 0 1 0 0 1.57</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
    
    <link name="wall_west">
      <pose>-5 0 1 0 0 1.57</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>10 0.2 2</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
    
    <!-- Optional: Doorway in the south wall -->
    <link name="door_jamb_1">
      <pose>-1.5 -5 1 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.1 1.5</size>
          </box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
          <diffuse>0.6 0.4 0.2 1</diffuse>
        </material>
      </visual>
    </link>
    
    <link name="door_jamb_2">
      <pose>1.5 -5 1 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.2 0.1 1.5</size>
          </box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
          <diffuse>0.6 0.4 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
```

### Step 7.2: Creating Modular Environment Pieces

Create a modular environment system where you can combine different pieces. Create `src/humanoid_simple_robot/config/modular_environment.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="modular_environment">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sky -->
    <scene>
      <ambient>0.4 0.4 0.4</ambient>
      <background>0.7 0.7 0.7</background>
      <shadows>true</shadows>
    </scene>
    
    <!-- Modular components -->
    
    <!-- Include the parametric room -->
    <include>
      <uri>model://parametric_room</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
    
    <!-- Multiple rooms -->
    <include>
      <uri>model://parametric_room</uri>
      <pose>15 0 0 0 0 0</pose>
    </include>
    
    <!-- Connection between rooms -->
    <model name="corridor">
      <static>true</static>
      <link name="floor">
        <pose>7.5 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>5 2 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>5 2 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

## Part 8: Validation and Assessment

### Step 8.1: Environment Validation Checklist

After setting up your environments, verify the following:

1. **Basic Environment**: 
   - [ ] Robot spawns without errors
   - [ ] All static obstacles are visible
   - [ ] Dynamic objects (movable box) respond appropriately
   - [ ] Sensors detect obstacles correctly

2. **Indoor Environment**:
   - [ ] Room structure is coherent
   - [ ] Furniture is properly placed
   - [ ] Doorways are navigable
   - [ ] Lighting effects are visible

3. **Outdoor Environment**:
   - [ ] Terrain variations are navigable
   - [ ] Stairs can be climbed (if properly designed)
   - [ ] Sloped surfaces are traversable
   - [ ] Challenge elements work as intended

4. **Navigation System**:
   - [ ] Robot avoids obstacles in all environments
   - [ ] Robot reaches designated goal positions
   - [ ] Sensors provide accurate data in all environments
   - [ ] Performance is consistent across environments

### Step 8.2: Self-Assessment Questions

After completing the exercise, answer these questions:

1. How would you modify the indoor environment to include elevators or stairs between floors?
2. What considerations are needed when designing an environment for humanoid robot manipulation tasks?
3. How can you create a dynamic environment where obstacles move or change positions?
4. What techniques would you use to simulate outdoor conditions like grass, gravel, or mud surfaces?
5. How would you evaluate whether your robot can successfully navigate your designed environments?

## Part 9: Troubleshooting Common Issues

### Issue 1: Robot Falls Through Terrain

**Symptoms:** Robot falls infinitely through the ground.

**Solutions:**
1. Verify all ground planes have proper `<collision>` elements
2. Check that static models have `<static>true</static>` tag
3. Ensure all collision geometries are properly defined

### Issue 2: Performance Problems

**Symptoms:** Slow simulation, dropped frames.

**Solutions:**
1. Simplify collision geometries for static objects
2. Reduce the number of objects in the environment
3. Adjust physics parameters for performance
4. Use simpler visual representations where possible

### Issue 3: Navigation Difficulties

**Symptoms:** Robot struggles to navigate through doorways or tight spaces.

**Solutions:**
1. Ensure doorways are wide enough (at least 1.2x robot width)
2. Check that sensor ranges are appropriate for the environment
3. Verify that the navigation algorithm accounts for robot dimensions

### Issue 4: Sensor Interference

**Symptoms:** Sensors provide incorrect data due to environment objects.

**Solutions:**
1. Verify that sensor models don't interfere with environment geometry
2. Check sensor mounting positions on the robot
3. Ensure appropriate sensor fields of view for the environment

## Bonus Challenge

Create a "maze environment" that tests your robot's pathfinding capabilities. Design a maze with multiple paths, dead ends, and one correct path to a goal. Implement a pathfinding algorithm that can navigate the maze and reach the goal.

## Exercise Completion

Congratulations! You have successfully:
- Created multiple complex 3D environments for humanoid robotics
- Designed both indoor and outdoor environments with varied challenges
- Implemented environment-specific navigation challenges
- Created modular environment components for reusability
- Validated that the robot can navigate in different environments
- Learned to troubleshoot common environment design issues

These skills are essential for creating realistic and challenging simulation environments for validating humanoid robot capabilities.