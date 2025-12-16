# Troubleshooting Guide: Physical AI & Humanoid Robotics Book

## Overview

This guide provides educators and students with systematic approaches to identify, diagnose, and resolve common issues encountered throughout the Physical AI & Humanoid Robotics curriculum. Each section corresponds to specific modules and includes step-by-step troubleshooting procedures.

## General Troubleshooting Principles

### 1. Systematic Approach
1. **Identify the problem** - Describe the issue in specific terms
2. **Reproduce the issue** - Verify the problem is consistent
3. **Isolate the component** - Determine which part of the system is failing
4. **Check dependencies** - Verify all required components are available
5. **Test incrementally** - Make small changes and test after each
6. **Document the solution** - Record what worked for future reference

### 2. Common Debugging Techniques
- Use `ros2 topic list`, `ros2 node list`, and `ros2 service list` to verify system state
- Check logs with `ros2 topic echo` and `rqt_console`
- Use `print` statements or ROS logging to trace program execution
- Verify configuration files and parameter values

---

## Module 1: ROS 2 Fundamentals Troubleshooting

### Issue 1: ROS 2 Installation Problems
**Symptoms**: 
- `ros2` command not found
- Package installation fails
- Environment not set up correctly

**Solutions**:
1. **Verify Ubuntu version**: Ensure using Ubuntu 22.04 LTS
   ```bash
   lsb_release -a
   ```

2. **Check ROS 2 repository**: 
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. **Install ROS 2 Iron**:
   ```bash
   sudo apt update
   sudo apt install ros-iron-desktop
   sudo apt install python3-colcon-common-extensions
   ```

4. **Source the environment**:
   ```bash
   source /opt/ros/iron/setup.bash
   echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
   ```

### Issue 2: Node Communication Problems
**Symptoms**:
- Nodes cannot communicate with each other
- Topics not publishing/subscribing
- "No data received" messages

**Solutions**:
1. **Check domain ID**: Ensure all nodes are using the same domain
   ```bash
   echo $ROS_DOMAIN_ID
   export ROS_DOMAIN_ID=0  # If not set
   ```

2. **Verify network setup**: For multiple machines, ensure proper network configuration
   ```bash
   export ROS_LOCALHOST_ONLY=0  # For multi-machine setups
   ```

3. **Check topic names**: Ensure exact matching of topic names
   ```bash
   ros2 topic list  # Verify topic names
   ros2 topic info <topic_name>  # Check topic details
   ```

4. **Check message types**: Ensure publisher and subscriber use compatible message types
   ```bash
   ros2 interface show std_msgs/msg/String  # Check message definition
   ```

### Issue 3: URDF Model Not Loading in RViz
**Symptoms**:
- Robot model not displaying in RViz
- TF transforms not showing
- URDF parsing errors

**Solutions**:
1. **Validate URDF syntax**:
   ```bash
   check_urdf /path/to/robot.urdf
   ```

2. **Check file paths**: Ensure all mesh and material paths are correct
   ```bash
   # Use package:// paths for mesh files
   <mesh filename="package://my_robot/meshes/link1.stl"/>
   ```

3. **Launch robot state publisher**:
   ```bash
   ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(find_my_robot)/urdf/robot.urdf'
   ```

4. **Verify RViz configuration**: Check that RobotModel and TF displays are properly configured

### Issue 4: Parameter Not Loading
**Symptoms**:
- Node starts with default parameter values
- Parameter file seems to be ignored
- "Parameter not found" warnings

**Solutions**:
1. **Check parameter file format**: Ensure YAML format is correct
   ```yaml
   my_node:
     ros__parameters:
       param_name: value
   ```

2. **Pass parameter file correctly**:
   ```bash
   ros2 run my_package my_node --ros-args --params-file config/params.yaml
   ```

3. **Declare parameters in code**: Ensure all parameters are declared
   ```python
   self.declare_parameter('param_name', default_value)
   ```

---

## Module 2: Digital Twin Simulation Troubleshooting

### Issue 1: Gazebo Not Starting
**Symptoms**:
- Gazebo fails to launch
- Graphics errors during startup
- Segmentation fault

**Solutions**:
1. **Check graphics drivers**: Ensure GPU drivers are properly installed
   ```bash
   nvidia-smi  # For NVIDIA GPUs
   glxinfo | grep OpenGL  # Check OpenGL support
   ```

2. **Install Gazebo Garden**:
   ```bash
   sudo apt install ros-iron-gazebo-*
   sudo apt install gazebo
   ```

3. **Check environment variables**:
   ```bash
   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models
   export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo/worlds
   ```

4. **Try software rendering if needed**:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1  # As a last resort for compatibility
   ```

### Issue 2: URDF Not Loading in Gazebo
**Symptoms**:
- Robot model doesn't appear in simulation
- Collision or visual properties not working
- Warnings about missing plugins

**Solutions**:
1. **Add Gazebo-specific tags to URDF**:
   ```xml
   <gazebo reference="link_name">
     <material>Gazebo/Blue</material>
   </gazebo>
   ```

2. **Include joint plugins**:
   ```xml
   <gazebo>
     <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
       <joint_name>joint1</joint_name>
     </plugin>
   </gazebo>
   ```

3. **Verify inertial properties**: Ensure all links have proper inertial, mass, and collision properties

### Issue 3: Sensor Data Not Publishing
**Symptoms**:
- No sensor data on ROS topics
- Sensor plugins not working
- Simulation running but no sensor output

**Solutions**:
1. **Verify sensor plugin configuration**:
   ```xml
   <gazebo reference="camera_link">
     <sensor name="camera" type="camera">
       <always_on>true</always_on>
       <visualize>true</visualize>
       <camera name="head">
         <horizontal_fov>1.089</horizontal_fov>
         <image>
           <width>640</width>
           <height>480</height>
           <format>R8G8B8</format>
         </image>
       </camera>
     </sensor>
   </gazebo>
   ```

2. **Check topic names**: Gazebo sensor topics are often prefixed with robot namespace
   ```bash
   ros2 topic list | grep camera
   ros2 topic echo /robot_name/camera/image_raw
   ```

3. **Install sensor packages**:
   ```bash
   sudo apt install ros-iron-gazebo-ros-pkgs
   ```

### Issue 4: Physics Simulation Issues
**Symptoms**:
- Robot falling through ground
- Joints behaving unexpectedly
- Unstable or unrealistic physics

**Solutions**:
1. **Check inertial properties**: Ensure mass, inertia, and center of mass are properly defined
   ```xml
   <inertial>
     <mass value="1.0"/>
     <origin xyz="0 0 0" rpy="0 0 0"/>
     <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
   </inertial>
   ```

2. **Verify collision meshes**: Make sure collision geometry is properly defined
   ```xml
   <collision>
     <origin xyz="0 0 0" rpy="0 0 0"/>
     <geometry>
       <box size="0.1 0.1 0.1"/>
     </geometry>
   </collision>
   ```

3. **Tune physics parameters**: Adjust parameters in world file
   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_factor>1</real_time_factor>
     <real_time_update_rate>1000</real_time_update_rate>
   </physics>
   ```

### Issue 5: Unity-ROS Connection Problems
**Symptoms**:
- Unity and ROS nodes cannot communicate
- Connection timeouts
- No robot state updates in Unity

**Solutions**:
1. **Check ROS-TCP-Connector**: Ensure both Unity and ROS components are installed
   - Unity package: Install from Unity Package Manager or GitHub
   - ROS package: Install from apt or build from source

2. **Verify network settings**:
   ```csharp
   // In Unity, verify IP and port
   [System.Serializable]
   public class ROSConnectionSettings
   {
       public string hostname = "127.0.0.1";
       public int port = 10000;
   }
   ```

3. **Check firewall settings**: Ensure Unity can connect to ROS network
   ```bash
   # Test connection
   telnet localhost 10000
   ```

---

## Module 3: AI-Robot Brain Troubleshooting

### Issue 1: Isaac Sim Installation Problems
**Symptoms**:
- Isaac Sim fails to launch
- CUDA compatibility issues
- Graphics errors

**Solutions**:
1. **Verify CUDA compatibility**: Check NVIDIA GPU compatibility
   ```bash
   nvidia-smi
   nvcc --version  # Check CUDA version
   ```

2. **Install proper CUDA version**: Isaac Sim requires specific CUDA versions
   ```bash
   # Check Isaac Sim requirements for exact CUDA version needed
   ```

3. **Install Isaac Sim dependencies**:
   ```bash
   # Follow NVIDIA's official installation guide for your specific Isaac Sim version
   ```

4. **Check system requirements**: Ensure adequate RAM, storage, and driver versions

### Issue 2: Isaac ROS Perception Node Issues
**Symptoms**:
- Perception nodes not publishing data
- GPU acceleration not working
- High CPU usage instead of GPU

**Solutions**:
1. **Verify Isaac ROS installation**:
   ```bash
   # Check if Isaac ROS packages are properly installed
   ros2 pkg list | grep isaac
   ```

2. **Check GPU access**:
   ```bash
   nvidia-smi  # Verify GPU access
   # Ensure Isaac ROS nodes can access GPU
   ```

3. **Install Isaac ROS GEMs** (if needed):
   ```bash
   # Follow Isaac ROS documentation for GEM installation
   ```

### Issue 3: VSLAM Not Working Properly
**Symptoms**:
- VSLAM not producing maps
- Poor localization accuracy
- High computational resource usage

**Solutions**:
1. **Verify camera calibration**: Ensure cameras are properly calibrated
   ```bash
   ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108 image:=/camera/image_raw camera:=/camera
   ```

2. **Check VSLAM parameters**: Tune parameters for your environment
   ```bash
   # Adjust parameters like detection threshold, tracking window, etc.
   ```

3. **Verify lighting conditions**: Ensure adequate lighting for visual features

### Issue 4: Jetson Deployment Issues
**Symptoms**:
- AI models not running on Jetson
- Performance significantly worse than expected
- Memory errors during inference

**Solutions**:
1. **Check Jetson model**: Verify hardware capabilities match model requirements
   ```bash
   # Use jetson-stats to check utilization
   jtop
   ```

2. **Optimize models for Jetson**: Use TensorRT or other optimization tools
   ```bash
   # Convert models to Jetson-optimized formats
   ```

3. **Check power modes**: Ensure Jetson is in appropriate power mode for performance
   ```bash
   sudo nvpmodel -q  # Query current power mode
   sudo nvpmodel -m 0  # Set to max performance mode if needed
   ```

### Issue 5: Nav2 Navigation Problems
**Symptoms**:
- Navigation controller not reaching goals
- Robot getting stuck or oscillating
- Costmap not updating properly

**Solutions**:
1. **Tune navigation parameters**: Adjust parameters in costmap and controller configs
   ```yaml
   # Example parameters
   local_costmap:
     resolution: 0.05  # Adjust based on robot size
     robot_radius: 0.3  # Match your robot's size
   ```

2. **Verify transforms**: Ensure TF tree is complete and updates properly
   ```bash
   ros2 run tf2_tools view_frames
   ros2 run rqt_tf_tree rqt_tf_tree
   ```

3. **Check sensor data**: Ensure navigation sensors are providing proper data

---

## Module 4: Vision-Language-Action Troubleshooting

### Issue 1: Whisper Speech Recognition Problems
**Symptoms**:
- Poor speech recognition accuracy
- Audio input not detected
- High latency in processing

**Solutions**:
1. **Verify audio input**:
   ```bash
   arecord -l  # List available audio devices
   # Test audio recording
   arecord -d 3 -f cd test.wav && aplay test.wav
   ```

2. **Install Whisper properly**:
   ```bash
   pip3 install openai-whisper
   # Or install with specific PyTorch version if needed
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install openai-whisper
   ```

3. **Optimize audio preprocessing**: Add noise reduction and proper audio formatting

### Issue 2: LLM Integration Issues
**Symptoms**:
- API requests failing
- High token usage
- Slow response from LLMs
- Incorrect response formatting

**Solutions**:
1. **Check API key**: Verify OpenAI API key is set and valid
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Implement proper error handling**:
   ```python
   import openai
   try:
       response = openai.ChatCompletion.create(...)
   except openai.error.RateLimitError:
       # Handle rate limit
   except openai.error.APIError as e:
       # Handle other API errors
   ```

3. **Optimize prompts**: Reduce token usage with efficient prompts
   ```python
   # Use structured prompts to get concise responses
   prompt = f"Convert command '{command}' to JSON action sequence: "
   ```

### Issue 3: Vision-Language Integration Problems
**Symptoms**:
- Vision and language systems not working together
- Context from vision not used in language processing
- Mismatched data timing

**Solutions**:
1. **Implement proper synchronization**:
   ```python
   # Use message filters to synchronize vision and language inputs
   from message_filters import ApproximateTimeSynchronizer, Subscriber
   
   # Synchronize vision and language inputs
   ```

2. **Create unified context representation**:
   ```python
   # Structure vision data in a way that language models can understand
   context = {
       "detected_objects": [...],
       "spatial_relationships": [...],
       "command": "user_command"
   }
   ```

### Issue 4: Multi-Modal Fusion Issues
**Symptoms**:
- Different modalities producing conflicting actions
- System not properly integrating multiple inputs
- Timing issues between modalities

**Solutions**:
1. **Implement fusion logic**:
   ```python
   class MultiModalFusion:
       def __init__(self):
           self.vision_buffer = []
           self.language_buffer = []
           self.audio_buffer = []
       
       def update_context(self, modalities):
           # Implement fusion logic here
           pass
   ```

2. **Add temporal alignment**:
   ```python
   # Ensure modalities are temporally aligned
   # Use timestamps to match corresponding inputs
   ```

### Issue 5: VLA System Performance Problems
**Symptoms**:
- System too slow for real-time operation
- High computational resource usage
- Memory leaks in long-running operations

**Solutions**:
1. **Optimize for performance**:
   ```python
   # Use threading for non-blocking operations
   # Implement caching for repeated operations
   # Optimize data structures for faster access
   ```

2. **Profile system performance**:
   ```bash
   # Use profiling tools to identify bottlenecks
   ros2 run topic_tools throttle messages /input_topic 5 /throttled_topic
   ```

3. **Implement resource management**:
   ```python
   # Monitor and limit resource usage
   import psutil
   import gc
   
   # Implement garbage collection and memory management
   ```

---

## General System Troubleshooting

### Issue 1: System Performance Problems
**Symptoms**:
- Slow system response
- High memory or CPU usage
- Frequent crashes or freezes

**Solutions**:
1. **Monitor system resources**:
   ```bash
   htop  # Check CPU and memory usage
   df -h  # Check disk space
   free -h  # Check available memory
   ```

2. **Optimize launch files**:
   ```xml
   <!-- Only launch required nodes -->
   <!-- Use appropriate CPU scheduling -->
   ```

3. **Adjust simulation settings**:
   ```xml
   <!-- Reduce physics update rate if needed -->
   <!-- Simplify models for performance -->
   ```

### Issue 2: Network and Communication Issues
**Symptoms**:
- Nodes on different machines cannot communicate
- High network latency
- Intermittent connection problems

**Solutions**:
1. **Check network setup**:
   ```bash
   # Verify all machines are on same network
   ping other_machine_ip
   # Check firewall settings
   sudo ufw status
   ```

2. **Configure ROS networking**:
   ```bash
   # Set ROS_IP and ROS_HOSTNAME appropriately
   export ROS_IP=your_machine_ip
   export ROS_HOSTNAME=your_machine_name
   ```

3. **Use appropriate QoS settings** for network-tolerant communication

---

## Quick Reference Commands

### ROS 2 Commands
```bash
ros2 topic list                      # List all topics
ros2 node list                       # List all nodes
ros2 service list                    # List all services
ros2 topic echo /topic_name          # Echo messages from topic
ros2 node info /node_name           # Get info about a node
ros2 param list /node_name           # List parameters of a node
ros2 run rqt_graph rqt_graph         # Visualize ROS graph
```

### System Diagnostics
```bash
source /opt/ros/iron/setup.bash      # Source ROS environment
printenv | grep ROS                  # Check ROS environment variables
colcon build --symlink-install       # Build ROS workspace
ros2 doctor                          # Check ROS installation
```

### Simulation Commands
```bash
gazebo                               # Launch Gazebo
gz sim                               # New Gazebo Garden command
rviz2                                # Launch RViz
```

---

## When to Seek Help

### Contact Resources
- **ROS Answers**: For ROS-specific issues
- **Isaac Sim Forum**: For NVIDIA Isaac issues
- **Course Instructor**: For curriculum-specific problems
- **Student Community**: For peer support and collaboration

### Information to Include When Seeking Help
1. **Complete error message**: Copy the full error text
2. **System specifications**: Ubuntu version, ROS version, hardware specs
3. **Steps to reproduce**: Clear sequence of actions that lead to the issue
4. **What you've tried**: Solutions you've already attempted
5. **Expected vs. actual behavior**: What you expected to happen vs. what actually happened