# Interactive Code Examples and Simulations: Physical AI & Humanoid Robotics

## Overview

This document outlines interactive code examples and simulations that should be implemented in the Physical AI & Humanoid Robotics curriculum. These interactive elements enhance learning by allowing students to experiment with code and see immediate results, making complex concepts more tangible and understandable.

## Module 1: ROS 2 Fundamentals Interactive Examples

### 1. Interactive Publisher/Subscriber Demonstration

#### Example 1: Real-time Message Inspector
```python
# publisher_subscriber_demo.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class InteractivePublisher(Node):
    def __init__(self):
        super().__init__('interactive_publisher')
        self.publisher = self.create_publisher(String, 'interactive_topic', 10)
        self.i = 0
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello from Publisher {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.i += 1

class InteractiveSubscriber(Node):
    def __init__(self):
        super().__init__('interactive_subscriber')
        self.subscription = self.create_subscription(
            String,
            'interactive_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    
    publisher = InteractivePublisher()
    subscriber = InteractiveSubscriber()
    
    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(subscriber)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        publisher.get_logger().info('Shutting down...')
    finally:
        publisher.destroy_node()
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Web-Based Interactive Version (HTML/JS)
```html
<!DOCTYPE html>
<html>
<head>
    <title>ROS 2 Publisher/Subscriber Interactive Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .control-panel { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .message-log { background: #f8f8f8; border: 1px solid #ccc; height: 300px; overflow-y: scroll; padding: 10px; margin: 10px 0; }
        .message-published { color: blue; }
        .message-received { color: green; }
        button { padding: 8px 15px; margin: 5px; }
    </style>
</head>
<body>
    <h1>ROS 2 Publisher/Subscriber Interactive Demo</h1>
    
    <div class="control-panel">
        <h3>Publisher Controls</h3>
        <button id="startPubBtn">Start Publishing</button>
        <button id="stopPubBtn">Stop Publishing</button>
        <button id="customMsgBtn">Send Custom Message</button>
        <input type="text" id="customMsgInput" placeholder="Enter custom message">
        
        <h3>Subscriber Controls</h3>
        <button id="clearLogBtn">Clear Log</button>
    </div>
    
    <div class="message-log" id="messageLog">
        <!-- Messages will appear here -->
        <div>Simulation started. Click "Start Publishing" to begin.</div>
    </div>

    <script>
        let publishInterval;
        let messageCounter = 0;
        
        document.getElementById('startPubBtn').addEventListener('click', startPublishing);
        document.getElementById('stopPubBtn').addEventListener('click', stopPublishing);
        document.getElementById('customMsgBtn').addEventListener('click', sendCustomMessage);
        document.getElementById('clearLogBtn').addEventListener('click', clearLog);
        
        function startPublishing() {
            if (publishInterval) clearInterval(publishInterval);
            publishInterval = setInterval(publishMessage, 1000);
            logMessage("Publisher started", "system");
        }
        
        function stopPublishing() {
            if (publishInterval) {
                clearInterval(publishInterval);
                publishInterval = null;
                logMessage("Publisher stopped", "system");
            }
        }
        
        function publishMessage() {
            messageCounter++;
            const message = `Hello from Publisher ${messageCounter}`;
            logMessage(`Published: "${message}"`, "message-published");
            
            // Simulate subscriber receiving the message after a delay
            setTimeout(() => {
                logMessage(`Received: "${message}"`, "message-received");
            }, 200);
        }
        
        function sendCustomMessage() {
            const input = document.getElementById('customMsgInput');
            const message = input.value.trim();
            if (message) {
                logMessage(`Published: "${message}"`, "message-published");
                
                setTimeout(() => {
                    logMessage(`Received: "${message}"`, "message-received");
                }, 200);
                
                input.value = '';
            }
        }
        
        function logMessage(text, className) {
            const log = document.getElementById('messageLog');
            const messageDiv = document.createElement('div');
            messageDiv.className = className;
            messageDiv.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
            log.appendChild(messageDiv);
            log.scrollTop = log.scrollHeight;
        }
        
        function clearLog() {
            document.getElementById('messageLog').innerHTML = '';
        }
    </script>
</body>
</html>
```

### 2. URDF Model Builder Interactively

#### Example 2: Interactive URDF Generator
```python
# interactive_urdf_builder.py
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

class URDFBuilder:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.root = ET.Element("robot", name=robot_name)
        self.links = {}
        self.joints = {}
    
    def add_link(self, name, mass=1.0, x=0.5, y=0.5, z=0.5):
        """Add a box-shaped link to the robot"""
        link = ET.SubElement(self.root, "link", name=name)
        
        # Visual element
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        box = ET.SubElement(geometry, "box", size=f"{x} {y} {z}")
        material = ET.SubElement(visual, "material", name=f"{name}_mat")
        color = ET.SubElement(material, "color", rgba="0.8 0.4 0.1 1.0")
        
        # Collision element
        collision = ET.SubElement(link, "collision")
        col_geom = ET.SubElement(collision, "geometry")
        col_box = ET.SubElement(col_geom, "box", size=f"{x} {y} {z}")
        
        # Inertial element
        inertial = ET.SubElement(link, "inertial")
        mass_elem = ET.SubElement(inertial, "mass", value=str(mass))
        origin = ET.SubElement(inertial, "origin", xyz="0 0 0")
        inertia = ET.SubElement(inertial, "inertia", 
                              ixx="0.1", ixy="0", ixz="0", 
                              iyy="0.1", iyz="0", izz="0.1")
        
        self.links[name] = link
        return link
    
    def add_joint(self, name, parent, child, joint_type="fixed", xyz="0 0 0", rpy="0 0 0"):
        """Add a joint between two links"""
        joint = ET.SubElement(self.root, "joint", name=name, type=joint_type)
        
        parent_elem = ET.SubElement(joint, "parent", link=parent)
        child_elem = ET.SubElement(joint, "child", link=child)
        origin = ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
        
        self.joints[name] = joint
        return joint
    
    def generate_urdf(self):
        """Generate URDF string"""
        rough_string = ET.tostring(self.root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_urdf(self, filename):
        """Save URDF to file"""
        with open(filename, 'w') as f:
            f.write(self.generate_urdf())

# Example usage of URDF builder
if __name__ == "__main__":
    # Create a simple robot with a chassis and wheels
    robot = URDFBuilder("interactive_robot")
    
    # Add chassis
    robot.add_link("chassis", mass=2.0, x=0.8, y=0.5, z=0.2)
    
    # Add wheels
    robot.add_link("left_wheel", mass=0.5, x=0.1, y=0.1, z=0.1)
    robot.add_link("right_wheel", mass=0.5, x=0.1, y=0.1, z=0.1)
    
    # Connect wheels to chassis with continuous joints
    robot.add_joint("left_wheel_joint", "chassis", "left_wheel", 
                   joint_type="continuous", xyz="0 0.3 -0.1", rpy="0 0 0")
    robot.add_joint("right_wheel_joint", "chassis", "right_wheel", 
                   joint_type="continuous", xyz="0 -0.3 -0.1", rpy="0 0 0")
    
    # Save the URDF
    robot.save_urdf("interactive_robot.urdf")
    print("URDF file generated: interactive_robot.urdf")
    print("\nGenerated URDF:")
    print(robot.generate_urdf())
```

#### Web-Based Interactive URDF Builder
```html
<!DOCTYPE html>
<html>
<head>
    <title>Interactive URDF Builder</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .builder-panel { display: flex; }
        .controls { flex: 1; padding: 15px; background: #f0f0f0; margin-right: 15px; }
        .preview { flex: 2; padding: 15px; background: #f8f8f8; border: 1px solid #ccc; }
        .urdf-output { width: 100%; height: 300px; font-family: monospace; }
        button, input, select { margin: 5px; padding: 5px; }
    </style>
</head>
<body>
    <h1>Interactive URDF Builder</h1>
    
    <div class="builder-panel">
        <div class="controls">
            <h3>Add Links</h3>
            <div>
                <input type="text" id="linkName" placeholder="Link name">
                <input type="number" id="linkMass" placeholder="Mass" value="1.0">
                <input type="number" id="linkX" placeholder="X size" value="0.5" step="0.1">
                <input type="number" id="linkY" placeholder="Y size" value="0.5" step="0.1">
                <input type="number" id="linkZ" placeholder="Z size" value="0.5" step="0.1">
                <button id="addLinkBtn">Add Link</button>
            </div>
            
            <h3>Add Joints</h3>
            <div>
                <input type="text" id="jointName" placeholder="Joint name">
                <select id="jointType">
                    <option value="fixed">Fixed</option>
                    <option value="continuous">Continuous</option>
                    <option value="revolute">Revolute</option>
                    <option value="prismatic">Prismatic</option>
                </select>
                <select id="jointParent"></select>
                <select id="jointChild"></select>
                <input type="text" id="jointXYZ" placeholder="XYZ (0 0 0)" value="0 0 0">
                <input type="text" id="jointRPY" placeholder="RPY (0 0 0)" value="0 0 0">
                <button id="addJointBtn">Add Joint</button>
            </div>
            
            <h3>Robot Settings</h3>
            <div>
                <input type="text" id="robotName" placeholder="Robot name" value="interactive_robot">
                <button id="generateURDFBtn">Generate URDF</button>
                <button id="resetBtn">Reset</button>
            </div>
        </div>
        
        <div class="preview">
            <h3>URDF Preview</h3>
            <textarea class="urdf-output" id="urdfOutput" readonly></textarea>
            <br>
            <button id="copyURDFBtn">Copy URDF to Clipboard</button>
        </div>
    </div>

    <script>
        let robotName = "interactive_robot";
        let links = [];
        let joints = [];
        
        document.getElementById('robotName').addEventListener('input', (e) => {
            robotName = e.target.value;
        });
        
        document.getElementById('addLinkBtn').addEventListener('click', addLink);
        document.getElementById('addJointBtn').addEventListener('click', addJoint);
        document.getElementById('generateURDFBtn').addEventListener('click', generateURDF);
        document.getElementById('resetBtn').addEventListener('click', resetBuilder);
        document.getElementById('copyURDFBtn').addEventListener('click', copyURDF);
        
        function addLink() {
            const name = document.getElementById('linkName').value.trim();
            const mass = document.getElementById('linkMass').value || '1.0';
            const x = document.getElementById('linkX').value || '0.5';
            const y = document.getElementById('linkY').value || '0.5';
            const z = document.getElementById('linkZ').value || '0.5';
            
            if (!name) {
                alert('Please enter a link name');
                return;
            }
            
            // Check for duplicate names
            if (links.some(link => link.name === name)) {
                alert('Link with this name already exists');
                return;
            }
            
            const link = {
                name: name,
                mass: mass,
                x: x,
                y: y,
                z: z
            };
            
            links.push(link);
            updateLinkSelects();
            document.getElementById('linkName').value = '';
            generateURDF();
        }
        
        function addJoint() {
            const name = document.getElementById('jointName').value.trim();
            const type = document.getElementById('jointType').value;
            const parent = document.getElementById('jointParent').value;
            const child = document.getElementById('jointChild').value;
            const xyz = document.getElementById('jointXYZ').value || '0 0 0';
            const rpy = document.getElementById('jointRPY').value || '0 0 0';
            
            if (!name || !parent || !child) {
                alert('Please fill in all joint fields');
                return;
            }
            
            // Check for duplicate names
            if (joints.some(joint => joint.name === name)) {
                alert('Joint with this name already exists');
                return;
            }
            
            const joint = {
                name: name,
                type: type,
                parent: parent,
                child: child,
                xyz: xyz,
                rpy: rpy
            };
            
            joints.push(joint);
            document.getElementById('jointName').value = '';
            generateURDF();
        }
        
        function updateLinkSelects() {
            const parentSelect = document.getElementById('jointParent');
            const childSelect = document.getElementById('jointChild');
            
            parentSelect.innerHTML = '';
            childSelect.innerHTML = '';
            
            links.forEach(link => {
                const option1 = document.createElement('option');
                option1.value = link.name;
                option1.textContent = link.name;
                parentSelect.appendChild(option1);
                
                const option2 = document.createElement('option');
                option2.value = link.name;
                option2.textContent = link.name;
                childSelect.appendChild(option2);
            });
        }
        
        function generateURDF() {
            let urdf = `<robot name="${robotName}">\n`;
            
            // Add links
            links.forEach(link => {
                urdf += `  <link name="${link.name}">\n`;
                urdf += `    <visual>\n`;
                urdf += `      <geometry>\n`;
                urdf += `        <box size="${link.x} ${link.y} ${link.z}"/>\n`;
                urdf += `      </geometry>\n`;
                urdf += `      <material name="${link.name}_mat">\n`;
                urdf += `        <color rgba="0.8 0.4 0.1 1.0"/>\n`;
                urdf += `      </material>\n`;
                urdf += `    </visual>\n`;
                urdf += `    <collision>\n`;
                urdf += `      <geometry>\n`;
                urdf += `        <box size="${link.x} ${link.y} ${link.z}"/>\n`;
                urdf += `      </geometry>\n`;
                urdf += `    </collision>\n`;
                urdf += `    <inertial>\n`;
                urdf += `      <mass value="${link.mass}"/>\n`;
                urdf += `      <origin xyz="0 0 0"/>\n`;
                urdf += `      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>\n`;
                urdf += `    </inertial>\n`;
                urdf += `  </link>\n\n`;
            });
            
            // Add joints
            joints.forEach(joint => {
                urdf += `  <joint name="${joint.name}" type="${joint.type}">\n`;
                urdf += `    <parent link="${joint.parent}"/>\n`;
                urdf += `    <child link="${joint.child}"/>\n`;
                urdf += `    <origin xyz="${joint.xyz}" rpy="${joint.rpy}"/>\n`;
                urdf += `  </joint>\n\n`;
            });
            
            urdf += `</robot>`;
            
            document.getElementById('urdfOutput').value = urdf;
        }
        
        function resetBuilder() {
            links = [];
            joints = [];
            document.getElementById('linkName').value = '';
            document.getElementById('jointName').value = '';
            document.getElementById('robotName').value = 'interactive_robot';
            document.getElementById('urdfOutput').value = '';
            updateLinkSelects();
            robotName = 'interactive_robot';
        }
        
        function copyURDF() {
            const urdfElement = document.getElementById('urdfOutput');
            urdfElement.select();
            document.execCommand('copy');
            alert('URDF copied to clipboard!');
        }
    </script>
</body>
</html>
```

## Module 2: Digital Twin Simulation Interactive Examples

### 3. Interactive Gazebo Environment Builder

#### Example 3: Web-Based Simulation Configuration
```python
# simulation_builder.py
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

class SimulationBuilder:
    def __init__(self, world_name="interactive_world"):
        self.world_name = world_name
        self.sdf_root = ET.Element("sdf", version="1.7")
        self.world = ET.SubElement(self.sdf_root, "world", name=world_name)
        
        # Add basic elements
        self.add_ground_plane()
        self.add_sky()
        self.add_sun()
    
    def add_ground_plane(self):
        """Add a ground plane to the simulation"""
        ground = ET.SubElement(self.world, "include")
        uri = ET.SubElement(ground, "uri")
        uri.text = "model://ground_plane"
        return ground
    
    def add_sky(self):
        """Add a sky to the simulation"""
        sky = ET.SubElement(self.world, "include")
        uri = ET.SubElement(sky, "uri")
        uri.text = "model://sky"
        return sky
    
    def add_sun(self):
        """Add a sun to the simulation"""
        sun = ET.SubElement(self.world, "include")
        uri = ET.SubElement(sun, "uri")
        uri.text = "model://sun"
        return sun
    
    def add_box_obstacle(self, name, size="1 1 1", position="0 0 0.5", color="0.4 0.4 0.4 1"):
        """Add a box obstacle to the simulation"""
        model = ET.SubElement(self.world, "model", name=name)
        pose = ET.SubElement(model, "pose")
        pose.text = f"{position} 0 0 0"  # x y z roll pitch yaw
        
        link = ET.SubElement(model, "link", name="link")
        
        collision = ET.SubElement(link, "collision", name="collision")
        geometry = ET.SubElement(collision, "geometry")
        box = ET.SubElement(geometry, "box")
        size_elem = ET.SubElement(box, "size")
        size_elem.text = size
        
        visual = ET.SubElement(link, "visual", name="visual")
        v_geometry = ET.SubElement(visual, "geometry")
        v_box = ET.SubElement(v_geometry, "box")
        v_size = ET.SubElement(v_box, "size")
        v_size.text = size
        
        material = ET.SubElement(visual, "material")
        ambient = ET.SubElement(material, "ambient")
        ambient.text = color
        diffuse = ET.SubElement(material, "diffuse")
        diffuse.text = color
        
        return model
    
    def add_room(self, name, width=5, depth=5, height=3):
        """Add a simple room with walls"""
        # Add floor
        floor = self.add_box_obstacle(f"{name}_floor", 
                                     size=f"{width} {depth} 0.1", 
                                     position=f"0 0 -0.05",
                                     color="0.8 0.8 0.8 1")
        
        # Add walls
        wall_height = height
        wall_thickness = 0.1
        
        # North wall
        self.add_box_obstacle(f"{name}_wall_north",
                             size=f"{width} {wall_thickness} {wall_height}",
                             position=f"0 {depth/2} {wall_height/2}",
                             color="0.6 0.6 0.6 1")
        
        # South wall
        self.add_box_obstacle(f"{name}_wall_south",
                             size=f"{width} {wall_thickness} {wall_height}",
                             position=f"0 {-depth/2} {wall_height/2}",
                             color="0.6 0.6 0.6 1")
        
        # East wall
        self.add_box_obstacle(f"{name}_wall_east",
                             size=f"{wall_thickness} {depth} {wall_height}",
                             position=f"{width/2} 0 {wall_height/2}",
                             color="0.6 0.6 0.6 1")
        
        # West wall
        self.add_box_obstacle(f"{name}_wall_west",
                             size=f"{wall_thickness} {depth} {wall_height}",
                             position=f"{-width/2} 0 {wall_height/2}",
                             color="0.6 0.6 0.6 1")
        
        return floor
    
    def generate_sdf(self):
        """Generate SDF string"""
        rough_string = ET.tostring(self.sdf_root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_sdf(self, filename="interactive_world.sdf"):
        """Save SDF to file"""
        with open(filename, 'w') as f:
            f.write(self.generate_sdf())

# Example usage
if __name__ == "__main__":
    sim = SimulationBuilder("my_interactive_world")
    
    # Add a room
    sim.add_room("main_room", width=6, depth=4, height=3)
    
    # Add some obstacles in the room
    sim.add_box_obstacle("table", size="0.8 0.6 0.7", position="1 0.5 0.35", color="0.8 0.6 0.4 1")
    sim.add_box_obstacle("chair", size="0.4 0.4 0.5", position="-1 -0.5 0.25", color="0.5 0.3 0.1 1")
    sim.add_box_obstacle("bookshelf", size="0.3 0.2 1.5", position="-2.5 1.5 0.75", color="0.7 0.5 0.3 1")
    
    # Save the simulation
    sim.save_sdf("interactive_environment.sdf")
    print("Simulation world generated: interactive_environment.sdf")
    print("\nGenerated SDF:")
    print(sim.generate_sdf())
```

## Module 3: AI-Robot Brain Interactive Examples

### 4. Interactive Perception Pipeline

#### Example 4: Real-time AI Perception Demonstration
```python
# interactive_perception_demo.py
import numpy as np
import cv2
from collections import deque
import time
import threading

class InteractivePerceptionDemo:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Use default camera
        self.running = False
        self.detections = []
        self.fps_counter = deque(maxlen=30)  # For FPS calculation
        
        # Perception modes
        self.modes = {
            'color': self.color_detection,
            'motion': self.motion_detection,
            'shape': self.shape_detection,
            'face': self.face_detection
        }
        self.current_mode = 'color'
        
        # Parameters for different modes
        self.params = {
            'color': {
                'lower_color': np.array([0, 100, 100]),  # Red color lower range
                'upper_color': np.array([10, 255, 255]), # Red color upper range
                'min_area': 500
            },
            'motion': {
                'min_area': 500,
                'threshold': 25
            },
            'shape': {
                'min_area': 500
            },
            'face': {
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'min_size': (30, 30)
            }
        }
        
        # Load face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Background subtractor for motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2()
    
    def color_detection(self, frame):
        """Detect objects of specific colors"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (HSV wraps around)
        mask1 = cv2.inRange(hsv, self.params['color']['lower_color'], self.params['color']['upper_color'])
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.params['color']['min_area']:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'type': 'color',
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'area': area,
                    'confidence': min(1.0, area / 10000)  # Normalize confidence
                })
        
        return detections, mask
    
    def motion_detection(self, frame):
        """Detect motion in the frame"""
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Apply threshold to remove shadows
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.params['motion']['min_area']:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'type': 'motion',
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'area': area,
                    'confidence': min(1.0, area / 10000)
                })
        
        return detections, fg_mask
    
    def shape_detection(self, frame):
        """Detect basic shapes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.params['shape']['min_area']:
                # Approximate the contour to find the number of sides
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Determine shape based on number of vertices
                shape = "unidentified"
                if len(approx) == 3:
                    shape = "triangle"
                elif len(approx) == 4:
                    # Check if it's a square or rectangle
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
                elif len(approx) == 5:
                    shape = "pentagon"
                else:
                    shape = "circle"
                
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'type': 'shape',
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'area': area,
                    'shape': shape,
                    'confidence': min(1.0, area / 10000)
                })
        
        return detections, thresh
    
    def face_detection(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.params['face']['scale_factor'],
            minNeighbors=self.params['face']['min_neighbors'],
            minSize=self.params['face']['min_size']
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'type': 'face',
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'area': w * h,
                'confidence': min(1.0, (w * h) / 50000)  # Normalize based on face size
            })
        
        return detections, gray
    
    def run(self):
        """Run the interactive perception demo"""
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Record start time for FPS calculation
            start_time = time.time()
            
            # Process frame based on current mode
            detections, overlay = self.modes[self.current_mode](frame)
            self.detections = detections
            
            # Draw detections on frame
            output_frame = frame.copy()
            for det in detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"{det['type']}: {det.get('shape', '')}"
                confidence_text = f"Conf: {det['confidence']:.2f}"
                
                cv2.putText(output_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(output_frame, confidence_text, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Calculate and display FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            self.fps_counter.append(fps)
            avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
            
            # Display mode and FPS
            status_text = f"Mode: {self.current_mode.upper()} | FPS: {avg_fps:.1f}"
            cv2.putText(output_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(output_frame, "Press 'c' for Color, 'm' for Motion, 's' for Shape, 'f' for Face", 
                       (10, output_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Interactive Perception Demo', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_mode = 'color'
            elif key == ord('m'):
                self.current_mode = 'motion'
            elif key == ord('s'):
                self.current_mode = 'shape'
            elif key == ord('f'):
                self.current_mode = 'face'
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.running = False

if __name__ == "__main__":
    demo = InteractivePerceptionDemo()
    print("Interactive Perception Demo starting...")
    print("Controls:")
    print("  'c' - Color detection")
    print("  'm' - Motion detection") 
    print("  's' - Shape detection")
    print("  'f' - Face detection")
    print("  'q' - Quit")
    demo.run()
```

## Module 4: Vision-Language-Action Interactive Examples

### 5. Interactive VLA System Simulator

#### Example 5: Web-Based VLA System Simulator
```html
<!DOCTYPE html>
<html>
<head>
    <title>Vision-Language-Action Interactive Simulator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .vla-simulator {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .simulator-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .vla-architecture {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .component {
            flex: 1;
            min-width: 250px;
            margin: 10px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            transition: background-color 0.3s;
        }
        .vision { background-color: #e3f2fd; border: 2px solid #2196f3; }
        .language { background-color: #f1f8e9; border: 2px solid #8bc34a; }
        .action { background-color: #ffebee; border: 2px solid #f44336; }
        .component h3 {
            margin-top: 0;
        }
        .input-area {
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .command-input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin: 10px 0;
        }
        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 20px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background-color: #2196f3;
            color: white;
        }
        button:hover {
            background-color: #0b7dda;
        }
        .robot-display {
            margin: 20px 0;
            padding: 20px;
            background-color: #fff;
            border: 2px solid #ddd;
            border-radius: 8px;
            text-align: center;
        }
        .robot {
            width: 200px;
            height: 300px;
            background: linear-gradient(to bottom, #666, #ccc);
            margin: 0 auto;
            border-radius: 10px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .robot:after {
            content: "🤖";
            font-size: 80px;
        }
        .vision-feed {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .camera-feed {
            width: 300px;
            height: 200px;
            background-color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            border-radius: 8px;
        }
        .detection {
            position: absolute;
            border: 2px solid red;
            color: red;
        }
        .log-area {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            height: 150px;
            overflow-y: auto;
        }
        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #2196f3;
            background-color: #f0f8ff;
        }
        .vision-log { border-left-color: #2196f3; background-color: #e3f2fd; }
        .language-log { border-left-color: #8bc34a; background-color: #f1f8e9; }
        .action-log { border-left-color: #f44336; background-color: #ffebee; }
        .prediction {
            margin: 10px 0;
            padding: 10px;
            background-color: #e8f5e8;
            border-radius: 4px;
            border: 1px solid #4caf50;
        }
        .scenario-panel {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .scenario-btn {
            background-color: #9e9e9e;
            margin: 5px;
        }
        .scenario-btn.active {
            background-color: #4caf50;
        }
    </style>
</head>
<body>
    <div class="vla-simulator">
        <div class="simulator-header">
            <h1>Vision-Language-Action Interactive Simulator</h1>
            <p>Experiment with AI-driven robotic systems in a simulated environment</p>
        </div>
        
        <div class="vla-architecture">
            <div class="component vision">
                <h3>👁️ Vision System</h3>
                <p>Processes visual input</p>
                <p>Detects objects and environment</p>
                <p>Provides spatial context</p>
            </div>
            
            <div class="component language">
                <h3>💬 Language System</h3>
                <p>Interprets natural language</p>
                <p>Understands commands</p>
                <p>Plans actions</p>
            </div>
            
            <div class="component action">
                <h3>⚙️ Action System</h3>
                <p>Executes robot behaviors</p>
                <p>Controls movement</p>
                <p>Manipulates objects</p>
            </div>
        </div>
        
        <div class="scenario-panel">
            <h3>Environment Scenarios</h3>
            <button class="scenario-btn" data-scenario="kitchen">Kitchen Environment</button>
            <button class="scenario-btn" data-scenario="office">Office Environment</button>
            <button class="scenario-btn" data-scenario="living">Living Room</button>
            <button class="scenario-btn" data-scenario="outdoor">Outdoor Scene</button>
        </div>
        
        <div class="vision-feed">
            <div class="camera-feed" id="mainCamera">
                <div>📷 Main Camera Feed</div>
            </div>
            <div class="camera-feed" id="gripperCamera">
                <div>📷 Gripper Camera</div>
            </div>
        </div>
        
        <div class="robot-display">
            <h3>🤖 Humanoid Robot</h3>
            <div class="robot"></div>
            <p>Current State: Idle</p>
        </div>
        
        <div class="input-area">
            <h3>🗣️ Voice Command Input</h3>
            <input type="text" class="command-input" id="commandInput" placeholder="Enter a command, e.g., 'Pick up the red cup'">
            <div class="buttons">
                <button id="processCommandBtn">Process Command</button>
                <button id="voiceInputBtn">🎤 Use Voice Input</button>
                <button id="resetBtn">Reset Simulation</button>
            </div>
        </div>
        
        <div class="input-area">
            <h3>🎯 Predefined Commands</h3>
            <div class="buttons">
                <button class="command-btn" data-command="Go to the kitchen">Go to Kitchen</button>
                <button class="command-btn" data-command="Find the red cup">Find Red Cup</button>
                <button class="command-btn" data-command="Pick up the book">Pick Up Book</button>
                <button class="command-btn" data-command="Bring me the pen">Bring Pen</button>
                <button class="command-btn" data-command="Wave to the person">Wave to Person</button>
            </div>
        </div>
        
        <div class="log-area">
            <h3>System Log</h3>
            <div id="logEntries">
                <div class="log-entry">Simulation started - Ready to receive commands</div>
            </div>
        </div>
    </div>

    <script>
        // Current simulation state
        let currentScenario = 'kitchen';
        let detectedObjects = [];
        let robotState = 'idle';
        
        // DOM Elements
        const commandInput = document.getElementById('commandInput');
        const processCommandBtn = document.getElementById('processCommandBtn');
        const voiceInputBtn = document.getElementById('voiceInputBtn');
        const resetBtn = document.getElementById('resetBtn');
        const logEntries = document.getElementById('logEntries');
        const robotDisplay = document.querySelector('.robot');
        const mainCamera = document.getElementById('mainCamera');
        const gripperCamera = document.getElementById('gripperCamera');
        
        // Predefined commands
        document.querySelectorAll('.command-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                commandInput.value = btn.getAttribute('data-command');
                processCommand();
            });
        });
        
        // Scenario selection
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentScenario = btn.getAttribute('data-scenario');
                updateScenario();
            });
        });
        
        // Event listeners
        processCommandBtn.addEventListener('click', processCommand);
        resetBtn.addEventListener('click', resetSimulation);
        
        // Voice input simulation (since real voice input requires special setup)
        voiceInputBtn.addEventListener('click', () => {
            alert('Voice input simulation would start here. In a real implementation, this would use the Web Speech API.');
            // Simulate voice input with a predefined command
            commandInput.value = "Pick up the red cup from the table";
        });
        
        // Initialize the simulation
        updateScenario();
        logMessage("VLA Simulator initialized", "system", "system");
        
        function updateScenario() {
            // Update camera feeds based on scenario
            mainCamera.innerHTML = `📷 ${getScenarioName(currentScenario)} View`;
            gripperCamera.innerHTML = '📷 Gripper Camera';
            
            // Generate simulated objects based on scenario
            detectedObjects = generateObjectsForScenario(currentScenario);
            
            // Update robot state display
            robotDisplay.style.background = getScenarioBackground(currentScenario);
        }
        
        function getScenarioName(scenario) {
            const names = {
                'kitchen': 'Kitchen',
                'office': 'Office',
                'living': 'Living Room',
                'outdoor': 'Outdoor'
            };
            return names[scenario] || 'Environment';
        }
        
        function getScenarioBackground(scenario) {
            const backgrounds = {
                'kitchen': 'linear-gradient(to bottom, #ffab91, #ffccbc)',
                'office': 'linear-gradient(to bottom, #c5cae9, #e8eaf6)',
                'living': 'linear-gradient(to bottom, #b3e5fc, #e1f5fe)',
                'outdoor': 'linear-gradient(to bottom, #e1bee7, #f3e5f5)'
            };
            return backgrounds[scenario] || 'linear-gradient(to bottom, #666, #ccc)';
        }
        
        function generateObjectsForScenario(scenario) {
            const objects = {
                'kitchen': [
                    {name: 'red cup', type: 'cup', color: 'red', position: {x: 45, y: 35, w: 15, h: 20}},
                    {name: 'blue plate', type: 'plate', color: 'blue', position: {x: 70, y: 45, w: 20, h: 5}},
                    {name: 'banana', type: 'fruit', color: 'yellow', position: {x: 20, y: 60, w: 8, h: 12}}
                ],
                'office': [
                    {name: 'black laptop', type: 'laptop', color: 'black', position: {x: 30, y: 25, w: 30, h: 20}},
                    {name: 'blue pen', type: 'pen', color: 'blue', position: {x: 65, y: 50, w: 5, h: 15}},
                    {name: 'white paper', type: 'paper', color: 'white', position: {x: 50, y: 70, w: 25, h: 20}}
                ],
                'living': [
                    {name: 'red book', type: 'book', color: 'red', position: {x: 40, y: 40, w: 15, h: 20}},
                    {name: 'blue cushion', type: 'cushion', color: 'blue', position: {x: 60, y: 60, w: 20, h: 15}},
                    {name: 'green plant', type: 'plant', color: 'green', position: {x: 15, y: 20, w: 10, h: 25}}
                ],
                'outdoor': [
                    {name: 'green tree', type: 'tree', color: 'green', position: {x: 20, y: 15, w: 15, h: 40}},
                    {name: 'brown bench', type: 'bench', color: 'brown', position: {x: 60, y: 65, w: 30, h: 15}},
                    {name: 'yellow flower', type: 'flower', color: 'yellow', position: {x: 80, y: 80, w: 5, h: 8}}
                ]
            };
            
            return objects[scenario] || [];
        }
        
        function processCommand() {
            const command = commandInput.value.trim();
            if (!command) {
                alert('Please enter a command');
                return;
            }
            
            logMessage(`Processing command: "${command}"`, "system", "system");
            
            // Simulate vision processing
            simulateVisionProcessing();
            
            // Simulate language processing
            setTimeout(() => simulateLanguageProcessing(command), 1000);
            
            // Simulate action execution
            setTimeout(() => simulateActionExecution(command), 2000);
        }
        
        function simulateVisionProcessing() {
            logMessage("Starting vision processing...", "vision", "vision");
            
            // Add simulated detections to the main camera view
            detectedObjects.forEach(obj => {
                const detection = document.createElement('div');
                detection.className = 'detection';
                detection.style.position = 'absolute';
                detection.style.left = obj.position.x + '%';
                detection.style.top = obj.position.y + '%';
                detection.style.width = obj.position.w + '%';
                detection.style.height = obj.position.h + '%';
                detection.style.border = '2px solid red';
                detection.style.color = 'red';
                detection.style.display = 'flex';
                detection.style.alignItems = 'center';
                detection.style.justifyContent = 'center';
                detection.style.fontSize = '10px';
                detection.style.pointerEvents = 'none';
                detection.textContent = obj.name;
                
                mainCamera.appendChild(detection);
            });
            
            logMessage(`Detected ${detectedObjects.length} objects`, "vision", "vision");
        }
        
        function simulateLanguageProcessing(command) {
            logMessage(`Processing language: "${command}"`, "language", "language");
            
            // Simple command interpretation
            let action = "unknown";
            let target = "unknown";
            
            if (command.toLowerCase().includes("go to")) {
                action = "navigate";
                if (command.toLowerCase().includes("kitchen")) target = "kitchen";
                else if (command.toLowerCase().includes("office")) target = "office";
                else if (command.toLowerCase().includes("living")) target = "living room";
            } else if (command.toLowerCase().includes("pick up") || command.toLowerCase().includes("grasp") || command.toLowerCase().includes("take")) {
                action = "grasp";
                // Extract target object from command
                detectedObjects.forEach(obj => {
                    if (command.toLowerCase().includes(obj.name)) {
                        target = obj.name;
                    }
                });
            } else if (command.toLowerCase().includes("wave")) {
                action = "greet";
                target = "person";
            }
            
            logMessage(`Parsed: Action=${action}, Target=${target}`, "language", "language");
            
            // Show prediction
            const predictionDiv = document.createElement('div');
            predictionDiv.className = 'prediction';
            predictionDiv.innerHTML = `
                <strong>Language Understanding Result:</strong><br>
                Command: "${command}"<br>
                Action: ${action}<br>
                Target: ${target}
            `;
            logEntries.appendChild(predictionDiv);
        }
        
        function simulateActionExecution(command) {
            logMessage(`Executing action for: "${command}"`, "action", "action");
            
            // Update robot state
            robotState = 'executing';
            document.querySelector('.robot').parentElement.querySelector('p').textContent = 'Current State: Executing Action';
            
            // Simulate robot movement
            const robot = document.querySelector('.robot');
            robot.style.transform = 'scale(1.1)';
            robot.style.background = 'linear-gradient(to bottom, #4CAF50, #81C784)';
            
            setTimeout(() => {
                robot.style.transform = 'scale(1)';
                robot.style.background = getScenarioBackground(currentScenario);
                robotState = 'idle';
                document.querySelector('.robot').parentElement.querySelector('p').textContent = 'Current State: Idle';
                logMessage(`Action completed for: "${command}"`, "action", "action");
            }, 2000);
        }
        
        function logMessage(message, type, category) {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${type}-log`;
            logEntry.innerHTML = `<strong>[${new Date().toLocaleTimeString()}]</strong> ${message}`;
            logEntries.appendChild(logEntry);
            
            // Scroll to bottom
            logEntries.scrollTop = logEntries.scrollHeight;
        }
        
        function resetSimulation() {
            // Clear detection boxes
            const detections = mainCamera.querySelectorAll('.detection');
            detections.forEach(d => d.remove());
            
            // Reset robot
            robotDisplay.style.transform = 'scale(1)';
            robotDisplay.style.background = getScenarioBackground(currentScenario);
            document.querySelector('.robot').parentElement.querySelector('p').textContent = 'Current State: Idle';
            
            // Clear log
            logEntries.innerHTML = '<div class="log-entry">Simulation reset - Ready to receive commands</div>';
            
            // Reset command input
            commandInput.value = '';
            
            robotState = 'idle';
        }
    </script>
</body>
</html>
```

These interactive examples provide hands-on experience with the core concepts of each module. Students can experiment with the code, modify parameters, and observe real-time results. This approach helps solidify understanding of complex robotics and AI concepts through active engagement rather than passive reading.