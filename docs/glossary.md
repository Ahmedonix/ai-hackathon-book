# Comprehensive Glossary: Physical AI & Humanoid Robotics

## A

**Action (ROS)**: A communication pattern in ROS 2 that provides feedback, goals, and results for long-running tasks. Unlike services, actions can be preempted and provide continuous feedback during execution.

**Action Server**: A ROS component that handles action goals, provides feedback, and returns results once an action is completed.

**Actionlib**: The ROS library used for implementing and using action-based communication between nodes.

**AI Planning**: The process of creating a sequence of actions to achieve a specific goal using artificial intelligence algorithms.

**Algorithm**: A step-by-step procedure for solving problems or performing tasks in robotics and AI.

**API (Application Programming Interface)**: A set of rules that allows different software applications to communicate with each other.

**Asynchronous**: A programming pattern where operations don't block program execution, allowing other tasks to run concurrently.

**Autonomous**: A system that can operate independently without human intervention.

## B

**Behavior Tree**: A hierarchical structure used in robotics and AI to represent and organize complex behaviors and decision-making processes.

**Bounding Box**: A rectangle that encloses an object in image processing, used for object detection and localization.

**Bringup**: The process of starting all necessary nodes and systems to make a robot operational.

**Bridge**: A component that connects two different systems or protocols, often used to connect ROS and other systems like Unity.

## C

**Callback**: A function that is executed when a specific event occurs, commonly used for handling ROS topic messages or service requests.

**Camera Matrix**: A matrix that describes the internal parameters of a camera, used in computer vision for image processing.

**Cartesian Space**: A coordinate system where positions are defined by X, Y, Z coordinates in 3D space.

**Changelog**: A document that records changes made to software, including new features, bug fixes, and improvements.

**Clipping Range**: The distance range (near to far) within which objects are rendered in computer graphics and simulation.

**Cloud Computing**: The delivery of computing services over the internet, used for processing intensive AI tasks.

**Collaborative Robot (Cobot)**: A robot designed to work safely alongside humans in shared spaces.

**Collision Detection**: The process of determining when objects in a simulation make contact with each other.

**Command and Control**: The systems and processes used to direct and manage robotic systems.

**Computer Vision**: A field of artificial intelligence that trains computers to interpret and understand visual information from the world.

**Concurrent**: Multiple processes or operations happening at the same time, often used in multi-threaded programming.

**Control Loop**: A continuous cycle of sensing, processing, and acting that allows robots to respond to their environment.

**Coordinate Frame**: A reference system used to define positions and orientations of objects in space.

**Cubic Spline**: A smooth curve that passes through a series of points, used in path planning for robot motion.

## D

**Debugging**: The process of finding and fixing errors in software code.

**Deep Learning**: A subset of machine learning based on artificial neural networks with multiple layers.

**Deep Neural Network (DNN)**: A neural network with multiple layers between input and output, used for complex pattern recognition.

**Depth Camera**: A camera that captures both RGB and depth information for 3D scene understanding.

**Differential Drive**: A common robot drive configuration with two independently controlled wheels on either side.

**Digital Twin**: A virtual replica of a physical robot or system used for simulation and analysis.

**Discrete Event Simulation**: A simulation where system states change at specific time points rather than continuously.

**Domain Randomization**: A technique for training AI systems with varied simulation environments to improve real-world performance.

**Docker**: A platform that enables containerization of applications, useful for consistent deployment across systems.

## E

**Edge Computing**: Processing data near the source (on the robot) rather than in the cloud to reduce latency.

**Embodied AI**: Artificial intelligence integrated into physical systems like robots.

**Encoder**: A device that measures the position or speed of rotating components like motors.

**Environment Mapping**: The process of creating a representation of the robot's surroundings.

**Episode**: A complete sequence of interaction between an agent and its environment in reinforcement learning.

**Event Handling**: The process of responding to specific occurrences in a program, such as sensor data arrival.

**Executor**: In ROS 2, a component that manages the execution of nodes and callbacks.

## F

**Feedback Controller**: A control system that uses sensor data to adjust robot behavior and reduce errors.

**Field Robot**: A robot designed to operate in unstructured outdoor environments.

**Focal Length**: The distance between a camera's optical center and its sensor, affecting the field of view.

**Forward Kinematics**: The process of calculating the position of a robot's end-effector based on joint angles.

**F-Stop**: A measure of camera lens aperture size, affecting exposure and depth of field.

## G

**Gazebo**: A physics-based simulation environment for robotics development and testing.

**Geometric Constraints**: Limitations on the position and movement of robot components based on physical structure.

**Gimbal**: A mount that allows an object to remain level independent of platform movement, used in camera stabilization.

**GPU (Graphics Processing Unit)**: A specialized processor designed for rendering graphics and parallel computation.

**Ground Plane**: The surface on which robots operate in simulation environments.

**Ground Truth**: The actual state of a system used as a reference for evaluating sensor accuracy.

**Gyroscope**: A sensor that measures angular velocity and helps determine orientation.

## H

**Hardware-in-the-loop**: A testing approach that includes actual hardware components in a simulated environment.

**Heuristic**: A practical approach to problem-solving that is not guaranteed to be optimal but is sufficient for immediate goals.

**Homogeneous Coordinates**: A mathematical representation used in robotics for transformations in 3D space.

**Human-Robot Interaction (HRI)**: The study and implementation of communication and collaboration between humans and robots.

**Humanoid**: A robot designed with human-like characteristics, especially in form and movement capabilities.

**Hyperparameter**: A parameter in a learning algorithm that is set before the learning process begins.

## I

**ICP (Iterative Closest Point)**: An algorithm used for aligning 3D point clouds, important for SLAM.

**Image Space**: The coordinate system of a digital image, measured in pixels.

**Imitation Learning**: A machine learning approach where an agent learns to perform tasks by observing expert demonstrations.

**Impedance Control**: A control strategy that regulates the relationship between force and position in robotic systems.

**Inertial Measurement Unit (IMU)**: A device that measures and reports velocity, orientation, and gravitational forces.

**Inference**: The process of using a trained model to make predictions on new data.

**Institutional Safety**: Safety protocols and standards that apply to robotic systems in educational and research institutions.

**Interface**: A shared boundary between components that allows them to communicate and interact.

**Intermediate Representation**: A simplified version of code used internally by compilers and interpreters.

**Intrinsic Parameters**: Camera properties like focal length and optical center that affect image formation.

**Isaac Sim**: NVIDIA's simulation environment for robotics and AI development.

**Isaac ROS**: NVIDIA's collection of ROS 2 packages for robotics perception and manipulation.

## J

**Jacobian**: A matrix that describes the relationship between joint velocities and end-effector velocities in robotics.

**Joint**: A connection between two rigid bodies that allows relative movement between them.

**Joint Limits**: The minimum and maximum allowable values for robot joint positions.

**Joint Space**: The configuration space defined by robot joint angles or positions.

**Jupyter Notebook**: An interactive computing environment for developing and sharing code and documentation.

## K

**Kinematic Chain**: A series of rigid bodies connected by joints that enable controlled motion.

**Kinematics**: The study of motion without considering the forces that cause it.

**Kinetic Energy**: The energy possessed by a robot due to its motion.

**Knowledge Graph**: A structured representation of knowledge that shows relationships between entities.

**Kubernetes**: An open-source platform for automating deployment, scaling, and management of containerized applications.

## L

**Laser Scanner**: A sensor that measures distances using laser light, often used for navigation and mapping.

**Latency**: The delay between an action and its result, important in controlling real-time systems.

**Launch File**: A ROS configuration file that starts multiple nodes with specified parameters and settings.

**LIDAR (Light Detection and Ranging)**: A sensing method that measures distances using laser light.

**Linear Actuator**: A device that creates linear motion, often used in robot joints and mechanisms.

**Linear Interpolation**: A method of estimating values between two known values by drawing a straight line.

**Linux**: An open-source operating system commonly used in robotics for its stability and real-time capabilities.

**LiDAR**: See LIDAR.

**Load Balancing**: Distributing computational work across multiple systems to optimize performance.

**Local Area Network (LAN)**: A network that connects devices within a limited area like a laboratory.

**Localization**: The process of determining a robot's position and orientation in its environment.

**Long Short-Term Memory (LSTM)**: A type of neural network architecture designed for processing sequential data.

## M

**Machine Learning**: A subset of artificial intelligence that enables systems to learn and improve from experience.

**Manipulation**: The capability of a robot to physically interact with objects in its environment.

**Map**: A representation of an environment used for navigation and path planning.

**Marker**: A visual element in RViz that provides graphical representation of data like poses or paths.

**Master**: In ROS 1, a central component that enables communication between distributed nodes (replaced by DDS in ROS 2).

**Middleware**: Software that provides services and support for applications, such as ROS for robotics.

**Motion Capture**: The process of recording human movements for use in animation or robot programming.

**Motion Planning**: The process of determining a valid path for a robot to move from start to goal.

**Multimodal**: Systems that process multiple types of sensory input, such as vision, audio, and tactile.

## N

**Navigation**: The capability of a robot to move from one place to another while avoiding obstacles.

**Network Topology**: The arrangement of nodes and connections in a communication network.

**Neural Network**: A computational model inspired by biological neural networks, used in machine learning.

**Node**: In ROS, a process that performs computation related to robot operations.

**Notification**: A message sent to inform other systems about events or changes in a system.

**NumPy**: A Python library for numerical computing, commonly used in robotics and AI.

## O

**Object Detection**: The process of identifying and locating objects within images or scenes.

**Occupancy Grid**: A 2D representation of an environment where each cell represents the probability of occupancy.

**Odometry**: The process of estimating position based on motion sensors and wheel encoders.

**Open Source**: Software with source code that is freely available for use, modification, and distribution.

**OpenCV**: An open-source computer vision library with functions for image and video processing.

**Operator**: A person who controls or monitors robotic systems.

**Optimization**: The process of adjusting parameters to improve system performance.

**Orientability**: The capability of determining and maintaining orientation in 3D space.

**OS (Operating System)**: Software that manages computer hardware and software resources, such as Ubuntu.

## P

**Package**: In ROS, a modular unit that contains libraries, nodes, and other resources.

**Parameter**: A value that can be adjusted to change the behavior of a system.

**Path Planning**: The process of determining a valid route for a robot to follow.

**Perception**: The ability of a robot to interpret sensory information from its environment.

**PID Controller**: A control loop mechanism using proportional, integral, and derivative terms.

**Point Cloud**: A set of data points in 3D space, typically generated by 3D scanning devices.

**Pose**: The position and orientation of an object in 3D space.

**Pr2**: A humanoid robot platform developed by Willow Garage for research purposes.

**Practical**: Relating to real-world application rather than theory, emphasizing hands-on implementation.

**Precision**: The quality of being exact and accurate, important in robot movements and measurements.

**Predictive**: Systems that anticipate future states or events based on current data.

**Processor**: A component that performs computations, such as a CPU or GPU.

**Programming Language**: A formal language used to write instructions for computers, such as Python or C++.

**Protocol**: A set of rules governing communication between systems.

**Proximity Sensor**: A sensor that detects nearby objects without physical contact.

## Q

**QoS (Quality of Service)**: In ROS 2, policies that define how messages are delivered between nodes.

**Quaternion**: A mathematical representation of rotations in 3D space, avoiding issues like gimbal lock.

## R

**Radar**: A system that uses radio waves to detect objects and measure distances.

**Real-time**: Systems that must respond to inputs within strict time constraints.

**Recurrent Neural Network (RNN)**: A neural network architecture designed for sequential data processing.

**Reinforcement Learning**: A machine learning paradigm where agents learn to make decisions through rewards.

**Remote**: Control or monitoring of systems from a distance.

**Repository**: A storage location for software code and related files.

**Representation**: The way information is structured and stored within a system.

**Request-Response**: A communication pattern where one system requests information and another responds.

**Resource Management**: The process of allocating and managing computational resources efficiently.

**Return on Investment (ROI)**: A measure of the profitability of an investment, relevant for robotics projects.

**Robot**: An autonomous or semi-autonomous machine that can perform tasks automatically or with guidance.

**Robotics**: The interdisciplinary field encompassing mechanical, electrical, and computer engineering for robot design.

**Robotics Middleware**: Software infrastructure that facilitates communication between robot software components.

**Robotics Operating System (ROS)**: A flexible framework for writing robot software (ROS 1) or collection of tools and conventions (ROS 2).

**ROS 2**: The second generation of the Robot Operating System with improved security, reliability, and real-time capabilities.

**ROS Bridge**: A component that connects ROS with other systems or frameworks, such as web browsers.

**ROS Distribution**: A version of ROS with specific sets of packages and dependencies.

**ROS Ecosystem**: The collection of tools, packages, and community resources available for ROS development.

**ROS Package**: A modular unit containing libraries, nodes, and other resources organized together.

**Robot State Publisher**: A ROS node that publishes transformations between coordinate frames for visualization.

**RViz**: The 3D visualization tool for ROS that allows visualization of robot models and sensor data.

**Runtime**: The time during which a program is executing rather than being compiled.

## S

**Safety**: Measures and protocols designed to prevent harm to humans, robots, or the environment.

**Scene Understanding**: The process of interpreting and reasoning about the content and layout of a 3D environment.

**Scheduling**: The process of ordering and timing tasks for optimal system performance.

**Script**: A program written in a high-level language, often used for automation and system control.

**SDK (Software Development Kit)**: A collection of tools, documentation, and examples for developing software.

**Search Algorithm**: A method for finding solutions to problems by exploring possible options.

**Semantic Segmentation**: The process of categorizing each pixel in an image according to its meaning.

**Sensor**: A device that detects and responds to physical inputs such as light, heat, or motion.

**Service**: In ROS, a communication pattern where a client sends a request and receives a response.

**Service Robot**: A robot that performs useful tasks in human environments, such as cleaning or delivery.

**Simulation**: The imitation of real-world processes in a virtual environment.

**SLAM (Simultaneous Localization and Mapping)**: The process of building a map while determining position within it.

**SMACH (State Machine for Autonomous Systems)**: A behavior architecture for programming robot states.

**Sociable Robot**: A robot designed to interact with humans in socially acceptable ways.

**Software Component**: A modular unit of software that performs specific functions.

**Software Development**: The process of designing, creating, testing, and maintaining software applications.

**Solver**: A computational method for finding solutions to mathematical or optimization problems.

**Space Robot**: A robot designed for operation in space environments.

**Spatial**: Relating to space and the position of objects within it.

**Spin**: In ROS, the process of executing pending callbacks in the main thread.

**Standard Operating Procedure (SOP)**: A set of step-by-step instructions for performing specific tasks.

**State Machine**: A model of computation used to design algorithms that respond to inputs based on current state.

**Static**: Unchanging or not moving, often used to describe fixed parameters or positions.

**Subsumption Architecture**: A behavior-based approach to robot programming with multiple behavior layers.

**Supervised Learning**: A machine learning approach using labeled training data to learn input-output mappings.

**System Integration**: The process of combining different subsystems into a complete functional system.

**Systematic**: Done according to a system or plan, thorough and methodical.

## T

**Task Planning**: The process of determining a sequence of actions to achieve a specific goal.

**TCP/IP**: A suite of communication protocols used to interconnect network devices.

**Teleoperation**: Control of a robot by a human operator from a remote location.

**Template**: A pre-formatted document or code structure used as a starting point.

**TensorFlow**: An open-source machine learning framework developed by Google.

**Terminology**: The body of terms used in a particular subject area.

**Testbed**: An environment used for testing new technologies or procedures.

**Testing**: The process of evaluating software to ensure it meets specified requirements.

**Thread**: A lightweight process that can execute concurrently with others.

**Throughput**: The rate at which a system processes tasks, important for real-time systems.

**Time Complexity**: A measure of the computational complexity characterizing an algorithm's operation time.

**Topic**: In ROS, a communication channel where messages are published and subscribed.

**Transform**: A mathematical operation that changes the position, orientation, or scale of objects.

**Trajectory**: The path followed by a moving system over time.

**Training**: The process of teaching a machine learning model using example data.

**Transportation**: The movement of objects or people from one location to another.

**Tunable**: Capable of being adjusted or fine-tuned for optimal performance.

**Type Checking**: The process of verifying that values have the correct data types.

## U

**UDP (User Datagram Protocol)**: A communications protocol for sending data without requiring connection establishment.

**UI (User Interface)**: The space where interactions occur between human users and a computer system.

**Underactuated**: A robotic system with fewer control inputs than degrees of freedom.

**Undocumented**: Code or systems that lack adequate explanatory documentation.

**Unit Testing**: Testing individual components of software in isolation.

**Unity**: A popular game engine used for creating simulation environments in robotics.

**Unity Robotics Hub**: NVIDIA's package for integrating Unity with ROS for robotics simulation.

**Universal Robot Description Format (URDF)**: An XML format for representing robot models in ROS.

**Unstructured Environment**: An environment without predetermined layout or organization.

**URDF**: See Universal Robot Description Format.

**URLError**: An error related to network or URL access in software applications.

**UX (User Experience)**: The overall experience of a person using a product or system.

## V

**Variance**: A measure of how spread out values are in a dataset, important in sensor accuracy assessment.

**Vector**: A quantity with both magnitude and direction, used in robotics for position and force.

**Velocity**: The rate of change of displacement, including both speed and direction.

**Version Control**: A system for tracking and managing changes to code and documents.

**Video Capture**: The process of recording video from cameras or simulation environments.

**Virtual Reality (VR)**: A simulated experience that can be similar to or completely different from the real world.

**Vision System**: A system that processes visual information from cameras or other optical sensors.

**Vision-Language-Action (VLA)**: A system combining computer vision, natural language processing, and robotic action.

**Visual Servoing**: Controlling robot motion based on visual feedback.

**VSLAM (Visual Simultaneous Localization and Mapping)**: SLAM using visual sensors as primary input.

## W

**Warehouse Automation**: The use of robots and automated systems in warehouse environments.

**Web Interface**: A user interface accessed through a web browser.

**Wheel Encoder**: A sensor that measures the rotation of wheels for odometry calculations.

**Whisper**: OpenAI's speech recognition system for converting audio to text.

**Workspace**: The space in which a robot can operate, either physical or virtual.

**World File**: A file defining the environment in simulation software like Gazebo.

## X

**XACRO**: An XML macro language for the Robot Operating System that extends URDF.

## Y

**Yaw**: Rotation around the vertical (Z) axis of a robot or vehicle.

## Z

**Z-Buffer**: A computer graphics technique for determining object visibility.

**Zero Configuration Networking**: Automatic network setup without manual configuration.

## Acronyms and Abbreviations

**API**: Application Programming Interface

**DDS**: Data Distribution Service (middleware used in ROS 2)

**DNN**: Deep Neural Network

**FPS**: Frames Per Second (camera rate)

**GPU**: Graphics Processing Unit

**HRI**: Human-Robot Interaction

**IMU**: Inertaneous Measurement Unit

**IoT**: Internet of Things

**LAN**: Local Area Network

**LIDAR**: Light Detection and Ranging

**LLM**: Large Language Model

**ML**: Machine Learning

**NLP**: Natural Language Processing

**QoS**: Quality of Service

**RGB**: Red, Green, Blue color model

**RGB-D**: Color + Depth camera

**ROS**: Robot Operating System

**RPM**: Revolutions Per Minute

**RTK**: Real-Time Kinematic (GPS positioning)

**SBC**: Single Board Computer

**SDK**: Software Development Kit

**SOP**: Standard Operating Procedure

**TCP**: Transmission Control Protocol

**UDP**: User Datagram Protocol

**URDF**: Universal Robot Description Format

**USB**: Universal Serial Bus

**VLA**: Vision-Language-Action

**VSLAM**: Visual Simultaneous Localization and Mapping

**WLAN**: Wireless Local Area Network

**XML**: Extensible Markup Language

**YAML**: YAML Ain't Markup Language