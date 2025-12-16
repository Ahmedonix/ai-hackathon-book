# Practical Hands-On Exercise: Unity Integration for Humanoid Robotics Simulation

## Exercise Overview

In this hands-on exercise, you'll learn to integrate Unity with your ROS 2-based humanoid robot simulation. You'll create a high-fidelity visualization layer that complements your Gazebo simulation and provides enhanced visual representation of your robot and its environment.

## Learning Objectives

By the end of this exercise, you will be able to:
1. Set up Unity with the ROS TCP Connector for communication
2. Create a Unity scene with a humanoid robot model
3. Implement real-time synchronization between Gazebo and Unity
4. Visualize sensor data in Unity
5. Implement a user interface for robot control in Unity
6. Debug and optimize Unity-ROS communication

## Prerequisites

Before starting this exercise, you should have:
- Completed the Gazebo setup exercise
- Completed the sensor simulation exercise
- Completed the environment design exercise
- A working ROS 2 Iron installation
- A functioning humanoid robot simulation
- Unity Hub with Unity 2022.3 LTS installed
- Basic C# programming knowledge

## Part 1: Setting Up Unity Development Environment

### Step 1.1: Install Required Unity Packages

First, let's ensure you have the right tools in your Unity project.

1. Open Unity Hub and create a new 3D project called "HumanoidRobotVisualization"
2. In the Project window, click "Packages" → "Package Manager"
3. Install the following packages:
   - **ROS TCP Connector** from GitHub: https://github.com/Unity-Technologies/ROS-TCP-Connector
   - **Universal Render Pipeline** (URP) if not already using it

### Step 1.2: Verify ROS TCP Endpoint Installation

On your ROS 2 side, ensure you have the TCP endpoint installed:

```bash
# Install the ROS TCP endpoint package
sudo apt install ros-iron-ros-tcp-endpoint

# Alternatively, you can install it directly from source
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
cd ROS-TCP-Endpoint
colcon build
source install/setup.bash
```

## Part 2: Creating the ROS Connection Manager in Unity

### Step 2.1: Create Connection Manager Script

In your Unity project, create a new C# script named `ROSConnectionManager.cs` in `Assets/Scripts/`:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using System.Collections.Concurrent;
using System.Threading;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("Connection Settings")]
    [Tooltip("IP address of the ROS system")]
    public string rosIPAddress = "127.0.0.1";
    
    [Tooltip("Port number for ROS communication")]
    public int rosPort = 10000;
    
    [Header("Connection Options")]
    public bool autoConnectOnStart = true;
    public bool logConnectionStatus = true;
    
    private static ROSConnectionManager s_Instance;
    private ROSConnection m_ROSConnection;
    private ConcurrentQueue<System.Action> m_ActionQueue = new ConcurrentQueue<System.Action>();
    
    // Singleton pattern implementation
    public static ROSConnectionManager Instance
    {
        get
        {
            if (s_Instance == null)
            {
                var go = new GameObject("ROSConnectionManager");
                s_Instance = go.AddComponent<ROSConnectionManager>();
            }
            return s_Instance;
        }
    }
    
    void Awake()
    {
        // Ensure singleton pattern
        if (s_Instance == null)
        {
            s_Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else if (s_Instance != this)
        {
            Destroy(gameObject);
            return;
        }
    }
    
    void Start()
    {
        if (autoConnectOnStart)
        {
            ConnectToROS();
        }
    }
    
    public void ConnectToROS()
    {
        if (m_ROSConnection == null)
        {
            m_ROSConnection = gameObject.AddComponent<ROSConnection>();
        }
        
        m_ROSConnection.Initialize(rosIPAddress, rosPort);
        
        if (logConnectionStatus)
        {
            Debug.Log($"Attempting to connect to ROS at {rosIPAddress}:{rosPort}");
        }
    }
    
    public void DisconnectFromROS()
    {
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Close();
            m_ROSConnection = null;
            
            if (logConnectionStatus)
            {
                Debug.Log("Disconnected from ROS");
            }
        }
    }
    
    public ROSConnection GetConnection()
    {
        return m_ROSConnection;
    }
    
    public bool IsConnected()
    {
        return m_ROSConnection != null;
    }
    
    public void EnqueueAction(System.Action action)
    {
        m_ActionQueue.Enqueue(action);
    }
    
    void Update()
    {
        // Process actions queued from other threads on the main thread
        while (m_ActionQueue.TryDequeue(out System.Action action))
        {
            try
            {
                action?.Invoke();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Exception when executing queued action: {e}");
            }
        }
    }
    
    void OnApplicationQuit()
    {
        DisconnectFromROS();
    }
}
```

### Step 2.2: Create the Humanoid Robot Model

Create a humanoid robot model representation in Unity. First, let's create a script to handle the robot model: `HumanoidRobotModel.cs`:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using System.Collections.Generic;

[RequireComponent(typeof(ROSConnectionManager))]
public class HumanoidRobotModel : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotRoot;  // Main game object containing the robot hierarchy
    
    [Header("Joint Mapping")]
    [Tooltip("Assign each joint in your robot with its corresponding GameObject in the scene")]
    public List<JointMapping> jointMappings;
    
    [Header("Synchronization Settings")]
    [Range(1, 100)]
    public int updatesPerSecond = 30;  // Rate of synchronization with ROS
    
    [Header("Visualization Settings")]
    public bool visualizeJointLimits = true;
    public bool visualizeSensorData = true;
    
    private ROSConnection m_ROSConnection;
    private float m_LastUpdate;
    private JointStateMsg m_LastJointState;
    
    [System.Serializable]
    public class JointMapping
    {
        [Tooltip("The name of the joint as it appears in ROS/URDF")]
        public string jointName;
        
        [Tooltip("The Transform of the joint in the Unity scene")]
        public Transform jointTransform;
        
        [Tooltip("Which axis is affected by this joint (for revolute joints)")]
        public JointAxis axis = JointAxis.X;
        
        [Tooltip("Offset in degrees")] 
        public float offset = 0f;
        
        [Tooltip("Minimum joint limit in degrees")]
        public float minLimit = -180f;
        
        [Tooltip("Maximum joint limit in degrees")]
        public float maxLimit = 180f;
    }
    
    public enum JointAxis { X, Y, Z }
    
    void Start()
    {
        // Ensure the connection manager exists
        var connManager = GetComponent<ROSConnectionManager>();
        if (connManager == null)
        {
            connManager = gameObject.AddComponent<ROSConnectionManager>();
        }
        
        m_ROSConnection = connManager.GetConnection();
        if (m_ROSConnection == null)
        {
            Debug.LogError("No ROS connection available. Please ensure ROSConnectionManager is configured properly.");
            return;
        }
        
        // Subscribe to joint states topic
        m_ROSConnection.Subscribe<JointStateMsg>("/joint_states", OnJointStateReceived);
        
        // Start synchronization
        m_LastUpdate = Time.time;
        
        Debug.Log("Humanoid Robot Model Initialized");
    }
    
    void Update()
    {
        // Update at the specified rate
        float timeSinceLastUpdate = Time.time - m_LastUpdate;
        if (timeSinceLastUpdate >= 1.0f / updatesPerSecond)
        {
            ProcessJointState();
            m_LastUpdate = Time.time;
        }
    }
    
    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Store the latest joint state - process in main thread
        m_LastJointState = jointState;
    }
    
    void ProcessJointState()
    {
        if (m_LastJointState == null) return;
        
        // Process each joint state and update the corresponding Unity transform
        for (int i = 0; i < m_LastJointState.name.Length; i++)
        {
            string jointName = m_LastJointState.name[i];
            double jointPosition = m_LastJointState.position[i];
            
            // Find the corresponding joint in our mapping
            JointMapping mapping = jointMappings.Find(jm => jm.jointName == jointName);
            if (mapping != null && mapping.jointTransform != null)
            {
                // Convert from radians to degrees
                float jointAngle = (float)(jointPosition * Mathf.Rad2Deg) + mapping.offset;
                
                // Apply joint limits if enabled
                if (visualizeJointLimits)
                {
                    jointAngle = Mathf.Clamp(jointAngle, mapping.minLimit, mapping.maxLimit);
                }
                
                // Apply rotation based on axis
                Vector3 rotation = mapping.jointTransform.localEulerAngles;
                switch (mapping.axis)
                {
                    case JointAxis.X:
                        rotation.x = jointAngle;
                        break;
                    case JointAxis.Y:
                        rotation.y = jointAngle;
                        break;
                    case JointAxis.Z:
                        rotation.z = jointAngle;
                        break;
                }
                
                mapping.jointTransform.localEulerAngles = rotation;
            }
            else
            {
                // Optional: Log unknown joints
                // Debug.Log($"Joint '{jointName}' not found in Unity model");
            }
        }
    }
    
    private void OnValidate()
    {
        if (jointMappings == null) return;
        
        // Validate joint mappings
        for (int i = 0; i < jointMappings.Count; i++)
        {
            if (jointMappings[i] == null)
            {
                jointMappings[i] = new JointMapping();
            }
        }
    }
}
```

## Part 3: Implementing Sensor Data Visualization

### Step 3.1: Create Sensor Visualization Script

Create `SensorVisualizer.cs` to handle sensor data visualization:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;
using System.Collections.Generic;

public class SensorVisualizer : MonoBehaviour
{
    [Header("LiDAR Visualization")]
    public GameObject laserPointPrefab;
    public Color laserColor = Color.red;
    public float laserPointSize = 0.05f;
    public int maxLaserPoints = 1000;
    
    [Header("Camera Feed")]
    public UnityEngine.UI.RawImage cameraFeedDisplay;
    public int cameraImageWidth = 640;
    public int cameraImageHeight = 480;
    
    [Header("IMU Visualization")]
    public Transform imuVisualizer;
    public float imuScale = 0.1f;
    
    private ROSConnection m_ROSConnection;
    private List<GameObject> m_LaserPoints = new List<GameObject>();
    private Texture2D m_CameraTexture;
    
    void Start()
    {
        var connManager = FindObjectOfType<ROSConnectionManager>();
        if (connManager != null)
        {
            m_ROSConnection = connManager.GetConnection();
        }
        
        if (m_ROSConnection == null)
        {
            Debug.LogError("No ROS connection found. Ensure ROSConnectionManager is properly set up.");
            return;
        }
        
        // Subscribe to various sensor topics
        m_ROSConnection.Subscribe<LaserScanMsg>("/scan", OnLaserScanReceived);
        m_ROSConnection.Subscribe<ImageMsg>("/camera/image_raw", OnCameraImageReceived);
        m_ROSConnection.Subscribe<ImuMsg>("/imu", OnImuReceived);
        m_ROSConnection.Subscribe<OdomMsg>("/odom", OnOdometryReceived);
        
        // Initialize camera texture if needed
        if (cameraFeedDisplay != null)
        {
            m_CameraTexture = new Texture2D(cameraImageWidth, cameraImageHeight, TextureFormat.RGB24, false);
            cameraFeedDisplay.texture = m_CameraTexture;
        }
        
        Debug.Log("Sensor Visualizer Initialized");
    }
    
    void OnLaserScanReceived(LaserScanMsg scanMsg)
    {
        // Use the connection manager's action queue to ensure thread safety
        ROSConnectionManager.Instance.EnqueueAction(() => ProcessLaserScan(scanMsg));
    }
    
    void OnCameraImageReceived(ImageMsg imageMsg)
    {
        ROSConnectionManager.Instance.EnqueueAction(() => ProcessCameraImage(imageMsg));
    }
    
    void OnImuReceived(ImuMsg imuMsg)
    {
        ROSConnectionManager.Instance.EnqueueAction(() => ProcessImuData(imuMsg));
    }
    
    void OnOdometryReceived(OdomMsg odomMsg)
    {
        ROSConnectionManager.Instance.EnqueueAction(() => ProcessOdometryData(odomMsg));
    }
    
    void ProcessLaserScan(LaserScanMsg scanMsg)
    {
        // Clear existing laser points
        foreach (var point in m_LaserPoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
        m_LaserPoints.Clear();
        
        // Only visualize a subset of points to maintain performance
        float angleStep = scanMsg.angle_increment * (scanMsg.ranges.Length / 100f);  // Only visualize ~100 points
        if (angleStep < 1) angleStep = 1;  // Ensure we sample at least every degree
        
        // Create new laser points for valid ranges
        for (int i = 0; i < scanMsg.ranges.Length; i += (int)angleStep)
        {
            float range = (float)scanMsg.ranges[i];
            
            // Only visualize if range is valid (not NaN or Infinity)
            if (float.IsNaN(range) || float.IsInfinity(range) || range > scanMsg.range_max || range < scanMsg.range_min)
                continue;
            
            // Calculate angle for this laser beam
            float angle = (float)(scanMsg.angle_min + (i * scanMsg.angle_increment));
            
            // Calculate position in 2D space (relative to robot)
            float x = range * Mathf.Cos(angle);
            float y = 0f;
            float z = range * Mathf.Sin(angle);
            
            // Create laser point
            if (laserPointPrefab != null)
            {
                GameObject laserPoint = Instantiate(laserPointPrefab, transform);
                
                // Position the point
                laserPoint.transform.localPosition = new Vector3(x, y, z);
                laserPoint.transform.localScale = Vector3.one * laserPointSize;
                
                // Apply color if the renderer supports it
                var renderer = laserPoint.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = new Material(renderer.material);
                    renderer.material.color = laserColor;
                }
                
                m_LaserPoints.Add(laserPoint);
                
                // Limit number of points for performance
                if (m_LaserPoints.Count >= maxLaserPoints)
                    break;
            }
        }
    }
    
    void ProcessCameraImage(ImageMsg imageMsg)
    {
        if (cameraFeedDisplay == null) return;
        
        // Check if we need to resize the texture
        if (m_CameraTexture.width != (int)imageMsg.width || m_CameraTexture.height != (int)imageMsg.height)
        {
            DestroyImmediate(m_CameraTexture);
            m_CameraTexture = new Texture2D((int)imageMsg.width, (int)imageMsg.height, TextureFormat.RGB24, false);
            cameraFeedDisplay.texture = m_CameraTexture;
        }
        
        // Convert ROS Image message to Unity Texture
        // This assumes RGB8 format for simplicity
        if (imageMsg.encoding == "rgb8")
        {
            Color32[] colors = new Color32[imageMsg.data.Length / 3];
            
            for (int i = 0; i < colors.Length; i++)
            {
                int dataIndex = i * 3;
                colors[i] = new Color32(
                    imageMsg.data[dataIndex],
                    imageMsg.data[dataIndex + 1],
                    imageMsg.data[dataIndex + 2],
                    255
                );
            }
            
            m_CameraTexture.SetPixels32(colors);
            m_CameraTexture.Apply();
        }
        else
        {
            // For other formats, you would need to handle accordingly
            Debug.LogWarning($"Unsupported image format: {imageMsg.encoding}");
        }
    }
    
    void ProcessImuData(ImuMsg imuMsg)
    {
        if (imuVisualizer == null) return;
        
        // Extract orientation from quaternion
        Quaternion orientation = new Quaternion(
            (float)imuMsg.orientation.x,
            (float)imuMsg.orientation.y,
            (float)imuMsg.orientation.z,
            (float)imuMsg.orientation.w
        );
        
        // Apply to the visualization object
        imuVisualizer.rotation = orientation;
        imuVisualizer.localScale = Vector3.one * imuScale;
    }
    
    void ProcessOdometryData(OdomMsg odomMsg)
    {
        // Update robot position based on odometry
        transform.position = new Vector3(
            (float)odomMsg.pose.pose.position.x,
            (float)odomMsg.pose.pose.position.y,
            (float)odomMsg.pose.pose.position.z
        );
        
        transform.rotation = new Quaternion(
            (float)odomMsg.pose.pose.orientation.x,
            (float)odomMsg.pose.pose.orientation.y,
            (float)odomMsg.pose.pose.orientation.z,
            (float)odomMsg.pose.pose.orientation.w
        );
    }
    
    void OnDestroy()
    {
        // Clean up laser points
        foreach (var point in m_LaserPoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
        
        // Clean up texture
        if (m_CameraTexture != null)
        {
            DestroyImmediate(m_CameraTexture);
        }
    }
}
```

## Part 4: Creating Robot Control Interface

### Step 4.1: Create Control Interface Script

Create `RobotControlInterface.cs` to allow user interaction with the robot:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotControlInterface : MonoBehaviour
{
    [Header("Movement Controls")]
    public float linearVelocity = 1.0f;
    public float angularVelocity = 1.0f;
    
    [Header("UI Elements")]
    public UnityEngine.UI.Button forwardButton;
    public UnityEngine.UI.Button backwardButton;
    public UnityEngine.UI.Button leftButton;
    public UnityEngine.UI.Button rightButton;
    public UnityEngine.UI.Button stopButton;
    public UnityEngine.UI.Slider linearSpeedSlider;
    public UnityEngine.UI.Slider angularSpeedSlider;
    public UnityEngine.UI.Text statusText;
    
    [Header("Control Mode")]
    public bool useKeyboardControls = true;
    public bool useUIControls = true;
    
    private ROSConnection m_ROSConnection;
    private bool m_IsConnected = false;
    private Vector3 m_CurrentVelocity;
    
    void Start()
    {
        var connManager = FindObjectOfType<ROSConnectionManager>();
        if (connManager != null)
        {
            m_ROSConnection = connManager.GetConnection();
            m_IsConnected = m_ROSConnection != null;
        }
        
        SetupUI();
        
        if (statusText != null)
        {
            statusText.text = m_IsConnected ? "Connected to ROS" : "Not Connected to ROS";
        }
        
        Debug.Log("Robot Control Interface Initialized");
    }
    
    void SetupUI()
    {
        if (forwardButton != null)
            forwardButton.onClick.AddListener(() => OnDirectionButtonClicked(Vector3.forward));
        
        if (backwardButton != null)
            backwardButton.onClick.AddListener(() => OnDirectionButtonClicked(Vector3.back));
        
        if (leftButton != null)
            leftButton.onClick.AddListener(() => OnDirectionButtonClicked(Vector3.left));
        
        if (rightButton != null)
            rightButton.onClick.AddListener(() => OnDirectionButtonClicked(Vector3.right));
        
        if (stopButton != null)
            stopButton.onClick.AddListener(StopRobot);
        
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.onValueChanged.AddListener(OnLinearSpeedChanged);
            linearSpeedSlider.SetValueWithoutNotify(linearVelocity);
        }
        
        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);
            angularSpeedSlider.SetValueWithoutNotify(angularVelocity);
        }
    }
    
    void Update()
    {
        if (useKeyboardControls)
        {
            HandleKeyboardInput();
        }
    }
    
    void HandleKeyboardInput()
    {
        Vector3 velocity = Vector3.zero;
        
        // Linear movement (WASD or arrow keys)
        if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
            velocity += Vector3.forward;
        if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            velocity += Vector3.back;
        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            velocity += Vector3.left;
        if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            velocity += Vector3.right;
        
        // Angular movement (Q and E for turning)
        if (Input.GetKey(KeyCode.Q))
            m_CurrentVelocity = new Vector3(0, angularVelocity, 0);  // Turn left
        else if (Input.GetKey(KeyCode.E))
            m_CurrentVelocity = new Vector3(0, -angularVelocity, 0); // Turn right
        else if (velocity.magnitude > 0.1f)
            // Only set linear velocity if not rotating
            m_CurrentVelocity = velocity.normalized * linearVelocity;
        else
            // Stop if no keys are pressed
            m_CurrentVelocity = Vector3.zero;
        
        // Send velocity command if connected
        if (m_IsConnected && m_ROSConnection != null)
        {
            SendVelocityCommand();
        }
    }
    
    void OnDirectionButtonClicked(Vector3 direction)
    {
        if (!useUIControls) return;
        
        m_CurrentVelocity = direction * linearVelocity;
        
        if (m_IsConnected && m_ROSConnection != null)
        {
            SendVelocityCommand();
        }
    }
    
    void OnLinearSpeedChanged(float value)
    {
        linearVelocity = value;
    }
    
    void OnAngularSpeedChanged(float value)
    {
        angularVelocity = value;
    }
    
    void StopRobot()
    {
        m_CurrentVelocity = Vector3.zero;
        
        if (m_IsConnected && m_ROSConnection != null)
        {
            SendVelocityCommand();
        }
    }
    
    void SendVelocityCommand()
    {
        if (m_ROSConnection == null) return;
        
        // Create a Twist message for differential drive control
        var twist = new TwistMsg();
        
        // Note: This assumes a simplified control model
        // You may need to adapt this based on your actual robot's control interface
        twist.linear = new Vector3Msg(m_CurrentVelocity.x, 0, m_CurrentVelocity.z);
        twist.angular = new Vector3Msg(0, 0, -m_CurrentVelocity.y);
        
        // Publish to the robot's velocity command topic
        m_ROSConnection.Publish("/cmd_vel", twist);
    }
    
    public void SetLinearVelocity(float velocity)
    {
        linearVelocity = velocity;
        if (linearSpeedSlider != null)
            linearSpeedSlider.value = velocity;
    }
    
    public void SetAngularVelocity(float velocity)
    {
        angularVelocity = velocity;
        if (angularSpeedSlider != null)
            angularSpeedSlider.value = velocity;
    }
}
```

### Step 4.2: Create a UI Canvas

Create a simple UI canvas to hold your control elements:

```csharp
// Create UICanvasManager.cs
using UnityEngine;
using UnityEngine.UI;

public class UICanvasManager : MonoBehaviour
{
    [Header("Canvas References")]
    public Button forwardButton;
    public Button backwardButton;
    public Button leftButton;
    public Button rightButton;
    public Button stopButton;
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Text statusText;
    public Text infoText;
    
    [Header("Layout Settings")]
    public float buttonSize = 60f;
    public float spacing = 20f;
    
    void Start()
    {
        SetupUIElements();
    }
    
    void SetupUIElements()
    {
        // Configure buttons with appropriate positioning
        if (forwardButton != null)
            ConfigureButton(forwardButton, Vector2.zero, Vector2.up, "↑");
        
        if (backwardButton != null)
            ConfigureButton(backwardButton, Vector2.zero, Vector2.down, "↓");
        
        if (leftButton != null)
            ConfigureButton(leftButton, Vector2.zero, Vector2.left, "←");
        
        if (rightButton != null)
            ConfigureButton(rightButton, Vector2.zero, Vector2.right, "→");
        
        if (stopButton != null)
            ConfigureButton(stopButton, Vector2.zero, Vector2.zero, "STOP");
        
        // Setup sliders
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.minValue = 0f;
            linearSpeedSlider.maxValue = 2f;
            linearSpeedSlider.wholeNumbers = false;
        }
        
        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.minValue = 0f;
            angularSpeedSlider.maxValue = 2f;
            angularSpeedSlider.wholeNumbers = false;
        }
        
        // Setup info text
        if (infoText != null)
        {
            infoText.text = "Humanoid Robot Control\nUse WASD or arrow keys to move\nQ/E to rotate left/right\nConnect to ROS to control robot";
        }
    }
    
    void ConfigureButton(Button button, Vector2 position, Vector2 direction, string text)
    {
        if (button == null) return;
        
        // Get the text component and set the label
        Text buttonText = button.GetComponentInChildren<Text>();
        if (buttonText != null)
        {
            buttonText.text = text;
            buttonText.fontSize = 24;
        }
        
        // Set button size
        RectTransform rectTransform = button.GetComponent<RectTransform>();
        if (rectTransform != null)
        {
            rectTransform.sizeDelta = new Vector2(buttonSize, buttonSize);
        }
    }
}
```

## Part 5: Implementing Scene Setup and Configuration

### Step 5.1: Create Scene Setup Script

Create `SceneSetup.cs` to handle the initial unity scene setup:

```csharp
using UnityEngine;
using System.Collections;

public class SceneSetup : MonoBehaviour
{
    [Header("Scene Configuration")]
    public bool setupDefaultLighting = true;
    public bool setupDefaultCamera = true;
    public bool setupDefaultPostProcessing = true;
    
    [Header("Lighting Settings")]
    public Light mainDirectionalLight;
    public Color ambientLightColor = new Color(0.4f, 0.4f, 0.4f, 1);
    public float directionalLightIntensity = 1.0f;
    
    [Header("Camera Settings")]
    public Camera mainCamera;
    public float cameraFOV = 60f;
    public float cameraDistance = 10f;
    public float cameraHeight = 5f;
    
    [Header("Environment")]
    public GameObject groundPlane;
    public Material groundMaterial;
    
    void Start()
    {
        InitializeScene();
    }
    
    void InitializeScene()
    {
        // Setup lighting
        if (setupDefaultLighting)
        {
            SetupLighting();
        }
        
        // Setup camera
        if (setupDefaultCamera)
        {
            SetupCamera();
        }
        
        // Setup environment
        SetupEnvironment();
        
        Debug.Log("Scene Setup Complete");
    }
    
    void SetupLighting()
    {
        // Create or configure main directional light
        if (mainDirectionalLight == null)
        {
            GameObject lightObj = new GameObject("MainDirectionalLight");
            mainDirectionalLight = lightObj.AddComponent<Light>();
            mainDirectionalLight.type = LightType.Directional;
            mainDirectionalLight.transform.rotation = Quaternion.Euler(50, -120, 0);
        }
        
        mainDirectionalLight.intensity = directionalLightIntensity;
        
        // Set ambient light
        RenderSettings.ambientLight = ambientLightColor;
        
        // Enable shadows for better visual quality
        mainDirectionalLight.shadows = LightShadows.Soft;
        mainDirectionalLight.shadowStrength = 0.5f;
    }
    
    void SetupCamera()
    {
        // Get or create main camera
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
            if (mainCamera == null)
            {
                GameObject cameraObj = new GameObject("MainCamera");
                mainCamera = cameraObj.AddComponent<Camera>();
                mainCamera.tag = "MainCamera";
            }
        }
        
        // Configure camera properties
        mainCamera.fieldOfView = cameraFOV;
        
        // Position camera to view the robot
        if (mainCamera.transform.parent == null)
        {
            // Position camera behind and above the scene center
            Vector3 cameraPos = new Vector3(0, cameraHeight, -cameraDistance);
            mainCamera.transform.position = cameraPos;
            mainCamera.transform.LookAt(Vector3.zero);
        }
    }
    
    void SetupEnvironment()
    {
        // Create or configure ground plane
        if (groundPlane == null)
        {
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "GroundPlane";
            ground.GetComponent<Renderer>().material = groundMaterial ?? CreateDefaultGroundMaterial();
            groundPlane = ground;
        }
    }
    
    Material CreateDefaultGroundMaterial()
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.7f, 0.7f, 0.7f);
        mat.SetFloat("_Metallic", 0.1f);
        mat.SetFloat("_Smoothness", 0.1f);
        return mat;
    }
    
    public void ResetScene()
    {
        // Reset any runtime changes made to the scene
        if (mainCamera != null)
        {
            Vector3 cameraPos = new Vector3(0, cameraHeight, -cameraDistance);
            mainCamera.transform.position = cameraPos;
            mainCamera.transform.LookAt(Vector3.zero);
        }
    }
    
    void OnValidate()
    {
        // Validate configuration values
        if (cameraDistance <= 0) cameraDistance = 5f;
        if (cameraHeight <= 0) cameraHeight = 3f;
        if (cameraFOV <= 0) cameraFOV = 60f;
    }
}
```

## Part 6: Testing the Unity Integration

### Step 6.1: Create a Test Scene

Now let's put everything together in a test scene:

1. In Unity, create a new scene (File → New Scene)
2. Create an empty GameObject called "RobotSystem"
3. Add the following components:
   - ROSConnectionManager
   - HumanoidRobotModel
   - SensorVisualizer
   - RobotControlInterface
   - SceneSetup

### Step 6.2: Configure the Robot Model

1. Create a simple humanoid robot in your scene using basic geometric shapes:
   - Capsule for torso
   - Capsules for limbs
   - Sphere for head
2. Arrange them hierarchically with proper parenting
3. Create JointMapping entries in the HumanoidRobotModel component for each joint

### Step 6.3: Configure Connection Parameters

1. Set the ROS IP address to "127.0.0.1" (localhost)
2. Set the port to 10000 (default for ROS TCP Connector)
3. Make sure the joint names match those in your URDF model

### Step 6.4: Testing Procedure

1. First, start the ROS TCP endpoint:
```bash
# In terminal 1:
source /opt/ros/iron/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000
```

2. Start your Gazebo simulation:
```bash
# In terminal 2:
source ~/humanoid_ws/install/setup.bash
ros2 launch humanoid_simple_robot sensor_demo.launch.py  # or your robot's launch file
```

3. Press Play in Unity and verify:
   - The Unity-ROS connection is established
   - Robot model updates based on joint states from Gazebo
   - Sensor data is visualized (LiDAR points, camera feed, IMU)
   - Robot controls work (if published to appropriate topics)

## Part 7: Advanced Unity Features

### Step 7.1: Creating Custom Shaders for Robot Visualization

Create a custom shader to visualize robot status. Create `Assets/Shaders/RobotStatus.shader`:

```
Shader "Custom/RobotStatus"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _StatusIntensity ("Status Intensity", Range(0, 2)) = 1.0
        _StatusColor ("Status Color", Color) = (0, 1, 0, 1)  // Green for OK
        _Glossiness ("Smoothness", Range(0,1)) = 0.5
        _Metallic ("Metallic", Range(0,1)) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;

        struct Input
        {
            float2 uv_MainTex;
        };

        half _Glossiness;
        half _Metallic;
        fixed4 _Color;
        fixed4 _StatusColor;
        float _StatusIntensity;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
            
            // Add status visualization effect
            fixed3 statusEffect = _StatusColor.rgb * _StatusIntensity;
            
            o.Albedo = c.rgb + statusEffect;
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha = c.a;
        }
        ENDCG
    }
    Fallback "Diffuse"
}
```

### Step 7.2: Creating Animation Controller for Robot

Create a simple animation controller that responds to sensor data:

```csharp
// RobotAnimationController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class RobotAnimationController : MonoBehaviour
{
    [Header("Animation Parameters")]
    public Animator animator;
    public float stepThreshold = 0.5f;
    public float balanceThreshold = 0.2f;
    
    [Header("Sensor Data")]
    public float lastLaserDistance = float.MaxValue;
    public Vector3 lastImuOrientation = Vector3.zero;
    
    private ROSConnection m_ROSConnection;
    private Vector3 m_LastPosition;
    private float m_DistanceTraveled = 0f;
    private bool m_IsMoving = false;
    
    void Start()
    {
        var connManager = FindObjectOfType<ROSConnectionManager>();
        if (connManager != null)
        {
            m_ROSConnection = connManager.GetConnection();
        }
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<LaserScanMsg>("/scan", OnLaserDataReceived);
            m_ROSConnection.Subscribe<ImuMsg>("/imu", OnImuDataReceived);
        }
        
        m_LastPosition = transform.position;
        
        if (animator != null)
        {
            animator.SetBool("IsMoving", false);
            animator.SetFloat("Speed", 0f);
        }
    }
    
    void Update()
    {
        UpdateAnimationParameters();
    }
    
    void UpdateAnimationParameters()
    {
        if (animator == null) return;
        
        // Calculate movement speed
        Vector3 currentPosition = transform.position;
        float deltaTime = Time.deltaTime;
        
        if (deltaTime > 0)
        {
            float speed = Vector3.Distance(m_LastPosition, currentPosition) / deltaTime;
            m_DistanceTraveled += Vector3.Distance(m_LastPosition, currentPosition);
            
            // Update movement flags
            m_IsMoving = speed > 0.01f;
            
            // Update animation parameters
            animator.SetBool("IsMoving", m_IsMoving);
            animator.SetFloat("Speed", speed);
            animator.SetFloat("DistanceTraveled", m_DistanceTraveled);
            
            // Check for obstacles from laser data
            bool obstacleNearby = lastLaserDistance < 1.0f;
            animator.SetBool("ObstacleNearby", obstacleNearby);
            
            // Check balance from IMU data
            float tiltAmount = Mathf.Abs(lastImuOrientation.x) + Mathf.Abs(lastImuOrientation.y);
            bool isUnbalanced = tiltAmount > balanceThreshold;
            animator.SetBool("IsUnbalanced", isUnbalanced);
        }
        
        m_LastPosition = currentPosition;
    }
    
    void OnLaserDataReceived(LaserScanMsg scanMsg)
    {
        // Find the minimum distance in front of the robot
        float minDistance = float.MaxValue;
        int frontSectorStart = (int)(scanMsg.ranges.Length * 0.45f);  // Front 10% of scan
        int frontSectorEnd = (int)(scanMsg.ranges.Length * 0.55f);
        
        for (int i = frontSectorStart; i < frontSectorEnd && i < scanMsg.ranges.Length; i++)
        {
            float range = (float)scanMsg.ranges[i];
            if (!float.IsNaN(range) && !float.IsInfinity(range) && range < minDistance)
            {
                minDistance = range;
            }
        }
        
        lastLaserDistance = minDistance;
    }
    
    void OnImuDataReceived(ImuMsg imuMsg)
    {
        // Convert quaternion to Euler angles to determine tilt
        Quaternion q = new Quaternion(
            (float)imuMsg.orientation.x,
            (float)imuMsg.orientation.y,
            (float)imuMsg.orientation.z,
            (float)imuMsg.orientation.w
        );
        
        Vector3 eulerAngles = q.eulerAngles;
        
        // Normalize angles to -180 to 180 range
        if (eulerAngles.x > 180) eulerAngles.x -= 360;
        if (eulerAngles.y > 180) eulerAngles.y -= 360;
        if (eulerAngles.z > 180) eulerAngles.z -= 360;
        
        lastImuOrientation = eulerAngles;
    }
}
```

## Part 8: Performance Optimization

### Step 8.1: Optimization Settings

To ensure good performance when visualizing sensor data:

1. Limit the number of LiDAR points visualized
2. Reduce update rates for non-critical visualizations
3. Use object pooling for frequently created/destroyed objects
4. Implement Level of Detail (LOD) for complex visualizations

### Step 8.2: Creating an Optimization Manager

```csharp
// PerformanceOptimizer.cs
using UnityEngine;

public class PerformanceOptimizer : MonoBehaviour
{
    [Header("LOD Settings")]
    public int lodDistance1 = 10;  // Distance to switch to LOD 1
    public int lodDistance2 = 20;  // Distance to switch to LOD 2
    public int maxLaserPoints = 500;  // Maximum points to visualize
    
    [Header("Update Rate Settings")]
    [Range(1, 100)]
    public int sensorVisualizationRate = 30;  // Updates per second for sensor viz
    [Range(1, 100)]
    public int robotPositionUpdateRate = 60;  // Updates per second for robot position
    
    private float m_LastSensorUpdate = 0f;
    private float m_LastPositionUpdate = 0f;
    
    void Update()
    {
        float currentTime = Time.time;
        
        // Update sensor visualization at reduced rate
        if (currentTime - m_LastSensorUpdate >= 1.0f / sensorVisualizationRate)
        {
            UpdateSensorVisualization();
            m_LastSensorUpdate = currentTime;
        }
        
        // Update robot position at appropriate rate
        if (currentTime - m_LastPositionUpdate >= 1.0f / robotPositionUpdateRate)
        {
            UpdateRobotPosition();
            m_LastPositionUpdate = currentTime;
        }
    }
    
    void UpdateSensorVisualization()
    {
        // This would contain calls to update sensor visualization
        // with the appropriate detail level based on distance
    }
    
    void UpdateRobotPosition()
    {
        // This would update robot position from ROS data
        // at the appropriate frequency
    }
    
    public int GetEffectiveLaserPoints(float distanceToRobot)
    {
        if (distanceToRobot > lodDistance2)
            return maxLaserPoints / 4;  // Lowest detail
        else if (distanceToRobot > lodDistance1)
            return maxLaserPoints / 2;  // Medium detail
        else
            return maxLaserPoints;      // Full detail
    }
}
```

## Part 9: Troubleshooting and Debugging

### Common Issues and Solutions

#### Issue 1: Connection Problems
- **Symptoms**: No data received from ROS, connection status shows disconnected
- **Solutions**: 
  - Verify ROS TCP endpoint is running: `ros2 run ros_tcp_endpoint default_server_endpoint`
  - Check IP and port settings match between Unity and ROS
  - Ensure firewall isn't blocking the connection

#### Issue 2: Robot Not Updating in Unity
- **Symptoms**: Robot position/orientation doesn't change despite ROS data
- **Solutions**:
  - Check that joint names match between URDF and Unity JointMapping
  - Verify that joint state topic is being published: `ros2 topic echo /joint_states`
  - Ensure the robot model hierarchy is properly set up

#### Issue 3: Performance Problems
- **Symptoms**: Slow frame rate, laggy response
- **Solutions**:
  - Reduce number of visualized LiDAR points
  - Lower sensor visualization update rate
  - Simplify robot model geometry
  - Use Unity Profiler to identify bottlenecks

#### Issue 4: Sensor Data Not Visualized
- **Symptoms**: LiDAR points, camera feed, or IMU visualization not appearing
- **Solutions**:
  - Verify sensor topics are being published by ROS nodes
  - Check topic names match between ROS and Unity subscriptions
  - Ensure message types are correct

### Debugging Tools

Add debugging utilities to help troubleshoot:

```csharp
// DebugVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class DebugVisualizer : MonoBehaviour
{
    [Header("Debug Settings")]
    public bool showDebugInfo = true;
    public bool logIncomingMessages = false;
    public int maxLogEntries = 50;
    public float messageDisplayDuration = 5f;
    
    private System.Collections.Generic.List<string> m_DebugMessages = 
        new System.Collections.Generic.List<string>();
    private System.Collections.Generic.List<float> m_MessageTimestamps = 
        new System.Collections.Generic.List<float>();
    
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUIStyle style = new GUIStyle();
        style.normal.textColor = Color.white;
        style.fontSize = 16;
        
        string debugInfo = GetDebugInfo();
        Rect rect = new Rect(10, 10, 400, 300);
        GUI.Label(rect, debugInfo, style);
    }
    
    string GetDebugInfo()
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        
        sb.AppendLine("=== Unity ROS Debug Info ===");
        sb.AppendLine($"Connected: {ROSConnectionManager.Instance.IsConnected()}");
        sb.AppendLine($"ROS IP: {ROSConnectionManager.Instance.rosIPAddress}");
        sb.AppendLine($"ROS Port: {ROSConnectionManager.Instance.rosPort}");
        sb.AppendLine($"Time Scale: {Time.timeScale}");
        sb.AppendLine($"Frame Count: {Time.frameCount}");
        sb.AppendLine($"Frame Rate: {1.0f / Time.deltaTime:F1} FPS");
        
        sb.AppendLine("\nRecent Messages:");
        for (int i = 0; i < m_DebugMessages.Count; i++)
        {
            if (Time.time - m_MessageTimestamps[i] < messageDisplayDuration)
            {
                sb.AppendLine($"  {m_DebugMessages[i]}");
            }
        }
        
        return sb.ToString();
    }
    
    public void LogDebugMessage(string message)
    {
        if (logIncomingMessages)
        {
            m_DebugMessages.Add($"{Time.time:F2}: {message}");
            m_MessageTimestamps.Add(Time.time);
            
            // Limit message count
            while (m_DebugMessages.Count > maxLogEntries)
            {
                m_DebugMessages.RemoveAt(0);
                m_MessageTimestamps.RemoveAt(0);
            }
        }
    }
    
    void Start()
    {
        // Subscribe to connection status changes
        ROSConnection rosConn = ROSConnectionManager.Instance.GetConnection();
        if (rosConn != null)
        {
            rosConn.OnConnected += () => LogDebugMessage("Connected to ROS");
            rosConn.OnDisconnected += () => LogDebugMessage("Disconnected from ROS");
        }
    }
}
```

## Part 10: Assessment and Validation

### Step 10.1: Validation Checklist

After completing the Unity integration exercise, verify the following:

1. **Connection**: Unity successfully connects to ROS 2 via TCP
2. **Robot Model**: Robot model updates in Unity based on Gazebo joint states
3. **Sensor Visualization**: All sensors (LiDAR, Camera, IMU) visualize correctly in Unity
4. **Control Interface**: Robot can be controlled through Unity UI
5. **Performance**: System runs smoothly without significant performance degradation
6. **Synchronization**: Unity visualization matches Gazebo simulation in real-time

### Step 10.2: Self-Assessment Questions

Answer these questions to ensure understanding:

1. How does the ROS TCP Connector facilitate communication between Unity and ROS 2?
2. What are the advantages of using Unity for visualization compared to RViz?
3. How would you add a new sensor type to your Unity visualization system?
4. What performance considerations are important when visualizing multiple sensors?
5. How would you implement a virtual reality interface for teleoperating the humanoid robot?

## Bonus Challenge

Enhance your Unity interface by implementing:
1. A virtual reality environment where you can walk around your simulated robot
2. Advanced visual effects like bloom, depth of field, or motion blur for more realistic rendering
3. A recording system to capture and replay robot behaviors
4. AI visualization tools to show the robot's decision-making process

## Exercise Completion

Congratulations! You have successfully implemented Unity integration for your humanoid robot simulation. You've learned to:
- Set up communication between Unity and ROS 2
- Create a synchronized visualization of your robot
- Visualize multiple sensor types in Unity
- Implement user controls for robot operation
- Optimize performance for real-time visualization
- Debug common integration issues

This integration allows for high-quality visualization and intuitive user interaction with your simulated humanoid robot, enhancing both the development and presentation of your robotics work.