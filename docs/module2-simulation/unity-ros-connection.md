# Connecting Unity Visualization to ROS 2 Nodes

## Overview

This section focuses on connecting Unity visualizations to ROS 2 nodes, completing the full simulation pipeline. With Gazebo handling physics and sensor simulation, and ROS 2 processing the data, Unity serves as the high-fidelity visualization layer that provides realistic graphics and intuitive human-robot interaction interfaces.

## Understanding the Unity-ROS 2 Connection

### 1. Architecture Overview

The Unity-ROS 2 connection involves:

- **ROS TCP Connector**: Acts as a bridge between ROS 2 and Unity via TCP/IP
- **Message Serialization**: Converts ROS 2 message types to Unity-compatible formats
- **Bidirectional Communication**: Allows Unity to both receive visualization data and send control commands
- **Synchronization**: Keeps Unity visualization synchronized with the simulation state

### 2. Network Communication Model

Unity communicates with ROS 2 through a TCP connection:

```
ROS 2 Nodes ── TCP ── ROS TCP Connector ── Unity
   │                    │                    │
Sensor Data          Protocol            Visualization
  │                 Conversion             │
  │                    │                    │
Commands ←──────────────┘←──────────────────┘
```

## Setting Up the ROS TCP Connector

### 1. Unity Package Installation

First, ensure the ROS TCP Connector package is installed in your Unity project:

1. Install the ROS TCP Connector package:
   - In Unity Package Manager, add from git URL:
   - `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`

### 2. ROS TCP Endpoint Setup

Start the ROS TCP endpoint on the ROS side:

```bash
# Terminal 1: Source ROS environment
source /opt/ros/iron/setup.bash

# Terminal 2: Start the ROS TCP endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=127.0.0.1 -p ROS_TCP_PORT:=10000
```

## Unity Connection Scripts

### 1. Basic Connection Manager

Create a connection manager script in Unity:

```csharp
// Scripts/ROSConnectionManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using System.Collections.Generic;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("Connection Settings")]
    [Tooltip("IP address of the ROS 2 system")]
    public string rosIPAddress = "127.0.0.1";
    
    [Tooltip("Port number for ROS communication")]
    public int rosPort = 10000;
    
    [Header("Performance Settings")]
    [Tooltip("Whether to log connection status messages")]
    public bool logStatusMessages = true;
    
    // Singleton instance
    public static ROSConnectionManager Instance { get; private set; }
    
    // Connection reference
    private ROSConnection m_ROSConnection;
    
    // Message queues for thread safety
    private Queue<System.Action> m_EnqueuedActions = new Queue<System.Action>();
    private object m_LockObject = new object();

    void Awake()
    {
        // Singleton pattern
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
        InitializeConnection();
    }

    private void InitializeConnection()
    {
        m_ROSConnection = ROSConnection.instance;
        if (m_ROSConnection == null)
        {
            m_ROSConnection = gameObject.AddComponent<ROSConnection>();
        }
        
        m_ROSConnection.Initialize(rosIPAddress, rosPort);
        
        if (logStatusMessages)
        {
            Debug.Log($"ROS Connection initialized to {rosIPAddress}:{rosPort}");
        }
    }

    public ROSConnection GetConnection()
    {
        return m_ROSConnection;
    }

    public void EnqueueAction(System.Action action)
    {
        lock (m_LockObject)
        {
            m_EnqueuedActions.Enqueue(action);
        }
    }

    void Update()
    {
        // Process enqueued actions from ROS callbacks
        lock (m_LockObject)
        {
            while (m_EnqueuedActions.Count > 0)
            {
                var action = m_EnqueuedActions.Dequeue();
                action?.Invoke();
            }
        }
    }

    private void OnApplicationQuit()
    {
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Close();
        }
    }
}
```

### 2. Robot State Visualizer

Create a script to visualize robot state in Unity:

```csharp
// Scripts/RobotStateVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class RobotStateVisualizer : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotModel;  // The robot model to visualize
    
    [Header("Joint Mapping")]
    public JointMapping[] jointMappings;
    
    [Header("Visualization Settings")]
    [Range(0.1f, 10.0f)]
    public float updateRate = 30.0f;  // Updates per second
    private float m_LastUpdateTime;
    
    private ROSConnection m_ROSConnection;

    [System.Serializable]
    public class JointMapping
    {
        public string jointName;
        public Transform jointTransform;
        public Vector3 rotationAxis = Vector3.right;
        [Range(0, 360)] public float angleOffset = 0;
    }

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            // Subscribe to joint states
            m_ROSConnection.Subscribe<JointStateMsg>("joint_states", OnJointStateReceived);
            
            // Subscribe to pose updates
            m_ROSConnection.Subscribe<PoseMsg>("robot_pose", OnPoseReceived);
            
            Debug.Log("Robot State Visualizer: Subscribed to ROS topics");
        }
        else
        {
            Debug.LogError("Robot State Visualizer: ROS Connection not available");
        }
        
        m_LastUpdateTime = Time.time;
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        // Process the joint state in the main thread to avoid threading issues
        ROSConnectionManager.Instance.EnqueueAction(() => UpdateJointPositions(msg));
    }
    
    void OnPoseReceived(PoseMsg msg)
    {
        // Process the pose update in the main thread
        ROSConnectionManager.Instance.EnqueueAction(() => UpdateRobotPose(msg));
    }

    void UpdateJointPositions(JointStateMsg jointState)
    {
        if (robotModel == null) return;
        
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];
            
            // Find the corresponding joint in our mapping
            foreach (var mapping in jointMappings)
            {
                if (mapping?.jointTransform != null && mapping.jointName == jointName)
                {
                    // Convert radians to degrees and apply rotation
                    float rotationInDegrees = jointPosition * Mathf.Rad2Deg + mapping.angleOffset;
                    
                    // Apply rotation using the specified axis
                    mapping.jointTransform.localRotation = Quaternion.AngleAxis(rotationInDegrees, mapping.rotationAxis);
                    break; // Found the joint, no need to continue searching
                }
            }
        }
    }

    void UpdateRobotPose(PoseMsg pose)
    {
        if (robotModel == null) return;
        
        // Update position
        robotModel.transform.position = new Vector3(
            (float)pose.position.x,
            (float)pose.position.y,
            (float)pose.position.z
        );
        
        // Update orientation
        robotModel.transform.rotation = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.y,
            (float)pose.orientation.z,
            (float)pose.orientation.w
        );
    }

    void Update()
    {
        // Additional update logic if needed
    }
}
```

## Sensor Data Visualization in Unity

### 1. LiDAR Visualization

Create a script to visualize LiDAR data in Unity:

```csharp
// Scripts/LiDARVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class LiDARVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    public GameObject laserPointPrefab;  // Prefab for individual laser points
    public Color laserColor = Color.red;
    public float maxRange = 30.0f;  // Maximum range of the LiDAR
    public float pointSize = 0.05f;  // Size of each laser point
    
    [Header("Performance Settings")]
    [Range(1, 100)]
    public int updateEveryNthScan = 10;  // Only process every Nth scan for performance
    
    private List<GameObject> m_LaserPoints = new List<GameObject>();
    private int m_ScanCounter = 0;
    private ROSConnection m_ROSConnection;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<LaserScanMsg>("scan", OnLaserScanReceived);
            Debug.Log("LiDAR Visualizer: Subscribed to /scan topic");
        }
        
        // Create a material for the laser points
        if (laserPointPrefab != null)
        {
            var renderer = laserPointPrefab.GetComponent<Renderer>();
            if (renderer != null && renderer.material != null)
            {
                renderer.material.color = laserColor;
            }
        }
    }

    void OnLaserScanReceived(LaserScanMsg scanMsg)
    {
        m_ScanCounter++;
        
        // Only process every Nth scan to manage performance
        if (m_ScanCounter % updateEveryNthScan != 0)
            return;
        
        // Process the scan in the main thread
        ROSConnectionManager.Instance.EnqueueAction(() => UpdateLaserVisualization(scanMsg));
    }

    void UpdateLaserVisualization(LaserScanMsg scanMsg)
    {
        // Clear previous laser points
        foreach (var point in m_LaserPoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
        m_LaserPoints.Clear();
        
        // Create new laser points based on scan data
        for (int i = 0; i < scanMsg.ranges.Length; i++)
        {
            float range = (float)scanMsg.ranges[i];
            
            // Only create points for valid measurements within range
            if (float.IsFinite(range) && range > scanMsg.range_min && range < maxRange)
            {
                // Calculate angle for this laser beam
                float angle = (float)(scanMsg.angle_min + (i * scanMsg.angle_increment));
                
                // Calculate position in 2D space (assuming level scanning)
                float x = range * Mathf.Cos(angle);
                float y = 0f;  // Height offset
                float z = range * Mathf.Sin(angle);
                
                // Create a laser point at this position
                if (laserPointPrefab != null)
                {
                    GameObject laserPoint = Instantiate(laserPointPrefab, transform);
                    laserPoint.transform.localPosition = new Vector3(x, y, z);
                    
                    // Scale the point based on distance (optional)
                    laserPoint.transform.localScale = Vector3.one * pointSize;
                    
                    m_LaserPoints.Add(laserPoint);
                }
            }
        }
        
        Debug.Log($"Updated LiDAR visualization with {m_LaserPoints.Count} points");
    }

    private void OnDestroy()
    {
        // Clean up laser points when object is destroyed
        foreach (var point in m_LaserPoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
    }
}
```

### 2. Camera Feed Visualization

Create a script to display camera feeds in Unity:

```csharp
// Scripts/CameraFeedVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using UnityEngine.UI;

public class CameraFeedVisualizer : MonoBehaviour
{
    [Header("Camera Feed Settings")]
    public RawImage displayImage;  // UI element to display the camera feed
    public int targetWidth = 640;
    public int targetHeight = 480;
    
    [Header("Performance Settings")]
    [Range(1, 30)]
    public int maxFPS = 10;  // Maximum frame rate for camera updates
    private float m_LastUpdateTime;
    
    private Texture2D m_CameraTexture;
    private Color32[] m_CameraColors;
    private ROSConnection m_ROSConnection;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<ImageMsg>("camera/image_raw", OnCameraImageReceived);
            Debug.Log("Camera Feed Visualizer: Subscribed to /camera/image_raw topic");
        }
        
        // Initialize texture
        InitializeTexture();
    }

    void InitializeTexture()
    {
        if (m_CameraTexture != null)
        {
            DestroyImmediate(m_CameraTexture);
        }
        
        m_CameraTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        m_CameraColors = new Color32[targetWidth * targetHeight];
        
        if (displayImage != null)
        {
            displayImage.texture = m_CameraTexture;
        }
    }

    void OnCameraImageReceived(ImageMsg imageMsg)
    {
        // Only update at specified frame rate to manage performance
        if (Time.time - m_LastUpdateTime < 1.0f / maxFPS)
            return;
        
        m_LastUpdateTime = Time.time;
        
        // Process the image in the main thread
        ROSConnectionManager.Instance.EnqueueAction(() => ProcessCameraImage(imageMsg));
    }

    void ProcessCameraImage(ImageMsg imageMsg)
    {
        // Check if image dimensions match our target
        if ((int)imageMsg.width != targetWidth || (int)imageMsg.height != targetHeight)
        {
            targetWidth = (int)imageMsg.width;
            targetHeight = (int)imageMsg.height;
            InitializeTexture();
        }
        
        // Handle different image encodings
        if (imageMsg.encoding == "rgb8" || imageMsg.encoding == "bgr8")
        {
            // Process RGB/BGR image data
            for (int i = 0; i < m_CameraColors.Length && (i * 3 + 2) < imageMsg.data.Length; i++)
            {
                int dataIndex = i * 3;
                
                // For RGB encoding
                if (imageMsg.encoding == "rgb8")
                {
                    m_CameraColors[i].r = imageMsg.data[dataIndex];
                    m_CameraColors[i].g = imageMsg.data[dataIndex + 1];
                    m_CameraColors[i].b = imageMsg.data[dataIndex + 2];
                }
                // For BGR encoding
                else if (imageMsg.encoding == "bgr8")
                {
                    m_CameraColors[i].b = imageMsg.data[dataIndex];  // Blue
                    m_CameraColors[i].g = imageMsg.data[dataIndex + 1];  // Green
                    m_CameraColors[i].r = imageMsg.data[dataIndex + 2];  // Red
                }
                
                m_CameraColors[i].a = 255;  // Alpha is always 255
            }
        }
        else
        {
            Debug.LogWarning($"Unsupported image encoding: {imageMsg.encoding}");
            return;
        }
        
        // Update the texture with new colors
        m_CameraTexture.SetPixels32(m_CameraColors);
        m_CameraTexture.Apply();
    }
}
```

## Unity Control Interface

### 1. Robot Control Interface

Create a script to send control commands from Unity to ROS 2:

```csharp
// Scripts/RobotController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class RobotController : MonoBehaviour
{
    [Header("Control Settings")]
    public float linearVelocity = 1.0f;
    public float angularVelocity = 1.0f;
    
    [Header("Input Settings")]
    public KeyCode forwardKey = KeyCode.W;
    public KeyCode backwardKey = KeyCode.S;
    public KeyCode leftKey = KeyCode.A;
    public KeyCode rightKey = KeyCode.D;
    public KeyCode stopKey = KeyCode.Space;
    
    private ROSConnection m_ROSConnection;
    private Vector3 m_CurrentVelocity;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            Debug.Log("Robot Controller: Ready to send commands");
        }
    }

    void Update()
    {
        // Handle keyboard input
        HandleKeyboardInput();
        
        // Send velocity commands periodically
        if (m_CurrentVelocity.magnitude > 0.01f)
        {
            SendVelocityCommand();
        }
    }

    void HandleKeyboardInput()
    {
        // Reset velocity
        Vector3 newVelocity = Vector3.zero;
        
        // Handle movement keys
        if (Input.GetKey(forwardKey))
        {
            newVelocity.z = linearVelocity;
        }
        if (Input.GetKey(backwardKey))
        {
            newVelocity.z = -linearVelocity;
        }
        if (Input.GetKey(leftKey))
        {
            newVelocity.x = -linearVelocity;
        }
        if (Input.GetKey(rightKey))
        {
            newVelocity.x = linearVelocity;
        }
        
        // Handle rotation keys
        if (Input.GetKey(KeyCode.Q))  // Rotate left
        {
            newVelocity.y = angularVelocity;
        }
        if (Input.GetKey(KeyCode.E))  // Rotate right
        {
            newVelocity.y = -angularVelocity;
        }
        
        // Update current velocity
        m_CurrentVelocity = newVelocity;
        
        // Handle stop key
        if (Input.GetKeyDown(stopKey))
        {
            m_CurrentVelocity = Vector3.zero;
            StopRobot();
        }
    }

    void SendVelocityCommand()
    {
        if (m_ROSConnection != null)
        {
            var twist = new TwistMsg();
            
            // Convert Unity coordinates to ROS coordinates (Unity: X=Right, Y=Up, Z=Forward; ROS: X=Forward, Y=Left, Z=Up)
            twist.linear.x = m_CurrentVelocity.z;    // Unity Z (forward) -> ROS X (forward)
            twist.linear.y = -m_CurrentVelocity.x;   // Unity X (right) -> ROS Y (left)
            twist.linear.z = m_CurrentVelocity.y;    // Unity Y (up) -> ROS Z (up)
            
            twist.angular.x = 0;
            twist.angular.y = 0;
            twist.angular.z = m_CurrentVelocity.y;   // Unity Y rotation -> ROS Z rotation
            
            m_ROSConnection.Publish("cmd_vel", twist);
        }
    }

    void StopRobot()
    {
        if (m_ROSConnection != null)
        {
            var twist = new TwistMsg();
            m_ROSConnection.Publish("cmd_vel", twist);
        }
    }
}
```

### 2. Advanced Control Interface with UI

Create a more sophisticated control interface using Unity UI:

```csharp
// Scripts/AdvancedRobotController.cs
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;

public class AdvancedRobotController : MonoBehaviour
{
    [Header("UI Elements")]
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Button moveForwardButton;
    public Button moveBackwardButton;
    public Button turnLeftButton;
    public Button turnRightButton;
    public Button stopButton;
    public Button goToPositionButton;
    
    [Header("Navigation Settings")]
    public InputField xPositionInput;
    public InputField yPositionInput;
    public InputField zPositionInput;
    
    private ROSConnection m_ROSConnection;
    private float m_LinearSpeed = 0.5f;
    private float m_AngularSpeed = 0.5f;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        // Setup UI event handlers
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.onValueChanged.AddListener(OnLinearSpeedChanged);
        }
        
        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);
        }
        
        if (moveForwardButton != null)
        {
            moveForwardButton.onClick.AddListener(() => MoveRobot(1, 0, 0));
        }
        
        if (moveBackwardButton != null)
        {
            moveBackwardButton.onClick.AddListener(() => MoveRobot(-1, 0, 0));
        }
        
        if (turnLeftButton != null)
        {
            turnLeftButton.onClick.AddListener(() => MoveRobot(0, 0, 1));
        }
        
        if (turnRightButton != null)
        {
            turnRightButton.onClick.AddListener(() => MoveRobot(0, 0, -1));
        }
        
        if (stopButton != null)
        {
            stopButton.onClick.AddListener(StopRobot);
        }
        
        if (goToPositionButton != null)
        {
            goToPositionButton.onClick.AddListener(GoToPosition);
        }
        
        // Set default values
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.value = m_LinearSpeed;
        }
        
        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.value = m_AngularSpeed;
        }
    }

    void OnLinearSpeedChanged(float value)
    {
        m_LinearSpeed = value;
    }

    void OnAngularSpeedChanged(float value)
    {
        m_AngularSpeed = value;
    }

    void MoveRobot(int forwardBackward, int leftRight, int turn)
    {
        if (m_ROSConnection != null)
        {
            var twist = new TwistMsg();
            
            // Set linear and angular velocities based on direction and speed settings
            twist.linear.x = forwardBackward * m_LinearSpeed;
            twist.linear.y = leftRight * m_LinearSpeed;
            twist.linear.z = 0;
            
            twist.angular.x = 0;
            twist.angular.y = 0;
            twist.angular.z = turn * m_AngularSpeed;
            
            m_ROSConnection.Publish("cmd_vel", twist);
        }
    }

    void StopRobot()
    {
        if (m_ROSConnection != null)
        {
            var twist = new TwistMsg();
            m_ROSConnection.Publish("cmd_vel", twist);
        }
    }

    void GoToPosition()
    {
        if (m_ROSConnection != null)
        {
            // Parse input fields
            float.TryParse(xPositionInput.text, out float x);
            float.TryParse(yPositionInput.text, out float y);
            float.TryParse(zPositionInput.text, out float z);
            
            // Create a goal pose message
            var goal = new PoseStampedMsg();
            goal.header = new Std.Msgs.HeaderMsg();
            goal.header.stamp = new builtin_interfaces.TimeMsg();
            goal.header.frame_id = "map";
            
            goal.pose.position = new geometry_msgs.PointMsg(x, y, z);
            goal.pose.orientation = new geometry_msgs.QuaternionMsg(0, 0, 0, 1); // Default orientation
            
            // Publish the navigation goal
            m_ROSConnection.Publish("goal_pose", goal);
        }
    }
}
```

## Visualization Optimization

### 1. Performance Optimized Visualizer

Create a performance-optimized visualizer for large amounts of data:

```csharp
// Scripts/PerformanceOptimizedVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;
using System.Threading.Tasks;

public class PerformanceOptimizedVisualizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public int maxPoints = 10000;  // Maximum number of visualization points
    public float updateInterval = 0.1f;  // Update every 100ms
    private float m_LastUpdateTime;
    
    [Header("Visualization Settings")]
    public GameObject pointPrefab;
    public Material pointMaterial;
    
    private List<GameObject> m_ActivePoints = new List<GameObject>();
    private Queue<GameObject> m_Pool = new Queue<GameObject>();
    private ROSConnection m_ROSConnection;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<LaserScanMsg>("scan", OnLaserScanReceived);
        }
        
        // Pre-populate object pool for performance
        PrepopulatePool();
    }

    void PrepopulatePool()
    {
        // Create a pool of objects to reuse
        int poolSize = Mathf.Min(maxPoints, 2000);  // Limit pool size for memory efficiency
        for (int i = 0; i < poolSize; i++)
        {
            GameObject point = CreatePointCloudObject();
            point.SetActive(false);
            m_Pool.Enqueue(point);
        }
    }

    GameObject GetOrCreatePoint()
    {
        if (m_Pool.Count > 0)
        {
            GameObject point = m_Pool.Dequeue();
            point.SetActive(true);
            return point;
        }
        else
        {
            return CreatePointCloudObject();
        }
    }

    GameObject CreatePointCloudObject()
    {
        GameObject point;
        if (pointPrefab != null)
        {
            point = Instantiate(pointPrefab, transform);
        }
        else
        {
            // Create a simple sphere if no prefab is provided
            point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.transform.SetParent(transform);
            point.GetComponent<Renderer>().material = pointMaterial;
        }
        
        // Set a small scale for point cloud visualization
        point.transform.localScale = Vector3.one * 0.02f;
        
        // Remove collider to improve performance
        var collider = point.GetComponent<Collider>();
        if (collider != null)
            DestroyImmediate(collider);
        
        return point;
    }

    void OnLaserScanReceived(LaserScanMsg scanMsg)
    {
        if (Time.time - m_LastUpdateTime < updateInterval)
            return;
            
        m_LastUpdateTime = Time.time;
        
        // Process scan asynchronously to avoid blocking the main thread
        ROSConnectionManager.Instance.EnqueueAction(() => UpdatePointCloud(scanMsg));
    }

    void UpdatePointCloud(LaserScanMsg scanMsg)
    {
        // Disable all active points first
        foreach (var point in m_ActivePoints)
        {
            if (point != null)
            {
                point.SetActive(false);
                m_Pool.Enqueue(point);
            }
        }
        m_ActivePoints.Clear();
        
        // Determine how many points to actually visualize
        int totalPoints = scanMsg.ranges.Length;
        int step = Mathf.Max(1, totalPoints / maxPoints);  // Only visualize every Nth point
        
        for (int i = 0; i < totalPoints; i += step)
        {
            float range = (float)scanMsg.ranges[i];
            
            // Only visualize valid measurements
            if (float.IsFinite(range) && range > scanMsg.range_min && range < scanMsg.range_max)
            {
                // Calculate angle and position
                float angle = (float)(scanMsg.angle_min + (i * scanMsg.angle_increment));
                float x = range * Mathf.Cos(angle);
                float z = range * Mathf.Sin(angle);
                
                // Get a point from pool or create new
                GameObject point = GetOrCreatePoint();
                point.transform.localPosition = new Vector3(x, 0, z);
                
                m_ActivePoints.Add(point);
                
                // Limit the number of points for performance
                if (m_ActivePoints.Count >= maxPoints)
                    break;
            }
        }
        
        Debug.Log($"Updated point cloud: {m_ActivePoints.Count} points");
    }

    void OnDestroy()
    {
        // Clean up pooled objects
        foreach (var point in m_Pool)
        {
            if (point != null)
                DestroyImmediate(point);
        }
        
        foreach (var point in m_ActivePoints)
        {
            if (point != null)
                DestroyImmediate(point);
        }
    }
}
```

## Integration Testing

### 1. Connection and Visualization Test Script

Create a script to test the Unity-ROS 2 connection:

```csharp
// Scripts/ConnectionTester.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using System.Collections;
using System.Collections.Generic;

public class ConnectionTester : MonoBehaviour
{
    [Header("Test Settings")]
    public float testInterval = 2.0f;
    public int messageCount = 0;
    public int receivedMessageCount = 0;
    
    [Header("UI Display")]
    public UnityEngine.UI.Text statusText;
    public UnityEngine.UI.Text messageStatsText;
    
    private ROSConnection m_ROSConnection;
    private bool m_ConnectionEstablished = false;
    private float m_LastTestTime;
    private List<float> m_LatencyHistory = new List<float>();

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            // Subscribe to a test topic
            m_ROSConnection.Subscribe<StringMsg>("test_feedback", OnTestFeedbackReceived);
        }
        
        m_LastTestTime = Time.time;
        
        StartCoroutine(TestConnectionRoutine());
    }

    IEnumerator TestConnectionRoutine()
    {
        while (true)
        {
            // Send a test message
            if (m_ROSConnection != null)
            {
                var testMsg = new StringMsg();
                testMsg.data = $"Unity heartbeat at {Time.time}";
                
                m_ROSConnection.Publish("test_input", testMsg);
                messageCount++;
                
                // Wait for potential response
                yield return new WaitForSeconds(0.5f);
            }
            
            yield return new WaitForSeconds(testInterval);
        }
    }

    void OnTestFeedbackReceived(StringMsg msg)
    {
        receivedMessageCount++;
        
        // Calculate potential latency
        if (float.TryParse(msg.data.Split(' ')[3], out float sentTime))
        {
            float latency = Time.time - sentTime;
            m_LatencyHistory.Add(latency);
            
            // Keep only recent measurements
            if (m_LatencyHistory.Count > 20)
                m_LatencyHistory.RemoveAt(0);
        }
    }

    void Update()
    {
        UpdateUI();
        
        // Update connection status
        if (m_ROSConnection != null)
        {
            m_ConnectionEstablished = true;
        }
    }

    void UpdateUI()
    {
        if (statusText != null)
        {
            string status = m_ConnectionEstablished ? "CONNECTED" : "DISCONNECTED";
            status += $" - Messages Sent: {messageCount}, Received: {receivedMessageCount}";
            
            // Calculate average latency
            float avgLatency = 0;
            if (m_LatencyHistory.Count > 0)
            {
                avgLatency = m_LatencyHistory.Average();
                status += $", Avg Latency: {avgLatency:F3}s";
            }
            
            statusText.text = status;
        }
        
        if (messageStatsText != null)
        {
            messageStatsText.text = $"Sent: {messageCount}, Received: {receivedMessageCount}";
        }
    }
}

// Extension for calculating average on List<float>
public static class ListExtensions
{
    public static float Average(this List<float> list)
    {
        if (list.Count == 0) return 0;
        
        float sum = 0;
        foreach (float value in list)
        {
            sum += value;
        }
        return sum / list.Count;
    }
}
```

## Troubleshooting Common Issues

### 1. Connection Problems

**Common Issues and Solutions:**

1. **No connection established:**
   - Verify ROS TCP endpoint is running: `ros2 run ros_tcp_endpoint default_server_endpoint`
   - Check IP address and port in Unity match the ROS endpoint
   - Ensure firewall isn't blocking the connection port

2. **Messages not received in Unity:**
   - Verify topic names match between publisher and subscriber
   - Check that message types are properly supported
   - Test with simple data first (String messages)

3. **Performance issues with visualization:**
   - Reduce update rates for high-frequency topics
   - Implement object pooling for visual elements
   - Use Level of Detail (LOD) techniques

4. **Threading issues:**
   - Always update Unity objects from the main thread
   - Use the action queuing pattern as shown in examples
   - Be careful with multithreading in Unity callbacks

### 2. Debugging Techniques

Add debugging utilities:

```csharp
// Scripts/DebugVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class DebugVisualizer : MonoBehaviour
{
    [Header("Debug Settings")]
    public bool showDebugInfo = true;
    public int maxMessagesToDisplay = 50;
    
    private Queue<string> m_DebugMessages = new Queue<string>();
    private ROSConnection m_ROSConnection;

    void Start()
    {
        m_ROSConnection = ROSConnectionManager.Instance.GetConnection();
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Subscribe<JointStateMsg>("joint_states", (msg) => 
            {
                AddDebugMessage($"Joint States: {msg.name.Length} joints, pos[0]={msg.position[0]}");
            });
        }
    }

    public void AddDebugMessage(string message)
    {
        string timestampedMessage = $"[{Time.time:F2}] {message}";
        m_DebugMessages.Enqueue(timestampedMessage);
        
        // Limit message count
        if (m_DebugMessages.Count > maxMessagesToDisplay)
            m_DebugMessages.Dequeue();
    }

    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        int lineCount = 0;
        foreach (string msg in m_DebugMessages)
        {
            GUI.Label(new Rect(10, 10 + (lineCount * 20), 600, 20), msg);
            lineCount++;
        }
    }
}
```

## Best Practices

### 1. Performance Optimization

- **Throttle updates**: Don't update visualization at the same rate as physics
- **Use object pooling**: Reuse visualization objects rather than creating/destroying them
- **Implement LOD**: Reduce detail based on distance or importance
- **Minimize callbacks**: Batch process messages when possible
- **Monitor performance**: Use Unity Profiler to identify bottlenecks

### 2. Thread Safety

- Always use the main thread for Unity operations
- Use the action queuing pattern for thread-safe operations
- Be careful with static variables in multi-threaded environments

### 3. Error Handling

- Implement connection status monitoring
- Gracefully handle disconnections
- Provide fallback behaviors when data is unavailable
- Log errors appropriately without spamming the console

## Next Steps

With Unity visualization properly connected to ROS 2 nodes, you'll next document how to test humanoid motion in simulation. This will include creating test scenarios and validation procedures to ensure your humanoid robot behaves correctly in the simulated environment.

The complete simulation pipeline is now established: Gazebo handles physics, ROS 2 processes data, and Unity provides high-fidelity visualization, creating a comprehensive development and testing environment for your humanoid robot.