# Setting up Unity Robotics Hub for Visualization

## Overview

This section provides detailed instructions for setting up the Unity Robotics Hub for your humanoid robot visualization projects. The Unity Robotics Hub provides essential tools and packages that enable seamless integration between Unity and ROS 2 systems, making it possible to create high-fidelity visualizations for your humanoid robot.

## Prerequisites

Before setting up the Unity Robotics Hub, ensure you have:

1. **Unity Hub** installed (version 3.0 or later)
2. **Unity Editor** (version 2021.3 LTS or later)
3. **ROS 2 Iron** properly installed and configured
4. **Git** installed for package management
5. **Visual Studio** or compatible IDE for scripting

## Installing Unity Robotics Hub Components

### 1. Installing ROS-TCP-Connector

The ROS-TCP-Connector is the primary package for communication between Unity and ROS 2:

1. Open your Unity project or create a new one
2. Open the Package Manager (Window → Package Manager)
3. Click the "+" button in the top left corner
4. Select "Add package from git URL..."
5. Enter the following URL:
   ```
   https://github.com/Unity-Technologies/ROS-TCP-Connector.git
   ```
6. Click "Add" to install the package

Alternatively, you can install via manifest:
1. Navigate to your project's `Packages` folder
2. Open `manifest.json` in a text editor
3. Add the package to the dependencies section:
   ```json
   {
     "dependencies": {
       "com.unity.robotics.ros-tcp-connector": "https://github.com/Unity-Technologies/ROS-TCP-Connector.git",
       ...
     }
   }
   ```

### 2. Installing Robotics Package

For additional robotics-specific utilities:

1. In the Package Manager, click "+" → "Add package from git URL..."
2. Add the robotics package:
   ```
   https://github.com/Unity-Technologies/Unity-Robotics-Hub.git?path=/packages/URDF-Importer
   ```
   This provides URDF import capabilities for your robot models.

### 3. Installing URDF Importer

To import your ROS robot models into Unity:

1. In Package Manager, add from git URL:
   ```
   https://github.com/Unity-Technologies/URDF-Importer.git
   ```
2. This enables direct import of URDF files into Unity scenes

## Unity Project Setup for Robotics

### 1. Creating a Robotics Project

Create a new Unity project specifically for robotics visualization:

1. Open Unity Hub
2. Click "New Project"
3. Select the "3D (Built-in Render Pipeline)" template
4. Name your project (e.g., "HumanoidRobotVisualization")
5. Choose a location to save the project
6. Click "Create Project"

### 2. Project Configuration

Configure your Unity project for robotics workflows:

1. Go to Edit → Project Settings
2. Under "Player" settings:
   - Set "Run in Background" to true (for network communication)
   - Under "Resolution and Presentation", set "Fullscreen Mode" to "Windowed" for development
3. Under "Quality" settings:
   - Adjust to balance visual quality with performance requirements
4. Under "Physics" settings:
   - Set Timestep appropriately (e.g., 0.02 for 50 FPS)
   - Adjust Solver iterations as needed

### 3. Importing Robot Models

If you have URDF files for your humanoid robot:

1. Create an "ImportedRobot" folder in your Unity Assets
2. Copy your URDF files to the Assets folder
3. If using the URDF Importer package, select the URDF file in the Project window
4. The importer will automatically create the robot model in Unity
5. Check that all joints and links are correctly imported

For more complex robots, you might need to provide the entire package containing meshes and materials alongside the URDF file.

## Setting Up ROS Connection

### 1. Basic Connection Setup

Create a simple connection script to establish ROS communication:

```csharp
// Scripts/ROSConnectionManager.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    [Tooltip("IP address of the ROS system")]
    public string rosIPAddress = "127.0.0.1";

    [Tooltip("Port number for ROS communication")]
    public int rosPort = 10000;

    private ROSConnection m_ROSConnection;

    void Start()
    {
        // Establish ROS connection
        m_ROSConnection = ROSConnection.instance;
        
        if (m_ROSConnection != null)
        {
            m_ROSConnection.Initialize(rosIPAddress, rosPort);
            Debug.Log($"Connected to ROS at {rosIPAddress}:{rosPort}");
        }
        else
        {
            Debug.LogError("Failed to initialize ROS connection");
        }
    }

    public ROSConnection GetConnection()
    {
        return m_ROSConnection;
    }
}
```

### 2. Adding the Connection Manager to Your Scene

1. Create an empty GameObject in your scene (GameObject → Create Empty)
2. Name it "ROSConnectionManager"
3. Add the ROSConnectionManager script to this object
4. Adjust the IP address and port as needed for your setup

### 3. Testing the Connection

Create a simple test to verify the connection:

```csharp
// Scripts/ConnectionTester.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class ConnectionTester : MonoBehaviour
{
    public float publishRate = 1.0f; // Publish once per second
    private float lastPublishTime;

    void Start()
    {
        lastPublishTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastPublishTime > publishRate)
        {
            // Publish a test message
            var testMsg = new StringMsg();
            testMsg.data = $"Unity heartbeat at {Time.time}";
            
            ROSConnection.instance.Publish("/unity_test", testMsg);
            lastPublishTime = Time.time;
            
            Debug.Log($"Published test message: {testMsg.data}");
        }
    }
}
```

## Creating a Robot Visualization Scene

### 1. Basic Scene Setup

1. Create a new scene (File → New Scene)
2. Save it as "HumanoidVisualization"
3. Set up basic lighting:
   - Add a Directional Light (GameObject → Light → Directional Light)
   - Position it to illuminate your robot from above
4. Set up a camera:
   - Position the Main Camera to view the robot
   - Adjust field of view and clipping planes as needed

### 2. Robot Visualization Script

Create a script to visualize robot state in Unity:

```csharp
// Scripts/HumanoidVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class HumanoidVisualizer : MonoBehaviour
{
    [Header("Robot Joint Mappings")]
    public Transform head;
    public Transform leftHip;
    public Transform rightHip;
    public Transform leftKnee;
    public Transform rightKnee;
    public Transform leftAnkle;
    public Transform rightAnkle;
    
    // Store joint names to match with ROS messages
    private string[] jointNames;
    private Transform[] jointTransforms;

    void Start()
    {
        // Initialize joint mapping arrays
        jointNames = new string[] {
            "head_joint",
            "left_hip_joint", 
            "right_hip_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_joint",
            "right_ankle_joint"
        };
        
        jointTransforms = new Transform[] {
            head, leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle
        };
        
        // Subscribe to joint states from ROS
        ROSConnection.instance.Subscribe<JointStateMsg>("joint_states", OnJointStateReceived);
    }

    void OnJointStateReceived(JointStateMsg msg)
    {
        // Update each joint based on ROS message
        for (int i = 0; i < msg.name.Length; i++)
        {
            string jointName = msg.name[i];
            double jointPosition = msg.position[i];
            
            // Find corresponding Unity transform and update rotation
            for (int j = 0; j < jointNames.Length; j++)
            {
                if (jointNames[j] == jointName && jointTransforms[j] != null)
                {
                    // Convert radians to degrees and apply rotation
                    float rotationInDegrees = (float)(jointPosition * Mathf.Rad2Deg);
                    
                    // Apply rotation - adjust axis as needed for your robot
                    jointTransforms[j].Rotate(Vector3.right, rotationInDegrees);
                }
            }
        }
    }
    
    void Update()
    {
        // Any additional visualization updates can go here
    }
}
```

### 3. Visualization Setup

1. Attach the HumanoidVisualizer script to your robot model root
2. Assign the appropriate joint transforms in the Inspector
3. Make sure your robot model hierarchy matches the joint names in your ROS system

## Setting Up Sensor Visualization

### 1. Camera Sensor Visualization

Create a script to visualize camera data from ROS:

```csharp
// Scripts/CameraVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class CameraVisualizer : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera displayCamera;
    public Material cameraMaterial;
    
    private Texture2D texture2D;
    private int imageWidth = 640;
    private int imageHeight = 480;
    
    void Start()
    {
        // Initialize texture for camera display
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        if (cameraMaterial != null)
        {
            cameraMaterial.mainTexture = texture2D;
        }
        
        // Subscribe to camera image topic
        ROSConnection.instance.Subscribe<ImageMsg>("camera/image_raw", OnCameraImageReceived);
    }
    
    void OnCameraImageReceived(ImageMsg imageMsg)
    {
        if (imageMsg.width != imageWidth || imageMsg.height != imageHeight)
        {
            // Handle different image sizes if needed
            imageWidth = (int)imageMsg.width;
            imageHeight = (int)imageMsg.height;
            texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            
            if (cameraMaterial != null)
            {
                cameraMaterial.mainTexture = texture2D;
            }
        }
        
        // Convert ROS image data to Unity texture
        Color32[] colors = new Color32[imageMsg.data.Length / 3];
        
        for (int i = 0; i < colors.Length; i++)
        {
            // Assuming RGB format - adjust based on your camera's encoding
            colors[i].r = imageMsg.data[i * 3];
            colors[i].g = imageMsg.data[i * 3 + 1];
            colors[i].b = imageMsg.data[i * 3 + 2];
            colors[i].a = 255; // Alpha is always 255 in RGB
        }
        
        texture2D.SetPixels32(colors);
        texture2D.Apply();
    }
}
```

### 2. LiDAR Visualization

Create a visualization for LiDAR data:

```csharp
// Scripts/LiDARVisualizer.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class LiDARVisualizer : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public GameObject laserPointPrefab;  // Small sphere prefab to represent laser points
    public float maxRange = 10.0f;
    public Color laserColor = Color.red;
    
    private GameObject[] laserPoints;
    private int lastLaserCount = 0;

    void Start()
    {
        ROSConnection.instance.Subscribe<LaserScanMsg>("scan", OnLaserScanReceived);
    }
    
    void OnLaserScanReceived(LaserScanMsg scanMsg)
    {
        int numPoints = scanMsg.ranges.Length;
        
        // Instantiate laser point objects if needed
        if (laserPoints == null || laserPoints.Length != numPoints)
        {
            ClearPreviousLaserPoints();
            laserPoints = new GameObject[numPoints];
            
            for (int i = 0; i < numPoints; i++)
            {
                laserPoints[i] = Instantiate(laserPointPrefab, Vector3.zero, Quaternion.identity, transform);
                laserPoints[i].GetComponent<Renderer>().material.color = laserColor;
                laserPoints[i].SetActive(false);
            }
        }
        
        // Update laser point positions based on scan data
        for (int i = 0; i < numPoints; i++)
        {
            float range = scanMsg.ranges[i];
            
            if (range < maxRange && !float.IsPositiveInfinity(range) && !float.IsNaN(range))
            {
                // Calculate angle for this laser beam
                float angle = scanMsg.angle_min + (i * scanMsg.angle_increment);
                
                // Calculate position in Unity coordinate system
                float x = range * Mathf.Cos(angle);
                float y = range * Mathf.Sin(angle);
                
                laserPoints[i].transform.localPosition = new Vector3(x, 0, y);
                laserPoints[i].SetActive(true);
            }
            else
            {
                laserPoints[i].SetActive(false);
            }
        }
    }
    
    void ClearPreviousLaserPoints()
    {
        if (laserPoints != null)
        {
            foreach (GameObject point in laserPoints)
            {
                if (point != null)
                {
                    DestroyImmediate(point);
                }
            }
        }
    }
    
    // Cleanup when object is destroyed
    void OnDestroy()
    {
        ClearPreviousLaserPoints();
    }
}
```

## Testing the Setup

### 1. Running a Basic Test

1. Make sure your ROS 2 environment is sourced:
   ```bash
   source /opt/ros/iron/setup.bash
   ```

2. Start a simple ROS node that publishes joint states or other data:
   ```bash
   # Example: publish joint states with fake data
   ros2 run joint_state_publisher joint_state_publisher
   ```

3. In Unity, click "Play" to start the scene
4. Check the Unity console for connection messages and errors

### 2. Verification Steps

Verify that the connection is working:

1. Check that the Unity console shows successful connection messages
2. Verify that ROS is receiving heartbeat messages from Unity
3. Test that joint states or other sensor data are updating the Unity visualization
4. Ensure that frame rates are acceptable for your application

## Troubleshooting Common Issues

### 1. Connection Problems

If Unity can't connect to ROS:
- Verify that ROS_IP/ROS_HOSTNAME is set correctly in your ROS environment
- Check that the ROS TCP Connector is running (ros2 run ros_tcp_endpoint default_server_endpoint)
- Ensure the IP address and port are correctly configured in Unity
- Check firewall settings that might block the connection

### 2. Performance Issues

If visualization performance is poor:
- Reduce the update frequency of robot visualization
- Simplify robot model geometry for visualization
- Use less resource-intensive shaders
- Limit the number of visualized sensor data points

### 3. Coordinate System Issues

If robot visualization doesn't match ROS coordinates:
- Verify that Unity's coordinate system (left-handed) is properly aligned with ROS (right-handed)
- Check ROS coordinate transforms
- Adjust joint rotation axes in visualization scripts as needed

## Optimizing for Production Use

### 1. Configuration Management

Create a configuration system to manage different deployment scenarios:

```csharp
// Scripts/RobotVisualizationConfig.cs
using UnityEngine;

[CreateAssetMenu(fileName = "RobotVisualizationConfig", menuName = "Robotics/Visualization Config")]
public class RobotVisualizationConfig : ScriptableObject
{
    [Header("Network Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Visualization Settings")]
    public float jointUpdateRate = 30.0f;  // Updates per second
    public int maxLiDARPoints = 1000;
    
    [Header("Performance Settings")]
    public bool useLODs = true;
    public int maxTextureSize = 512;
    
    [Header("Robot Configuration")]
    public string[] jointNames;
    public JointType[] jointTypes;
}

public enum JointType
{
    Revolute,
    Prismatic,
    Fixed
}
```

### 2. Resource Management

Implement proper resource management for long-running applications:

```csharp
// Scripts/VisualizationResourceManagement.cs
using UnityEngine;
using System.Collections.Generic;

public class VisualizationResourceManagement : MonoBehaviour
{
    private List<GameObject> spawnedObjects = new List<GameObject>();
    private List<Texture2D> createdTextures = new List<Texture2D>();

    public void RegisterSpawnedObject(GameObject obj)
    {
        spawnedObjects.Add(obj);
    }
    
    public void RegisterCreatedTexture(Texture2D tex)
    {
        createdTextures.Add(tex);
    }
    
    void OnApplicationQuit()
    {
        CleanupResources();
    }
    
    public void CleanupResources()
    {
        // Destroy spawned objects
        foreach (GameObject obj in spawnedObjects)
        {
            if (obj != null)
                DestroyImmediate(obj);
        }
        spawnedObjects.Clear();
        
        // Unload textures
        foreach (Texture2D tex in createdTextures)
        {
            if (tex != null)
                DestroyImmediate(tex);
        }
        createdTextures.Clear();
    }
}
```

## Next Steps

With the Unity Robotics Hub properly set up, you'll next learn how to integrate ROS 2 with both Gazebo and Unity simulators. This involves setting up the communication architecture that allows data to flow between all components of your simulation environment.

The Unity setup you've completed provides the foundation for creating high-fidelity visualizations that complement the physics-accurate Gazebo simulation for your humanoid robot.