# Common utility functions and libraries for humanoid robotics

"""
This module contains common utility functions and libraries used across all modules
of the Physical AI & Humanoid Robotics Book. These utilities provide standardized
tools for working with ROS 2, handling robot-specific data, and performing common
operations in humanoid robotics applications.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from geometry_msgs.msg import Point, Quaternion, Pose
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


def create_header(frame_id: str = "base_link", stamp: Optional[Time] = None) -> Header:
    """
    Create a standard ROS 2 header with the given frame ID and timestamp.
    
    Args:
        frame_id: The coordinate frame this header refers to
        stamp: The timestamp for this header (uses current time if None)
        
    Returns:
        A populated Header message
    """
    header = Header()
    header.frame_id = frame_id
    
    if stamp is None:
        # In a real ROS 2 node, you would get the time from the node's clock
        # This is a placeholder implementation
        header.stamp.sec = 0
        header.stamp.nanosec = 0
    else:
        header.stamp = stamp
        
    return header


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion representation.
    
    Args:
        roll: Rotation around X-axis in radians
        pitch: Rotation around Y-axis in radians
        yaw: Rotation around Z-axis in radians
        
    Returns:
        A Quaternion representing the same rotation as the input Euler angles
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    
    return q


def euler_from_quaternion(quaternion: Quaternion) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion: The quaternion to convert
        
    Returns:
        A tuple of (roll, pitch, yaw) in radians
    """
    # Convert quaternion to Euler angles using the standard formula
    # Handle singularities at poles
    sinr_cosp = 2 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
    cosr_cosp = 1 - 2 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
    pitch = math.asin(sinp)

    siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    """
    Normalize a quaternion to unit length.
    
    Args:
        quaternion: The quaternion to normalize
        
    Returns:
        A normalized quaternion
    """
    length = math.sqrt(
        quaternion.x * quaternion.x +
        quaternion.y * quaternion.y +
        quaternion.z * quaternion.z +
        quaternion.w * quaternion.w
    )
    
    if length == 0:
        # Return identity quaternion if input is zero
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    
    return Quaternion(
        x=quaternion.x / length,
        y=quaternion.y / length,
        z=quaternion.z / length,
        w=quaternion.w / length
    )


def transform_point(point: Point, translation: Point, rotation: Quaternion) -> Point:
    """
    Transform a point by a given translation and rotation.
    
    Args:
        point: The point to transform
        translation: The translation vector to apply
        rotation: The rotation quaternion to apply
        
    Returns:
        The transformed point
    """
    # Convert quaternion to rotation matrix
    # First normalize the quaternion
    rot = normalize_quaternion(rotation)
    
    # Convert quaternion to rotation matrix
    matrix = np.array([
        [1 - 2*(rot.y**2 + rot.z**2), 2*(rot.x*rot.y - rot.z*rot.w), 2*(rot.x*rot.z + rot.y*rot.w)],
        [2*(rot.x*rot.y + rot.z*rot.w), 1 - 2*(rot.x**2 + rot.z**2), 2*(rot.y*rot.z - rot.x*rot.w)],
        [2*(rot.x*rot.z - rot.y*rot.w), 2*(rot.y*rot.z + rot.x*rot.w), 1 - 2*(rot.x**2 + rot.y**2)]
    ])
    
    # Apply rotation to the point
    rotated_point = matrix @ np.array([point.x, point.y, point.z])
    
    # Apply translation
    transformed_point = Point()
    transformed_point.x = rotated_point[0] + translation.x
    transformed_point.y = rotated_point[1] + translation.y
    transformed_point.z = rotated_point[2] + translation.z
    
    return transformed_point


def calculate_distance_3d(point1: Point, point2: Point) -> float:
    """
    Calculate the 3D distance between two points.
    
    Args:
        point1: The first point
        point2: The second point
        
    Returns:
        The Euclidean distance between the points
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def calculate_pose_difference(pose1: Pose, pose2: Pose) -> Tuple[float, float]:
    """
    Calculate the position and orientation differences between two poses.
    
    Args:
        pose1: The first pose
        pose2: The second pose
        
    Returns:
        A tuple of (position_difference, orientation_difference) in meters and radians
    """
    pos_diff = calculate_distance_3d(pose1.position, pose2.position)
    
    # Convert quaternions to Euler angles to get orientation difference
    euler1 = euler_from_quaternion(pose1.orientation)
    euler2 = euler_from_quaternion(pose2.orientation)
    
    # Calculate orientation difference (simplified as the sum of absolute differences)
    orient_diff = abs(euler1[0] - euler2[0]) + abs(euler1[1] - euler2[1]) + abs(euler1[2] - euler2[2])
    
    return pos_diff, orient_diff


def interpolate_pose(start_pose: Pose, end_pose: Pose, t: float) -> Pose:
    """
    Linearly interpolate between two poses based on parameter t (0.0 to 1.0).
    
    Args:
        start_pose: The starting pose
        end_pose: The ending pose
        t: Interpolation parameter (0.0 = start_pose, 1.0 = end_pose)
        
    Returns:
        The interpolated pose
    """
    # Interpolate position
    interp_pose = Pose()
    interp_pose.position.x = start_pose.position.x + t * (end_pose.position.x - start_pose.position.x)
    interp_pose.position.y = start_pose.position.y + t * (end_pose.position.y - start_pose.position.y)
    interp_pose.position.z = start_pose.position.z + t * (end_pose.position.z - start_pose.position.z)
    
    # For orientation, we'd normally use spherical linear interpolation (SLERP)
    # but for simplicity, we'll use linear interpolation of the quaternion components
    # followed by normalization
    interp_pose.orientation.x = start_pose.orientation.x + t * (end_pose.orientation.x - start_pose.orientation.x)
    interp_pose.orientation.y = start_pose.orientation.y + t * (end_pose.orientation.y - start_pose.orientation.y)
    interp_pose.orientation.z = start_pose.orientation.z + t * (end_pose.orientation.z - start_pose.orientation.z)
    interp_pose.orientation.w = start_pose.orientation.w + t * (end_pose.orientation.w - start_pose.orientation.w)
    
    # Normalize the resulting quaternion
    interp_pose.orientation = normalize_quaternion(interp_pose.orientation)
    
    return interp_pose


def validate_joint_positions(
    joint_names: List[str], 
    positions: List[float], 
    limits: Dict[str, Tuple[float, float]]
) -> bool:
    """
    Validate that joint positions are within their specified limits.
    
    Args:
        joint_names: List of joint names
        positions: List of joint positions (same length as joint_names)
        limits: Dictionary mapping joint names to (min, max) position tuples
        
    Returns:
        True if all positions are within limits, False otherwise
    """
    if len(joint_names) != len(positions):
        raise ValueError("Joint names and positions lists must have the same length")
    
    for name, pos in zip(joint_names, positions):
        if name not in limits:
            return False  # Joint not in limits dictionary
        
        min_limit, max_limit = limits[name]
        if pos < min_limit or pos > max_limit:
            return False
    
    return True


def map_range(
    value: float, 
    input_min: float, 
    input_max: float, 
    output_min: float, 
    output_max: float
) -> float:
    """
    Map a value from one range to another range.
    
    Args:
        value: The input value to be mapped
        input_min: Minimum value of the input range
        input_max: Maximum value of the input range
        output_min: Minimum value of the output range
        output_max: Maximum value of the output range
        
    Returns:
        The mapped value in the output range
    """
    # Clamp input value to input range
    value = max(input_min, min(input_max, value))
    
    # Calculate the normalized position of the value in the input range
    input_range = input_max - input_min
    if input_range == 0:
        return output_min  # Avoid division by zero
    
    normalized = (value - input_min) / input_range
    
    # Map to the output range
    output_range = output_max - output_min
    return output_min + normalized * output_range


def is_valid_pose(pose: Pose) -> bool:
    """
    Check if a pose contains valid (finite) values.
    
    Args:
        pose: The pose to validate
        
    Returns:
        True if all pose values are finite, False otherwise
    """
    # Check position values
    if not (math.isfinite(pose.position.x) and 
            math.isfinite(pose.position.y) and 
            math.isfinite(pose.position.z)):
        return False
    
    # Check orientation values
    if not (math.isfinite(pose.orientation.x) and 
            math.isfinite(pose.orientation.y) and 
            math.isfinite(pose.orientation.z) and 
            math.isfinite(pose.orientation.w)):
        return False
    
    return True


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to be within the specified range.
    
    Args:
        value: The value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        The clamped value
    """
    return max(min_val, min(max_val, value))


# Add more utility functions as needed...