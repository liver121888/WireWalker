import os
import json
import numpy as np
import math

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    # Normalize quaternion
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Compute rotation matrix
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2*qy**2 - 2*qz**2
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw
    
    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx**2 - 2*qz**2
    R[1, 2] = 2*qy*qz - 2*qx*qw
    
    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx**2 - 2*qy**2
    
    return R

def get_points_save_path(filename):
    """ Get absolute path to save file in ../../assets/points/. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(script_dir, "../../assets/points"))
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

def get_xml_save_path(filename):
    """ Get absolute path to save file in ../../assets/urdf/wires. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(script_dir, "../../assets/urdf/wires"))
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

def round_precision(value, precision=1e-3):
    """ Round a value to the nearest precision level without zeroing small values. """
    return np.round(value, decimals=int(-np.log10(precision)))

def save_points(filename, points):
    file_path = get_points_save_path(filename)
    with open(file_path, 'w') as f:
        json.dump(points, f, indent=4)
    print(f"Points saved to {file_path}")

def load_points(filename):
    file_path = get_points_save_path(filename)
    with open(file_path, 'r') as f:
        points = json.load(f)
    print(f"Points loaded from {file_path}")
    return points

def save_xml(filename, xml_string):
    file_path = get_xml_save_path(filename)
    with open(file_path, 'w') as f:
        f.write(xml_string)
    print(f"XML saved to {file_path}")
