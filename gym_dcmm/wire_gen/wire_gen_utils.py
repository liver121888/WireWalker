import os
import json
import numpy as np

def get_points_save_path(filename):
    """ Get absolute path to save file in ../../assets/points/. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(script_dir, "../../assets/points"))
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

def get_xml_save_path(filename):
    """ Get absolute path to save file in ../../assets/urdf/tracks. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(script_dir, "../../assets/urdf/tracks"))
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

def save_xml(filename, xml_string):
    file_path = get_xml_save_path(filename)
    with open(file_path, 'w') as f:
        f.write(xml_string)
    print(f"XML saved to {file_path}")
