import matplotlib.pyplot as plt
import numpy as np
import importlib
mpl_toolkits = importlib.import_module('mpl_toolkits')
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math
from wire_gen_utils import quaternion_to_rotation_matrix, load_points

def plot_frame(ax, position, quaternion, scale=0.01, linewidth=1.5):
    """
    Plot a coordinate frame at the given position with given quaternion orientation
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to plot on
    position : tuple or list
        (x, y, z) position of the frame
    quaternion : tuple or list
        (qw, qx, qy, qz) quaternion defining orientation
    scale : float
        Size of the coordinate frame
    linewidth : float
        Width of the coordinate frame lines
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    # print(R)

    # Create axis vectors
    x_axis = scale * R[:, 0]
    y_axis = scale * R[:, 1]
    z_axis = scale * R[:, 2]
    
    # Plot coordinate frame
    origin = np.array(position)
    
    # X-axis (red)
    x_end = origin + x_axis
    # print(x_end)
    ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 
            color='r', alpha=0.7, linewidth=linewidth)
    
    # Y-axis (green)
    y_end = origin + y_axis
    # print(origin)
    # print(y_end)
    ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 
            color='g', alpha=0.7, linewidth=linewidth)
    
    # Z-axis (blue)
    z_end = origin + z_axis
    ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 
            color='b', alpha=0.7, linewidth=linewidth)

    # print(origin, x_end, y_end, z_end)

def visualize_wire_with_frames(args):
    """
    Visualize the wire with quaternion frames
    """
    
    # Load the points from the JSON file
    points_data = load_points(args.name + '.json') 
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the wire key 
    dict_key = args.name
    
    # Check if the data exists for this key
    if dict_key not in points_data:
        print(f"No data found for wire '{dict_key}'")
        return
    
    # Get the points for the specified wire
    wire_points = points_data[dict_key]
    
    # Extract coordinates
    x = [point['x'] for point in wire_points]
    y = [point['y'] for point in wire_points]
    z = [point['z'] for point in wire_points]
    
    # Plot the wire
    ax.plot(x, y, z, color='black', linewidth=3, alpha=1.0, label='Wire')
    ax.scatter(x, y, z, color='black', alpha=1.0, s=30)
    
    # Calculate and display wire length
    length = 0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dz = z[i] - z[i-1]
        length += math.sqrt(dx**2 + dy**2 + dz**2)
    
    print(f"Wire length: {length:.4f}")
    
    # Plot quaternion frames
    num_points = len(wire_points)
    frame_indices = range(0, num_points)
    
    # Add start and end point frames if they're not already included
    if 0 not in frame_indices:
        frame_indices = [0] + list(frame_indices)
    if (num_points - 1) not in frame_indices:
        frame_indices = list(frame_indices) + [num_points - 1]
    
    # Calculate frame scale as 1/20 of view_box
    frame_scale = args.view_box / 20.0
    print(f"Using frame scale: {frame_scale} (1/10 of view_box)")
    
    for i in frame_indices:
        point = wire_points[i]
        position = (point['x'], point['y'], point['z'])
        quaternion = (point['qw'], point['qx'], point['qy'], point['qz'])
        
        # Use calculated frame_scale instead of args.frame_scale
        plot_frame(ax, position, quaternion, scale=frame_scale, linewidth=3.0)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(f'Visualization of {dict_key} with quaternion frames (scale: {frame_scale})')
    
    # Add description of coordinate frames
    ax.text2D(0.02, 0.98, "Coordinate Frames: X (red), Y (green), Z (blue)", 
              transform=ax.transAxes, fontsize=10)
    
    # Add details about the wire
    ax.text2D(0.02, 0.94, f"Wire length: {length:.4f} units", 
              transform=ax.transAxes, fontsize=10)
    ax.text2D(0.02, 0.90, f"Number of points: {len(wire_points)}", 
              transform=ax.transAxes, fontsize=10)
    ax.text2D(0.02, 0.86, f"Frame scale: {frame_scale}", 
              transform=ax.transAxes, fontsize=10)
              
    # Add grid
    ax.grid(True)
    
    # Set the view to center on the mid point with specified view_box
    mid_point = (x[len(x)//2], y[len(x)//2], z[len(x)//2])
    half_box = args.view_box / 2
    
    # Set the axis limits centered on the first point
    ax.set_xlim(mid_point[0] - half_box, mid_point[0] + half_box)
    ax.set_ylim(mid_point[1] - half_box, mid_point[1] + half_box)
    ax.set_zlim(mid_point[2] - half_box, mid_point[2] + half_box)

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='straight', help='Name of the wire (key in the JSON file')
    parser.add_argument('--view_box', type=float, default=0.2, help='View box size')
    
    args = parser.parse_args()
    
    visualize_wire_with_frames(args)