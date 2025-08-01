import argparse
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from wire_gen_utils import save_points, save_xml

body_string="""
  <body name="wire_{idx}">
    <geom type="mesh" mesh="cosine" contype="0" conaffinity="0" group="2" rgba="0.0 0.8 0.0 1.0" pos="{p}" quat="{q}"/>
    <geom name="wire_{idx}" type="mesh" mesh="cosine_collision_0"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_1"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_2"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_3"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_4"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_5"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_6"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_7"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_8"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_9"  group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_10" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_11" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_12" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_13" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_14" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_15" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
    <geom type="mesh" mesh="cosine_collision_16" group="2" rgba="0.0 0.8 0.0 1" pos="{p}" quat="{q}"/>
  </body>
"""


def sample_points(args):
    """
    Sample points for a cosine wire.
    Each point is oriented using the analytical tangent vector.
    """
    dict_key = args.name
    start_point = args.start_point
    sample_num = args.sample_num
    wire_num = args.wire_num

    scale = 0.01
    amplitude = 4
    frequency = 1 / 4
    wire_length = 8 * math.pi

    # Convert start point to dictionary
    start_point = {
        'x': start_point[0],
        'y': start_point[1],
        'z': start_point[2],
        'qw': start_point[3],
        'qx': start_point[4],
        'qy': start_point[5],
        'qz': start_point[6]
    }

    start_points = {dict_key: []}
    sampled_points = {dict_key: []}

    for _ in range(wire_num):
        start_points[dict_key].append(start_point)

        for i in range(0, sample_num + 1):
            x_local = i / sample_num * wire_length
            y_local = 0
            z_local = -amplitude * math.cos(frequency * x_local) + amplitude
            # print(f"Sampled local point: {x_local}, {y_local}, {z_local}")
            local_pos = np.array([x_local, y_local, z_local])

            # Tangent from derivative -> -sin
            dz_dx = amplitude * frequency * math.sin(frequency * x_local)
            tangent = np.array([1.0, 0.0, dz_dx])
            tangent /= np.linalg.norm(tangent)

            rotation, _ = R.align_vectors([tangent], [[1, 0, 0]])
            quat = rotation.as_quat()  # [x, y, z, w]

            point_dict = {
                'x': start_point['x'] + scale * local_pos[0],
                'y': start_point['y'] + scale * local_pos[1],
                'z': start_point['z'] + scale * local_pos[2],
                'qw': quat[3],
                'qx': quat[0],
                'qy': quat[1],
                'qz': quat[2],
            }

            sampled_points[dict_key].append(point_dict)
        start_point = sampled_points[dict_key][-1]

    print(f"Sampled {len(sampled_points[dict_key])} points along the consine curve")
    for point in sampled_points[dict_key]:
        print(point)

    return start_points, sampled_points

def gen_xml(start_point, args):

    wire_num = args.wire_num
    dict_key = args.name

    xml_string = """<mujoco model="wire_{}">""".format(dict_key)

    for i in range(wire_num):
        xml_string += body_string.format(idx=i,
            p=" ".join(map(str,(start_point[dict_key][i]['x'], 
                     start_point[dict_key][i]['y'], 
                     start_point[dict_key][i]['z']))),
            q=" ".join(map(str,(start_point[dict_key][i]['qw'],
                                start_point[dict_key][i]['qx'],
                                start_point[dict_key][i]['qy'],
                                start_point[dict_key][i]['qz'])))
            )

    xml_string += """\n</mujoco>"""
    return xml_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # we acturally gen sample_num + 1 points
    parser.add_argument("--name", type=str, default='cosine', help="Name of the wire")
    parser.add_argument("--sample_num", type=int, default=10, help="Number of samples along the X")
    # must be float
    parser.add_argument('--start_point', type=float, nargs=7, 
                        default=[-0.02, 0.48, 0.45, 1.0, 0.0, 0.0, 0.0], 
                        help="Starting point of the wire: x y z qw qx qy qz")
    parser.add_argument('--wire_num', type=int, default=1, help="Number of wires to generate")
    args = parser.parse_args()

    args.points_filename = args.name + ".json"
    args.xml_filename = "wire_" + args.name + ".xml"

    start_points, points = sample_points(args)
    xml_string = gen_xml(start_points, args)
    save_points(args.points_filename, points)
    save_xml(args.xml_filename, xml_string)
