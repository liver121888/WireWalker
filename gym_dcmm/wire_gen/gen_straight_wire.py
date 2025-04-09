import argparse
import numpy as np
from wire_gen_utils import round_precision, save_points, save_xml

def sample_points(args):

    dict_key = args.name
    start_point = args.start_point
    sample_num = args.sample_num
    wire_num = args.wire_num
    precision = args.precision

    # Formula: z = c
    # x is of len 0.1
    # circle is of diameter 0.02

    # print(type(start_point))
    # Start point

    wire_length = 0.1

    # Sample points along the X-axis
    # don't include start point, include end point instead

    # change start point to point dict
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
            point_dict = {}
            x_pos = i/sample_num * wire_length  # Compute X position
            point_dict['x'] = round_precision(start_point['x'] + x_pos, precision)
            point_dict['y'] = round_precision(start_point['y'], precision)
            point_dict['z'] = round_precision(start_point['z'], precision)
            point_dict['qw'] = start_point['qw']
            point_dict['qx'] = start_point['qx']
            point_dict['qy'] = start_point['qy']
            point_dict['qz'] = start_point['qz']
            sampled_points[dict_key].append(point_dict)
        start_point = sampled_points[dict_key][-1]

    print(f"Sampled {len(sampled_points[dict_key])} points along X")
    for point in sampled_points[dict_key]:
        print(point)

    return start_points, sampled_points

def gen_xml(start_point, args):

    wire_num = args.wire_num
    dict_key = args.name

    xml_string = """<mujoco model="wire_{}">""".format(dict_key)

    straight_wire_string = """
    <body name="wire_{}" pos="{}" quat="{}">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.0 0.8 0.0 1" mesh="{}" />
        <geom name="wire_{}" type="mesh" group="2" rgba="0.0 0.8 0.0 1" mesh="{}" />
    </body>"""

    for i in range(wire_num):
        xml_string += straight_wire_string.format(i,
            " ".join(map(str,(start_point[dict_key][i]['x'], 
                     start_point[dict_key][i]['y'], 
                     start_point[dict_key][i]['z']))),
            " ".join(map(str,(start_point[dict_key][i]['qw'],
                                start_point[dict_key][i]['qx'],
                                start_point[dict_key][i]['qy'],
                                start_point[dict_key][i]['qz']))),
            dict_key,
            i,
            dict_key)

    xml_string += """\n</mujoco>"""
    return xml_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='straight', help="Name of the wire")
    # we acturally gen sample_num + 1 points
    parser.add_argument("--sample_num", type=float, default=10, help="Number of samples along the X")
    parser.add_argument("--precision", type=float, default=1e-3, help="Precision for coordinate rounding")
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
