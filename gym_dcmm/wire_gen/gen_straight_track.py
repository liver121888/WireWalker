import argparse
import os
import json
import numpy as np
from wire_gen_utils import round_precision, save_points, save_xml

dict_key = 'straight'


def sample_straight(start_point, sample_num, track_num, precision=1e-3):

    global dict_key

    # Formula: z = c
    # x is of len 0.1
    # circle is of diameter 0.02

    # print(type(start_point))
    # Start point

    straight_length = 0.1

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
    for _ in range(track_num):

        start_points[dict_key].append(start_point)
        for i in range(1, sample_num + 1):
            point_dict = {}
            x_pos = i/sample_num * straight_length  # Compute X position
            point_dict['x'] = round_precision(start_point['x'] + x_pos, precision)
            point_dict['y'] = round_precision(start_point['y'], precision)
            point_dict['z'] = round_precision(start_point['z'], precision)
            point_dict['qw'] = start_point['qw']
            point_dict['qx'] = start_point['qx']
            point_dict['qy'] = start_point['qy']
            point_dict['qz'] = start_point['qz']
            sampled_points[dict_key].append(point_dict)
        start_point = sampled_points[dict_key][-1]

    print(f"Sampled {len(sampled_points[dict_key])} points along Z-axis:")
    for point in sampled_points[dict_key]:
        print(point)

    return start_points, sampled_points

def xml_straight(start_point, track_num):

    global dict_key

    xml_string = """<mujoco model="track_straight">"""

    staight_track_string = """
    <body name="track_{}" pos="{}" quat="{}">
        <geom type="mesh" contype="0" conaffinity="0" group="3" rgba="0.0 0.8 0.0 1" mesh="straight" />
        <geom type="mesh" rgba="0.0 0.8 0.0 1" mesh="straight" />
    </body>"""

    for i in range(track_num):
        xml_string += staight_track_string.format(i,
            " ".join(map(str,(start_point[dict_key][i]['x'], 
                     start_point[dict_key][i]['y'], 
                     start_point[dict_key][i]['z']))),
            " ".join(map(str,(start_point[dict_key][i]['qw'],
                                start_point[dict_key][i]['qx'],
                                start_point[dict_key][i]['qy'],
                                start_point[dict_key][i]['qz']))))

    xml_string += """\n</mujoco>"""
    return xml_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_num", type=float, default=10, help="Number of samples along the Z-axis")
    parser.add_argument("--precision", type=float, default=1e-3, help="Precision for coordinate rounding")
    parser.add_argument("--points_filename", type=str, default="straight.json", help="Output json filename")
    parser.add_argument("--xml_filename", type=str, default="track_straight.xml", help="Output xml filename")
    parser.add_argument('--start_point', type=float, nargs=7, 
                        default=[-0.02, 0.48, 0.45, 1, 0, 0, 0], 
                        help="Starting point of the track: x y z qw qx qy qz")
    parser.add_argument('--track_num', type=int, default=1, help="Number of tracks to generate")
    args = parser.parse_args()

    start_points, points = sample_straight(args.start_point, args.sample_num, args.track_num, args.precision)
    xml_string = xml_straight(start_points, args.track_num)
    save_points(args.points_filename, points)
    save_xml(args.xml_filename, xml_string)
