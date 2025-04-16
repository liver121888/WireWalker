"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the RL environment for the DCMM task
"""

import os, sys

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./gym_dcmm/"))
import argparse
import math

# print(os.getcwd())
import configs.env.WireWalkerCfg as WireWalkerCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoWireWalker import MJ_WireWalker
from gym_dcmm.utils.ik_pkg.ik_base import IKBase
import copy
from termcolor import colored
from decorators import *
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from collections import deque
import json

# os.environ['MUJOCO_GL'] = 'egl'
np.set_printoptions(precision=8)

paused = True
cmd_lin_y = 0.0
cmd_lin_x = 0.0
cmd_ang = 0.0
trigger_delta = False
trigger_delta_hand = False
speed_delta = 0.1

def env_key_callback(keycode):
  print("chr(keycode): ", (keycode))
  global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
  if keycode == 265: # AKA: up
    cmd_lin_y += speed_delta
    print("up %f" % cmd_lin_y)
  if keycode == 264: # AKA: down
    cmd_lin_y -= speed_delta
    print("down %f" % cmd_lin_y)
  if keycode == 263: # AKA: left
    cmd_lin_x -= speed_delta
    print("left: %f" % cmd_lin_x)
  if keycode == 262: # AKA: right
    cmd_lin_x += speed_delta
    print("right %f" % cmd_lin_x) 
  if keycode == 52: # AKA: 4
    cmd_ang -= 0.2
    print("turn left %f" % cmd_ang)
  if keycode == 54: # AKA: 6
    cmd_ang += 0.2
    print("turn right %f" % cmd_ang)
  if chr(keycode) == ' ': # AKA: space
    if paused: paused = not paused
  if keycode == 334: # AKA + (on the numpad)
    trigger_delta = True
    delta_xyz = 0.1
  if keycode == 333: # AKA - (on the numpad)
    trigger_delta = True
    delta_xyz = -0.1


class WireWalkerVecEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.
        viewer: bool
            Whether to show the mujoco viewer.
        imshow_cam: bool
            Whether to show the camera image.
        object_eval: bool
            Use the evaluation object.
        camera_name: str
            The name of the camera.
        wire_name: str
            The name of the object.
        env_time: float
            The maximum time of the environment.
        steps_per_policy: int
            The number of steps per action.
        img_size: tuple
            The size of the image.
    """

    def __init__(
        self,
        task="tracing",
        camera_name=["top", "wrist"],
        render_per_step=False,
        render_mode="depth_array",
        wire_name="straight",
        img_size=(480, 640),
        imshow_cam=False,
        viewer=False,
        print_obs=False,
        print_info=False,
        print_reward=False,
        print_ctrl=False,
        print_contacts=False,
        wire_name_eval="",
        domain_rand=False,
        env_time=2.5,
        steps_per_policy=20,
        device="cuda:0",
    ):
        if task not in ["Tracking", "Catching", "Tracing"]:
            raise ValueError("Invalid task: {}".format(task))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.wire_name = wire_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts

        self.wire_train = True
        self.wire_name_eval = wire_name_eval
        if self.wire_name_eval:
            self.wire_train = False
        self.domain_rand = domain_rand

        # Initialize the environment
        self.WireWalker = MJ_WireWalker(
            viewer=viewer
        )
        # self.WireWalker.show_model_info()
        self.fps = 1 / (self.steps_per_policy * self.WireWalker.model.opt.timestep)
        # Randomize the Object Info
        self.random_mass = 0.25
        # self.object_static_time = 0.75
        # self.object_throw = False

        self.ee_link_name = "link_ee"

        self.WireWalker.model_xml_string = self._reset_scene()
        self.WireWalker.model = mujoco.MjModel.from_xml_string(
            self.WireWalker.model_xml_string
        )
        self.WireWalker.data = mujoco.MjData(self.WireWalker.model)
        self.floor_id = mujoco.mj_name2id(
            self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.base_id = mujoco.mj_name2id(
            self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, "ranger_base"
        )
        # Get the body id of the wire
        self.wire_id = mujoco.mj_name2id(
            self.WireWalker.model, mujoco.mjtObj.mjOBJ_BODY, "wire_0"
        )
        first_geom_id = next(
            geom_id
            for geom_id in range(self.WireWalker.model.ngeom)
            if self.WireWalker.model.geom_bodyid[geom_id] == self.wire_id
        )
        # Retrieve the group id for wire (assuming all geoms have same group)
        self.wire_group = self.WireWalker.model.geom_group[first_geom_id]
 
        # Set the camera configuration
        self.WireWalker.model.vis.global_.offwidth = WireWalkerCfg.cam_config["width"]
        self.WireWalker.model.vis.global_.offheight = WireWalkerCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.WireWalker.model, self.WireWalker.data
        )
        if self.WireWalker.open_viewer:
            if self.WireWalker.viewer:
                print("Close the previous viewer")
                self.WireWalker.viewer.close()
            self.WireWalker.viewer = mujoco.viewer.launch_passive(
                self.WireWalker.model,
                self.WireWalker.data,
                key_callback=env_key_callback,
            )
            # Modify the view position and orientation
            self.WireWalker.viewer.cam.lookat[0:2] = [0, 1]
            self.WireWalker.viewer.cam.distance = 5.0
            self.WireWalker.viewer.cam.azimuth = 180
            # self.viewer.cam.elevation = -1.57
        else:
            self.WireWalker.viewer = None

        # Observations
        """
        Base:
        Position of base (pos2d)                2,
        Velocity of base (v lin 2d)             2,

        Arm:
        joint positions (joint pos)             6,
        Position of hoop (ee pos3d)             3,
        Quat of hoop (ee quat)                  4,
        Velocity of hoop (ee v lin 3d)          3,

        Wire:
        Position of closest wire point to hoop  3,
        Quat of closest wire point to hoop      4,

        Total Observation Space Dim             27,
        """
        self.observation_space = spaces.Dict(
            {

                "base": spaces.Dict({
                    "pos2d": spaces.Box(-10, 10, shape=(2,), dtype=np.float32),
                    "v_lin_2d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),
                }),
                "arm": spaces.Dict({
                    "joint_pos": spaces.Box(low = np.array([self.WireWalker.model.jnt_range[i][0] for i in range(9, 15)]),
                                            high = np.array([self.WireWalker.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                }),
                "wire": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                })
            }
        )
        # Define the limit for the mobile base action
        base_low = np.array([-2, -2])
        base_high = np.array([2, 2])
        # Define the limit for the arm action
        arm_low = -0.015 * np.ones(4)
        arm_high = 0.015 * np.ones(4)

        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]

        # Actions
        """
        Base:
        base low/high                           2,

        Arm:
        not arm joints, is ee delta position    4,
        and delta roll. x, y, z, roll

        Total Action Space Dim                  6,
        """
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32),
            }
        )
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
        }
        # Combine the limits of the action space
        self.actions_low = np.concatenate([base_low, arm_low])
        self.actions_high = np.concatenate([base_high, arm_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)

        print(
            "##### {} Task: obs_dim: {}, act_dim: {}".format(
                self.task, self.obs_dim, self.act_dim
            )
        )

        # Init env params
        self.arm_limit = True
        self.terminated = False
        self.start_time = self.WireWalker.data.time
        self.catch_time = self.WireWalker.data.time - self.start_time
        self.reward_touch = 0
        self.reward_stability = 0
        self.env_time = env_time
        self.steps = 0 

        self.prev_ctrl = np.zeros(18)
        self.init_ctrl = True
        self.vel_init = False
        self.vel_history = deque(maxlen=4)

        self.waypoint_pos = []
        self.waypoint_quat = []
        self._load_waypoints()

        # get the distance between the end effector and the waypoint in wire
        self.info = self._get_info()

        self.contacts = {
            # Get contact point from the mujoco model
            "object_contacts": np.array([]),
            # "hand_contacts": np.array([]),
        }

        self.object_q = np.array([1, 0, 0, 0])
        self.object_pos3d = np.array([0, 0, 1.5])
        self.object_vel6d = np.array([0.0, 0.0, 1.25, 0.0, 0.0, 0.0])
        self.step_touch = False

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))

        # Random PID Params
        self.k_arm = np.ones(6)
        self.k_drive = np.ones(4)
        self.k_steer = np.ones(4)
        # self.k_hand = np.ones(1)
        # Random Obs & Act Params
        self.k_obs_base = WireWalkerCfg.k_obs_base
        self.k_obs_arm = WireWalkerCfg.k_obs_arm
        self.k_obs_wire = WireWalkerCfg.k_obs_wire
        # self.k_obs_hand = WireWalkerCfg.k_obs_hand
        self.k_obs_object = WireWalkerCfg.k_obs_object
        self.k_act = WireWalkerCfg.k_act

    def _load_waypoints(self):
        # Read the JSON file to get wire points
        with open(os.path.join(WireWalkerCfg.ASSET_PATH, 'points', self.wire_name + '.json'), 'r') as file:
            data = json.load(file)

        self.waypoint_pos = []
        self.waypoint_quat = []

        _waypoints = data[self.wire_name]
        for pt in _waypoints:
            self.waypoint_pos.append([pt["x"], pt["y"], pt["z"]])
            self.waypoint_quat.append([pt['qw'], pt['qx'], pt['qy'], pt['qz']])
        
        self.last_waypoint_idx = 0
        self.waypoint_num = len(self.waypoint_pos)
        print("Got", self.waypoint_num, "waypoints")

    def _get_contacts(self):
        # Contact information
        contacts = self.WireWalker.data.contact
        geom1_ids = np.array(
            [contacts[i].geom1 for i in range(self.WireWalker.data.ncon)]
        )
        geom2_ids = np.array(
            [contacts[i].geom2 for i in range(self.WireWalker.data.ncon)]
        )

        ## get the contact points of the object
        geom1_groups = np.array(
            [self.WireWalker.model.geom_group[geom1_id] for geom1_id in geom1_ids]
        )
        geom2_groups = np.array(
            [self.WireWalker.model.geom_group[geom2_id] for geom2_id in geom2_ids]
        )

        # Check for collisions with the wire
        # object_contacts = np.where((geom1_groups == self.wire_group) | (geom2_groups == self.wire_group))[0]

        # ## get the contact points of the object
        geom1_object = np.where(geom1_groups == self.wire_group)[0]
        geom2_object = np.where(geom2_groups == self.wire_group)[0]

        contacts_geom1 = np.array([])
        contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom1_ids[geom1_object]
        if geom2_object.size != 0:
            contacts_geom2 = geom2_ids[geom2_object]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))

        # geom1_names = [
        #     mujoco.mj_id2name(self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
        #     for geom1_id in geom1_ids
        # ]
        # geom2_names = [
        #     mujoco.mj_id2name(self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
        #     for geom2_id in geom2_ids
        # ]

        # print("geom1 names: ", geom1_names)
        # print("geom2 names: ", geom2_names)

        if len(object_contacts) > 0 and self.print_contacts:
            print("object_contacts: ", object_contacts)

        ## get the contact points of the base
        geom1_base = np.where(geom1_ids == self.base_id)[0]
        geom2_base = np.where(geom2_ids == self.base_id)[0]
        contacts_geom1 = np.array([])
        contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom1_ids[geom1_base]
        if geom2_base.size != 0:
            contacts_geom2 = geom2_ids[geom2_base]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))

        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("base_contacts: ", base_contacts)

        return {"object_contacts": object_contacts, "base_contacts": base_contacts}

    def _get_base_pos2d(self):
        return np.array(self.WireWalker.data.body("arm_base").xpos[0:2])

    def _get_base_vel(self):
        base_yaw = quat2theta(
            self.WireWalker.data.body("base_link").xquat[0],
            self.WireWalker.data.body("base_link").xquat[3],
        )
        global_base_vel = self.WireWalker.data.qvel[0:2]
        base_vel_x = (
            math.cos(base_yaw) * global_base_vel[0]
            + math.sin(base_yaw) * global_base_vel[1]
        )
        base_vel_y = (
            -math.sin(base_yaw) * global_base_vel[0]
            + math.cos(base_yaw) * global_base_vel[1]
        )
        return np.array([base_vel_x, base_vel_y])

    def _get_relative_ee_pos3d(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        base_yaw = quat2theta(
            self.WireWalker.data.body("base_link").xquat[0],
            self.WireWalker.data.body("base_link").xquat[3],
        )
        x, y = relative_position(
            self.WireWalker.data.body("arm_base").xpos[0:2],
            self.WireWalker.data.body(self.ee_link_name).xpos[0:2],
            base_yaw,
        )
        return np.array(
            [
                x,
                y,
                self.WireWalker.data.body(self.ee_link_name).xpos[2]
                - self.WireWalker.data.body("arm_base").xpos[2],
            ]
        )

    def _get_relative_ee_quat(self):
        # Caclulate the ee_quat w.r.t. the base_link
        quat = relative_quaternion(
            self.WireWalker.data.body("base_link").xquat,
            self.WireWalker.data.body(self.ee_link_name).xquat,
        )
        return np.array(quat)

    def _get_absolute_ee_pos3d(self):
        # Caclulate the absolute ee_pos3d
        return np.array([self.WireWalker.data.body(self.ee_link_name).xpos])

    def _get_absolute_ee_quat(self):
        # Caclulate the absolute ee_quat
        return np.array([self.WireWalker.data.body(self.ee_link_name).xquat])
    
    def _get_absolute_wire_pos3d(self):
        return np.array(self.waypoint_pos[min(self.last_waypoint_idx, self.waypoint_num-1)])
    
    def _get_absolute_wire_quat(self):
        return np.array(self.waypoint_quat[min(self.last_waypoint_idx, self.waypoint_num-1)])

    def _get_relative_ee_v_lin_3d(self):
        # Caclulate the ee_v_lin3d w.r.t. the base_link
        # In simulation, we can directly get the velocity of the end-effector
        base_vel = self.WireWalker.data.body("arm_base").cvel[3:6]
        global_ee_v_lin = self.WireWalker.data.body(self.ee_link_name).cvel[3:6]
        base_yaw = quat2theta(
            self.WireWalker.data.body("base_link").xquat[0],
            self.WireWalker.data.body("base_link").xquat[3],
        )
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0] - base_vel[0]) + math.sin(
            base_yaw
        ) * (global_ee_v_lin[1] - base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (
            global_ee_v_lin[0] - base_vel[0]
        ) + math.cos(base_yaw) * (global_ee_v_lin[1] - base_vel[1])
        # TODO: In the real world, we can only estimate it by differentiating the position
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2] - base_vel[2]])

    def _get_obs(self):
        ee_pos3d = self._get_relative_ee_pos3d()
        if self.init_pos:
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.init_pos = False
        # Add Obs Noise (Additive self.k_obs_base/arm/hand/object)
        obs = {
            "base": {
                "pos2d": self._get_base_pos2d()
                + np.random.normal(0, self.k_obs_base, 2),
                "v_lin_2d": self._get_base_vel()
                + np.random.normal(0, self.k_obs_base, 2),
            },
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),
                "ee_quat": self._get_relative_ee_quat()
                + np.random.normal(0, self.k_obs_arm, 4),
                "ee_v_lin_3d": (ee_pos3d - self.prev_ee_pos3d) * self.fps
                + np.random.normal(0, self.k_obs_arm, 3),
                "joint_pos": np.array(self.WireWalker.data.qpos[15:21])
                + np.random.normal(0, self.k_obs_arm, 6),
            },
            "wire": {
                "pos3d": self._get_absolute_wire_pos3d() + np.random.normal(0, self.k_obs_wire, 3),
                "quat": self._get_absolute_wire_quat() + np.random.normal(0, self.k_obs_wire, 4),
            }
        }
        self.prev_ee_pos3d = ee_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs
        # return obs_tensor

    def _get_info(self):

        # Time of the Mujoco environment
        env_time = self.WireWalker.data.time - self.start_time

        ee_distance = np.linalg.norm(
            self.WireWalker.data.body(self.ee_link_name).xpos - self._get_absolute_wire_pos3d()
        )
        base_distance = np.linalg.norm(
            # TODO: should modify to the distance between the base and the last waypoint (?
            self.WireWalker.data.body("arm_base").xpos[0:2] - self._get_absolute_wire_pos3d()[0:2]
        )
        # print("base_distance: ", base_distance)

        ee_abs_pose = self._get_absolute_ee_pos3d()
        ee_abs_quat = self._get_absolute_ee_quat()

        if self.print_info:
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
            print("ee_abs_pose: ", ee_abs_pose)
            print("ee_abs_quat: ", ee_abs_quat)

        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
            # Update imgs elsewhere
            "imgs": {},
        }

    def update_target_ctrl(self):
        self.action_buffer["base"].append(
            copy.deepcopy(self.WireWalker.target_base_vel[:])
        )
        self.action_buffer["arm"].append(
            copy.deepcopy(self.WireWalker.target_arm_qpos[:])
        )
        # self.action_buffer["hand"].append(copy.deepcopy(self.WireWalker.target_hand_qpos[:]))

    def _get_ctrl(self):
        # Map the action to the control
        mv_steer, mv_drive = self.WireWalker.move_base_vel(
            self.action_buffer["base"][0]
        )  # 8
        mv_arm = self.WireWalker.arm_pid.update(
            self.action_buffer["arm"][0],
            self.WireWalker.data.qpos[15:21],
            self.WireWalker.data.time,
        )  # 6

        ctrl = np.concatenate([mv_steer, mv_drive, mv_arm], axis=0)
        # print(ctrl.shape)

        # Add Action Noise (Scale with self.k_act)
        # noise level: self.k_act
        ctrl *= np.random.normal(1, self.k_act, ctrl.shape[0])
        if self.print_ctrl:
            print("##### ctrl:")
            print(
                "mv_steer: {}, \nmv_drive: {}, \nmv_arm: {}\n".format(
                    mv_steer, mv_drive, mv_arm
                )
            )
        return ctrl

    def _reset_scene(self):
        
        # Parse the XML string
        root = ET.fromstring(self.WireWalker.model_xml_string)

        # Find the <include> element with name="wire"
        include_body = root.find(".//include[@name='wire']")
        if include_body is not None:
            file_name = include_body.get("file")
            if file_name is not None:
                # Update the file attribute with the new file path
                # Modify the wire type

                if self.domain_rand:
                    wire_idx = np.random.choice([i for i in range(len(WireWalkerCfg.wire_names))])
                    print("wire_idx: ", wire_idx)
                    self.wire_name = WireWalkerCfg.wire_names[wire_idx]
                elif not self.wire_train:
                    self.wire_name = self.wire_name_eval
                # else: 
                    # train, use self.wire_name
                
                include_body.set("file", "assets/urdf/wires/wire_{}.xml".format(self.wire_name))

        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding="unicode")

        return xml_str


    def random_PID(self):

        # Random the PID Controller Params in WireWalker
        self.k_arm = np.random.uniform(0, 1, size=6)
        self.k_drive = np.random.uniform(0, 1, size=4)
        self.k_steer = np.random.uniform(0, 1, size=4)

        # Reset the PID Controller
        self.WireWalker.arm_pid.reset(
            self.k_arm * (WireWalkerCfg.k_arm[1] - WireWalkerCfg.k_arm[0])
            + WireWalkerCfg.k_arm[0]
        )
        self.WireWalker.steer_pid.reset(
            self.k_steer * (WireWalkerCfg.k_steer[1] - WireWalkerCfg.k_steer[0])
            + WireWalkerCfg.k_steer[0]
        )
        self.WireWalker.drive_pid.reset(
            self.k_drive * (WireWalkerCfg.k_drive[1] - WireWalkerCfg.k_drive[0])
            + WireWalkerCfg.k_drive[0]
        )

    def random_delay(self):
        # Random the Delay Buffer Params in WireWalker
        self.action_buffer["base"].set_maxlen(
            np.random.choice(WireWalkerCfg.act_delay["base"])
        )
        self.action_buffer["arm"].set_maxlen(
            np.random.choice(WireWalkerCfg.act_delay["arm"])
        )

        # Clear Buffer
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()


    def _reset_simulation(self):
        # Reset the data in Mujoco Simulation

        # will set self.wire_name
        # self.WireWalker.model_xml_string = self._reset_scene()
        # self.WireWalker.model = mujoco.MjModel.from_xml_string(
        #     self.WireWalker.model_xml_string
        # )
        # self.WireWalker.data = mujoco.MjData(self.WireWalker.model)
        # self.floor_id = mujoco.mj_name2id(
        #     self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        # )
        # self.base_id = mujoco.mj_name2id(
        #     self.WireWalker.model, mujoco.mjtObj.mjOBJ_GEOM, "ranger_base"
        # )
        # # Get the body id of the wire
        # self.wire_id = mujoco.mj_name2id(
        #     self.WireWalker.model, mujoco.mjtObj.mjOBJ_BODY, "wire_0"
        # )
        # first_geom_id = next(
        #     geom_id
        #     for geom_id in range(self.WireWalker.model.ngeom)
        #     if self.WireWalker.model.geom_bodyid[geom_id] == self.wire_id
        # )
        # self.wire_group = self.WireWalker.model.geom_group[first_geom_id]

        # self._load_waypoints()

        mujoco.mj_resetData(self.WireWalker.model, self.WireWalker.data)
        mujoco.mj_resetData(self.WireWalker.model_arm, self.WireWalker.data_arm)
        if self.WireWalker.model.na == 0:
            self.WireWalker.data.act[:] = None
        if self.WireWalker.model_arm.na == 0:
            self.WireWalker.data_arm.act[:] = None
        self.WireWalker.data.ctrl = np.zeros(self.WireWalker.model.nu)
        self.WireWalker.data_arm.ctrl = np.zeros(self.WireWalker.model_arm.nu)
        self.WireWalker.data.qpos[15:21] = WireWalkerCfg.arm_joints[:]
        # self.WireWalker.data.qpos[21:37] = WireWalkerCfg.hand_joints[:]
        self.WireWalker.data_arm.qpos[0:6] = WireWalkerCfg.arm_joints[:]
        # self.WireWalker.data.body(self.wire_name).xpos[0:3] = np.array([2, 2, 1])

        # Random 3D position TODO: Adjust to the fov
        # self.random_object_pose()
        # self.WireWalker.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    # velocity=np.zeros(6))
        # TODO: TESTING
        # self.WireWalker.set_throw_pos_vel(pose=np.array([0.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]),
        #                             velocity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # Random Gravity Why Though
        self.WireWalker.model.opt.gravity[2] = -9.81 #+ 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_PID()
        # Random Delay
        self.random_delay()
        # Forward Kinematics
        mujoco.mj_forward(self.WireWalker.model, self.WireWalker.data)
        mujoco.mj_forward(self.WireWalker.model_arm, self.WireWalker.data_arm)

    def reset(self):
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False
        self.steps = 0
        self.last_waypoint_idx = 0 #start from the start
        # Reset the time
        self.start_time = self.WireWalker.data.time
        self.catch_time = self.WireWalker.data.time - self.start_time

        ## Reset the target velocity of the mobile base
        self.WireWalker.target_base_vel = np.array([0.0, 0.0, 0.0])
        ## Reset the target joint positions of the arm
        self.WireWalker.target_arm_qpos[:] = WireWalkerCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        # self.WireWalker.target_hand_qpos[:] = WireWalkerCfg.hand_joints[:]
        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0

        # TODO: modify to the info we need, ideally the closest point to the hoop, then we cannot simply get wire_name
        self.info = self._get_info()

        # Get the observation and info
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        observation = self._get_obs()
        info = self._get_info()
        # Rendering
        imgs = self.render()
        info["imgs"] = imgs
        ctrl_delay = np.array(
            [len(self.action_buffer["base"]), len(self.action_buffer["arm"])]
        )
        info["ctrl_params"] = np.concatenate((self.k_arm, self.k_drive, ctrl_delay))

        return observation, info

    def norm_ctrl(self, ctrl, components):
        """
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        """
        ctrl_array = np.concatenate(
            [
                ctrl[component] * WireWalkerCfg.reward_weights["r_ctrl"][component]
                for component in components
            ]
        )
        return np.linalg.norm(ctrl_array)

    def compute_reward(self, obs, info, ctrl):
        # 初始化奖励和奖励信息字典
        reward = 0.0
        reward_info = {}
        
        # 1. 中心距离奖励 - 环与导线中心的距离
        ee_distance = info["ee_distance"]
        # center_dist_reward = ee_distance * WireWalkerCfg.reward_weights["r_center_dist"]
        # reward += center_dist_reward
        # reward_info["center_dist"] = center_dist_reward
        
        # 2. 精确度奖励 - 接近理想距离
        precision_reward_factor = min(1, np.exp(-50 * (ee_distance)**2))
        precision_reward = precision_reward_factor * WireWalkerCfg.reward_weights["r_precision"]
        reward += precision_reward
        reward_info["precision"] = precision_reward
        
        # 3. 碰撞惩罚 - 检测与导线的碰撞
        collision_reward = 0
        if self.contacts["object_contacts"].size > 0:
            collision_reward = WireWalkerCfg.reward_weights["r_collision"]
            reward += collision_reward
        reward_info["collision"] = collision_reward
        # 4. 进度奖励 - 沿着轨道的前进
        _num_pts_passed = self.advance_waypoint()
        progress_reward = _num_pts_passed * WireWalkerCfg.reward_weights["r_progress"]
        reward += progress_reward
        reward_info["progress"] = progress_reward
        goal_reward = 0
        if self.last_waypoint_idx >= self.waypoint_num - 1:
            goal_reward = WireWalkerCfg.reward_weights["r_goal"]
            print("Goal Reached!")
        reward += goal_reward
        reward_info["goal"] = goal_reward
        
        # 5. 约束奖励 - 关节限制
        constraint_reward = WireWalkerCfg.reward_weights["r_constraint"] if self.arm_limit else 0
        reward += constraint_reward
        reward_info["constraint"] = constraint_reward
        
        # 6. 控制惩罚 - 惩罚过大的控制输入
        reward_ctrl = self.norm_ctrl(ctrl, {"base", "arm"}) * WireWalkerCfg.reward_weights["r_ctrl"]["all"]
        reward += reward_ctrl
        
        reward_info["ctrl"] = reward_ctrl
        
        # 7. 时间惩罚 - 鼓励快速完成
        time_penalty = WireWalkerCfg.reward_weights["r_time"]
        reward += time_penalty
        reward_info["time"] = time_penalty
        
        # 如果打印奖励信息
        if self.print_reward:
            print("\n===== 奖励详情 =====")
            print(f"距离: {ee_distance:.4f}m")
            # print(f"中心距离奖励: {center_dist_reward:.4f}")
            print(f"精确度奖励: {precision_reward:.4f}")
            print(f"碰撞惩罚: {collision_reward:.4f}")
            print(f"进度奖励: {progress_reward:.4f}")
            print(f"目标奖励: {goal_reward:.4f}")
            print(f"约束奖励: {constraint_reward:.4f}")
            print(f"控制惩罚: {reward_ctrl:.4f}")
            print(f"时间惩罚: {time_penalty:.4f}")
            print(f"总奖励: {reward:.4f}")
            print("====================\n")
        
        # 保存奖励信息到info字典
        info["reward_info"] = reward_info
        
        return reward

    def _step_mujoco_simulation(self, action_dict):

        ## TODO: Low-Pass-Filter the Base Velocity
        self.WireWalker.target_base_vel[0:2] = action_dict["base"]
        action_arm = np.concatenate((action_dict["arm"], np.zeros(2)))
        result_QP, _ = self.WireWalker.move_ee_pose(action_arm)
        if result_QP[1]:
            self.arm_limit = True
            self.WireWalker.target_arm_qpos[:] = result_QP[0]
        else:
            # print("IK Failed!!!")
            self.arm_limit = False

        # Add Target Action to the Buffer
        self.update_target_ctrl()
        # Reset the Criteria for Successfully Touch
        self.step_touch = False
        for _ in range(self.steps_per_policy):
            # Update the control command according to the latest policy output
            self.WireWalker.data.ctrl[:] = self._get_ctrl()
            if self.render_per_step:
                # Rendering
                img = self.render()
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            # if self.WireWalker.data.time - self.start_time < self.object_static_time:
            #     self.WireWalker.set_throw_pos_vel(
            #         pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
            #         velocity=np.zeros(6),
            #     )
            #     self.WireWalker.data.ctrl[-1] = (
            #         self.random_mass * -self.WireWalker.model.opt.gravity[2]
            #     )
            # elif not self.object_throw:
            #     self.WireWalker.set_throw_pos_vel(
            #         pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
            #         velocity=self.object_vel6d[:],
            #     )
            #     self.WireWalker.data.ctrl[-1] = 0.0
            #     self.object_throw = True

            mujoco.mj_step(self.WireWalker.model, self.WireWalker.data)
            mujoco.mj_rnePostConstraint(self.WireWalker.model, self.WireWalker.data)

            # Update the contact information
            self.contacts = self._get_contacts()
            # Whether the base collides
            if self.contacts["base_contacts"].size != 0:
                self.terminated = True

            # terminate the episode in advance
            if self.terminated:
                break

    def advance_waypoint(self):
        ee_abs_pose = self._get_absolute_ee_pos3d().squeeze()
        _num_pts_advanced = 0
        
        while self.last_waypoint_idx < self.waypoint_num and \
                np.linalg.norm(ee_abs_pose - self.waypoint_pos[self.last_waypoint_idx]) < WireWalkerCfg.WAYPOINT_DIST_EPSILON:
            print("Moved past waypoint", self.last_waypoint_idx)
            self.last_waypoint_idx += 1
            _num_pts_advanced += 1
        return _num_pts_advanced

    def step(self, action):
        self.steps += 1
        self._step_mujoco_simulation(action)
        # Get the obs and info
        obs = self._get_obs()
        info = self._get_info()
        # TODO: change the termination condition
        # check if we finished the task
        
        if self.task == "Catching":
            if (
                info["ee_distance"] < WireWalkerCfg.distance_thresh
                and self.stage == "tracking"
            ):
                self.stage = "grasping"
            elif (
                info["ee_distance"] >= WireWalkerCfg.distance_thresh
                and self.stage == "grasping"
            ):
                self.terminated = True
        elif self.task == "Tracing":
            if self.last_waypoint_idx == self.waypoint_num:
                print("### Tracing Success!!!")
                self.terminated = True


        # Design the reward function
        # Before we update the info, we need to compute the reward
        reward = self.compute_reward(obs, info, action)
        # reward = 0 # temporarily
        # Update the info
        self.info["base_distance"] = info["base_distance"]
        self.info["ee_distance"] = info["ee_distance"]
        # Rendering
        imgs = self.render()
        # Update the imgs
        info["imgs"] = imgs
        ctrl_delay = np.array(
            [len(self.action_buffer["base"]), len(self.action_buffer["arm"])]
        )
        info["ctrl_params"] = np.concatenate((self.k_arm, self.k_drive, ctrl_delay))
        # The episode is truncated if the env_time is larger than the predefined time
        if self.task == "Catching":
            if info["env_time"] > self.env_time:
                # print("Catching Success!!!!!!")
                truncated = True
            else:
                truncated = False
        elif self.task == "Tracking":
            if self.step_touch:
                # print("Tracking Success!!!!!!")
                truncated = True
            else:
                truncated = False
        elif self.task == "Tracing":
            if info["env_time"] > self.env_time:
                truncated = True
            else:
                truncated = False

        terminated = self.terminated
        done = terminated or truncated
        if done:
            # TEST ONLY
            # print("### DONE ###")
            self.reset()
            # pass
        return obs, reward, terminated, truncated, info

    def preprocess_depth_with_mask(
        self,
        rgb_img,
        depth_img,
        depth_threshold=3.0,
        num_white_points_range=(5, 15),
        point_size_range=(1, 5),
    ):
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(
            num_white_points_range[0], num_white_points_range[1]
        )
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(
            point_size_range[0], point_size_range[1], size=num_white_points
        )
        # Create masks for all white points at once
        y, x = np.ogrid[: masked_depth_img.shape[0], : masked_depth_img.shape[1]]
        point_masks = (
            (x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2
        ) <= random_sizes**2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        # imgs_rgb = np.zeros((self.img_size[0], self.img_size[1], 3))
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                self.mujoco_renderer.render(self.render_mode, camera_name=camera_name)
                return imgs
            elif self.render_mode != "depth_rgb_array":
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name=camera_name
                )
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.render_mode == "depth_array":
                    img = self.WireWalker.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0 / img.max()))
                        cv.imshow(camera_name + "_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            else:
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name=camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name=camera_name
                )
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.WireWalker.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(
                        camera_name + "_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
                    )
                    cv.imshow(camera_name + "_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.WireWalker.viewer != None:
                self.WireWalker.viewer.sync()
        if self.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.WireWalker.viewer != None:
            self.WireWalker.viewer.close()

    # Ignore the comments in the following function
    def run_test(self):
        global cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
        self.reset()
        # action = np.zeros(18)
        action = np.zeros(6)
        total_reward = 0.0
        while True:
            # Note: action's dim = 18, which includes 2 for the base, 4 for the arm, and 12 for the hand
            # Keyboard control
            action[0:2] = np.array([cmd_lin_x, cmd_lin_y])
            if trigger_delta:
                print("delta_xyz: ", delta_xyz)
                action[2:6] = np.array([delta_xyz, delta_xyz, delta_xyz, delta_xyz])
                trigger_delta = False
            else:
                action[2:6] = np.zeros(4)

            # if trigger_delta_hand:
            #     print("delta_xyz_hand: ", delta_xyz_hand)
            #     action[6:18] = np.ones(12)*delta_xyz_hand
            #     trigger_delta_hand = False
            # else:
            #     action[6:18] = np.zeros(12)
            base_tensor = action[:2]
            arm_tensor = action[2:6]
            # hand_tensor = action[6:18]
            actions_dict = {
                "arm": arm_tensor,
                "base": base_tensor,
                # 'hand': hand_tensor
            }
            # print("self.WireWalker.data.body('link6'):", self.WireWalker.data.body('link6'))
            observation, reward, terminated, truncated, info = self.step(actions_dict)
            total_reward += reward
            print("ee dist", info["ee_distance"])
            # print(f"当前奖励: {reward:.4f}")
            # print(f"累计奖励: {total_reward:.4f}")

if __name__ == "__main__":
    os.chdir("../../")
    parser = argparse.ArgumentParser(description="Args for WireWalkerVecEnv")
    parser.add_argument(
        "--viewer", action="store_true", help="open the mujoco.viewer or not"
    )
    parser.add_argument(
        "--wire_name", type=str, default="straight", help="wire name"
    )
    parser.add_argument(
        "--imshow_cam", action="store_true", help="imshow the camera image or not"
    )
    parser.add_argument(
        "--task", type=str, default="Tracing", help="Task for the WireWalkerEnv"
    )
    parser.add_argument(
        "--print_reward", action="store_true", help="print the reward or not"
    )
    parser.add_argument(
        "--print_info", action="store_true", help="print the info or not"
    )
    parser.add_argument(
        "--print_contacts", action="store_true", help="print the contacts or not"
    )
    parser.add_argument(
        "--print_ctrl", action="store_true", help="print the ctrl or not"
    )
    parser.add_argument(
        "--print_obs", action="store_true", help="print the observation or not"
    )
    args = parser.parse_args()
    print("args: ", args)
    env = WireWalkerVecEnv(
        task=args.task,
        # TODO: modify to a list of body
        wire_name=args.wire_name,
        render_per_step=False,
        print_reward=args.print_reward,
        print_info=args.print_info,
        print_contacts=args.print_contacts,
        print_ctrl=args.print_ctrl,
        print_obs=args.print_obs,
        camera_name=["top"],
        render_mode="rgb_array",
        imshow_cam=args.imshow_cam,
        viewer=args.viewer,
        wire_eval=False,
        # last 5 minutes
        env_time=300,
        steps_per_policy=20,
    )
    env.run_test()
