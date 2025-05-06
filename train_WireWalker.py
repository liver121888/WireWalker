from __future__ import annotations

import hydra
import torch
import os
import random
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_trace import PPO_Trace
import configs.env.WireWalkerCfg as WireWalkerCfg
import time
import numpy as np
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    config.test = config.test
    model_path = None
    if (config.task == 'Tracing' and config.checkpoint_tracing):
        config.checkpoint_tracing = to_absolute_path(config.checkpoint_tracing)
        model_path = config.checkpoint_tracing

    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'

    # config.seed = random.seed(config.seed)
    if config.seed == -1:
        seed = int(time.time())  # or use random.SystemRandom().randint(...)
    else:
        seed = config.seed
    print("seed: ", seed)
    # Set the seed for random number generation
    random.seed(seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/WireWalkerVecWorld-v0'
    task = config.task
    print("===== top level config =====")
    print("config.num_envs: ", config.num_envs)
    print("config.wire_name: ", config.wire_name)
    print("config.wire_name_eval: ", config.wire_name_eval)
    print("config.domand_rand: ", config.domain_rand)
    print("============================")

    if config.domain_rand:
        wire_name_list = []
        for _ in range(config.num_envs):
            # Randomly select a wire name from the list
            wire_idx = random.randint(0, len(WireWalkerCfg.wire_names)-1)
            wire_name = WireWalkerCfg.wire_names[wire_idx]
            wire_name_list.append(wire_name)

        print("wire_name_list: ", wire_name_list)
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda wire_name=wire_name: gym.make(
                    env_name,
                    task=task, camera_name=["top"],
                    render_per_step=False, render_mode="rgb_array",
                    wire_name=wire_name,  # Use the specific wire name for this env
                    img_size=config.train.ppo.img_dim,
                    imshow_cam=config.imshow_cam, 
                    viewer=config.viewer,
                    print_obs=config.print_obs, print_info=config.print_info,
                    print_reward=config.print_reward, print_ctrl=config.print_ctrl,
                    print_contacts=config.print_contacts, wire_name_eval=config.wire_name_eval,
                    env_time=15, steps_per_policy=20,
                    device=config.rl_device,
                )
                for wire_name in wire_name_list
            ]
        )
    else:
        envs = gym.make_vec(env_name, num_envs=int(config.num_envs), 
                    task=task, camera_name=["top"],
                    render_per_step=False, render_mode = "rgb_array",
                    wire_name = config.wire_name,
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = config.print_obs, print_info = config.print_info,
                    print_reward = config.print_reward, print_ctrl = config.print_ctrl,
                    print_contacts = config.print_contacts, wire_name_eval = config.wire_name_eval,
                    env_time = 15, steps_per_policy = 20,
                    device=config.rl_device,)

    # print(envs)
    # domain_rand
    # <AsyncVectorEnv instance>
    # not domain_rand
    # AsyncVectorEnv(gym_dcmm/WireWalkerVecWorld-v0, 8)

    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('America/New_York')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    PPO = PPO_Trace
    
    agent = PPO(envs, output_dif, full_config=config)

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # connect to wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )

        agent.restore_train(model_path)
        agent.train()

        # close wandb
        wandb.finish()

if __name__ == '__main__':
    main()
