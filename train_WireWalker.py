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
# from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
# from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
# from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_trace import PPO_Trace

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
    # if config.task == 'Tracking' and config.checkpoint_tracking:
    #     config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
    #     model_path = config.checkpoint_tracking
    # elif (config.task == 'Catching_TwoStage' \
    #     or config.task == 'Catching_OneStage') \
    #     and config.checkpoint_catching:
    #     config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
    #     model_path = config.checkpoint_catching
    # el
    if (config.task == 'Tracing' and config.checkpoint_tracing):
        config.checkpoint_tracing = to_absolute_path(config.checkpoint_tracing)
        model_path = config.checkpoint_tracing


    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
    config.seed = random.seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/WireWalkerVecWorld-v0'
    # if config.task == 'Tracking':
    #     task = 'Tracking'
    # elif config.task == 'Catching':
    #     task = 'Catching'
    # else:
    task = 'Tracing'
    print("config.num_envs: ", config.num_envs)
    print("config.domand_rand: ", config.domain_rand)
    print("config.wire_name: ", config.wire_name)
    print("config.wire_name_eval: ", config.wire_name_eval)
    env = gym.make_vec(env_name, num_envs=int(config.num_envs), 
                    task=task, camera_name=["top"],
                    render_per_step=False, render_mode = "rgb_array",
                    wire_name = config.wire_name,
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = config.print_obs, print_info = config.print_info,
                    print_reward = config.print_reward, print_ctrl = config.print_ctrl,
                    print_contacts = config.print_contacts, wire_name_eval = config.wire_name_eval,
                    domain_rand = config.domain_rand,
                    env_time = 2.5, steps_per_policy = 20,
                    device=config.rl_device,)

    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('America/New_York')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    # PPO = PPO_Track if config.task == 'Tracking' else \
    #       PPO_Trace if config.task == 'Tracing' else \
    #       PPO_Catch_TwoStage if config.task == 'Catching_TwoStage' else \
    #       PPO_Catch_OneStage

    PPO = PPO_Trace
    
    agent = PPO(env, output_dif, full_config=config)

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
