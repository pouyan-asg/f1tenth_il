import torch
import gym
import numpy as np
import argparse
import yaml

from policies.agents.agent_mlp import AgentPolicyMLP
from policies.experts.expert_waypoint_follower import ExpertWaypointFollower

import utils.env_utils as env_utils

from bc import bc
from dagger import dagger
from hg_dagger import hg_dagger
    

def initialization(il_config):

    """"
    initialization function for the imitation learning training script.
    It sets up the environment, agent, and expert based on the provided configuration.
    Args:
        il_config (dict): A dictionary containing the imitation learning configuration.
    Returns:
        seed (int): The random seed for reproducibility.
        learner_agent (AgentPolicyMLP): The agent policy initialized based on the configuration.
        expert (ExpertWaypointFollower): The expert policy initialized based on the configuration.
        env (gym.Env): The environment initialized based on the configuration.
        start_pose (np.ndarray): The starting pose of the agent in the environment.
        observation_shape (int): The shape of the observations used by the agent.
        downsampling_method (str): The method used for downsampling observations.
        render (bool): Whether to render the environment during training.
        render_mode (str): The mode of rendering for the environment.
    
    config files:
    - il_config.yaml: configurations for the IL training.
        - map/example_map/config_example_map.yaml: Cconfigurations for the environemnt map.
            - example_map.png: The map image used in the environment.
            - example_waypoints.csv: The waypoints for the example map.
    """

    seed = il_config['random_seed']
    # np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the environment
    map_conf = None

    if il_config['environment']['random_generation'] == False:
        if il_config['environment']['map_config_location'] == None:
            # If no environment is specified but random generation is off, use the default gym environment
            with open('map/example_map/config_example_map.yaml') as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            # If an environment is specified and random generation is off, use the specified environment
            with open(il_config['environment']['map_config_location']) as file:
                map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        # Convert the map configuration dictionary to an object with attribute-style access to its keys.
        map_conf = argparse.Namespace(**map_conf_dict)

        import f110_gym
        env = gym.make('f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
        # env = gym.make('f110_gym:f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
        env.add_render_callback(env_utils.render_callback)
    else:
        # TODO: If random generation is on, generate random environment
        pass
    
    # obs, step_reward, done, info = env.reset(np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]]))

    # Initialize the agent
    if il_config['policy_type']['agent']['model'] == 'mlp':
        """
        - observation shape: 1080
            -- The LiDAR sensor returns 1080 distance measurements in one 2D scan.
                --- FOV = 270 degrees (common for LiDAR sensors)
                --- Angular resolution = 0.25 degrees per measurement
                --- number of distance values in one scan: 270/0.25 = 1080
                    ---- The number of laser beams (rays) cast per LiDAR scan.
                    ---- Each value is a float distance to the nearest object in that direction.
        - hidden_dim: 256
            -- A common heuristic size in deep learning.
                --- e.g. 64, 128, 256, 512, etc.
        - action space: 2 (steering angle and speed)
        """
        learner_agent = AgentPolicyMLP(observ_dim = il_config['policy_type']['agent']['observation_shape'],
                               hidden_dim = il_config['policy_type']['agent']['hidden_dim'],
                               action_dim = 2,
                               lr = il_config['policy_type']['agent']['learning_rate'],
                               device = device)
    else:
        #TODO: Implement other model (Transformer)
        pass


    # Initialize the expert
    if il_config['policy_type']['expert']['behavior']  == 'waypoint_follower':
        expert = ExpertWaypointFollower(map_conf)
    else:
        # TODO: Implement other expert behavior (Lane switcher and hybrid)
        pass
    
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])
    # observation_gap = int(1080/il_config['policy_type']['agent']['observation_shape'])
    observation_shape = il_config['policy_type']['agent']['observation_shape']
    downsampling_method = il_config['policy_type']['agent']['downsample_method']

    render = il_config['environment']['render']
    render_mode = il_config['environment']['render_mode']

    return seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode
    

def train(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    if il_algo == 'bc':
        bc(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='train')
    elif il_algo == 'dagger':
        dagger(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)
    elif il_algo == 'hg-dagger':
        hg_dagger(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)
    else:
        # TODO: Implement other IL algorithms (BC, HG DAgger, etc.)
        pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--algo', type=str, default='bc', help='imitation learning algorithm to use')
    arg_parser.add_argument('--config', type=str, default='il_config.yaml', help='the yaml file containing the training configuration')
    parsed_args = arg_parser.parse_args()

    il_algo = parsed_args.algo
    il_config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)  # Load the YAML configuration file

    # Initialize
    seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode = initialization(il_config)
    
    with open('logs/initialize_log.txt', 'w') as file:
        file.write(f"Seed: {seed}\n")
        file.write(f"Agent: {learner_agent}\n")
        file.write(f"Expert: {expert}\n")
        file.write(f"Environment: {env}\n")
        file.write(f"Start Pose: {start_pose}\n")
        file.write(f"Observation Shape: {observation_shape}\n")
        file.write(f"Downsampling Method: {downsampling_method}\n")
        file.write(f"Render: {render}\n")
        file.write(f"Render Mode: {render_mode}\n")
        # file.write(f"Env action space: {env.action_space}\n")
        # file.write(f"Env observation space: {env.observation_space}\n")
        file.write(f"Env reward: {env.reward_range}\n")
        file.write(f"Env meta data: {env.metadata}\n")

    # Train
    train(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode)

    
    
