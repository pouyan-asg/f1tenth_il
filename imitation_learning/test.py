# # import gym
# # print(gym.envs.registry)


# import f110_gym  # this must be first to trigger registration
# import gym

# print(gym.envs.registry)  # Show what's in the registry



import yaml
import gym
import f110_gym
import numpy as np
import pygame
import argparse


# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("F1Tenth Keyboard Control")

il_config = yaml.load(open('il_config.yaml'), Loader=yaml.FullLoader)  # Load the YAML configuration file

seed = il_config['random_seed']


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

start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])

# env = gym.make('f110-v0', map='maps/levine_2020', map_ext='.png', num_agents=1)
env = gym.make('f110-v0', map=map_conf.map_path, map_ext=map_conf.map_ext, num_agents=1)
obs = env.reset(poses=start_pose)
done = False

# Control parameters
steer = 0.0
speed = 1.0
steer_delta = 0.1
speed_delta = 0.1

def get_keyboard_action():
    global steer, speed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        steer += steer_delta
    if keys[pygame.K_RIGHT]:
        steer -= steer_delta
    if keys[pygame.K_UP]:
        speed += speed_delta
    if keys[pygame.K_DOWN]:
        speed -= speed_delta
    # Clamp values
    steer = np.clip(steer, -1.0, 1.0)
    speed = np.clip(speed, 0.0, 5.0)
    return np.array([[steer, speed]])

while not done:
    action = get_keyboard_action()
    obs, reward, done, info = env.step(action)
    env.render(mode='human_fast')
    pygame.time.wait(30)  # ~30 FPS

env.close()
pygame.quit()

