from gym.envs.registration import register
import gym

register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	)