from mimetypes import init
import gym
import torch
import numpy as np
import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils
from dataset import Dataset
from pathlib import Path


def bc(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose):
    algo_name = "BehavioralCloning"
    best_model = learner_agent
    longest_distance_travelled = 0  # The longest distance travelled by the agent so far
    path = "logs/{}".format(algo_name)  # Path to save the checkpoint logs
    num_of_saved_models = 0  # Number of saved models during the training
    resume_pose = start_pose  # The pose to resume from in the next iteration
    is_last_round_done = False
    max_traj_len = 10000  # number of evaluation time internally 
    best_model_saving_threshold = 500000  # The threshold for number of samples during training
    num_of_samples_increment = 500  # Number of samples to collect from the expert in each iteration
    n_batch_updates_per_iter = 1000  # Number of batch updates (trainign loop) per iteration
    train_batch_size = 64  # sample batch size for training the agent

    if purpose == "train":
        n_iter = 100
    elif purpose == "bootstrap":
        n_iter = 1
    else:
        raise ValueError("purpose must be either 'train' or 'bootstrap'")
    
    if render:
        eval_batch_size = 1
    else:
        eval_batch_size = 10  # Number of evaluation iteration loops

    # np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Dataset()

    log = {'Number of Samples': [], 
           'Number of Expert Queries': [], 
           'Mean Distance Travelled': [],
           'STDEV Distance Travelled': [],
           'Mean Reward': [],
           'STDEV Reward': []}

    # Perform BC
    for iter in range(n_iter + 1):
        if purpose == "train":
            print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))
        else:
            print("- "*15 + "\nbootstrap using BC:")
        
        # Evaluate the agent's performance after each data collection
        # No evaluation at the initial iteration normally
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)

            mean_travelled_distances, \
            stdev_travelled_distances, \
            mean_reward, \
            stdev_reward = agent_utils.eval(env, 
                                            learner_agent, 
                                            start_pose, 
                                            max_traj_len, 
                                            eval_batch_size, 
                                            observation_shape, 
                                            downsampling_method, 
                                            render, 
                                            render_mode)
            
            log['Mean Distance Travelled'].append(mean_travelled_distances)
            log['STDEV Distance Travelled'].append(stdev_travelled_distances)
            log['Mean Reward'].append(mean_reward)
            log['STDEV Reward'].append(stdev_reward)
            
            # Replace the best model if the current model is better
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled) and (log['Number of Samples'][-1] < best_model_saving_threshold):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = learner_agent
            
            # For Sim2Real: save checkpoints
            # If the agent's mean distance travelled is greater than 100 (a threshold for good performance)
            if (log['Mean Distance Travelled'][-1] > 100):
                curr_dist = log['Mean Distance Travelled'][-1]
                current_expsamples = log['Number of Expert Queries'][-1]
                model_path = Path(path + f'/{algo_name}_svidx_{str(num_of_saved_models)}_dist_{int(curr_dist)}_expsamp_{int(current_expsamples)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # e.g. BehavioralCloning_svidx_0_dist_109_expsamp_18000.pkl
                torch.save(learner_agent.state_dict(), model_path)
                num_of_saved_models += 1

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))

            print("- "*15)

            # DELETE IT WHEN DOING SIM2REAL
            # if log['Number of Samples'][-1] > 3000:
            #     break
        
        if iter == n_iter:
            break

        tlad = 0.82461887897713965  # Lookahead distance for PurePursuit expert
        vgain = 0.90338203837889  # Velocity gain for PurePursuit expert

        # Collect data from the expert
        print("Collecting data from the expert...")
        traj = {"observs": [], "poses_x": [], "poses_y": [], "poses_theta": [], "scans": [], "actions": [], "reward": 0}
        done = False
        obs, step_reward, done, info = env.reset(poses=resume_pose)

        """
        obs: {'ego_idx': 0, 'scans': [array([...])], 'poses_x': [0.7], 'poses_y': [0.0], 
            'poses_theta': [1.37079632679], 'linear_vels_x': [0.0], 'linear_vels_y': [0.0], 
            'ang_vels_z': [0.0], 'collisions': array([0.]), 'lap_times': array([0.01]), 
            'lap_counts': array([0.])}
        step_reward: 0.01
        done: False/True
        """
    
        # Start rendering
        if render:
            if env.renderer is None:
                env.render()
        
        if purpose == "train":
            step_num = num_of_samples_increment
        else:
            step_num = 500

        # collecting 500 samples from the expert (like an expert policy)
        for j in range(step_num):
            traj["observs"].append(obs)
            scan = agent_utils.downsample_and_extract_lidar(obs, observation_shape, downsampling_method)

            # Add Sim2Real noise
            sim2real_noise = np.random.uniform(-0.25, 0.25, scan.shape)
            scan = scan + sim2real_noise

            traj["scans"].append(scan)
            traj["poses_x"].append(obs["poses_x"][0])
            traj["poses_y"].append(obs["poses_y"][0])
            traj["poses_theta"].append(obs["poses_theta"][0])

            # Expert's action regarding current pose
            speed, steer = expert.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)
            action = np.array([[steer, speed]])

            obs, step_reward, done, info = env.step(action)

            # Update rendering
            if render:
                env.render(mode=render_mode)

            traj["actions"].append(action)
            traj["reward"] += step_reward

            if done:
                is_last_round_done = True
                break
        
        # To evenly sampling using expert by resuming at the last pose in the next iteration
        if is_last_round_done:
            resume_pose = start_pose
        else:
            resume_pose = np.array([[obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]]])
        
        # converting a list of 1D (or 2D) NumPy arrays into a single 2D (or 3D) 
        # NumPy array by stacking them vertically.
        traj["observs"] = np.vstack(traj["observs"])  # (500, 1)
        traj["poses_x"] = np.vstack(traj["poses_x"])  # (500, 1)
        traj["poses_y"] = np.vstack(traj["poses_y"])  # (500, 1)
        traj["poses_theta"] = np.vstack(traj["poses_theta"])  # (500, 1)
        traj["scans"] = np.vstack(traj["scans"])  # (500, 1080)
        traj["actions"] = np.vstack(traj["actions"])  # (500, 2)

        # Adding to datasets
        print("Adding to dataset...")
        dataset.add(traj)

        log['Number of Samples'].append(dataset.get_num_of_total_samples())
        log['Number of Expert Queries'].append(dataset.get_num_of_total_samples()) # TODO: change it to the number of expert queries

        # Train the agent
        print("Training agent...")
        for _ in range(n_batch_updates_per_iter):
            train_batch = dataset.sample(train_batch_size)
            learner_agent.train(train_batch["scans"], train_batch["actions"])
        
        # exit the loop and return the initial policy and logs
        if purpose == "bootstrap":
            return learner_agent, log, dataset

    # Save log and the best model
    agent_utils.save_log_and_model(log, best_model, algo_name)