from mimetypes import init
import gym
import torch
import numpy as np
import utils.agent_utils as agent_utils
import utils.expert_utils as expert_utils
import utils.env_utils as env_utils
from dataset import Dataset
from bc import bc
from pathlib import Path


def hg_dagger(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    """
    learner_agent: a NN which is trained to imitate the expert
    expert: a PurePursuit expert which is used to provide expert actions
    env: the gym environment in which the agent is trained
    """
    
    algo_name = "HGDAgger"
    best_model = learner_agent
    longest_distance_travelled = 0
    num_of_expert_queries = 0
    init_traj_len = 50
    max_traj_len = 3500  # nummber of steps in teh environment for collecting data (trajectories)
    n_batch_updates_per_iter = 1000  # Number of training loop per iteration
    eval_max_traj_len = 10000  # number of agent evaluation time internally 
    train_batch_size = 64  # sample batch size for training the agent

    # For Sim2Real
    path = "logs/{}".format(algo_name)
    num_of_saved_models = 0

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

    n_iter = 10    # Number of Epochs 267

    n_rollout = 5

    tlad = 0.82461887897713965  # Lookahead distance for PurePursuit expert
    vgain = 0.90338203837889  # Velocity gain for PurePursuit expert

    # Epochs
    for iter in range(n_iter + 1):
        """
        the loop runs for 'n_iter' times in which for each iteration (epoch) 
        'max_traj_len' data is collected and also 'n_batch_updates_per_iter'
        times the agent is trained on the collected data.
        """
        print("-"*30 + ("\ninitial:" if iter == 0 else "\niter {}:".format(iter)))

        # Evaluation
        if iter > 0:
            print("Evaluating agent...")
            print("- "*15)

            # log["Iteration"].append(iter)
            mean_travelled_distances, \
            stdev_travelled_distances, \
            mean_reward, \
            stdev_reward = agent_utils.eval(env, 
                                            learner_agent, 
                                            start_pose, 
                                            eval_max_traj_len, 
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
            if (log['Mean Distance Travelled'][-1] > longest_distance_travelled):
                longest_distance_travelled = log['Mean Distance Travelled'][-1]
                best_model = learner_agent

            # saving checkpoints with a minimum ctiteria (distance travelled > 100)
            if (log['Mean Distance Travelled'][-1] > 100):
                curr_dist = log['Mean Distance Travelled'][-1]
                current_expsamples = log['Number of Expert Queries'][-1]
                model_path = Path(path + f'/{algo_name}_svidx_{str(num_of_saved_models)}_dist_{int(curr_dist)}_expsamp_{int(current_expsamples)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True) 
                torch.save(learner_agent.state_dict(), model_path)
                num_of_saved_models += 1

            print("Number of Samples: {}".format(log['Number of Samples'][-1]))
            print("Number of Expert Queries: {}".format(log['Number of Expert Queries'][-1]))
            print("Distance Travelled: {} (+/- {})".format(log['Mean Distance Travelled'][-1], log['STDEV Distance Travelled'][-1]))
            print("Reward: {} (+/- {})".format(log['Mean Reward'][-1], log['STDEV Reward'][-1]))
            print("- "*15)

            # DELETE IT WHEN DOING SIM2REAL
            # if log['Number of Expert Queries'][-1] > 3000:
            #     break
        
        if iter == n_iter:
            break

        if iter == 0:
            # Bootstrap using BC for first 500 samples
            learner_agent, log, dataset = bc(seed, learner_agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose='bootstrap')
        else:
            # Reset environment
            done = False
            observ, step_reward, done, info = env.reset(poses=start_pose)

            """
            observation:
                {'ego_idx': 0, 'scans': [array([])], 'poses_x': [], 'poses_y': [], 'poses_theta': [], 
                'linear_vels_x': [], 'linear_vels_y': [], 'ang_vels_z': [], 'collisions': array([]), 
                'lap_times': array([]), 'lap_counts': array([])}
            reward = 0.01
            info: {'checkpoint_done': array([False])}
            """

            # Start rendering
            if render:
                if env.renderer is None:
                    env.render()

            # Timestep of rollout
            traj = {"observs": [], "poses_x": [], "poses_y": [], 
                    "poses_theta": [], "scans": [], "actions": [], "reward": 0}
            
            for _ in range(max_traj_len):    
                # Extract useful observations
                raw_lidar_scan = observ["scans"][0]
                downsampled_scan = agent_utils.downsample_and_extract_lidar(observ, observation_shape, downsampling_method)

                # Add Sim2Real noise
                sim2real_noise = np.random.uniform(-0.25, 0.25, downsampled_scan.shape)
                downsampled_scan = downsampled_scan + sim2real_noise

                linear_vels_x = observ["linear_vels_x"][0]

                poses_x = observ["poses_x"][0]
                poses_y = observ["poses_y"][0]
                poses_theta = observ["poses_theta"][0]

                # Expert's action regarding current pose (synthetic query)
                curr_expert_speed, curr_expert_steer = expert.plan(poses_x, 
                                                                   poses_y, 
                                                                   poses_theta, 
                                                                   tlad, 
                                                                   vgain)
                
                expert_action = np.array([[curr_expert_steer, curr_expert_speed]])

                agent_action_raw = learner_agent.get_action(downsampled_scan)
                curr_agent_steer = agent_action_raw[0]
                curr_agent_speed = agent_action_raw[1]
                agent_action = np.expand_dims(agent_action_raw, axis=0)

                # Decide if learner(agent) or expert has control
                if (np.abs(curr_agent_steer - curr_expert_steer) > 0.1) or (np.abs(curr_agent_speed - curr_expert_speed) > 1):
                    """
                    poses_x = observ["poses_x"][0]
                    poses_y = observ["poses_y"][0]
                    poses_theta = observ["poses_theta"][0]

                    curr_expert_speed, curr_expert_steer = expert.plan(poses_x, poses_y, poses_theta, tlad, vgain)
                    curr_action = np.array([[curr_expert_steer, curr_expert_speed]])
                    """
                    curr_action = expert_action

                    traj["observs"].append(observ)
                    traj["scans"].append(downsampled_scan)
                    traj["poses_x"].append(observ["poses_x"][0])
                    traj["poses_y"].append(observ["poses_y"][0])
                    traj["poses_theta"].append(observ["poses_theta"][0])
                    traj["actions"].append(curr_action)
                    traj["reward"] += step_reward
                else:
                    """
                    curr_action_raw = agent.get_action(downsampled_scan)
                    curr_action = np.expand_dims(curr_action_raw, axis=0)
                    """
                    curr_action = agent_action
                
                observ, reward, done, _ = env.step(curr_action)

                    # Update rendering
                if render:
                    env.render(mode=render_mode)
                
                if done:
                    break
            
            print("Adding to dataset...")
            if len(traj["observs"]) > 0:
                traj["observs"] = np.vstack(traj["observs"])
                traj["poses_x"] = np.vstack(traj["poses_x"])
                traj["poses_y"] = np.vstack(traj["poses_y"])
                traj["poses_theta"] = np.vstack(traj["poses_theta"])
                traj["scans"] = np.vstack(traj["scans"])
                traj["actions"] = np.vstack(traj["actions"])
                dataset.add(traj)

            log['Number of Samples'].append(dataset.get_num_of_total_samples())
            log['Number of Expert Queries'].append(dataset.get_num_of_total_samples())

            print("Training agent...")
            for _ in range(n_batch_updates_per_iter):
                train_batch = dataset.sample(train_batch_size)
                learner_agent.train(train_batch["scans"], train_batch["actions"])
    
    agent_utils.save_log_and_model(log, best_model, algo_name)