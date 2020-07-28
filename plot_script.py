import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import math

def smooth(data, k):
    if isinstance(data, pd.DataFrame):
        num_episodes = data.shape[1]
        num_runs = data.shape[0]
    
        smoothed_data = np.zeros((num_runs, num_episodes))

        for i in range(num_episodes):
            if i < k:
                smoothed_data[:, i] = np.mean(data[:, :i+1], axis = 1)   
            else:
                smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis = 1)    

        return smoothed_data
    else:
        num_episodes = len(data)
        num_runs = 1

        smoothed_data = np.zeros((num_runs, num_episodes))

        for i in range(num_episodes):
            if i < k:
                smoothed_data[:, i] = np.mean(data[:i+1])
            else:
                smoothed_data[:, i] = np.mean(data[i-k:i+1])
        
        return smoothed_data


# Function to plot result
def plot_result(data_name_array, direct=False, k=5):
    plt_agent_sweeps = []
    
    fig, ax = plt.subplots(figsize=(8,6))
    max_list = []

    for data_name in data_name_array:
        # load data
        if not direct:
            filename = 'sum_reward_{}'.format(data_name).replace('.','')
            sum_reward_data = np.load('{}/{}.npy'.format("results/", filename))

        # smooth data
        else:
            sum_reward_data = data_name_array[data_name]

        smoothed_sum_reward = smooth(data=sum_reward_data, k=k)
        max_list.append(max(smoothed_sum_reward[0]))
        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis = 0)

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:], label=data_name)
        plt_agent_sweeps.append(graph_current_agent_sum_reward)

    max_to_hundred = int(math.ceil(max(max_list) / 100.0)) * 100
    
    ax.legend(handles=plt_agent_sweeps, fontsize = 13)
    ax.set_title("Learning Curve", fontsize = 15)
    ax.set_xlabel('Episodes', fontsize = 14)
    ax.set_ylabel("Sum of\nreward\nduring\nepisode", rotation=0, labelpad=40, fontsize = 14)
    ax.set_ylim([-200, max_to_hundred])
    plt.show()     
