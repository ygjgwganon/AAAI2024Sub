#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load packages
import gymnasium as gym
import numpy as np
import random
import pandas as pd
import highway_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib import pyplot as plt
from sb3_contrib import TRPO
from statistics import mean, variance
import ot as POT
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import dataframe_image as dfi
gd.persistence_graphical_tools._gudhi_matplotlib_use_tex=False


# --------------------------------------------------------------------------------

#Make the Environments
env = gym.make("highway-fast-v0")

env_highway = gym.make("highway-fast-v0")
env_roundabout = gym.make("roundabout-v0")
env_intersection = gym.make("intersection-v0")
config = { #to configure observation type to kinematics
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}
env_roundabout.configure(config)
env_intersection.configure(config)

env_highway.reset()
env_roundabout.reset()
env_intersection.reset()

# --------------------------------------------------------------------------------
#making containers to store envirionments and file names 
env_make = [env_highway,env_roundabout, env_intersection]
env_names = ["highway_","roundabout_","intersection_"]
model_names = ["dqn/model","PPO/model","TRPO/model"]
fig_name = ["DQN","PPO","TRPO","RAND"]
filenames = []
for envnames in env_names :
    for modelnames in model_names:
        filenames.append(envnames + modelnames)

# --------------------------------------------------------------------------------
#main body to do reward and action analysis and store
random.seed(1)
import pandas as pd
j = 0
ep = 100
ts = 200
fig, axs = plt.subplots(3,4, figsize = (10,4))
tot_tab = []
for i in [0,3,6]:
    
    act_tab = pd.DataFrame()
    print("file names",filenames[i],filenames[i+1],filenames[i+2])
    print("env name",env_make[j])
   
    model_dqn = DQN.load(filenames[i], env = env_make[j])
    model_ppo = PPO.load(filenames[i+1], env = env_make[j])
    model_trpo = TRPO.load(filenames[i+2], env = env_make[j])
    state_dqn, action_dqn, reward_dqn = simulate_env(ep, ts, env_make[j], "DQN")
    state_ppo, action_ppo, reward_ppo = simulate_env(ep, ts, env_make[j], "PPO")
    state_trpo, action_trpo, reward_trpo = simulate_env(ep, ts, env_make[j], "TRPO")
    state_rand, action_rand, reward_rand = simulate_env(ep, ts, env_make[j],"Random")  
    l_state = [state_dqn,state_ppo,state_trpo, state_rand]
    l_reward = [reward_dqn, reward_ppo, reward_trpo, reward_rand]
    l_action = [action_dqn,action_ppo,action_trpo,action_rand]
    for k in range(4):
        bd_mean, bd = mean_dist(l_state[k], "bnd")
        wd_mean, wd = mean_dist(l_state[k],"wd")
        
        table = [mean(l_reward[k]),variance(l_reward[k])**0.5,
                 mean(bd_mean),variance(bd_mean)**0.5,
                 mean(wd_mean), variance(wd_mean)**0.5]
                
        axs[j][k].scatter(bd_mean, l_reward[k], c=np.random.rand(ep), alpha = 0.8, s=20)
#         axs[j][k].axhline(mean(reward_dqn), label = "Mean reward for policy")
#         axs[j][k].legend(loc = "upper right")
#         axs[j][k].set_xlabel("Mean Episode BND")
#         axs[j][k].set_ylabel("Mean Episode Rewards")
#         axs[j][k].set_title(fig_name[k])
        tot_tab.append(table)
        act_bd = action_analysis(l_state[k],l_action[k],fig_name[k])
        act_tab= pd.concat([act_tab,act_bd], ignore_index=True)
    act_bd = act_tab
    test= act_bd.groupby(["algorithm","actions"])            .agg({"actions":"size","BND":"mean"})             .rename(columns = {"actions":"Counts","BND":"Mean BND"})      
    tab = test.style.bar(subset=["Mean BND","Counts"], cmap="viridis")
    tab.set_table_styles(
        [
          {"selector": "td, th", "props": [("border", "1px solid grey !important")]},
        ]
    )
    dfi.export(tab, "action_count_"+env_names[j]+".png")
    j = j+1
print(pd.DataFrame(tot_tab))

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


# In[ ]:




