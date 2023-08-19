#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import gymnasium as gym
import gym as gym2
import numpy as np
import time
import random
import pandas as pd
import highway_env
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib import pyplot as plt
import mujoco
from sb3_contrib import TRPO, QRDQN
from statistics import mean, variance
import ot as POT
from gudhi.wasserstein import wasserstein_distance
import dataframe_image as dfi
import pandoc
import random


#main body

import gudhi as gd
import random as rand
gd.persistence_graphical_tools._gudhi_matplotlib_use_tex=False


def new_action(action):
    
    U = random.uniform(0,1)
    if U < 0.5:
        new_action = 4
    else :
        new_action = action
    return new_action

# fig,axs = plt.subplots(2,5, figsize =(10,4))
# plot of state and their persistence
def persis(perturb, env):
    l_diag = []
    fig,axs = plt.subplots(2,5, figsize =(10,4))
    for i in range(5):
        if perturb == "Yes":
            env.config["lanes_count"] = rand.randint(2,6)
            env.config["vehicles_density"] = rand.uniform(0.5,1.5)
        else :
            env.config["lanes_count"] = 4

        state,_= env.reset()
        rips_complex = gd.RipsComplex(points = state,max_edge_length = 10)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        diag = simplex_tree.persistence()
        axs[0,i].imshow(env.render())
        axs[0,i].axis("off")
        axs[0,i].title.set_text("State Snapshot")
        gd.plot_persistence_diagram(diag,axes = axs[1,i]).set_ylabel("")
        axs[1,i].title.set_text("")
        axs[1,i].set_xlabel("")
        l_diag.append(diag)
    
    return fig,l_diag

#Computing the Bottleneck distance between sequence of persistence diagrams produced by sequence of states. 


# model_dqn = DQN.load("highway_dqn/model", env =env)
# model_ppo = PPO.load("highway_PPO/model", env= env)
# model_a2c = A2C.load("highway_A2C/model", env =env)
# model_trpo= TRPO.load("highway_TRPO/model",env=env)
# model_qrdqn= QRDQN.load("highway_QRDQN/model", env= env)  
#saving sequence of states. 
def simulate_env(episodes, timesteps, env,mod) : 
#     env = gym.make("highway-fast-v0")
    total_state = []
    total_reward = []
    total_action = []
    for episode in range(episodes):
        ep_state=[]
        ep_action = []
        sum_reward = 0
#     env.config["lanes_count"] = random.randint(1,5)
        state, info = env.reset()
        print("Episode start:", episode)
        for timestep in range(timesteps) :
            env.render()
            if(mod == "PPO"):
                action,obs = model_ppo.predict(state)
                print("PPO")
            elif(mod=="DQN"):
                action,obs = model_dqn.predict(state, deterministic = True)
                print("DQN")
            elif(mod=="A2C") :
                action,obs = model_a2c.predict(state)
                print("a2c")
            elif(mod=="TRPO"):
                action,obs = model_trpo.predict(state)
            elif(mod=="QRDQN"):
                action, obs= model_qrdqn.predict(state)
            else:
                action = env.action_space.sample()
            state, reward, truncated, terminated, info = env.step(action)
            print(action)
            ep_action.append(action)
            ep_state.append(state)
            sum_reward += reward
            print("episode Timestep :", episode,timestep)
        
            if terminated or truncated :
                break
        total_action.append(ep_action)
        total_reward.append(sum_reward)
        total_state.append(ep_state)
    env.close()
    return total_state, total_action, total_reward


def simulate_pertenv(episodes, timesteps, env,mod) : 
    total_state = []
    total_reward = []
    total_action = []
    for episode in range(episodes):
        ep_state=[]
        ep_action = []
        sum_reward = 0
        env.config["lanes_count"] = rand.randint(2,4)
        env.config["vehicles_density"] = rand.uniform(0.5,1.5)
        state, info = env.reset()
        print("Episode start:", episode)
        for timestep in range(timesteps) :
            env.render()
            if(mod == "PPO"):
                action,obs = model_ppo.predict(state)
                print("PPO")
            elif(mod=="DQN"):
                action,obs = model_dqn.predict(state, deterministic = True)
                print("DQN")
            elif(mod=="A2C") :
                action,obs = model_a2c.predict(state)
                print("a2c")
            elif(mod=="TRPO"):
                action,obs = model_trpo.predict(state)
            elif(mod=="QRDQN"):
                action, obs= model_qrdqn.predict(state)
            else:
                action = env.action_space.sample()
            state, reward, truncated, terminated, info = env.step(action)
            print(action)
            ep_action.append(action)
            ep_state.append(state)
            sum_reward += reward
            print("episode Timestep :", episode,timestep)
        
            if terminated or truncated :
                break
        total_action.append(ep_action)
        total_reward.append(sum_reward)
        total_state.append(ep_state)
    env.close()
    return total_state, total_action, total_reward

def simulate_pertenvTDA(episodes, timesteps, env,mod) : 
    total_state = []
    total_reward = []
    total_action = []
    for episode in range(episodes):
        ep_state=[]
        ep_action = []
        sum_reward = 0
        env.config["lanes_count"] = rand.randint(2,4)
        env.config["vehicles_density"] = rand.uniform(0.5,1.5)
        state, info = env.reset()
        print("Episode start:", episode)
        for timestep in range(timesteps) :
            env.render()
            if(mod == "PPO"):
                action,obs = model_ppo.predict(state)
                print("PPO")
            elif(mod=="DQN"):
                action,obs = model_dqn.predict(state, deterministic = True)
                print("DQN")
            elif(mod=="A2C") :
                action,obs = model_a2c.predict(state)
                print("a2c")
            elif(mod=="TRPO"):
                action,obs = model_trpo.predict(state)
            elif(mod=="QRDQN"):
                action, obs= model_qrdqn.predict(state)
            else:
                action = env.action_space.sample()
            
#             if action == 2 or action == 3:
#                 action_count = action_count+1
#                 print("Old Action",action)
#                 if action_count >3 :
#                     action = new_action(action)
#                     print("New acton", action)

            
            if action == 2 or action ==3 :
                action = new_action(action)
            
            state, reward, truncated, terminated, info = env.step(action)
            print(action)
            ep_action.append(action)
            ep_state.append(state)
            sum_reward += reward
            print("episode Timestep :", episode,timestep)
        
            if terminated or truncated :
                break
        total_action.append(ep_action)
        total_reward.append(sum_reward)
        total_state.append(ep_state)
    env.close()
    return total_state, total_action, total_reward

#Extracting states with high or low rewards
# THis is to answer the question "Is there any impact of starting orientation on the trajectory/reward and can,
# Tda capture that. 

def hilo(episodes, timesteps, env, reward_threshold, hilo) : 
    low_instate = []
    high_instate =[]
    high_reward = []
    low_reward = []
    for episode in range(episodes):
        ep_state=[]
        sum_reward = 0
#     env.config["lanes_count"] = random.randint(1,5)
        state, info = env.reset()
        instate= state
        print("Episode start:", episode)
        for timestep in range(timesteps) :
            env.render()
            action = env.action_space.sample()
            state, reward, truncated, terminated, info = env.step(action)
            ep_state.append(state)
            sum_reward += reward
            print("episode Timestep :", episode,timestep)
        
            if terminated or truncated :
                break
        
        if sum_reward>=reward_threshold:
            high_reward.append(sum_reward)
            high_instate.append(instate)
        else :
            low_reward.append(sum_reward)
            low_instate.append(instate)
    env.close()
    if hilo=="High":
        return high_instate, high_reward
    else :
        return low_instate, low_reward

#computing PD for each state in the states array. 
def compute_pd(states):
    pd= []
    for i in range(len(states)):
        rips_complex = gd.RipsComplex(points = states[i], max_edge_length = 10)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 1)
        diag = simplex_tree.persistence()
        pd.append(diag)
    
    return pd

def bd_space(diag):
    birth =[]
    death = []
    bd_space = []
    for i in range(len(diag)) : 
        b,d = diag[i][1]
        bd_space.append([b,d])
#         birth.append(b)
#         death.append(d)
    return bd_space



#compute bottle neck distance and plot. 

#bottle neck lagged 
def bottle_lag(pd):
    bnd_i = []
    for i in range(len(pd)): 
        bnd_j = []
        j=i+1
        while j <len(pd) :
            bn_distance = gd.bottleneck_distance(bd_space(pd[i]),bd_space(pd[j]))
            bnd_j.append(bn_distance)
            j = j+1
        bnd_i.append(bnd_j)
    return bnd_i

#bottleneck consecutive.
def bottle_consec(pd):
    bnd_i = []
    for i in range(len(pd)-1):
        bn_distance = gd.bottleneck_distance(bd_space(pd[i]),bd_space(pd[i+1]))
#         print(i)
        bnd_i.append(bn_distance)
    return bnd_i

def wd_consec(pd):
    wd_i = []
    for i in range(len(pd)-1):
        dgm1 = np.array(bd_space(pd[i]))
        dgm2 = np.array(bd_space(pd[i+1]))
        wd_distance = wasserstein_distance(dgm1,dgm2,order=1., internal_p=2.)
        wd_i.append(wd_distance)
    return wd_i



#mean bd_distance and list of bd

def mean_dist(state,dist_type):
    td =[]
    td_mean = []
    if dist_type == "bnd":
        for i in range(len(state)) : 
            persistence = compute_pd(state[i])
            dis = bottle_consec(persistence)
            if bool(dis):
                dis = dis
            else:
                dis = [0]
            td.append(dis)
            td_mean.append(mean(dis))
        return td_mean,td
    elif dist_type =="wd":
        for i in range(len(state)) : 
            persistence = compute_pd(state[i])
            dis = wd_consec(persistence)
            if bool(dis):
                dis = dis
            else:
                dis = [0]
            td.append(dis)
            td_mean.append(mean(dis))
        return td_mean,td
    

#action analysis
def action_analysis(state, action,name):
    act_bd = pd.DataFrame()
    n = len(state)
    for i in range(n):
        persistence = compute_pd(state[i])
        bd = bottle_consec(persistence)
#     print(len(bd))
        act = action[i].copy()
#     print(len(act))
        act.pop(-1)
        actions = [x.tolist() for x in act]
        dct = {"actions" : actions,"BND":bd,"algorithm" :name}
        df = pd.DataFrame(data= dct)
        act_bd= pd.concat([act_bd,df], ignore_index=True)
    return act_bd

