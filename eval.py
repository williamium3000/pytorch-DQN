import sys

import agent
import Q_network
import experience_replay
import gym
import numpy as np
import torch
def evaluate(times, env, agent, render=False):
    with torch.no_grad():
        eval_reward = []
        for i in range(times):
            obs = env.reset()
            episode_reward = 0
            while True:
                action = agent.predict(obs)  # 预测动作，只选最优动作
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if render:
                    env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
    return np.mean(eval_reward)
opt = {
    "LEARN_FREQ" : 3, # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
    "MEMORY_SIZE" : 20000,    # replay memory的大小，越大越占用内存
    "MEMORY_WARMUP_SIZE" : 300,  # replay_memory 里需要预存一些经验数据，再开启训练
    "BATCH_SIZE" : 64,   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
    "LEARNING_RATE" : 0.001, # 学习率
    "GAMMA" : 0.99, # reward 的衰减因子，一般取 0.9 到 0.999 不等
    "E_GREEDY" : 0.1,
    "E_GREEDY_DECREMENT" : 1e-6,
    "max_episode" : 1000
}

if __name__ == "__main__":
    env_name = "CartPole-v0"
    # env_name = "MountainCar-v0"
    env = gym.make(env_name)
    num_act = env.action_space.n
    num_obs = env.observation_space.shape[0]
    dqn_agent = agent.DQN_agent(num_act, num_obs, opt["GAMMA"], opt["LEARNING_RATE"], opt["E_GREEDY"], opt["E_GREEDY_DECREMENT"])
    dqn_agent.load("{}.pth".format(env_name))
    print("evaluate on {} episode: reward {}".format(30, evaluate(5, env, dqn_agent, True)))