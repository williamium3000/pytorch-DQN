import sys
import os
sys.path.append("DQN")
import Q_network
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
import copy
class DQN_agent():
    def __init__(self, num_act, dim_obs, gamma, lr, e_greedy, e_greed_decrement):
        self.model = Q_network.Q_network(dim_obs, num_act)
        self.target_model = Q_network.Q_network(dim_obs, num_act)
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.Loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_obs = dim_obs
        self.num_act = num_act
        self.gamma = gamma
        self.lr = lr
        self.global_step = 0
        self.update_target_steps = 1000 # 每隔1000个training steps再把model的参数复制到target_model中
        self.optim = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=500, gamma=0.05)
        self.e_greedy = e_greedy  # 有一定概率随机选取动作，探索
        self.e_greedy_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低
    def sample(self, obs):
        sample = np.random.rand()
        if sample < self.e_greedy:
            action = np.random.choice(self.num_act)
        else:
            action = self.predict(obs)
        self.e_greedy = max(0.02, self.e_greedy - self.e_greedy_decrement)
        return action
    def predict(self, obs):
        self.model.eval()
        with torch.no_grad():
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype = torch.float32)
            # print("obs.shape:{}".format(obs.shape))
            self.model.to(self.device)
            obs = obs.to(self.device)
            pred_Q = self.model(obs)
            pred_Q = pred_Q.cpu()
            # print("pred_Q.shape:{}".format(pred_Q.shape))
            pred_Q = np.squeeze(pred_Q, axis=0)
            # print("pred_Q.shape:{}".format(pred_Q.shape))
            act = np.argmax(pred_Q).item()  # 选择Q最大的下标，即对应的动作
        return act
    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.model.to(self.device)
        self.model.train()
        self.global_step += 1
        act = np.expand_dims(act, -1)
        reward = np.expand_dims(reward, -1)
        terminal = np.expand_dims(terminal, -1)
        obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.int64), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        obs, act, reward, next_obs, terminal = obs.to(self.device), act.to(self.device), reward.to(self.device), next_obs.to(self.device), terminal.to(self.device)
        self.target_model.to(self.device)
        next_pred_value = self.target_model(next_obs)
        best_value = torch.max(next_pred_value, -1, keepdim = True)[0]
        target = reward + (1.0 - terminal) * self.gamma * best_value
        y = self.model(obs)
        y = torch.gather(y, 1, act)
        # print(y[0], target[0])
        loss = self.Loss(y, target)
        self.optim.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optim.step()
        # self.scheduler.step()
        



    def save(self, name):
        torch.save(self.model, name + ".pth")
    def load(self, path):
        self.model = torch.load(path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        self.sync_target()

    def sync_target(self):
        print("sync model to target model")
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.eval()

    