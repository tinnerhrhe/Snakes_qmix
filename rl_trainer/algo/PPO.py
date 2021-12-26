import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
import torch.nn.functional as F

from torch.distributions import Normal  # Multivariate
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Actor(nn.Module):
    def __init__(self):  # (n+2p-f)/s + 1
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)  # 20104 -> 20108
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)  # 20108 -> 18816
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # 18816 -> 16632
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # 16632 -> 14464

        self.linear_1 = nn.Linear(3584, 256)  # 14464 = 3584
        self.linear_2 = nn.Linear(256, 64)
        self.MU = nn.Linear(64, 3)
        self.STD = nn.Linear(64, 3)

        self.action_scale = 2
        self.action_bias = 2
        self.epsilon = 1e-6

    def forward(self, tensor_cv):  # ,batch_size
        # CV
        x = F.relu(self.conv1(tensor_cv))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        batch_size = x.size()[0]
        x = x.reshape(x.size()[0], 1, 3584)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x)).reshape(x.size()[0], 64)

        mean = self.MU(x)
        std = self.STD(x).clamp(-20, 2)
        std = std.exp()
        dist = Normal(mean, std)

        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)

        entropy = -torch.exp(log_prob) * log_prob

        return action.clamp(0, 3.99), log_prob, entropy


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)  # 20104 -> 20108
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)  # 20108 -> 18816
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # 18816 -> 16632
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)  # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(3584, 128)  # 14464 = 3584  896
        self.linear_CNN_2_1 = nn.Linear(128, 128)
        #
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)
        #
        self.linear_3 = nn.Linear(128, 64)  # 512
        self.linear_4 = nn.Linear(64, 3)

    def forward(self, tensor_cv, h_state=(torch.zeros(1, 4, 128).to(device),
                                          torch.zeros(1, 4, 128).to(device))):
        # CV
        x = F.relu(self.conv1_1(tensor_cv))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x))
        batch_size = x.size()[0]
        x = x.reshape(x.size()[0], 1, 3584)
        #
        x = F.relu(self.linear_CNN_1_1(x))
        x = F.relu(self.linear_CNN_2_1(x))
        # x,h_state = self.lstm(x,h_state)

        x = x.reshape(x.size()[0], 1, 128)  # 512

        z = torch.relu(self.linear_3(x))
        out = torch.tanh(self.linear_4(z)).reshape(x.size()[0], 3)

        return out, (h_state[0].data, h_state[1].data)


class PPO:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory = Memory()

        self.c_loss = 0
        self.a_loss = 0

        self.eps_clip = 0.2
        self.K_epochs = 4

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):
        self.memory.states.append(obs)
        obs = torch.Tensor([obs]).to(self.device)
        action, action_logprob, _ = self.actor(obs)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob[0].cpu().detach().numpy())
        return action.cpu().detach().numpy()

    def update(self, new_lr):
        if new_lr != self.a_lr:
            print("new_lr", new_lr)
            self.a_lr = new_lr
            self.c_lr = new_lr
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.c_lr)
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):  # 反转迭代
            # print("1 - is_terminal",(1-is_terminal))
            # if is_terminal:
            #    discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - is_terminal))
            rewards.insert(0, discounted_reward)  # 插入列表

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor. stack: Concatenates sequence of tensors along a new dimension. #stack指定去掉1维
        batch_size = len(self.memory.actions)
        old_states = torch.tensor(self.memory.states).reshape(batch_size, 4, 20, 10).to(
            device).detach()  # torch.squeeze(, 1)
        # old_actions = torch.tensor(self.memory.actions).to(device).detach()
        old_logprobs = torch.tensor(self.memory.logprobs).reshape(batch_size, 1, 3).to(device).detach()

        # importance sampling -> 多次学习

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values : 整序列训练...
            _, logprobs, dist_entropy = self.actor(old_states)
            state_values, _ = self.critic(old_states)

            # Finding the ratio e^(pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:     # Critic    (r+γV(s')-V(s))
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            # Jθ'(θ) = E min { (P(a|s,θ)/P(a|s,θ') Aθ'(s,a)) ,
            #                         clip(P(a|s,θ)/P(a|s,θ'),1-ε,1+ε)Aθ'(s,a) }              θ' demonstration
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -(torch.min(surr1, surr2)).mean()  # + 5*self.a_lr*dist_entropy
            critic_loss = self.SmoothL1Loss(state_values, rewards)  # 0.5*

            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.a_loss += actor_loss.item()
            self.c_loss += critic_loss.item()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # self.memory.clear_memory()

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path", base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("new_lr: ", self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + ".pth")  # + str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + ".pth")  # + str(episode)
        torch.save(self.critic.state_dict(), model_critic_path)