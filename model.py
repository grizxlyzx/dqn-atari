import os
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 shape,
                 bias=False,
                 act_func=nn.ReLU,
                 out_act=nn.ReLU):
        super(MLP, self).__init__()
        assert len(shape) > 1
        layers = []
        for i in range(len(shape) - 2):
            layers.append(nn.Linear(shape[i], shape[i+1], bias))
            layers.append(act_func())
        layers.append(nn.Linear(shape[-2], shape[-1], bias))
        if out_act is not None:
            layers.append(out_act())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def save_weights(self, fn):
        torch.save(self.state_dict(), fn)

    def load_weights(self, fn):
        if os.path.exists(fn):
            self.load_state_dict(torch.load(fn))
        else:
            print(f'MLP: Warning, load weights failed, path not exist: "{fn}"')


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque([], maxlen=capacity)  # 队列,先进先出

    def add(self, ob, ob_nx, action, reward, reward_nx, done):
        self.buffer.appendleft([ob, ob_nx, action, reward, reward_nx, done])

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        ob, ob_nx, action, reward, reward_nx, done = zip(*transitions)
        return np.array(ob), np.array(ob_nx), action, reward, reward_nx, done

    def calc_return(self, gamma):
        if len(self.buffer) <= 1:
            return
        for i in range(1, len(self.buffer)):
            if not self.buffer[i][4]:  # if NOT term
                self.buffer[i][2] = gamma * self.buffer[i - 1][2]
            else:
                return

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class DQN:
    def __init__(self,
                 net,
                 device,
                 gamma=0.9):
        self.device = device
        self.gamma = gamma
        self.net = net.to(device)

    def __call__(self, x):
        return self.net(x)

    def choose_action(self, ob, eps=0.05):
        ob = torch.tensor(ob).to(self.device)
        q_vals = self(ob).detach().cpu().numpy()
        if np.random.random() > eps:
            action = np.argmax(q_vals)
        else:
            action = np.random.randint(low=0, high=len(q_vals))
        return action

    def calc_1_step_td_loss(self, ob, a, r, ob_nx, done, tgt_net):
        ob = torch.tensor(ob).to(self.device)
        a = torch.tensor(a, dtype=torch.int64).view(-1, 1).to(self.device)
        r = torch.tensor(r, dtype=torch.float).view(-1, 1).to(self.device)
        ob_nx = torch.tensor(ob_nx, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        q_a = self(ob).gather(1, a)
        max_q_a_nx = tgt_net(ob_nx).max(1)[0].view(-1, 1).detach()
        q_tgt = r + self.gamma * max_q_a_nx * (1 - done)
        loss = F.mse_loss(q_a, q_tgt)
        return loss



class DQNConv(nn.Module):
    def __init__(self, in_channels=4, n_actions=14, num_hidden=64 * 8 * 8,
                 device=0,
                 gamma=0.95):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(num_hidden, 512)
        self.head = nn.Linear(512, n_actions)
        self.num_hidden = num_hidden
        self.device = device
        self.gamma = gamma
        self.to(device)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(torch.flatten(x, 1)))
        return self.head(x)

    def choose_action(self, ob, eps=0.05):
        ob = torch.tensor(ob).to(self.device)
        ob = torch.permute(ob, dims=(2, 0, 1)).unsqueeze(0)
        q_vals = self.forward(ob).detach().cpu().numpy()
        if np.random.random() > eps:
            action = np.argmax(q_vals)
        else:
            action = np.random.randint(low=0, high=len(q_vals[0]))
        return action

    def calc_1_step_td_loss(self, ob, a, r, ob_nx, done, tgt_net):
        ob = torch.tensor(ob, dtype=torch.float).to(self.device)
        ob = torch.permute(ob, dims=(0, 3, 1, 2))
        a = torch.tensor(a, dtype=torch.int64).view(-1, 1).to(self.device)
        r = torch.tensor(r, dtype=torch.float).view(-1, 1).to(self.device)
        ob_nx = torch.tensor(ob_nx, dtype=torch.float).to(self.device)
        ob_nx = torch.permute(ob_nx, dims=(0, 3, 1, 2))
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(self.device)
        q_a = self.forward(ob).gather(1, a)
        max_q_a_nx = tgt_net(ob_nx).max(1)[0].view(-1, 1).detach()
        q_tgt = r + self.gamma * max_q_a_nx * (1 - done)
        loss = F.mse_loss(q_a, q_tgt)
        return loss

    def save_weights(self, fn):
        torch.save(self.state_dict(), fn)

    def load_weights(self, fn):
        if os.path.exists(fn):
            self.load_state_dict(torch.load(fn))
        else:
            print(f'MLP: Warning, load weights failed, path not exist: "{fn}"')
