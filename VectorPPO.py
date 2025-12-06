import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing,FrameStackObservation
from gymnasium.vector import SyncVectorEnv
import torch
from torch import nn
from collections import deque,namedtuple
from tqdm import tqdm
import numpy as np
from torch.distributions import Categorical

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    torch.set_default_device('cpu')
    device = 'cpu'


gym.register_envs(ale_py)

n_act = 4
batch_size = 32
epoch_num = 10000
trace_num = 1024
gamma = 0.99
lr = 3e-4
epsilon = 1
trace_buffer = deque()
lamb = 0.99

Transition = namedtuple('Transition',('state','action','old_prob','reward','done','next_state'))
samples = namedtuple('samples',('state','action','old_prob','value','reward','done','next_state'))
loss_fn = nn.SmoothL1Loss()

def make_env():
    def _init():
        env = gym.make(
            'ALE/Breakout-v5',
            render_mode=None,
            full_action_space=False,
            frameskip=1
        )
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            grayscale_obs=True,
            screen_size=84,
            terminal_on_life_loss=True
        )
        env = FrameStackObservation(env, 4)
        return env
    return _init

vec_env = SyncVectorEnv([make_env() for _ in range(batch_size)])




class PPO(nn.Module):
    def __init__(self,action_dim):
        super().__init__()
        self.observer = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4),#84x84/4 -> 20x20/32
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,stride=2),#20x20/32 -> 9x9/64
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1),#9x9/64 -> 7x7/64
            nn.Flatten()
        )
        self.actornet = nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU(inplace = True),
            nn.Linear(512,action_dim)#1?4?
        )
        self.criticnet = nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU(inplace = True),
            nn.Linear(512,1)
        )
    def act(self,x):
        obs = self.observer(x)
        return self.actornet(obs)
    def estimate(self,x):
        obs = self.observer(x)
        return self.criticnet(obs)

net = PPO(n_act)
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

for it in range(epoch_num):
    total_rew = 0
    terminated = torch.zeros(batch_size,dtype = torch.bool)
    truncated = torch.zeros(batch_size,dtype = torch.bool)
    state,_ = vec_env.reset()
    trace = []
    trace_buffer = deque()
    value = np.zeros(batch_size)
    for _ in tqdm(range(trace_num)): 
        logits = net.act(torch.from_numpy(state).float().to(device) / 255)
        dist = Categorical(logits=logits)
        action = dist.sample()
        prob_old = dist.log_prob(action).detach()
        next_state,reward,terminated,truncated,_ = vec_env.step(action)
        total_rew += reward
        trace.append(Transition(state,action,prob_old,reward,terminated | truncated,next_state))
        state = next_state
    trace = reversed(trace)
    for k in trace:
        value = k.reward + value *gamma
        state,action,prob_old,reward,done,next_state = k
        trace_buffer.append(samples(state,action,prob_old,value,reward,done,next_state))

    gae = 0
    for i in tqdm(range(len(trace_buffer)),desc = "epoch - %d|rew = %d" %(it,total_rew.sum())):
        batch_sample = trace_buffer[i]
        S_state = torch.from_numpy(batch_sample.state).to(device).float() / 255
        S_action = batch_sample.action.to(device)
        S_value = torch.from_numpy(batch_sample.value).to(device).float()
        S_reward = torch.from_numpy(batch_sample.reward).to(device).float()
        S_done = torch.from_numpy(batch_sample.done).to(device)
        S_next_state = torch.from_numpy(batch_sample.next_state).to(device).float() / 255
        S_prob_old = batch_sample.old_prob
        if S_done.any():
            gae = 0
        with torch.no_grad():
            td_target = gamma * net.estimate(S_next_state).squeeze(-1) * (~S_done) + S_reward
            TD_error = td_target - net.estimate(S_state).squeeze(-1)
        gae += TD_error * lamb * gamma
        
        logits = net.act(S_state)
        prob = Categorical(logits=logits).log_prob(S_action)
        ratio = torch.exp(prob - S_prob_old)
        sur1 = ratio * gae
        sur2 = torch.clamp(ratio,0.8,1.2) * gae
        
        actor_loss = -torch.min(sur1,sur2).mean()
        critic_loss = nn.functional.mse_loss(net.estimate(S_state).squeeze(-1),S_value).mean()
        
        totalloss = actor_loss + critic_loss
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
