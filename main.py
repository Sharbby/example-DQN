import torch
from torch import nn
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque,namedtuple
import random
from tqdm import tqdm
import ale_py
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py) 

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    torch.set_default_device('cpu')
    device = 'cpu'
#看看用什么，cuda还是cpu


Transition = namedtuple('Transition',('state','action','reward','nextstate','done'))
#一会要用的妙妙小元组

class DQN(nn.Module):
    def __init__(self,action_dim):
        super().__init__()
        self.Qnet = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4),#84x84/4 -> 20x20/32
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,stride=2),#20x20/32 -> 9x9/64
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1),#9x9/64 -> 7x7/64
            nn.Flatten(),#CNN提取特征，展平以后进行两层全连接，论文是这么写的
            nn.Linear(3136,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,action_dim)
        )
    def forward(self,x):
        return self.Qnet(x/255)#进入之前缩放到0-1方差弄小方便计算

class ReplayBuffer():
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self,*k):
        self.buffer.append(Transition(*k))
    def sample(self,num):
        tr = random.sample(self.buffer,num)
        state,action,reward,next_state,done =  zip(*tr)
        return (
            np.array(state),
            np.array(action,dtype=np.int32),
            np.array(reward,dtype=np.float32),
            np.array(next_state),
            np.array(done)
        )
    def __len__(self):
        return len(self.buffer)#经验回放池

env = gym.make(
    'ALE/Breakout-v5',
    full_action_space = False,
    frameskip = 1,
    render_mode=None
    )

env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84)
env = FrameStackObservation(env, 4)#这里如果手动做，务必注意球在两帧内有一帧是不渲染的


n_act = 4
update_frame = 4000
buffer = deque(maxlen = 4)
actionmap = env.unwrapped.get_action_meanings()[:4]
buffer_size = 1000000
epoch_num = 20000
gamma = 0.99
batch_size = 32
epsilon = 1
lr = 1e-4
trace_per_epoch = 10
eps_decay_frame = 1000
warm_up = 50000
end_frame = 20000000

net = DQN(n_act)
target_net = DQN(n_act)
target_net.load_state_dict(net.state_dict())
replaybuffer = ReplayBuffer(buffer_size)
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
loss_fn = nn.SmoothL1Loss()
return_list = []
buffer = deque(maxlen = 4)
framecount = 0
train_frame = 4
pbar = tqdm(total=epoch_num)

for i in range(epoch_num):
    total_reward = 0
    reward = 0
    action = 0
    terminated = False
    truncated = False
    state,_ = env.reset()
    
    while not (terminated or truncated):
        
        if (framecount % update_frame == 0):
            target_net.load_state_dict(net.state_dict())
            epsilon = max(0.01, epsilon - 0.001)#看情况更改，你要想快点看效果按这个来，基本一下午return在三四十，你要想正儿巴经复现就慢一点缩小
        if (len(replaybuffer) > warm_up) and (framecount % train_frame == 0):
            S_states,S_actions,S_rewards,S_next_states,S_dones = replaybuffer.sample(batch_size)
            S_states = torch.from_numpy(S_states).to(device)
            S_actions = torch.from_numpy(S_actions).to(device).unsqueeze(1)
            S_rewards = torch.from_numpy(S_rewards).to(device).unsqueeze(1)
            S_next_states = torch.from_numpy(S_next_states).to(device)
            S_dones = torch.from_numpy(S_dones).to(device).unsqueeze(1)
            q = net(S_states.float()).gather(1,S_actions)
            with torch.no_grad():
                best_actions = net(S_next_states.float()).argmax(1,keepdim = True)
                next_q = target_net(S_next_states.float()).gather(1,best_actions)
                target = S_rewards + (gamma * next_q * (~S_dones))
            loss = loss_fn(q,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#训练
            
        if random.random() < epsilon:
            action = random.randrange(n_act)
        else:
            action = net.forward(torch.from_numpy(state).unsqueeze(0).float().to(device)).argmax().item()
        #选动作↑
        next_state,reward,terminated,truncated,_ = env.step(action)
        reward = np.sign(reward)
        total_reward += reward
        framecount += 1
        replaybuffer.push(state,action,reward,next_state,truncated or terminated)
        state = next_state
    return_list.append(total_reward)#收集数据
    pbar.set_postfix(ret=f"{total_reward}",frm=f"{framecount}",eps=f"{epsilon:.2f}")
    pbar.update(1)
    if (framecount >= end_frame):
        break
    
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Epsilons')
plt.ylabel('Returns')
plt.title('DQN on breakout')
plt.savefig(r'./output')
