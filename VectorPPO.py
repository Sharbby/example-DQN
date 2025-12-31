import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing,FrameStackObservation
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Categorical
from tqdm import tqdm
from collections import deque

log = SummaryWriter(log_dir="logs/PPO")

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    torch.set_default_device('cpu')
    device = 'cpu'


gym.register_envs(ale_py)

class Actor(nn.Module):
    def __init__(self,action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4),#84x84/4 -> 20x20/32
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,stride=2),#20x20/32 -> 9x9/64
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1),#9x9/64 -> 7x7/64
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(inplace = True),
            nn.Linear(512,action_dim)
        )
    def forward(self,x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4),#84x84/4 -> 20x20/32
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,4,stride=2),#20x20/32 -> 9x9/64
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1),#9x9/64 -> 7x7/64
            nn.Flatten(),
            nn.Linear(3136,512),
            nn.ReLU(inplace = True),
            nn.Linear(512,1)
        )
    def forward(self,x):
        return self.net(x)
    
action_dim = 4
epoch_num = 10000
batch_size = 16
n_steps = 512
gamma = 0.99
lmbda = 0.98
train_per_epoch = 10
eps = 0.1
ent_fac = 0.01
    
actor = Actor(action_dim)
actor_optimizer = optim.AdamW(params=actor.parameters())
critic = Critic()
critic_optimizer = optim.AdamW(params=critic.parameters())

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

vec_env = SyncVectorEnv([make_env() for _ in range(batch_size)])#向量化

for it in tqdm(range(epoch_num)):
    #sample
    state,_ = vec_env.reset()
    trace_dict = {
        'state':[],
        'action':[],
        'old_probs':[],
        'next_state':[],
        'reward':[],
        'done':[],
        'gae':deque(),
        'target':deque()
        }
    state = torch.from_numpy(state).to(device)
    video = state.unsqueeze(2)
    state = state.float() / 255
    for i in range(n_steps):
        logit = actor.forward(state)
        prob = Categorical(logits=logit)
        action = prob.sample()
        old_prob = prob.log_prob(action).detach()
        next_state,reward,terminated,truncated,_ = vec_env.step(action)
        reward = torch.from_numpy(reward).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        video = torch.concat((video,next_state.unsqueeze(2)[:,-1,:,:,:].unsqueeze(2)),dim=1)
        next_state = next_state.float() / 255
        trace_dict['state'].append(state)
        trace_dict['action'].append(action)
        trace_dict['old_probs'].append(old_prob)
        trace_dict['next_state'].append(next_state)
        trace_dict['reward'].append(reward)
        trace_dict['done'].append(torch.from_numpy((truncated | terminated)))
        state = next_state
        
    trace_dict['state'] = torch.stack(trace_dict['state']).float()
    trace_dict['action'] = torch.stack(trace_dict['action']).float()
    trace_dict['old_probs'] = torch.stack(trace_dict['old_probs']).float()
    trace_dict['next_state'] = torch.stack(trace_dict['next_state']).float()
    trace_dict['reward'] = torch.stack(trace_dict['reward']).float()
    trace_dict['done'] = torch.stack(trace_dict['done']).to(device) 
    
    if (it%100 == 0):
        log.add_video("Breakout/video_%d"%it,video.repeat(1,1,3,1,1).to('cpu'),fps=16)
    log.add_scalar('Breakout/train',trace_dict['reward'].sum(), it)
    #train
    criticloss = 0
    actorloss = 0
    gae = 0
    for i in reversed(range(n_steps)):
        TD_target = critic.forward(trace_dict['next_state'][i])*gamma*(~trace_dict['done'][i]) + trace_dict['reward'][i]
        TD_delta = TD_target - critic.forward(trace_dict['state'][i])
        gae = gae*(~trace_dict['done'][i])*gamma*lmbda + TD_delta
        trace_dict['gae'].appendleft(gae.detach())
        trace_dict['target'].appendleft(TD_target.detach())
    #for j in range(train_per_epoch):
    actorloss = 0
    criticloss = 0
    advantages = torch.stack(list(trace_dict['gae']))
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for i in range(n_steps):
        log_logits = actor.forward(trace_dict['state'][i])
        new_log_probs = Categorical(logits=log_logits).log_prob(trace_dict['action'][i])
        ratio = torch.exp(new_log_probs - trace_dict['old_probs'][i])
        surr1 = ratio * trace_dict['gae'][i]
        surr2 = torch.clamp(ratio,1-eps,1+eps)*trace_dict['gae'][i]
        entropy = Categorical(logits=log_logits).entropy().mean()
        actorloss += -torch.min(surr1,surr2).mean() - ent_fac*entropy
        criticloss += torch.nn.functional.mse_loss(critic.forward(trace_dict['state'][i]),trace_dict['target'][i])
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actorloss.backward()
    criticloss.backward()
    actor_optimizer.step()
    critic_optimizer.step()
            
