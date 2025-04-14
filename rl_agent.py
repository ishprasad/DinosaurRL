import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.99999
        self.action_dim = action_dim
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        batch_state = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        batch_done = torch.FloatTensor(batch_done).to(self.device)
        
        current_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(batch_next_state).max(1)[0].detach()
        target_q = batch_reward + (1 - batch_done) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_policy(self, episode, save_dir="policies"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f"{save_dir}/policy_episode_{episode}.pt"
        torch.save(self.target_net.state_dict(), filename)
        print(f"Saved policy to {filename}")


        
def get_state(env):
    dino_y = env.dino_y
    dino_velocity = env.dino_velocity
    
    if env.obstacles:
        nearest_obs = env.obstacles[0]
        obs_x, obs_y, obs_type = nearest_obs
        distance = obs_x - 50  
        obs_type_val = 1 if obs_type == "cactus" else 2
    else:
        distance = SCREEN_WIDTH
        obs_y = SCREEN_HEIGHT
        obs_type_val = 0
    
    state = [
        dino_y / SCREEN_HEIGHT,
        dino_velocity / 15,  
        distance / SCREEN_WIDTH,
        obs_y / SCREEN_HEIGHT,
        obs_type_val / 2
    ]
    return np.array(state)

def save_policy(agent, episode, filepath="target_policy.pth"):
    torch.save(agent.target_net.state_dict(), filepath)
    print(f"Saved target policy to {filepath} at episode {episode}")

