import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class Agent:
    def __init__(
        self,
        env: gym.Env,
        gamma=0.99,
        alpha=0.0003,
        initial_epsilon=1,
        min_epsilon=0.1,
        decay_rate=0.9999,
        batch_size=64,
        n_rollouts=2000,
        capacity=100000,
        device: torch.device = torch.device("cpu"),
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.alpha = alpha 
        self.epsilon = initial_epsilon 
        self.batch_size = batch_size 
        self.n_rollouts = n_rollouts 

        self.epsilon = 1 
        self.min_epsilon = min_epsilon 
        self.decay_rate = decay_rate  

        self.replay_memory = ReplayMemory(capacity, env, device)
        self.q_network = QNetwork(
            env.observation_space.shape[0], env.action_space.n
        ).to(device)
        self.target_network = deepcopy(self.q_network)
        self.optimizer = AdamW(self.q_network.parameters(), lr=alpha)

        self.n_states = env.observation_space.shape[0]

        self.n_time_steps = 0  
        self.episodes = 0  
        self.n_updates = 0  
        self.best_reward = -np.inf 

    def get_action(self, obs, greedy=False):

        if not greedy and np.random.rand() < self.epsilon:  
            return np.random.randint(self.env.action_space.n)  
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) 
        self.q_network.eval()  
        with torch.no_grad():
            q_values: torch.Tensor = self.q_network(obs)  
            return q_values.argmax().item() 

    def sample_experience(self):

        return self.replay_memory.sample(self.batch_size) 

    def update_target(self):

        self.target_network.load_state_dic

    def collect_rollouts(self):
        terminated = False
        truncated = False
        rewards = 0  
        episodes = 0 
        for _ in range(self.n_rollouts):
            action = self.get_action(obs, greedy=False)  
            next_obs, reward, terminated, truncated, _ = self.env.step(action) 
            self.replay_memory.push(
                obs, action, next_obs, reward, terminated, truncated
            )  
            obs = next_obs  
            rewards += reward  
            self.n_time_steps += 1  
            if terminated or truncated:  
                episodes += 1
                self.episodes += 1

            self.epsilon = max(
                self.min_epsilon, self.decay_rate * self.epsilon
            )  

        return rewards / episodes  

    def learn(self, epochs):

        self.q_network.train()

        average_loss = 0
        for i in range(epochs):
            obs, action, next_obs, reward, terminated, truncated = (
                self.sample_experience()
            )  
            q_values: torch.Tensor = self.q_network(obs) 
            next_q_values = self.target_network(next_obs)  

            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)  
            next_q_value = next_q_values.max(1).values  
            target = reward + self.gamma * next_q_value * (1 - terminated) * (
                1 - truncated
            )

            loss = F.smooth_l1_loss(q_value, target) 

            self.optimizer.zero_grad()  
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10) 
            self.optimizer.step() 

            average_loss += (loss.item() - average_loss) / (i + 1) 
            self.n_updates += 1  

            if self.n_updates % 1000 == 0:  
                self.update_target()

        return average_loss  


class QNetwork(nn.Module):
    def __init__(self, nvec_s, nvec_u):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(nvec_s, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, nvec_u)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayMemory:
    def __init__(self, capacity, env: gym.Env, device: torch.device):

        self.position = 0
        self.size = 0
        self.capacity = capacity
        self.device = device

        self.n_states = env.observation_space.shape  
        self.n_actions = env.action_space.n 

        self.states = np.zeros((capacity, *self.n_states))
        self.actions = np.zeros((capacity))
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, *self.n_states))
        self.terminated = np.zeros(capacity)
        self.truncated = np.zeros(capacity)

    def push(self, state:np.ndarray, action:int, next_state:np.ndarray, reward:float, terminated: bool, truncated:bool):

        self.states[self.position] = state.flatten()
        self.actions[self.position] = action
        self.next_states[self.position] = next_state.flatten()
        self.rewards[self.position] = reward
        self.terminated[self.position] = terminated
        self.truncated[self.position] = truncated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        indices = np.random.choice(self.size, batch_size, replace=False)

        states = torch.tensor(self.states[indices], dtype = torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype = torch.int64, device=self.device)
        next_states = torch.tensor(self.next_states[indices], dtype = torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype = torch.float32, device=self.device)
        terminated = torch.tensor(self.terminated[indices], dtype = torch.float32, device=self.device)
        truncated = torch.tensor(self.truncated[indices], dtype = torch.float32, device=self.device)

        return states, actions, next_states, rewards, terminated, truncated

    def __len__(self):
        return len(self.size)   