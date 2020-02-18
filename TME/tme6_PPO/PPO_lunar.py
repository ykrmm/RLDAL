import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from torch import nn
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class NN_V(torch.nn.Module):
    def __init__(self,inSize,outSize,layers = []):
        super().__init__()
        self.activ=nn.LeakyReLU()
        self.activ=nn.Tanh()
        self.tan=nn.Tanh()
        self.layers = nn.ModuleList([])
        for x in layers : 
            self.layers.append(nn.Linear(inSize,x))
            inSize = x
        self.layers.append(nn.Linear(inSize,outSize))
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            x  = self.tan(x)
            x = self.layers[i](x)
        #x=self.tan(x)
        return x

class NN_Pi(torch.nn.Module):
    def __init__(self,inSize,outSize,layers = []):
        super().__init__()
        self.activ=nn.LeakyReLU()

        self.softM=nn.Softmax(dim=0)
        #self.softM==nn.LogSoftmax()
        #self.softM=nn.Sigmoid()
        self.layers = nn.ModuleList([])
        for x in layers : 
            self.layers.append(nn.Linear(inSize,x))
            inSize = x
        self.layers.append(nn.Linear(inSize,outSize))
    def forward(self,x):
        #print("start forward",x)

        x = self.layers[0](x)
        #print("before loop",x)
        for i in range(1,len(self.layers)):
            x  = self.activ(x)
            x = self.layers[i](x)
        #print("before softM",x)
        x=self.softM(x)
        return x


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__=="__main__":
    env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = 4
    render = True
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 1000        # max training episodes
    max_timesteps = 500         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    rewards = []
    lengths = []
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0


    reward_a2c=[]
    reward_random=[]
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        render=(i_episode % 100 == 0 and i_episode > 0)
        rsum=0
        for t in range(500):
            timestep += 1
                
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            rsum += reward
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
                
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        reward_a2c.append(rsum)

        avg_length += t
        #print(rsum)
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
                
        
        avg_length = int(avg_length/log_interval)
        running_reward = int((running_reward/log_interval))
        
        rewards.append(running_reward)
        lengths.append(avg_length)
        if i_episode % log_interval == 0:    
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0

    envm = gym.make('LunarLander-v2')
    envm.seed(0)
    agent = RandomAgent(env.action_space)
    envm.seed(0)

    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    print("allo")
    for i in range(max_episodes):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        reward_random.append(rsum)



    plt.figure()
    plt.plot(range(len(reward_a2c)),reward_a2c,label="PPO")
    plt.plot(range(len(reward_a2c)),reward_random,label="Random")
    plt.legend()
    plt.savefig("reward_ppo.png")

    a2c_cumu=[0]
    random_cumu=[0]
    for i in range(len(reward_a2c)):
        a2c_cumu.append(reward_a2c[i]+a2c_cumu[-1])
        random_cumu.append(reward_random[i]+random_cumu[-1])

    plt.figure()
    plt.plot(range(len(a2c_cumu)),a2c_cumu,label="PPO")
    plt.plot(range(len(a2c_cumu)),random_cumu,label="Random")
    plt.legend()
    plt.savefig("reward_cumu_ppo.png")    
    env.close()