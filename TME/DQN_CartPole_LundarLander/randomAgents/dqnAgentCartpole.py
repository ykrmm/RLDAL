import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from collections import deque # self.D = deque(maxlen=capacity) self.D.append((st,at,st1,rt,done))
import torch
from torch import nn


class NN_Q(torch.nn.Module):
    def __init__(self,inSize,outSize,layers = []):
        super(NN_Q,self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers : 
            self.layers.append(nn.Linear(inSize,x))
            inSize = x
        self.layers.append(nn.Linear(inSize,outSize))
    def forward(self,x):
        x = self.layers[0](x)
        for i in range(1,len(self.layers)):
            x  = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x

class DQN_Agent(object):
    def __init__(self,env,learning_rate,discount,inSize,batchsize,layers=[],loss=nn.SmoothL1Loss(),epsilon=0.1,capacity = 1000,c_step = 20):
        action_space=env.action_space
        self.env=env
        self.action_space = action_space # right or left
        self.state = env.states
        self.Q = np.zeros((len(self.state),self.action_space.n))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.discount = discount
        self.lastobs = None 
        self.lasta = None
        self.model = NN_Q(inSize,outSize=self.action_space.n,layers=layers)
        self.loss = loss
        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        
        self.c_step = c_step # Step before Q_hat = Q 
        self.D = deque(maxlen=capacity) # transition history 
        self.Q_hat = None

    def act(self, observation, reward, done):
        
        tmp=self.env.state2str(observation)
        self.obs = self.env.states[tmp]
        self.reward = reward
        if self.lasta is not None: # On entraine pas si c'est on est à la première étape
            self.D.append((self.lastobs,self.lasta,self.reward,self.obs,done))
            els = np.random.choice(range(len(self.D)),self.batchsize)
            l = [self.D[i] for i in els]
            lso,la,ls1,lr,ld = map(list,zip(*l))
            # Entrainement neurone 
                # entraine
            #

        if np.random.random() < 1 - self.epsilon and self.lasta is not None: # Sinon au debut ça bug pour la première action
            _,action = torch.max(self.model(self.obs)) 
            
        else : # action random
            action = np.random.randint(self.action_space.n) 
        
        
        self.lastobs = self.obs 
        self.lasta = action
        return action



if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    #agent = #DQN_Agent(env.action_space)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
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

    print("done")
    env.close()