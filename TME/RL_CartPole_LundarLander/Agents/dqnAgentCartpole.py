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
    def __init__(self,env,learning_rate,discount,batchsize,layers=[],loss=nn.SmoothL1Loss(),epsilon=0.1,capacity = 1000,c_step = 20):
        action_space=env.action_space
        self.env=env
        self.action_space = action_space # right or left
        #self.state = env.states
        #self.Q = np.zeros((len(self.state),self.action_space.n))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.discount = discount
        self.lastobs = None 
        self.lasta = None
        self.Q = NN_Q(4,outSize=self.action_space.n,layers=layers)
        self.loss = loss
        self.optim = torch.optim.Adam(self.Q.parameters(),lr=self.learning_rate)
        
        self.c_step = c_step # Step before Q_hat = Q 
        self.D = deque(maxlen=capacity) # transition history 
        self.Q_hat = NN_Q(4,outSize=self.action_space.n,layers=layers) # pour éviter les maj pdt le backward 
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.count=0
    def act(self, observation, reward, done):
        self.obs=observation
        #tmp=self.env.state2str(observation)
        #self.obs = self.env.states[tmp]
        self.reward = reward
        if self.lasta is not None: # On entraine pas si c'est on est à la première étape
            self.D.append((self.lastobs,self.lasta,self.reward,self.obs,done))
            els = np.random.choice(range(len(self.D)),self.batchsize)
            l = [self.D[i] for i in els]
            #print(l[0])
            lso,la,lr,ls1,ld = map(list,zip(*l))
            #print(ls1)
            lr_asT=torch.Tensor(lr)
            ld_asT=torch.BoolTensor(ld)
            lso_asT=torch.Tensor(lso)
            ls1_asT=torch.Tensor(ls1)
            print(ld_asT.shape)
            with torch.no_grad():
                print("ls1",ls1_asT.shape)
                q=self.Q_hat(ls1_asT)
                val,ind=torch.max(q[range(self.batchsize),la],0)
            
                ypred=torch.where(ld_asT,lr_asT,lr_asT+self.discount*val)

            print("ypred",ypred.shape)
            
            ytar=self.Q(lso_asT)
            val,ind=torch.max(ytar[range(self.batchsize),la],0)
            print("ytar",ytar.shape)
            loss=self.loss(ypred,val)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            # Entrainement neurone 
                # entraine
            #

        if np.random.random() < 1 - self.epsilon and self.lasta is not None: # Sinon au debut ça bug pour la première action
            tmp=torch.from_numpy(self.obs)
            print("tmp",tmp.shape)
            tmp=self.Q(tmp.float())
            action = torch.max(tmp) #marche pas ça
            print(action)
            
        else : # action random
            action = np.random.randint(self.action_space.n) 
        
        
        self.lastobs = self.obs 
        self.lasta = action
        self.count+=1
        if self.count==self.c_step:
            self.count=0
            self.Q_hat.load_state_dict(self.Q.state_dict())

        return action



if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = DQN_Agent(env,learning_rate=0.01,discount=0.9,batchsize=20,layers=[12,8])

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