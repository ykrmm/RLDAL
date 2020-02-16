import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
from torch import nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')

#https://pastebin.com/ZH4gEELb

class NN_V(torch.nn.Module):
    def __init__(self,inSize,outSize,layers = []):
        super().__init__()
        self.activ=nn.LeakyReLU()
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


class A2C_Agent(object):
    def __init__(self,env,learning_rate_V,learning_rate_Pi,discount,layers=[],loss=nn.SmoothL1Loss(),T_max=200):
        action_space=env.action_space
        self.env=env
        self.action_space = action_space # right or left

        self.learning_rate_V = learning_rate_V
        self.learning_rate_Pi= learning_rate_Pi
        self.discount = discount
        self.lastobs = None 
        self.lasta = None
        self.V = NN_V(4,outSize=1,layers=[100]).to(device)
        self.loss = loss
        self.loss=nn.MSELoss()
        self.optimV = torch.optim.Adam(self.V.parameters(),lr=self.learning_rate_V)
        
        self.pi=NN_Pi(4,outSize=self.action_space.n,layers=layers).to(device)
        self.optimPi=torch.optim.Adam(self.pi.parameters(),lr=self.learning_rate_Pi)


        
        self.stock_states=[]
        self.T_max=T_max
        self.t=0
        self.t_start=self.t

    def act(self, observation, reward, done):
        
        self.obs=observation
        self.reward = reward
        tmp=torch.from_numpy(self.obs)
        #print(tmp)
        tmp=self.pi(tmp.float().to(device))
        #print(tmp)
        try:
            m=torch.distributions.categorical.Categorical(probs=tmp)
            action=m.sample()
            action=action.cpu().item()
        #print(action)
        except Exception as e:
            print(e)
            print("tmp",tmp)
        if self.lasta is not None:
            self.stock_states.insert(0,{"state":torch.from_numpy(self.obs),"reward":self.reward,"action":self.lasta})
        self.lasta=action
        self.lastobs=obs
        self.t+=1
        if self.t-self.t_start==self.T_max:
            self.update(terminal=False)
            self.t_start=self.t
        return action


    def update(self,terminal):


        if terminal:
            R=torch.zeros(1).to(device)
        else:
            with torch.no_grad():
                R=self.V(self.stock_states[0]['state'].float().to(device))

        lossV=0
        lossPi=0
        for i in range(1,len(self.stock_states)):
            #print("R=",R)
            res_V=self.V(self.stock_states[i]["state"].float().to(device))
            #print("res_V",res_V)

            #gradV=(R-res_V).pow(2)
            gradV=self.loss(R.detach(),res_V)
            lossV+=gradV
            #print("R- resV",R-res_V)

            res_Pi=self.pi(self.stock_states[i]["state"].float().to(device))
            #print("res_Pi",res_Pi)
            #print("choix res_Pi",res_Pi[self.stock_states[i]['action']])
            #print("log du choix",res_Pi[self.stock_states[i]['action']].log())
            gradPi= res_Pi[self.stock_states[i]['action']].log()*(R.detach()-res_V.detach())
            lossPi+=gradPi
            R=self.discount*R+self.stock_states[i]["reward"]

            #print("loop ",i," lossPi=",lossPi)
        #print("terminal ?",terminal)
        #print("lossPi",lossPi)
        if type(lossPi) is not int:
            lossPi.backward()
            self.optimPi.step()
            self.optimPi.zero_grad()
        
        if type(lossV) is not int:
            lossV.backward()
            self.optimV.step()
            self.optimV.zero_grad()

        self.stock_states=[]
        




if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    #agent = #DQN_Agent(env.action_space)
    agent = A2C_Agent(env,learning_rate_V=0.01,learning_rate_Pi=0.001,discount=0.99,layers=[200])

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    episode_count=20000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        agent.t=0
        agent.t_start=0
        agent.lasta=None
        agent.lastobs=None
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
                agent.update(terminal=True)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()