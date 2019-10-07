import numpy as np
import random
import gym
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/RLADL/TME/grid_world_gym/env/gridworld')
import gridworld
from gym import wrappers, logger
import numpy as np
import copy

"""
Algo Q learning
"""

class Q_Learning(object):
    """The world's simplest agent!"""

    def __init__(self,env,learning_rate,discount,epsilon=0.1):
        action_space=env.action_space
        self.env=env
        self.action_space = action_space
        self.state = env.states
        self.Q = np.zeros((len(self.state),self.action_space.n))
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None 
        self.lasta = None
    def act(self, observation, reward, done):
        
        tmp=self.env.state2str(observation)
        self.obs = self.env.states[tmp]
        self.reward = reward
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q[self.obs])
        else : 
            action = np.random.randint(self.action_space.n)

        self.update_Q(action)
        return action

    def update_Q(self,action):
        if not self.lastobs==None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate*(self.reward +\
                self.discount*self.Q[st1][action]-self.Q[st][self.lasta])
        
        self.lastobs = self.obs 
        self.lasta = action
        




if __name__ == "__main__":
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.03, 3: 1, 4: 20, 5: -1, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
   
    statedic, mdp = env.getMDP()

#    print(env.states) 

    # recupere le mdp : statedic
    #print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    #state, transitions = list(mdp.items())[0]
    #print(state)  # un etat du mdp
    #print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = Q_Learning(env,learning_rate=10e-4,discount=0.999,epsilon=0.15)
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    
    env.seed()  # Initialiser le pseudo aleatoire

    
    episode_count = 3000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
    pass
        

            



    
