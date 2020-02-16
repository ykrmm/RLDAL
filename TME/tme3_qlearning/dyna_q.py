import numpy as np
import random
import gym
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/RLADL/TME/grid_world_gym/env/gridworld')
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import matplotlib.pyplot as plt

"""
Algo Dyna_Q
"""

class Dyna_Q(object):
    """Dyna_Q Agent"""

    def __init__(self,env,learning_rate,discount,planning_step,epsilon=0.1):
        self.env=env
        self.action_space=env.action_space
        self.Q = {}
        self.model = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None 
        self.lasta = None
        self.planning_step = planning_step
    def act(self, observation, reward, done):
        
        state=self.env.state2str(observation)
        self.obs = state           
        self.Q.setdefault(state,[0,0,0,0])
        self.model.setdefault(state,[(False,False),(False,False),(False,False),(False,False)])
        self.reward = reward
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q[self.obs])
        else : 
            action = np.random.randint(self.action_space.n)
        self.update_Q_Model(action) 
            
        return action

    def planning(self):
        for _ in range(min(self.planning_step,len(list(self.model.keys())))): # Au cas ou on a moins d'états dans notre modèle que de planning step.
            st = np.random.choice(list(self.model.keys()))
            action = np.random.randint(self.action_space.n)
            reward, st1 = self.model[st][action]
            cpt=0
            while reward==False and cpt<20:
                action = np.random.randint(self.action_space.n)
                reward, st1 = self.model[st][action]
                cpt+=1
            if st1 !=False:
                self.Q[st][action] += self.learning_rate*(reward +\
                    self.discount*np.max(self.Q[st1])-self.Q[st][action])


    def update_Q_Model(self,action):
        if not self.lastobs==None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate*(self.reward +\
                self.discount*np.max(self.Q[st1])-self.Q[st][self.lasta])
            self.model[st][self.lasta]=(self.reward,st1) # update model
            self.planning() # planning step
        self.lastobs = self.obs 
        self.lasta = action
        




if __name__ == "__main__":
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan10.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
   
    #statedic, mdp = env.getMDP()

    #    print(env.states) 

    # recupere le mdp : statedic
    #print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    #state, transitions = list(mdp.items())[0]
    #print(state)  # un etat du mdp
    #print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = Dyna_Q(env,learning_rate=10e-4,planning_step=5,discount=0.999,epsilon=0.1)
    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    
    env.seed()  # Initialiser le pseudo aleatoire

    
    episode_count = 2000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    all_rsum = []
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 1000 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        #rsum = 0
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
        all_rsum.append(rsum)

    print("done")
    all_rsum = np.array(all_rsum)
    print("Rsum moyen :",all_rsum.mean())
    env.close()
    np.save("dynaQ6.npy",all_rsum)
    pass
    plt.title("Reward cumulé Dyna_Q")
    plt.xlabel("iterations")
    plt.ylabel("reward cumulé")
    plt.plot(np.arange(episode_count), all_rsum, label="Dyna_Q")
    plt.legend()
    plt.show()
        

            



    
