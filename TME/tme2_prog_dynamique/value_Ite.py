import numpy as np
import random
import gym
import sys
sys.path.insert(0, '/Users/ykarmim/Documents/Cours/Master/M2/RLADL/TME/grid_world_gym/env/gridworld')
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from tqdm import tqdm

"""
Algo de policy et value iteration.
"""
class ValueIteration(object) :
    def __init__(self, env):
        self.action_space = env.action_space.n
        self.statedic, self.mdp = env.getMDP()
        self.policy = dict()
    
    def get_policy(self):
        return self.policy

    def fit(self,gamma, epsilon,nIt = 30000):
        V = np.random.randn(len(self.statedic))+1 
        V_bis = np.random.randn(len(self.statedic))+1
        # On cherche V_bis en fonction de V
        for i in range(nIt) :

            for tab, s in self.statedic.items() :
                if tab in self.mdp.keys() :
                    transitions = self.mdp[tab]
                else :
                    V_bis[s] = V[s]
                    pass

                argmaxA = 0
                maxA = -999
                for action, tuple in transitions.items() :
                    somme = 0
                    for prob, nextstate, reward, done in transitions[action] :
                        nextstate = self.statedic[nextstate]
                        somme += prob * (reward + gamma * V[nextstate])
                    
                    if somme > maxA :
                        argmaxA = action
                        maxA = somme
                V_bis[s] = argmaxA
            
            if (np.linalg.norm(V_bis - V) <= epsilon):
                V = V_bis
                break
            else :
                V = V_bis
                V_bis = np.random.randn(len(self.statedic))+1
        print("I",i)


        Pi = np.random.randint(self.action_space,size=len(self.statedic))

        for tab, s in self.statedic.items() :
            if tab in self.mdp.keys() :
                transitions = self.mdp[tab]
            else : 
                pass
            argmaxA = 0
            maxA = -999
            for action, tuple in transitions.items() :
                somme = 0
                for prob, nextstate, reward, done in transitions[action] :
                    nextstate = self.statedic[nextstate]
                    somme += prob * (reward + gamma * V[nextstate])

                if somme > maxA :
                    argmaxA = action
                    maxA = somme
            Pi[s] = argmaxA

        self.policy = Pi

    def act(self, observation, reward, done):
        
        #etat=self.env.state2str(observation)
        obs=self.statedic[gridworld.GridworldEnv.state2str(observation)]
        action = self.policy[obs]
        return action




if __name__ == "__main__":
    #env = gym.make()
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.005, 3: 10, 4: 1, 5: -1, 6: -1})
    env.seed(0)  # Initialise le seed du pseudo-random
    #print(env.action_space)  # Quelles sont les actions possibles
    #print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    #print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    #print(state)  # un etat du mdp
    #print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = ValueIteration(env)
    agent.fit(gamma=0.99,epsilon=0.0001)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    all_rsum=[]
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 500 == 0 and i > 0)  # afficher 1 episode sur 100
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
        all_rsum.append(rsum)

    print("done")
    print("Moyenne des rsum:",np.array(all_rsum).mean())
    print("Variance des rsum",np.array(all_rsum).std())
    env.close()
    pass
        

            



    
