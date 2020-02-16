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
class Policy_Iteration(object):

    def __init__(self, env):
        self.action_space = env.action_space.n
        self.statedic, self.mdp = env.getMDP()
        self.policy = dict()
        self.env = env
        
    def get_policy(self):
        return self.policy

    def fit(self, nIt = 10000, gamma = 0.99, epsilon = 1e-2):
        
        Pi = np.random.randint(self.action_space,size=len(self.statedic))
        
        for it in tqdm(range(nIt)) :

            Vs = []
            V = np.random.randn(len(self.statedic))+1 
            Vs.append(V)

            nb_while = 0
            while ((len(Vs)<2 or np.linalg.norm(V[-1]-V[-2]) <= epsilon) and nb_while < nIt) :
                V = Vs[-1]
                V_plus = np.random.randn(len(self.statedic)) 
                #l'array V_plus est modifié dans la boucle d'après 
                
                for tab, s in self.statedic.items() :
                    if tab in self.mdp.keys() :
                        transitions = self.mdp[tab]
                    else :
                        V_plus[s] = V[s]
                        pass

                    action = Pi[s]
                    v=0
                     
                    for prob, nextstate, reward, done in transitions[action]:
                        v+= prob * (reward + gamma * V[s])
                    V_plus[s] = v
                Vs.append(V_plus)
                nb_while +=1
            
            
            Pi_plus = np.random.randint(self.action_space,size=len(self.statedic)) 
            # L'array Pi_plus va être modifié dans la boucle d'après  
            for tab, s in self.statedic.items() :
                if tab in self.mdp.keys() :
                    transitions = self.mdp[tab]
                else :
                    Pi_plus[s] = Pi[s]
                    pass
                
                argmaxA = 0
                maxA = -999
                for action, tuple in transitions.items() :
                    somme = 0
                    for prob, nextstate, reward, done in transitions[action] :
                        nextstate = self.statedic[nextstate]
                        somme += prob * (reward + gamma * Vs[-1][nextstate])
                    
                    if somme > maxA :
                        argmaxA = action
                        maxA = somme
                Pi_plus[s] = argmaxA
            
            if (np.array_equiv(Pi,Pi_plus)):
                self.policy =  Pi
                break
            else :
                Pi = Pi_plus
        
        self.policy = Pi

    def act(self, observation, reward, done):
        
        #etat=self.env.state2str(observation)
        obs=self.statedic[gridworld.GridworldEnv.state2str(observation)]
        action = self.policy[obs]
        return action






if __name__ == "__main__":
    #env = gym.make()
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 2, 4: 1, 5: -1, 6: -1})
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
    agent = Policy_Iteration(env)
    agent.fit(gamma=0.99,epsilon=0.0001)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    

    # 0 case vide , #1 Mur , #2 Joueur, #3 case vert, #4 Jaune, #5 Rouge, #6 Rose 
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    all_rsum = []
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
    print("Rsum:",rsum)
    print("Moyenne des rsum:",np.array(all_rsum).mean())
    print("Variance des rsum",np.array(all_rsum).std())
    env.close()
    pass
        

            



    
