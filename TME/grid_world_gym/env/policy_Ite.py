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
Algo de policy et value iteration.
"""
class Policy_Iteration(object):
    """Agent based on policy iteration !"""

    def __init__(self,env,discount,eps = 10e-2):
        self.env = env
        self.action_space = env.action_space
        states,transition = env.getMDP()
        self.pi = {s:np.random.randint(env.action_space.n) for s in states} # Initialisation de la politique
        while 1:
            continuee = False    
            V={s:random.random() for s in states}
            v_temp={s:0 for s in states}
            #print("V",V)
            #print("V.values()",V.values())
            while np.linalg.norm(np.array(list(V.values()),dtype=float)-np.array(list(v_temp.values()),dtype=float))>eps: #critere de convergence
                v_temp={s:0 for s in states}
                for s in states:
                    if s in transition.keys():
                        for i in range(len(transition[s][self.pi[s]])):
                            etat_accessible = transition[s][self.pi[s]][i][1]

                            for pr in transition[s][self.pi[s]]:
                                if etat_accessible==pr[1]:
                                    proba = pr[0]
                                    reward = pr[2]
                            v_temp[s]+=proba*(reward + discount*V[s])

                V=v_temp.copy()


            for s in states:
                if s in transition.keys():
                    evaluation_action=dict()
                    for action in list(transition[s].keys()):
                        somme=0
                        for i in range(len(transition[s][action])):
                            etat_accessible = transition[s][action][i][1]
                            for pr in transition[s][self.pi[s]]:
                                if etat_accessible==pr[1]:
                                    proba = pr[0]
                                    reward = pr[2]
                            somme+=proba*(reward+discount* V[etat_accessible])
                        evaluation_action[action]=somme
                    action_choisie=sorted(evaluation_action.items(),key=lambda x:x[1])[-1][0]
                    
                    if action_choisie!=self.pi[s]:
                        self.pi[s]=action_choisie
                        continuee = True
            if not continuee:
                break

        
    def act(self, observation, reward, done):
        
        etat=self.env.state2str(observation)
        action = self.pi[etat]
        return action



class Value_Iteration(object):
    """Agent based on policy iteration !"""

    def __init__(self,env,discount,eps = 10e-2):
        self.env = env
        self.action_space = env.action_space
        states,transition = env.getMDP()
        self.pi = {s:np.random.randint(env.action_space.n) for s in states} # Initialisation de la politique
           
        V={s:random.random() for s in states}
        v_temp={s:0 for s in states}
        #print("V",V)
        #print("V.values()",V.values())
        while np.linalg.norm(np.array(list(V.values()),dtype=float)-np.array(list(v_temp.values()),dtype=float))>eps: #critere de convergence
            v_temp={s:0 for s in states}
            for s in states:
                if s in transition.keys():
                    val_max = np.inf
                    action_optimal = None
                    for a in transition[s].keys():

                        for i in range(len(transition[s][self.pi[s]])):
                            etat_accessible = transition[s][self.pi[s]][i][1]
                            for pr in transition[s][self.pi[s]]:
                                if etat_accessible==pr[1]:
                                    proba = pr[0]
                                    reward = pr[2]
                            valeur_a_test=proba*(reward +discount * V[s])
                            if valeur_a_test>val_max:
                                val_max=valeur_a_test
                                action_optimal=a
                    v_temp[s]=action_optimal

            V=v_temp.copy()

        for s in states:
            if s in transition.keys():
                evaluation_action=dict()
                for action in transition[s].keys():
                    somme=0
                    for i in range(len(transition[s][self.pi[s]])):
                        etat_accessible = transition[s][self.pi[s]][i][1]
                        for pr in transition[s][self.pi[s]]:
                            if etat_accessible==pr[1]:
                                print('laaaa')
                                proba = pr[0]
                                reward = pr[2]
                        print(type(somme),type(proba),type(reward),type(discount),type(V[etat_accessible]))
                        somme+=proba*(reward+discount* V[etat_accessible])
                    evaluation_action[action]=somme
                action_choisie=sorted(evaluation_action.items(),key=lambda x:x[1])[-1][0]
                self.pi[s]=action_choisie

        
    def act(self, observation, reward, done):
        
        etat=self.env.state2str(observation)
        action = self.pi[etat]
        return action


if __name__ == "__main__":
    #env = gym.make()
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = Value_Iteration(env,discount=0.99,eps=10e-4)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
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
        

            



    
