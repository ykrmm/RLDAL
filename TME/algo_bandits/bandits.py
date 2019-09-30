import numpy as np
import pandas as pd
import random 



with open("CTR.txt","r") as f:
    data = f.read()


data = pd.read_csv('CTR.txt', sep=":", header=None)
data.columns = ["num", "rpz", "annonceurs"]



def random_strat(data):
    reward = 0
    for i in range(len(data["num"])):
        choix = float(data["annonceurs"][i].split(";")[random.randint(0,9)])
        reward+=choix
    return reward

res = random_strat(data)

print("evaluation random :",res)
L = []
for i in range(5000):
    d = data["annonceurs"][i]
    d=d.split(";")
    d = [float(k) for k in d]
    L.append(d)

tab_ann = np.vstack(L)
def staticBest(tab_ann):
    somme = np.sum(tab_ann,axis=0)
    action = np.argmax(somme)
    return action



annonceurs = staticBest(tab_ann)
print("annonceur opt :",annonceurs)

def optimale(tab_ann,i):
    somme = np.sum(tab_ann[0:i+1],axis=0)
    action = np.argmax(somme)
    return action

annonceurs = optimale(tab_ann,5)
print("annonceur opt :",annonceurs)


def ucb(u,s,t):
    return np.argmax(u+np.sqrt(2*np.log(t)/s))

def evaluate_ucb(tab_ann):
    u=np.zeros(tab_ann.shape[1])
    s=np.zeros(tab_ann.shape[1])

    for i in range(len(s)):
        s[i]+=1
        u[i]+=tab_ann[i][i]
    for i in range(len(s)+1,len(tab_ann)):
        choix=ucb(u,s,i)
        u[choix]+=tab_ann[i][choix]
        s[choix]+=1
        
