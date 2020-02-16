import matplotlib.pyplot as plt

#matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = RandomAgent(env.action_space)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0
    all_rsum = []
    rsum_cumule = 0
    all_rsum_cumule = []

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
            rsum_cumule += reward
            
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        all_rsum_cumule.append(rsum_cumule)

    print("done")
    env.close()
    all_rsum = np.array(all_rsum)
    all_rsum_cumule = np.array(all_rsum_cumule)

    np.save("random_all_rsum.npy",all_rsum)
    np.save("random_all_rsum_cumule.npy",all_rsum_cumule)
    

    plt.title("Random Agent cartpole: reward cumulé")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.plot(np.arange(len(all_rsum_cumule)), all_rsum_cumule, label="Random")
    plt.legend()
    plt.show()