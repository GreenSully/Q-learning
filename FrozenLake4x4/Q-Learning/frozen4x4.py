import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

env = gym.make('FrozenLake-v0')

action_space_size=env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


rewards_all_episodes = []
exploration_rate_list=[]

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    if episode%1000==0:
        exploration_rate_list.append(exploration_rate)
        print("Exploration rate",exploration_rate)
    for step in range(max_steps_per_episode):

        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)


        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward
        if done == True:
            break
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

print(q_table)
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

plt.plot([i for i in range(1,int(num_episodes/1000)+1)],[sum(i/1000) for i in rewards_per_thousand_episodes],label="success rate")
plt.plot([i for i in range(1,int(num_episodes/1000)+1)],exploration_rate_list,label="exploration rate")
plt.legend()
#plt.show()
plt.savefig("frozenLake4x4.png")
