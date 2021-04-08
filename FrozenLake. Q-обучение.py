import gym
import numpy as np
env = gym.make("FrozenLake-v0")

gamma = 0.9
alpha = 0.2
epsilon = 1
total_reward = []
q = {}


def update_q_table(state, action, reward, new_state):
    qa = max(q[new_state, x] for x in range(env.action_space.n))
    q[(state, action)] += alpha*(reward + gamma*qa - q[(state, action)])


def epsilon_greedy_policy(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])
    return action


for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        q[(state, action)] = 0.0

count = 0
for i in range(2000):
    obs = env.reset()
    if i % 100 == 0 and count != 10:
        epsilon -= 0.1
        count += 1
    while True:
        #env.render()
        action = epsilon_greedy_policy(obs, epsilon)
        new_obs, reward, done, _ = env.step(action)
        update_q_table(obs, action, reward, new_obs)
        obs = new_obs
        if done:
            total_reward.append(reward)
            print("Mean reward %.2f, epsilon %.2f" % (np.mean(total_reward[-100:]), epsilon))
            break
